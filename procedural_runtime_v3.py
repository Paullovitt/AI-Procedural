from __future__ import annotations
from pathlib import Path
from collections import Counter, defaultdict
import json, lzma, math, re, random

WORD_RE = re.compile(r"[A-Za-zÀ-ÖØ-öø-ÿ0-9_]+(?:['’][A-Za-zÀ-ÖØ-öø-ÿ0-9_]+)?", re.UNICODE)
SLOT_RE = re.compile(r"^(?:e|a|v|r|x|obj|attr|val|rel)\d+$", re.I)

class BagacoSurfaceScorer:
    """Non-neural backoff surface scorer over the persisted Bagaço tables."""
    def __init__(self, root: str | Path, use_hot: bool = False):
        self.root = Path(root)
        self.use_hot = use_hot
        if use_hot:
            p = self.root / 'bagaco_model_gpu' / 'bagaco_hot_runtime.json.xz'
            with lzma.open(p, 'rt', encoding='utf8') as f:
                hot = json.load(f)
            tabs = hot['tables']
            self.tables = {k:{x['k']:int(x['n']) for x in v} for k,v in tabs.items()}
            self.struct = hot.get('quality_structural', {})
        else:
            q = self.root / 'model' / 'quality'
            self.tables = {}
            for name in ['tokens','p2','p3','p4','p5','open','close','connect']:
                d = {}
                with (q/f'{name}.jsonl').open(encoding='utf8') as f:
                    for line in f:
                        o=json.loads(line); d[o['k']]=int(o['n'])
                self.tables[name]=d
            with (q/'structural.json').open(encoding='utf8') as f:
                st=json.load(f)
            # structural.json from training uses mapping/hist forms. Normalize sentence hist.
            self.struct=st
        self.tok=self.tables['tokens']; self.p2=self.tables['p2']; self.p3=self.tables['p3']
        self.p4=self.tables.get('p4',{}); self.p5=self.tables.get('p5',{})
        self.total_tok=max(1,sum(self.tok.values())); self.V=max(1000,len(self.tok))
        self.sent_hist=self._sentence_hist()
        self.target_mean,self.target_median,self.target_p90=self._hist_summary(self.sent_hist)

    def _sentence_hist(self):
        if 'sent' in self.struct and isinstance(self.struct['sent'],list): return self.struct['sent']
        for key in ['sent_len','sentence_tokens']:
            if key in self.struct:
                x=self.struct[key]
                if isinstance(x,list): return x
                if isinstance(x,dict):
                    m=max(map(int,x)) if x else 0; h=[0]*(m+1)
                    for k,v in x.items(): h[int(k)]=int(v)
                    return h
        return []

    @staticmethod
    def _hist_summary(h):
        if not h: return (16.0,13.0,35.0)
        tot=sum(h); mean=sum(i*c for i,c in enumerate(h))/max(1,tot); cum=0; med=p90=0
        for i,c in enumerate(h):
            cum+=c
            if not med and cum>=.5*tot: med=i
            if not p90 and cum>=.9*tot: p90=i; break
        return mean,float(med),float(p90)

    @staticmethod
    def tokenize(text: str):
        return [x.lower() for x in WORD_RE.findall(text)]

    @staticmethod
    def is_slot(tok: str):
        return bool(SLOT_RE.match(tok))

    def score_tokens(self, words, max_order=4, slot_aware=True):
        """Average conditional log score. Unknown protected slots do not affect language score."""
        ws=list(words); total=0.0; weight=0.0
        valid=[not(slot_aware and self.is_slot(w)) for w in ws]
        for i,w in enumerate(ws):
            if not valid[i]: continue
            c=self.tok.get(w,0)
            total += .10*math.log((c+.5)/(self.total_tok+.5*self.V)); weight += .10
        tables={2:self.p2,3:self.p3,4:self.p4,5:self.p5}
        weights={2:1.0,3:.75,4:.45,5:.25}
        for n in range(2,min(max_order,5)+1):
            tab=tables[n]
            if not tab: continue
            pref_tab=self.tok if n==2 else tables[n-1]
            for i in range(n-1,len(ws)):
                idx=range(i-n+1,i+1)
                if not all(valid[j] for j in idx): continue
                gram='\t'.join(ws[i-n+1:i+1]); pref='\t'.join(ws[i-n+1:i])
                c=tab.get(gram,0); pc=pref_tab.get(pref,0)
                alpha=.05 if n==2 else .02
                if pc:
                    lp=math.log((c+alpha)/(pc+alpha*self.V))
                else:
                    lp=math.log(alpha/(self.total_tok+alpha*self.V))
                wgt=weights[n]; total += wgt*lp; weight += wgt
        return total/max(weight,1e-9)

    def score(self,text,max_order=4,slot_aware=True):
        return self.score_tokens(self.tokenize(text),max_order=max_order,slot_aware=slot_aware)

    def length_logprior(self,n):
        if not self.sent_hist or n>=len(self.sent_hist):
            # smooth tail around corpus p90
            return -abs(n-self.target_mean)/max(8.0,self.target_mean)
        tot=sum(self.sent_hist); c=self.sent_hist[n]
        return math.log((c+1)/(tot+len(self.sent_hist)))

    def supported_fraction(self,text,order=3,slot_aware=True):
        ws=self.tokenize(text); valid=[not(slot_aware and self.is_slot(w)) for w in ws]
        tab={2:self.p2,3:self.p3,4:self.p4,5:self.p5}[order]
        hit=den=0
        for i in range(order-1,len(ws)):
            ids=range(i-order+1,i+1)
            if not all(valid[j] for j in ids): continue
            den+=1; hit+=int('\t'.join(ws[i-order+1:i+1]) in tab)
        return hit/max(1,den)

class LearnedSurfaceSelector:
    """Scores candidate realizations, preserving semantic slots and penalizing local repetition."""
    def __init__(self, scorer: BagacoSurfaceScorer, seed=12345):
        self.s=scorer; self.rng=random.Random(seed)

    def choose(self,candidates,recent_openings=()):
        best=None
        recent=Counter(recent_openings)
        for text,meta in candidates:
            ws=self.s.tokenize(text); n=len(ws)
            lang=self.s.score_tokens(ws,max_order=4,slot_aware=True)
            lp=.22*self.s.length_logprior(n)
            opening=' '.join(w for w in ws[:3] if not self.s.is_slot(w))
            rep=.22*recent.get(opening,0)
            support=.30*self.s.supported_fraction(text,3,True)
            score=lang+lp+support-rep
            row=(score,self.rng.random(),text,meta,lang,lp,support,rep)
            if best is None or row[:2]>best[:2]: best=row
        return best

PROPERTY_TEMPLATES=[
    "No caso de {s}, o valor de {p} é {v}.",
    "Em relação a {s}, {p} corresponde a {v}.",
    "Para {s}, {p} é {v}.",
    "{s} apresenta {p} com valor {v}.",
    "Quanto a {s}, {p} assume {v}.",
    "Em {s}, {p} tem o valor {v}.",
    "O valor de {p} em {s} corresponde a {v}.",
]
PAIR_TEMPLATES=[
    "No caso de {s}, {p1} é {v1} e {p2} é {v2}.",
    "{s} apresenta {p1} como {v1} e {p2} como {v2}.",
    "Em relação a {s}, {p1} corresponde a {v1} e {p2} a {v2}.",
    "Para {s}, os valores de {p1} e {p2} são {v1} e {v2}, respetivamente.",
]
REL_TEMPLATES=[
    "Entre {a} e {b}, observa-se {r}.",
    "{a} apresenta a relação {r} com {b}.",
    "A relação {r} liga {a} a {b}.",
    "No caso de {a}, {r} estabelece ligação com {b}.",
    "{a} e {b} estão associados por {r}.",
    "Relativamente a {a}, {r} aponta para {b}.",
]

class SemanticRendererV2:
    """Generic symbolic renderer. Grammar family is generic/programmed; selection/style is corpus-scored."""
    def __init__(self,selector:LearnedSurfaceSelector): self.sel=selector
    @staticmethod
    def group_plan(facts):
        # Generic entity focus: properties by subject, then relations by source.
        by=defaultdict(list)
        for f in facts:
            key=f[1] if f[0]=='prop' else f[1]
            by[key].append(f)
        order=[]
        for k in sorted(by): order.extend(by[k])
        return order

    def render(self,facts,bundle=True):
        order=self.group_plan(facts); i=0; sentences=[]; represented=[]; recent=[]; picks=[]
        while i<len(order):
            f=order[i]
            cands=[]
            if f[0]=='prop':
                _,s,p,v=f
                for ti,t in enumerate(PROPERTY_TEMPLATES):
                    cands.append((t.format(s=s,p=p,v=v),{'facts':[f],'template':f'p{ti}'}))
                if bundle and i+1<len(order):
                    g=order[i+1]
                    if g[0]=='prop' and g[1]==s:
                        _,_,p2,v2=g
                        for ti,t in enumerate(PAIR_TEMPLATES):
                            cands.append((t.format(s=s,p1=p,v1=v,p2=p2,v2=v2),{'facts':[f,g],'template':f'pp{ti}'}))
            else:
                _,a,r,b=f
                for ti,t in enumerate(REL_TEMPLATES):
                    cands.append((t.format(a=a,r=r,b=b),{'facts':[f],'template':f'r{ti}'}))
            pick=self.sel.choose(cands,recent)
            score,_,text,meta,lang,lp,support,rep=pick
            sentences.append(text); represented.extend(meta['facts']); picks.append(pick)
            ws=self.sel.s.tokenize(text); opening=' '.join(w for w in ws[:3] if not self.sel.s.is_slot(w)); recent=(recent+[opening])[-12:]
            i += len(meta['facts'])
        return {'text':' '.join(sentences),'sentences':sentences,'represented':represented,'picks':picks}

def make_world(seed=1,n_entities=12,n_props=8,n_rels=4,n_facts=120):
    rng=random.Random(seed); ents=[f'e{i:03d}' for i in range(n_entities)]; props=[f'a{i:02d}' for i in range(n_props)]; vals=[f'v{i:02d}' for i in range(24)]; rels=[f'r{i:02d}' for i in range(n_rels)]
    facts=[]; seen=set()
    while len(facts)<n_facts:
        if rng.random()<.72:
            f=('prop',rng.choice(ents),rng.choice(props),rng.choice(vals))
        else:
            a,b=rng.sample(ents,2); f=('rel',a,rng.choice(rels),b)
        if f not in seen: seen.add(f); facts.append(f)
    return facts

def _join_items(items):
    if not items: return ''
    if len(items)==1: return items[0]
    if len(items)==2: return items[0]+' e '+items[1]
    return ', '.join(items[:-1])+' e '+items[-1]

def property_bundle_candidates(bundle):
    """Generic candidate family for N property facts sharing one subject."""
    s=bundle[0][1]
    pairs=[(f[2],f[3]) for f in bundle]
    fam=[]
    # Family 0: copular list
    clauses=[f'{p} é {v}' for p,v in pairs]
    fam.append((f'No caso de {s}, '+_join_items(clauses)+'.','b0'))
    # Family 1: correspondence; later clauses omit repeated verb naturally
    clauses=[f'{pairs[0][0]} corresponde a {pairs[0][1]}']+[f'{p} a {v}' for p,v in pairs[1:]]
    fam.append((f'Em relação a {s}, '+_join_items(clauses)+'.','b1'))
    # Family 2: presents/as
    clauses=[f'{p} como {v}' for p,v in pairs]
    fam.append((f'{s} apresenta '+_join_items(clauses)+'.','b2'))
    # Family 3: value list
    clauses=[f'{p} com valor {v}' for p,v in pairs]
    fam.append((f'Para {s}, '+_join_items(clauses)+'.','b3'))
    # Family 4: explicit values
    clauses=[f'o valor de {p} é {v}' for p,v in pairs]
    fam.append((f'Quanto a {s}, '+_join_items(clauses)+'.','b4'))
    return [(t,{'facts':list(bundle),'template':name}) for t,name in fam]

class CorpusStructurePlanner:
    """Chooses semantic bundle size from corpus sentence-length profile only; no lexical score."""
    def __init__(self, scorer:BagacoSurfaceScorer, max_bundle=4):
        self.s=scorer; self.max_bundle=max_bundle
        self.target=int(round(scorer.target_median or scorer.target_mean))

    def bundle(self, facts):
        # Same generic focus ordering, then bundle only facts with identical subject/type.
        by=defaultdict(list)
        for f in facts: by[f[1]].append(f)
        ordered=[]
        for key in sorted(by): ordered.extend(by[key])
        groups=[];i=0
        while i<len(ordered):
            f=ordered[i]
            if f[0]!='prop': groups.append([f]);i+=1;continue
            same=[];j=i
            while j<len(ordered) and ordered[j][0]=='prop' and ordered[j][1]==f[1] and len(same)<self.max_bundle:
                same.append(ordered[j]);j+=1
            # choose k solely by closeness of a neutral structural realization to learned median
            best=(10**9,1)
            for k in range(1,len(same)+1):
                probe=property_bundle_candidates(same[:k])[0][0]
                L=len(self.s.tokenize(probe))
                cand=(abs(L-self.target),-k,k)
                if cand<best: best=cand
            k=best[-1];groups.append(same[:k]);i+=k
        return groups

class SemanticRendererV3:
    """Separation of concerns: corpus structure planner fixes bundle; surface scorer only reranks equivalent wordings."""
    def __init__(self, selector:LearnedSurfaceSelector, structure:CorpusStructurePlanner):
        self.sel=selector;self.structure=structure
    def render(self,facts):
        groups=self.structure.bundle(facts);sentences=[];represented=[];picks=[];recent=[]
        for g in groups:
            if g[0][0]=='prop':
                cands=property_bundle_candidates(g)
            else:
                f=g[0];_,a,r,b=f
                cands=[(t.format(a=a,r=r,b=b),{'facts':[f],'template':f'r{ti}'}) for ti,t in enumerate(REL_TEMPLATES)]
            pick=self.sel.choose(cands,recent)
            sentences.append(pick[2]); represented.extend(pick[3]['facts']); picks.append(pick)
            ws=self.sel.s.tokenize(pick[2]); opening=' '.join(w for w in ws[:3] if not self.sel.s.is_slot(w));recent=(recent+[opening])[-12:]
        return {'text':' '.join(sentences),'sentences':sentences,'represented':represented,'picks':picks,'groups':groups}

def relation_bundle_candidates(bundle):
    """Neutral directional realization for N relation facts sharing the same source."""
    a=bundle[0][1]
    parts=[(f[2],f[3]) for f in bundle]
    # Keep direction explicit: source -> target.
    c0=[f'de {a} para {b}, regista-se {r}' for r,b in parts]
    c1=[f'{r} de {a} para {b}' for r,b in parts]
    c2=[f'entre {a} e {b}, regista-se {r}' for r,b in parts]
    fam=[
        ('; '.join(c0)+'.','rb0'),
        ('Registam-se '+_join_items(c1)+'.','rb1'),
        ('; '.join(c2)+'.','rb2'),
    ]
    return [(t,{'facts':list(bundle),'template':name}) for t,name in fam]

class CorpusStructurePlannerV2(CorpusStructurePlanner):
    """Corpus-driven bundle planner for both property and directional-relation facts."""
    def bundle(self,facts):
        by=defaultdict(lambda:{'prop':[],'rel':[]})
        for f in facts: by[f[1]][f[0]].append(f)
        ordered=[]
        for key in sorted(by):
            ordered.extend(by[key]['prop']); ordered.extend(by[key]['rel'])
        groups=[];i=0
        while i<len(ordered):
            f=ordered[i]; typ=f[0]; same=[];j=i
            while j<len(ordered) and ordered[j][0]==typ and ordered[j][1]==f[1] and len(same)<self.max_bundle:
                same.append(ordered[j]);j+=1
            best=None
            for k in range(1,len(same)+1):
                probe=(property_bundle_candidates(same[:k])[0][0] if typ=='prop' else relation_bundle_candidates(same[:k])[0][0])
                L=len(self.s.tokenize(probe)); cand=(abs(L-self.target),-k,k)
                if best is None or cand<best: best=cand
            k=best[-1];groups.append(same[:k]);i+=k
        return groups

class SemanticRendererV4(SemanticRendererV3):
    def render(self,facts):
        groups=self.structure.bundle(facts);sentences=[];represented=[];picks=[];recent=[]
        for g in groups:
            cands=property_bundle_candidates(g) if g[0][0]=='prop' else relation_bundle_candidates(g)
            pick=self.sel.choose(cands,recent)
            sentences.append(pick[2]); represented.extend(pick[3]['facts']); picks.append(pick)
            ws=self.sel.s.tokenize(pick[2]); opening=' '.join(w for w in ws[:3] if not self.sel.s.is_slot(w));recent=(recent+[opening])[-12:]
        return {'text':' '.join(sentences),'sentences':sentences,'represented':represented,'picks':picks,'groups':groups}

class LearnedSurfaceSelectorV2(LearnedSurfaceSelector):
    def __init__(self, scorer, seed=12345, length_weight=.22, support_weight=.30, repetition_weight=.22):
        super().__init__(scorer,seed)
        self.length_weight=length_weight;self.support_weight=support_weight;self.repetition_weight=repetition_weight
    def choose(self,candidates,recent_openings=()):
        best=None; recent=Counter(recent_openings)
        for text,meta in candidates:
            ws=self.s.tokenize(text);n=len(ws)
            lang=self.s.score_tokens(ws,max_order=4,slot_aware=True)
            lp=self.length_weight*self.s.length_logprior(n)
            opening=' '.join(w for w in ws[:3] if not self.s.is_slot(w))
            rep=self.repetition_weight*recent.get(opening,0)
            support=self.support_weight*self.s.supported_fraction(text,3,True)
            score=lang+lp+support-rep
            row=(score,self.rng.random(),text,meta,lang,lp,support,rep)
            if best is None or row[:2]>best[:2]:best=row
        return best

class LocalFluencyVerifier:
    """Detects likely local word-order anomalies; does not edit text or semantic slots."""
    def __init__(self, scorer:BagacoSurfaceScorer, threshold=.65, window=5):
        self.s=scorer; self.threshold=float(threshold); self.window=int(window)
    def gain_best_adjacent_swap(self, words):
        base=self.s.score_tokens(words,max_order=3,slot_aware=False);best=base;best_i=None
        for i in range(len(words)-1):
            if words[i]==words[i+1]: continue
            x=list(words);x[i],x[i+1]=x[i+1],x[i]
            v=self.s.score_tokens(x,max_order=3,slot_aware=False)
            if v>best:best=v;best_i=i
        return best-base,best_i
    def inspect_tokens(self, words):
        alarms=[]
        for start in range(0,max(0,len(words)-self.window+1)):
            win=list(words[start:start+self.window])
            if any(self.s.is_slot(w) for w in win): continue
            gain,idx=self.gain_best_adjacent_swap(win)
            if gain>self.threshold:
                alarms.append({'start':start,'gain':gain,'swap_local_index':idx,'window':win})
        return alarms
    def inspect(self,text):
        return self.inspect_tokens(self.s.tokenize(text))

class ConstructionMiner:
    """Discovers one-slot phrase frames by anti-unifying observed n-grams. No POS tags or semantic labels."""
    def __init__(self,min_variants=4,min_support=100):
        self.min_variants=min_variants;self.min_support=min_support;self.frames={}
    @staticmethod
    def skeleton(tokens,pos):
        x=list(tokens);x[pos]='*';return tuple(x)
    def fit(self, items):
        # items: iterable[(phrase,count)]
        raw={}
        for phrase,count in items:
            toks=tuple(phrase.split('\t'))
            for pos in range(len(toks)):
                sk=self.skeleton(toks,pos)
                d=raw.get(sk)
                if d is None:d={'support':0,'fillers':Counter(),'pos':pos};raw[sk]=d
                d['support']+=int(count);d['fillers'][toks[pos]]+=int(count)
        frames={}
        for sk,d in raw.items():
            v=len(d['fillers']);sup=d['support']
            if v<self.min_variants or sup<self.min_support:continue
            vals=list(d['fillers'].values());z=sum(vals);H=-sum((c/z)*math.log(c/z) for c in vals)
            Hn=H/max(math.log(v),1e-9)
            score=math.log1p(sup)*math.log1p(v)*Hn
            frames[sk]={'support':sup,'variants':v,'entropy':Hn,'score':score,'fillers':set(d['fillers'])}
        self.frames=frames;return self
    def matches(self, phrase):
        toks=tuple(phrase.split('\t'));out=[]
        for pos in range(len(toks)):
            sk=self.skeleton(toks,pos);d=self.frames.get(sk)
            if d is not None:out.append((pos,sk,d,toks[pos] not in d['fillers']))
        return out
    def top(self,n=50):
        return sorted(self.frames.items(),key=lambda kv:kv[1]['score'],reverse=True)[:n]

# ========================= v3: empirical long-tail discourse =========================
class EmpiricalLengthScheduler:
    """Deterministic low-discrepancy sampler over the learned sentence-length histogram."""
    def __init__(self, hist, seed=12345, q_low=0.05, q_high=0.99):
        self.hist=list(hist); self.seed=int(seed); self.q_low=float(q_low); self.q_high=float(q_high)
        total=max(1,sum(self.hist)); self.cdf=[]; c=0
        for n in self.hist:
            c+=n; self.cdf.append(c/total)
        self.phi=0.6180339887498949
        self.offset=((self.seed*0.7548776662466927)%1.0)
        self.i=0
    def _quantile(self,q):
        import bisect
        return bisect.bisect_left(self.cdf,min(max(q,0.0),1.0))
    def next(self):
        # Quasi-random sequence covers the empirical distribution more evenly than PRNG in small samples.
        u=(self.offset+self.i*self.phi)%1.0; self.i+=1
        q=self.q_low+(self.q_high-self.q_low)*u
        return max(1,self._quantile(q))


def _property_clause(f, style=0):
    _,s,p,v=f
    fam=(f'{p} é {v}', f'{p} corresponde a {v}', f'{p} tem o valor {v}', f'o valor de {p} é {v}')
    return fam[style%len(fam)]

def _relation_clause(f, style=0):
    _,a,r,b=f
    # Direction is always explicit in every family.
    fam=(f'{r} de {a} para {b}', f'de {a} para {b}, regista-se {r}', f'{r} liga {a} a {b}')
    return fam[style%len(fam)]

def focus_bundle_candidates(bundle):
    """Equivalent realizations for any facts sharing one focus/source. No rhetorical relation is invented."""
    focus=bundle[0][1]
    out=[]
    # Families differ only in syntax/punctuation; semantic clauses stay explicit.
    for style in range(4):
        clauses=[_property_clause(f,style) if f[0]=='prop' else _relation_clause(f,style) for f in bundle]
        # Three generic structural organizations; corpus scorer decides among them.
        if len(clauses)==1:
            bodies=[clauses[0]]
        else:
            bodies=[_join_items(clauses), '; '.join(clauses), ', '.join(clauses[:-1])+', e '+clauses[-1]]
        prefixes=(f'Quanto a {focus}, ', f'No caso de {focus}, ', f'Em relação a {focus}, ', f'Para {focus}, ')
        for bi,body in enumerate(bodies):
            text=prefixes[style]+body+'.'
            out.append((text,{'facts':list(bundle),'template':f'fb{style}_{bi}','focus':focus}))
    # De-duplicate exact strings while preserving order.
    seen=set(); uniq=[]
    for x in out:
        if x[0] not in seen: seen.add(x[0]); uniq.append(x)
    return uniq

class LearnedSurfaceSelectorV3(LearnedSurfaceSelectorV2):
    """Adds corpus opening prior to the existing non-neural n-gram scorer."""
    def __init__(self, scorer, seed=12345, length_weight=.16, support_weight=.34,
                 repetition_weight=.28, opening_weight=.12):
        super().__init__(scorer,seed,length_weight,support_weight,repetition_weight)
        self.opening_weight=float(opening_weight)
        self._open_total=max(1,sum(self.s.tables.get('open',{}).values()))
    def opening_score(self, words):
        # Longest prefix before the first protected semantic slot, max 4 tokens.
        p=[]
        for w in words[:5]:
            if self.s.is_slot(w): break
            p.append(w)
        best=0.0
        tab=self.s.tables.get('open',{})
        for n in range(1,min(4,len(p))+1):
            c=tab.get('\t'.join(p[:n]),0)
            if c: best=max(best, math.log1p(c)/math.log1p(self._open_total))
        return best
    def choose(self,candidates,recent_openings=()):
        best=None; recent=Counter(recent_openings)
        for text,meta in candidates:
            ws=self.s.tokenize(text);n=len(ws)
            lang=self.s.score_tokens(ws,max_order=4,slot_aware=True)
            lp=self.length_weight*self.s.length_logprior(n)
            opening=' '.join(w for w in ws[:3] if not self.s.is_slot(w))
            rep=self.repetition_weight*recent.get(opening,0)
            support=self.support_weight*self.s.supported_fraction(text,3,True)
            op=self.opening_weight*self.opening_score(ws)
            score=lang+lp+support+op-rep
            row=(score,self.rng.random(),text,meta,lang,lp,support,rep,op)
            if best is None or row[:2]>best[:2]:best=row
        return best

class EmpiricalStructurePlanner:
    """Uses the corpus length distribution to decide bundle size; only same-focus facts may share a sentence."""
    def __init__(self, scorer:BagacoSurfaceScorer, seed=12345, max_bundle=10, q_low=.05, q_high=.99):
        self.s=scorer; self.max_bundle=int(max_bundle)
        self.schedule=EmpiricalLengthScheduler(scorer.sent_hist,seed,q_low,q_high)
    def bundle(self,facts):
        by=defaultdict(list)
        for f in facts: by[f[1]].append(f)
        groups=[]; targets=[]
        for focus in sorted(by):
            fs=by[focus]; i=0
            while i<len(fs):
                target=self.schedule.next(); remain=fs[i:i+self.max_bundle]
                best=None
                for k in range(1,len(remain)+1):
                    # Structural choice is independent of lexical scorer: median length across equivalent candidates.
                    lens=sorted(len(self.s.tokenize(t)) for t,_ in focus_bundle_candidates(remain[:k]))
                    L=lens[len(lens)//2]
                    # Relative error prevents long targets from dominating; slight under-target penalty encourages tail reach.
                    err=abs(L-target)/max(8.0,target) + (0.04 if L<target else 0.0)
                    cand=(err,-k,k,L)
                    if best is None or cand<best: best=cand
                k=best[2]; groups.append(remain[:k]); targets.append(target); i+=k
        return groups,targets

class SemanticRendererV5:
    """Long-tail corpus-driven renderer with strict focus locality and semantic metadata."""
    def __init__(self, selector:LearnedSurfaceSelectorV3, structure:EmpiricalStructurePlanner):
        self.sel=selector; self.structure=structure
    def render(self,facts):
        groups,targets=self.structure.bundle(facts)
        sentences=[]; represented=[]; picks=[]; recent=[]; paragraphs=[]; cur_focus=None; cur=[]
        for g,target in zip(groups,targets):
            pick=self.sel.choose(focus_bundle_candidates(g),recent)
            text=pick[2]; focus=g[0][1]
            sentences.append(text);represented.extend(pick[3]['facts']);picks.append(pick)
            ws=self.sel.s.tokenize(text);opening=' '.join(w for w in ws[:3] if not self.sel.s.is_slot(w));recent=(recent+[opening])[-12:]
            if cur_focus is None: cur_focus=focus
            if focus!=cur_focus:
                paragraphs.append(' '.join(cur));cur=[];cur_focus=focus
            cur.append(text)
        if cur: paragraphs.append(' '.join(cur))
        return {'text':'\n\n'.join(paragraphs),'sentences':sentences,'paragraphs':paragraphs,
                'represented':represented,'picks':picks,'groups':groups,'targets':targets}

class ProtectedSlotVerifier:
    """Generic corruption gate for opaque semantic slots; checks every sentence against its immutable group."""
    def __init__(self):
        self.rx=re.compile(r'\b(?:e\d+|a\d+|v\d+|r\d+)\b',re.I)
    @staticmethod
    def expected_slots(facts):
        c=Counter()
        for f in facts:
            if f[0]=='prop': _,s,p,v=f; c.update([s,p,v])
            else: _,a,r,b=f; c.update([a,r,b])
        return c
    def inspect_sentence(self,text,facts):
        got=set(x.lower() for x in self.rx.findall(text))
        exp=set(x.lower() for x in self.expected_slots(facts))
        return got==exp
    def inspect_render(self,out):
        bad=[]
        for i,(s,g) in enumerate(zip(out['sentences'],out['groups'])):
            if not self.inspect_sentence(s,g): bad.append(i)
        return bad

class LearnedSurfaceSelectorV4(LearnedSurfaceSelectorV3):
    """Honors the corpus-sampled target length while reranking semantically equivalent candidates."""
    def __init__(self,*args,target_weight=.75,**kwargs):
        super().__init__(*args,**kwargs); self.target_weight=float(target_weight)
    def choose(self,candidates,recent_openings=()):
        best=None; recent=Counter(recent_openings)
        for text,meta in candidates:
            ws=self.s.tokenize(text);n=len(ws)
            lang=self.s.score_tokens(ws,max_order=4,slot_aware=True)
            lp=self.length_weight*self.s.length_logprior(n)
            opening=' '.join(w for w in ws[:3] if not self.s.is_slot(w))
            rep=self.repetition_weight*recent.get(opening,0)
            support=self.support_weight*self.s.supported_fraction(text,3,True)
            op=self.opening_weight*self.opening_score(ws)
            target=meta.get('target_length')
            tpen=0.0 if target is None else self.target_weight*abs(n-target)/max(8.0,float(target))
            score=lang+lp+support+op-rep-tpen
            row=(score,self.rng.random(),text,meta,lang,lp,support,rep,op,tpen)
            if best is None or row[:2]>best[:2]:best=row
        return best

class SemanticRendererV6(SemanticRendererV5):
    def render(self,facts):
        groups,targets=self.structure.bundle(facts)
        sentences=[];represented=[];picks=[];recent=[];paragraphs=[];cur_focus=None;cur=[]
        for g,target in zip(groups,targets):
            cands=[]
            for text,meta in focus_bundle_candidates(g):
                meta=dict(meta);meta['target_length']=target;cands.append((text,meta))
            pick=self.sel.choose(cands,recent)
            text=pick[2]; focus=g[0][1]
            sentences.append(text);represented.extend(pick[3]['facts']);picks.append(pick)
            ws=self.sel.s.tokenize(text);opening=' '.join(w for w in ws[:3] if not self.sel.s.is_slot(w));recent=(recent+[opening])[-12:]
            if cur_focus is None:cur_focus=focus
            if focus!=cur_focus:
                paragraphs.append(' '.join(cur));cur=[];cur_focus=focus
            cur.append(text)
        if cur:paragraphs.append(' '.join(cur))
        return {'text':'\n\n'.join(paragraphs),'sentences':sentences,'paragraphs':paragraphs,
                'represented':represented,'picks':picks,'groups':groups,'targets':targets}

class EmpiricalStructurePlannerV2(EmpiricalStructurePlanner):
    """Faster cached structural estimator plus a globally calibratable length scale."""
    def __init__(self, scorer, seed=12345, max_bundle=10, q_low=.05, q_high=.99, target_scale=1.0):
        super().__init__(scorer,seed,max_bundle,q_low,q_high)
        self.target_scale=float(target_scale); self._len_cache={}
    def _pattern_len(self, types):
        key=tuple(types)
        if key in self._len_cache:return self._len_cache[key]
        dummy=[]
        for i,t in enumerate(key):
            if t=='prop': dummy.append(('prop','e0',f'a{i}',f'v{i}'))
            else: dummy.append(('rel','e0',f'r{i}',f'e{i+1}'))
        lens=sorted(len(self.s.tokenize(t)) for t,_ in focus_bundle_candidates(dummy))
        L=lens[len(lens)//2]
        self._len_cache[key]=L;return L
    def bundle(self,facts):
        by=defaultdict(list)
        for f in facts:by[f[1]].append(f)
        groups=[];targets=[]
        for focus in sorted(by):
            fs=by[focus];i=0
            while i<len(fs):
                raw=self.schedule.next();target=max(1,int(round(raw*self.target_scale)))
                remain=fs[i:i+self.max_bundle];best=None;types=[]
                for k,f in enumerate(remain,1):
                    types.append(f[0]);L=self._pattern_len(types)
                    err=abs(L-target)/max(8.0,target)+(0.04 if L<target else 0.0)
                    cand=(err,-k,k,L)
                    if best is None or cand<best:best=cand
                k=best[2];groups.append(remain[:k]);targets.append(target);i+=k
        return groups,targets

class LearnedSurfaceSelectorV5(LearnedSurfaceSelectorV4):
    """Adds generic recent-structure memory; it penalizes repeated construction IDs, not words/rules."""
    def __init__(self,*args,template_repetition_weight=.22,**kwargs):
        super().__init__(*args,**kwargs);self.template_repetition_weight=float(template_repetition_weight)
    def choose(self,candidates,recent_openings=(),recent_templates=()):
        best=None; ro=Counter(recent_openings); rt=Counter(recent_templates)
        for text,meta in candidates:
            ws=self.s.tokenize(text);n=len(ws)
            lang=self.s.score_tokens(ws,max_order=4,slot_aware=True)
            lp=self.length_weight*self.s.length_logprior(n)
            opening=' '.join(w for w in ws[:3] if not self.s.is_slot(w))
            rep=self.repetition_weight*ro.get(opening,0)
            trep=self.template_repetition_weight*rt.get(meta.get('template'),0)
            support=self.support_weight*self.s.supported_fraction(text,3,True)
            op=self.opening_weight*self.opening_score(ws)
            target=meta.get('target_length')
            tpen=0.0 if target is None else self.target_weight*abs(n-target)/max(8.0,float(target))
            score=lang+lp+support+op-rep-trep-tpen
            row=(score,self.rng.random(),text,meta,lang,lp,support,rep,op,tpen,trep)
            if best is None or row[:2]>best[:2]:best=row
        return best

class RendererV7(SemanticRendererV6):
    def render(self,facts):
        groups,targets=self.structure.bundle(facts)
        sentences=[];represented=[];picks=[];recent=[];recent_t=[];paragraphs=[];cur_focus=None;cur=[]
        for g,target in zip(groups,targets):
            cands=[]
            for text,meta in focus_bundle_candidates(g):
                meta=dict(meta);meta['target_length']=target;cands.append((text,meta))
            pick=self.sel.choose(cands,recent,recent_t)
            text=pick[2];focus=g[0][1]
            sentences.append(text);represented.extend(pick[3]['facts']);picks.append(pick)
            ws=self.sel.s.tokenize(text);opening=' '.join(w for w in ws[:3] if not self.sel.s.is_slot(w))
            recent=(recent+[opening])[-12:];recent_t=(recent_t+[pick[3]['template']])[-12:]
            if cur_focus is None:cur_focus=focus
            if focus!=cur_focus:
                paragraphs.append(' '.join(cur));cur=[];cur_focus=focus
            cur.append(text)
        if cur:paragraphs.append(' '.join(cur))
        return {'text':'\n\n'.join(paragraphs),'sentences':sentences,'paragraphs':paragraphs,
                'represented':represented,'picks':picks,'groups':groups,'targets':targets}

RENDERER_V7_CONFIG={
    'name':'Renderer-V7',
    'max_bundle':12,
    'q_low':0.03,
    'q_high':0.995,
    'target_scale':1.10,
    'repetition_weight':0.8,
    'target_weight':1.0,
    'template_repetition_weight':0.08,
}

def build_promoted_renderer(root, seed=101, use_hot=False):
    scorer=BagacoSurfaceScorer(root,use_hot=use_hot)
    selector=LearnedSurfaceSelectorV5(
        scorer,seed=seed,
        repetition_weight=RENDERER_V7_CONFIG['repetition_weight'],
        target_weight=RENDERER_V7_CONFIG['target_weight'],
        template_repetition_weight=RENDERER_V7_CONFIG['template_repetition_weight'])
    planner=EmpiricalStructurePlannerV2(
        scorer,seed=seed,
        max_bundle=RENDERER_V7_CONFIG['max_bundle'],
        q_low=RENDERER_V7_CONFIG['q_low'],
        q_high=RENDERER_V7_CONFIG['q_high'],
        target_scale=RENDERER_V7_CONFIG['target_scale'])
    return scorer,RendererV7(selector,planner)
