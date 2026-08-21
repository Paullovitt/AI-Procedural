from __future__ import annotations
from pathlib import Path
from collections import Counter, defaultdict
from typing import Any
import json, math, difflib, unicodedata, re

from procedural_runtime_v4 import *


def _load_jsonl_table(path: Path):
    out={}
    with path.open(encoding='utf8') as f:
        for line in f:
            d=json.loads(line); out[d['k']]=int(d['n'])
    return out


def _norm_phrase(s: str):
    s=s.replace('\t',' ').lower()
    return ''.join(c for c in unicodedata.normalize('NFKD',s) if not unicodedata.combining(c))


def _cosine_counts(a,b):
    if not a or not b:return 0.0
    dot=sum(v*b.get(k,0) for k,v in a.items())
    na=math.sqrt(sum(v*v for v in a.values())); nb=math.sqrt(sum(v*v for v in b.values()))
    return dot/(na*nb) if na and nb else 0.0


class SafeWrapperInducer:
    """Discovers new focus wrappers from Bagaço without a hand-written synonym list.

    Semantic anchors are extracted automatically from the already verified V8 candidate family.
    A new wrapper is promoted only when it independently matches those anchors by:
      1) continuation-distribution similarity in the quality corpus,
      2) surface-form affinity to the nearest trusted anchor,
      3) independent support in the broader full model,
      4) same terminal grammatical shape as the anchor,
      5) not merely being a truncated substring of the anchor.
    This is intentionally conservative: unsafe discourse operators stay latent.
    """
    def __init__(self, root: str|Path, scorer: BagacoSurfaceScorer,
                 min_quality_support=30, min_continuations=3,
                 min_continuation_cosine=.70, min_surface_affinity=.50):
        self.root=Path(root); self.s=scorer
        self.min_quality_support=int(min_quality_support)
        self.min_continuations=int(min_continuations)
        self.min_continuation_cosine=float(min_continuation_cosine)
        self.min_surface_affinity=float(min_surface_affinity)
        q=self.root/'model'/'quality'; f=self.root/'model'/'full'
        self.qopen=_load_jsonl_table(q/'open.jsonl')
        self.fopen=_load_jsonl_table(f/'open.jsonl')
        self.qgrams={n:_load_jsonl_table(q/f'p{n}.jsonl') for n in range(2,6)}
        self.cont={m:defaultdict(dict) for m in range(1,5)}
        for n,tab in self.qgrams.items():
            m=n-1
            for k,v in tab.items():
                t=k.split('\t')
                if len(t)==n:self.cont[m]['\t'.join(t[:-1])][t[-1]]=v
        self.seeds=self._extract_verified_seed_wrappers()
        self.promoted=self._discover()

    def _extract_verified_seed_wrappers(self):
        # No phrase names are inserted here: they come from the existing verified renderer itself.
        dummy=[('prop','e999','a99','v99')]
        rows=focus_bundle_candidates(dummy)
        seeds=[]
        for text,_ in rows:
            ws=self.s.tokenize(text)
            try:i=ws.index('e999')
            except ValueError:continue
            ph='\t'.join(ws[:i])
            if ph and ph not in seeds:seeds.append(ph)
        return seeds

    def _discover(self):
        # One-token anchors are too broad to support semantic paraphrase induction.
        anchors=[s for s in self.seeds if 2<=len(s.split('\t'))<=4]
        sigs={s:self.cont[len(s.split('\t'))].get(s,{}) for s in anchors}
        rows=[]
        for ph,qsup in self.qopen.items():
            toks=ph.split('\t'); n=len(toks)
            if not (2<=n<=4) or ph in self.seeds or qsup<self.min_quality_support:continue
            fsup=self.fopen.get(ph,0)
            if fsup<=0:continue
            sig=self.cont[n].get(ph,{})
            if len(sig)<self.min_continuations:continue
            best_seed=None;best_sim=-1.0
            for seed in anchors:
                sim=_cosine_counts(sig,sigs.get(seed,{}))
                if sim>best_sim:best_sim=sim;best_seed=seed
            if best_seed is None:continue
            affinity=difflib.SequenceMatcher(None,_norm_phrase(ph),_norm_phrase(best_seed)).ratio()
            # The terminal token is learned from the anchor shape, not hard-coded Portuguese grammar.
            if toks[-1]!=best_seed.split('\t')[-1]:continue
            a=_norm_phrase(ph);b=_norm_phrase(best_seed)
            # Reject clipped/extended copies; we want independently observed constructions.
            if a in b or b in a:continue
            if best_sim<self.min_continuation_cosine or affinity<self.min_surface_affinity:continue
            support_conf=min(1.0,math.log1p(qsup)/math.log(1000.0))*min(1.0,math.log1p(fsup)/math.log(5000.0))
            confidence=best_sim*affinity*(0.65+0.35*support_conf)
            rows.append({'phrase':ph,'nearest_seed':best_seed,'quality_support':qsup,'full_support':fsup,
                         'continuations':len(sig),'continuation_cosine':best_sim,
                         'surface_affinity':affinity,'confidence':confidence})
        rows.sort(key=lambda r:(r['confidence'],r['quality_support']),reverse=True)
        return rows

    @staticmethod
    def display_phrase(ph):
        s=ph.replace('\t',' ')
        return s[:1].upper()+s[1:]

    def save(self,path: str|Path):
        data={'format':'Bagaco-Safe-Wrapper-Induction-v1','seed_wrappers':self.seeds,
              'thresholds':{'min_quality_support':self.min_quality_support,
                            'min_continuations':self.min_continuations,
                            'min_continuation_cosine':self.min_continuation_cosine,
                            'min_surface_affinity':self.min_surface_affinity},
              'promoted':self.promoted}
        Path(path).write_text(json.dumps(data,ensure_ascii=False,indent=2),encoding='utf8')
        return data


class InducedRealizationProposer:
    """Lets corpus-induced wrappers create new realizations while preserving the factual body verbatim."""
    def __init__(self, inducer: Any):
        self.inducer=inducer

    def propose(self,bundle,target_length=None):
        focus=bundle[0][1]
        base=focus_bundle_candidates(bundle)
        out=[];seen=set()
        for text,meta in base:
            if focus not in text:continue
            head,tail=text.split(focus,1)
            head_tokens=self.inducer.s.tokenize(head)
            base_wrapper='\t'.join(head_tokens)
            for wr in self.inducer.promoted:
                # Preserve the syntactic body that was already verified with the nearest
                # semantic anchor. This prevents arbitrary wrapper/body cross-products.
                if base_wrapper!=wr['nearest_seed']:
                    continue
                phrase=self.inducer.display_phrase(wr['phrase'])
                text2=f'{phrase} {focus}{tail}'
                if text2 in seen:continue
                seen.add(text2)
                m=dict(meta)
                m['template']='induced:'+wr['phrase']+':'+str(meta.get('template',''))
                m['source']='induced_wrapper'
                m['induced_wrapper']=wr['phrase']
                m['proposal_confidence']=float(wr['confidence'])
                if target_length is not None:m['target_length']=target_length
                out.append((text2,m))
        return out


class LearnedSurfaceSelectorV7(LearnedSurfaceSelectorV6):
    def __init__(self,*args,proposal_weight=.06,**kwargs):
        super().__init__(*args,**kwargs);self.proposal_weight=float(proposal_weight)
    def choose(self,candidates,recent_openings=(),recent_templates=(),paragraph_first=False) -> Any:
        best: Any=None; ro=Counter(recent_openings);rt=Counter(recent_templates)
        for text,meta in candidates:
            ws,n,lang,lp,opening,support,op,cscore,ccov,chits,pscore,pmeta=self._static_features(text,paragraph_first)
            rep=self.repetition_weight*ro.get(opening,0)
            trep=self.template_repetition_weight*rt.get(meta.get('template'),0)
            target=meta.get('target_length')
            tpen=0.0 if target is None else self.target_weight*abs(n-target)/max(8.0,float(target))
            pbonus=self.proposal_weight*float(meta.get('proposal_confidence',0.0))
            score=(lang+lp+support+op-rep-trep-tpen+
                   self.construction_weight*cscore+self.position_weight*pscore+pbonus)
            meta2=dict(meta);meta2['construction_score']=cscore;meta2['construction_coverage']=ccov
            meta2['construction_hits']=chits;meta2['position_evidence']=pmeta;meta2['paragraph_first']=bool(paragraph_first)
            meta2['proposal_bonus']=pbonus
            row=(score,self.rng.random(),text,meta2,lang,lp,support,rep,op,tpen,trep,cscore,pscore,pbonus)
            if best is None or row[:2]>best[:2]:best=row
        return best


class RendererV9(RendererV8):
    def __init__(self,selector,structure,proposer:InducedRealizationProposer):
        super().__init__(selector,structure);self.proposer=proposer
    def render(self,facts):
        groups,targets=self.structure.bundle(facts)
        sentences=[];represented=[];picks=[];recent=[];recent_t=[];paragraphs=[];cur_focus=None;cur=[]
        induced_candidates=0; induced_selected=0
        for g,target in zip(groups,targets):
            focus=g[0][1]; paragraph_first=(cur_focus is None or focus!=cur_focus)
            cands=[]
            for text,meta in focus_bundle_candidates(g):
                meta=dict(meta);meta['target_length']=target;meta.setdefault('source','verified_v8')
                cands.append((text,meta))
            new=self.proposer.propose(g,target);induced_candidates+=len(new);cands.extend(new)
            # Exact-string dedupe.
            ded=[];seen=set()
            for row in cands:
                if row[0] in seen:continue
                seen.add(row[0]);ded.append(row)
            pick=self.sel.choose(ded,recent,recent_t,paragraph_first=paragraph_first)
            text=pick[2]
            if pick[3].get('source')=='induced_wrapper':induced_selected+=1
            sentences.append(text);represented.extend(pick[3]['facts']);picks.append(pick)
            ws=self.sel.s.tokenize(text);opening=' '.join(w for w in ws[:3] if not self.sel.s.is_slot(w))
            recent=(recent+[opening])[-12:];recent_t=(recent_t+[pick[3]['template']])[-12:]
            if cur_focus is None:cur_focus=focus
            if focus!=cur_focus:
                paragraphs.append(' '.join(cur));cur=[];cur_focus=focus
            cur.append(text)
        if cur:paragraphs.append(' '.join(cur))
        return {'text':'\n\n'.join(paragraphs),'sentences':sentences,'paragraphs':paragraphs,
                'represented':represented,'picks':picks,'groups':groups,'targets':targets,
                'induced_candidates':induced_candidates,'induced_selected':induced_selected}


class StoredWrapperInducer:
    """Deployment loader for wrappers already promoted by shadow validation."""
    def __init__(self, root, scorer):
        self.root=Path(root); self.s=scorer
        data=json.loads((self.root/'LEARNED_REALIZATION_PROPOSALS_V1.json').read_text(encoding='utf8'))
        self.seeds=list(data.get('seed_wrappers',[]))
        self.promoted=list(data.get('promoted',[]))
    @staticmethod
    def display_phrase(ph):
        words=ph.replace('\t',' ').split()
        if not words:return ''
        return ' '.join([words[0].capitalize()]+words[1:])

def build_renderer_v9(root,seed=101,use_hot=False,proposal_weight=.06):
    root=Path(root)
    scorer=BagacoSurfaceScorer(root,use_hot=use_hot)
    grammar=InducedConstructionGrammar(root)
    # Full induction requires broad/full tables. Compact deployment loads only proposals
    # that were already independently validated and promoted.
    if (root/'model'/'quality'/'open.jsonl').exists() and (root/'model'/'full'/'open.jsonl').exists():
        inducer=SafeWrapperInducer(root,scorer)
    else:
        inducer=StoredWrapperInducer(root,scorer)
    proposer=InducedRealizationProposer(inducer)
    selector=LearnedSurfaceSelectorV7(
        scorer,seed=seed,grammar=grammar,
        repetition_weight=RENDERER_V8_CONFIG['repetition_weight'],
        target_weight=RENDERER_V8_CONFIG['target_weight'],
        template_repetition_weight=RENDERER_V8_CONFIG['template_repetition_weight'],
        construction_weight=RENDERER_V8_CONFIG['construction_weight'],
        position_weight=RENDERER_V8_CONFIG['position_weight'],
        proposal_weight=proposal_weight)
    planner=EmpiricalStructurePlannerV2(
        scorer,seed=seed,max_bundle=RENDERER_V8_CONFIG['max_bundle'],
        q_low=RENDERER_V8_CONFIG['q_low'],q_high=RENDERER_V8_CONFIG['q_high'],
        target_scale=RENDERER_V8_CONFIG['target_scale'])
    return scorer,grammar,inducer,RendererV9(selector,planner,proposer)
