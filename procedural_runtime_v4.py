from __future__ import annotations
from pathlib import Path
from collections import Counter
from typing import Any
import json, lzma, math

from procedural_runtime_v3 import *

class InducedConstructionGrammar:
    """Corpus-induced one-wildcard grammar. It never inserts lexical material; it only scores
    contexts around immutable semantic slots. Frames were promoted only after novel-variant
    generalization against the broader Bagaço model."""
    def __init__(self, root: str | Path):
        self.root=Path(root)
        with lzma.open(self.root/'LEARNED_CONSTRUCTIONS_V1.json.xz','rt',encoding='utf8') as f:
            self.model=json.load(f)
        self.frames={}
        self.max_log_gain={}
        stored_mx=self.model.get('max_log_gain',{})
        for ns,rows in self.model['frames'].items():
            n=int(ns); d={}
            mx=1.0
            for r in rows:
                sk=tuple(r['s'].split('\t')); d[sk]=r
                mx=max(mx,math.log1p(float(r['gain'])))
            self.frames[n]=d
            self.max_log_gain[n]=float(stored_mx.get(str(ns),mx))
        self.discourse=self.model['discourse']
        self.phrase_stats=self.discourse['phrase_stats']
        # Runtime-pruned grammars store the global prior so pruning cannot change scores.
        if 'baseline_first' in self.model:
            self.baseline_first=float(self.model['baseline_first'])
        else:
            first=sum(max(0,int(x['open'])-int(x['connect'])) for x in self.phrase_stats.values())
            opn=sum(int(x['open']) for x in self.phrase_stats.values())
            self.baseline_first=(first+1)/(opn+2)

    def slot_frame_score_tokens(self, words, is_slot):
        """Generalization score around protected slots, normalized to [0,~1]."""
        ws=list(words); vals=[]; hits=0; slots=0
        for i,w in enumerate(ws):
            if not is_slot(w): continue
            slots+=1; best=0.0
            for n in (2,3,4,5):
                tab=self.frames.get(n)
                if not tab: continue
                lo=max(0,i-n+1); hi=min(i,len(ws)-n)+1
                for start in range(lo,hi):
                    win=ws[start:start+n]
                    # Only the current semantic slot is abstracted. Other slots remain opaque,
                    # so a match genuinely demonstrates corpus generalization across this role.
                    pos=i-start; sk=tuple(win[:pos]+['*']+win[pos+1:])
                    r=tab.get(sk)
                    if r:
                        g=math.log1p(float(r['gain']))/self.max_log_gain[n]
                        # Reward independent shadow evidence as confidence, not as an answer label.
                        conf=min(1.0,math.log1p(int(r['novel_types']))/math.log(32.0))
                        best=max(best,g*(0.65+0.35*conf)); hits+=1
            vals.append(best)
        return (sum(vals)/max(1,len(vals))), (sum(v>0 for v in vals)/max(1,slots)), hits

    def slot_frame_score(self,text,scorer):
        return self.slot_frame_score_tokens(scorer.tokenize(text),scorer.is_slot)

    def opening_position_score_tokens(self, words, paragraph_first, is_slot):
        """Continuous, unlabeled discourse-position score learned from open vs connect counts.
        Discrete rhetorical clusters were rejected because model selection hit the search boundary."""
        ws=list(words)
        # Use lexical material before the first protected slot; if none, first four words.
        p=[]
        for w in ws[:5]:
            if is_slot(w): break
            p.append(w)
        if not p: return 0.0, None
        best: Any=None
        for n in range(1,min(4,len(p))+1):
            ph='\t'.join(p[:n]); r=self.phrase_stats.get(ph)
            if not r: continue
            o=int(r['open']); c=min(o,int(r['connect'])); first=max(0,o-c)
            pr=(first+1)/(o+2)
            if paragraph_first:
                score=math.log(pr/self.baseline_first)
            else:
                score=math.log((1-pr)/(1-self.baseline_first))
            # Reliability grows with corpus support and saturates.
            rel=min(1.0,math.log1p(o)/math.log(5000.0))
            row=(score*rel,n,ph,pr,o)
            if best is None or n>best[1]: best=row
        if best is None:return 0.0,None
        return best[0],{'phrase':best[2],'first_rate':best[3],'support':best[4]}

    def opening_position_score(self,text,scorer,paragraph_first):
        return self.opening_position_score_tokens(scorer.tokenize(text),paragraph_first,scorer.is_slot)


class LearnedSurfaceSelectorV6(LearnedSurfaceSelectorV5):
    """V7 scorer + induced grammar generalization + learned paragraph-position prior."""
    def __init__(self,*args,grammar:InducedConstructionGrammar,
                 construction_weight=.35,position_weight=.08,**kwargs):
        super().__init__(*args,**kwargs)
        self.grammar=grammar
        self.construction_weight=float(construction_weight)
        self.position_weight=float(position_weight)
        self._feature_cache={}

    def _static_features(self, text, paragraph_first):
        ws=self.s.tokenize(text)
        # Slot identities are intentionally erased: e001/e777 etc. are opaque symbols and
        # cannot legitimately change grammar/fluency scores.
        key=(tuple('__slot__' if self.s.is_slot(w) else w for w in ws), bool(paragraph_first))
        hit=self._feature_cache.get(key)
        if hit is not None: return hit
        n=len(ws)
        lang=self.s.score_tokens(ws,max_order=4,slot_aware=True)
        lp=self.length_weight*self.s.length_logprior(n)
        opening=' '.join(w for w in ws[:3] if not self.s.is_slot(w))
        support=self.support_weight*self.s.supported_fraction(text,3,True)
        op=self.opening_weight*self.opening_score(ws)
        cscore,ccov,chits=self.grammar.slot_frame_score_tokens(ws,self.s.is_slot)
        pscore,pmeta=self.grammar.opening_position_score_tokens(ws,paragraph_first,self.s.is_slot)
        hit=(ws,n,lang,lp,opening,support,op,cscore,ccov,chits,pscore,pmeta)
        self._feature_cache[key]=hit
        return hit

    def choose(self,candidates,recent_openings=(),recent_templates=(),paragraph_first=False) -> Any:
        best: Any=None; ro=Counter(recent_openings); rt=Counter(recent_templates)
        for text,meta in candidates:
            ws,n,lang,lp,opening,support,op,cscore,ccov,chits,pscore,pmeta=self._static_features(text,paragraph_first)
            rep=self.repetition_weight*ro.get(opening,0)
            trep=self.template_repetition_weight*rt.get(meta.get('template'),0)
            target=meta.get('target_length')
            tpen=0.0 if target is None else self.target_weight*abs(n-target)/max(8.0,float(target))
            score=(lang+lp+support+op-rep-trep-tpen+
                   self.construction_weight*cscore+self.position_weight*pscore)
            meta2=dict(meta); meta2['construction_score']=cscore;meta2['construction_coverage']=ccov
            meta2['construction_hits']=chits;meta2['position_evidence']=pmeta;meta2['paragraph_first']=bool(paragraph_first)
            row=(score,self.rng.random(),text,meta2,lang,lp,support,rep,op,tpen,trep,cscore,pscore)
            if best is None or row[:2]>best[:2]:best=row
        return best


class RendererV8(RendererV7):
    """Construction-aware renderer. Semantic grouping/planning remains unchanged from promoted V7."""
    def render(self,facts):
        groups,targets=self.structure.bundle(facts)
        sentences=[];represented=[];picks=[];recent=[];recent_t=[];paragraphs=[];cur_focus=None;cur=[]
        for g,target in zip(groups,targets):
            focus=g[0][1]
            paragraph_first=(cur_focus is None or focus!=cur_focus)
            cands=[]
            for text,meta in focus_bundle_candidates(g):
                meta=dict(meta);meta['target_length']=target;cands.append((text,meta))
            pick=self.sel.choose(cands,recent,recent_t,paragraph_first=paragraph_first)
            text=pick[2]
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


RENDERER_V8_CONFIG={
    **RENDERER_V7_CONFIG,
    'name':'Renderer-V8-InducedGrammar',
    'construction_weight':0.60,
    'position_weight':0.15,
    'grammar':'Bagaco-Induced-Constructions-v1',
}


def build_renderer_v8(root,seed=101,use_hot=False,construction_weight=None,position_weight=None):
    scorer=BagacoSurfaceScorer(root,use_hot=use_hot)
    grammar=InducedConstructionGrammar(root)
    selector=LearnedSurfaceSelectorV6(
        scorer,seed=seed,grammar=grammar,
        repetition_weight=RENDERER_V8_CONFIG['repetition_weight'],
        target_weight=RENDERER_V8_CONFIG['target_weight'],
        template_repetition_weight=RENDERER_V8_CONFIG['template_repetition_weight'],
        construction_weight=(RENDERER_V8_CONFIG['construction_weight'] if construction_weight is None else construction_weight),
        position_weight=(RENDERER_V8_CONFIG['position_weight'] if position_weight is None else position_weight),
    )
    planner=EmpiricalStructurePlannerV2(
        scorer,seed=seed,max_bundle=RENDERER_V8_CONFIG['max_bundle'],
        q_low=RENDERER_V8_CONFIG['q_low'],q_high=RENDERER_V8_CONFIG['q_high'],
        target_scale=RENDERER_V8_CONFIG['target_scale'])
    return scorer,grammar,RendererV8(selector,planner)
