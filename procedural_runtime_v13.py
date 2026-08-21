from __future__ import annotations
from pathlib import Path
from typing import Any
import re

from procedural_runtime_v4 import RENDERER_V8_CONFIG, InducedConstructionGrammar
from procedural_runtime_v5 import SafeWrapperInducer, StoredWrapperInducer, InducedRealizationProposer
from procedural_runtime_gpu import GpuBagacoSurfaceScorer
from procedural_runtime_v12 import DiversityAwareSelectorGPU, GraphDiscoursePlanner, RendererV12GPU


class LexicalizedDiversitySelectorGPU(DiversityAwareSelectorGPU):
    """Scores semantic candidates with known lexical forms on CUDA.

    Emitted candidates keep immutable semantic IDs. ``lexicalize_entities`` is opt-in:
    historical V13 behavior keeps entity IDs protected from language scoring, while
    prompt mode can score their readable labels without changing the semantic trace.
    """
    def __init__(self,*args,lexicon=None,lexicalize_entities=False,**kwargs):
        super().__init__(*args,**kwargs)
        self.lexicon={str(k).lower():str(v) for k,v in (lexicon or {}).items()}
        self.lexicalize_entities=bool(lexicalize_entities)

    def _score_words(self, text):
        orig=self.s.tokenize(text);out=[]
        for w in orig:
            lw=w.lower()
            allowed=lw[:1] in ('a','r','v') or (self.lexicalize_entities and lw[:1]=='e')
            if allowed and lw in self.lexicon:
                out.extend(self.s.tokenize(self.lexicon[lw]))
            else:
                out.append(w)
        return orig,out

    def _static_features_many(self,texts,paragraph_first):
        rows: list[Any]=[None]*len(texts);miss=[];orig_words=[];score_words=[];keys=[]
        for idx,text in enumerate(texts):
            ow,sw=self._score_words(text)
            if self.lexicalize_entities:
                key=(tuple(sw),bool(paragraph_first))
            else:
                key=(tuple('__entity__' if re.fullmatch(r'e\d+',w,re.I) else w for w in sw),bool(paragraph_first))
            hit=self._feature_cache.get(key)
            if hit is not None:rows[idx]=hit
            else:miss.append(idx);orig_words.append(ow);score_words.append(sw);keys.append(key)
        if miss:
            langs,supports=self.s.batch_language_support(score_words,max_order=5,slot_aware=True)
            for pos,ow,sw,key,lang,raw_support in zip(miss,orig_words,score_words,keys,langs,supports):
                n=len(sw);lp=self.length_weight*self.s.length_logprior(n)
                opening=' '.join(w for w in sw[:3] if not self.s.is_slot(w))
                support=self.support_weight*float(raw_support);op=self.opening_weight*self.opening_score(sw)
                cscore,ccov,chits=self.grammar.slot_frame_score_tokens(ow,self.s.is_slot)
                pscore,pmeta=self.grammar.opening_position_score_tokens(ow,paragraph_first,self.s.is_slot)
                hit=(ow,n,float(lang),lp,opening,support,op,cscore,ccov,chits,pscore,pmeta)
                self._feature_cache[key]=hit;rows[pos]=hit
        return rows


def build_renderer_v13_gpu(root,lexicon,seed=101,use_hot=False,proposal_weight=.24,position_weight=7.0,
                           diversity_weight=2.6,focus_diversity_weight=1.17,repetition_weight=1.1,
                           device=0,memory_limit_mb=4608,lexicalize_entities=False):
    root=Path(root);scorer=GpuBagacoSurfaceScorer(root,use_hot=use_hot,device=device,memory_limit_mb=memory_limit_mb)
    grammar=InducedConstructionGrammar(root)
    inducer=SafeWrapperInducer(root,scorer) if (root/'model'/'quality'/'open.jsonl').exists() and (root/'model'/'full'/'open.jsonl').exists() else StoredWrapperInducer(root,scorer)
    proposer=InducedRealizationProposer(inducer)
    selector=LexicalizedDiversitySelectorGPU(
        scorer,seed=seed,grammar=grammar,lexicon=lexicon,lexicalize_entities=lexicalize_entities,
        repetition_weight=float(repetition_weight),target_weight=RENDERER_V8_CONFIG['target_weight'],
        template_repetition_weight=RENDERER_V8_CONFIG['template_repetition_weight'],
        construction_weight=RENDERER_V8_CONFIG['construction_weight'],position_weight=float(position_weight),
        proposal_weight=float(proposal_weight),diversity_weight=float(diversity_weight),
        focus_diversity_weight=float(focus_diversity_weight))
    planner=GraphDiscoursePlanner(scorer,seed=seed,max_bundle=RENDERER_V8_CONFIG['max_bundle'],
        q_low=RENDERER_V8_CONFIG['q_low'],q_high=RENDERER_V8_CONFIG['q_high'],target_scale=RENDERER_V8_CONFIG['target_scale'])
    return scorer,grammar,inducer,RendererV12GPU(selector,planner,proposer)
