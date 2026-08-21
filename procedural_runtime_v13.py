from __future__ import annotations
from pathlib import Path
import re

from procedural_runtime_v3 import RENDERER_V7_CONFIG
from procedural_runtime_v4 import InducedConstructionGrammar, RENDERER_V8_CONFIG
from procedural_runtime_v5 import SafeWrapperInducer, StoredWrapperInducer, InducedRealizationProposer
from procedural_runtime_gpu import GpuBagacoSurfaceScorer
from procedural_runtime_v12 import DiversityAwareSelectorGPU, GraphDiscoursePlanner, RendererV12GPU

class LexicalizedDiversitySelectorGPU(DiversityAwareSelectorGPU):
    """Scores semantic candidates with their known lexical forms on CUDA.

    Entity identities remain protected slots. Concept/value/relation labels participate
    in the corpus score, while the emitted candidate keeps immutable semantic IDs.
    """
    def __init__(self,*args,lexicon=None,**kwargs):
        super().__init__(*args,**kwargs);self.lexicon={str(k).lower():str(v) for k,v in (lexicon or {}).items()}

    def _score_words(self, text):
        orig=self.s.tokenize(text);out=[]
        for w in orig:
            lw=w.lower()
            if lw[:1] in ('a','r','v') and lw in self.lexicon:
                out.extend(self.s.tokenize(self.lexicon[lw]))
            else: out.append(w)
        return orig,out

    def _static_features_many(self,texts,paragraph_first):
        rows=[None]*len(texts);miss=[];orig_words=[];score_words=[];keys=[]
        for idx,text in enumerate(texts):
            ow,sw=self._score_words(text)
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
                           device=0,memory_limit_mb=4608):
    root=Path(root);scorer=GpuBagacoSurfaceScorer(root,use_hot=use_hot,device=device,memory_limit_mb=memory_limit_mb)
    grammar=InducedConstructionGrammar(root)
    inducer=SafeWrapperInducer(root,scorer) if (root/'model'/'quality'/'open.jsonl').exists() and (root/'model'/'full'/'open.jsonl').exists() else StoredWrapperInducer(root,scorer)
    proposer=InducedRealizationProposer(inducer)
    selector=LexicalizedDiversitySelectorGPU(
        scorer,seed=seed,grammar=grammar,lexicon=lexicon,
        repetition_weight=float(repetition_weight),target_weight=RENDERER_V8_CONFIG['target_weight'],
        template_repetition_weight=RENDERER_V8_CONFIG['template_repetition_weight'],
        construction_weight=RENDERER_V8_CONFIG['construction_weight'],position_weight=float(position_weight),
        proposal_weight=float(proposal_weight),diversity_weight=float(diversity_weight),
        focus_diversity_weight=float(focus_diversity_weight))
    planner=GraphDiscoursePlanner(scorer,seed=seed,max_bundle=RENDERER_V8_CONFIG['max_bundle'],
        q_low=RENDERER_V8_CONFIG['q_low'],q_high=RENDERER_V8_CONFIG['q_high'],target_scale=RENDERER_V8_CONFIG['target_scale'])
    return scorer,grammar,inducer,RendererV12GPU(selector,planner,proposer)
