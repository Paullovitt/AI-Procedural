from __future__ import annotations

from pathlib import Path
from collections import Counter
import bisect

from procedural_runtime_v3 import focus_bundle_candidates
from procedural_runtime_v4 import RENDERER_V8_CONFIG, InducedConstructionGrammar
from procedural_runtime_v5 import SafeWrapperInducer, StoredWrapperInducer, InducedRealizationProposer
from procedural_runtime_gpu import GpuBagacoSurfaceScorer, attach_semantic_traces
from procedural_runtime_v12 import GraphDiscoursePlanner, DiversityAwareSelectorGPU
from procedural_runtime_v13 import LexicalizedDiversitySelectorGPU


class EmpiricalParagraphScheduler:
    """Low-discrepancy sampler over the paragraph sentence-count histogram learned from Bagaço."""
    def __init__(self,hist,seed=101):
        self.hist=list(hist or [0,1]);total=max(1,sum(self.hist));self.cdf=[];c=0
        for n in self.hist:c+=int(n);self.cdf.append(c/total)
        self.phi=0.6180339887498949;self.offset=(int(seed)*0.7548776662466927)%1.0;self.i=0
    def next_size(self):
        u=(self.offset+self.i*self.phi)%1.0;self.i+=1
        return max(1,bisect.bisect_left(self.cdf,u))
    def starts(self,n_sentences):
        starts={0};i=0
        while i<n_sentences:
            i+=self.next_size()
            if i<n_sentences:starts.add(i)
        return starts


class RendererV14GPU:
    """V12 surface engine with corpus-learned multi-focus paragraph boundaries."""
    def __init__(self,selector,structure,proposer,paragraph_scheduler):
        self.sel=selector;self.structure=structure;self.proposer=proposer;self.paragraph_scheduler=paragraph_scheduler

    def render(self,facts,focus_order_hint=None):
        groups,targets=self.structure.bundle(facts,focus_order_hint=focus_order_hint)
        starts=self.paragraph_scheduler.starts(len(groups))
        prepared=[];induced_candidates=0
        for gi,(g,target) in enumerate(zip(groups,targets)):
            focus=g[0][1];paragraph_first=(gi in starts);cands=[]
            for text,meta in focus_bundle_candidates(g):
                meta=dict(meta);meta['target_length']=target;meta.setdefault('source','verified_v8');cands.append((text,meta))
            new=self.proposer.propose(g,target);induced_candidates+=len(new);cands.extend(new)
            ded=[];seen=set()
            for row in cands:
                if row[0] in seen:continue
                seen.add(row[0]);ded.append(row)
            prepared.append((g,target,focus,paragraph_first,ded))

        bypos={False:[],True:[]}
        for _,_,_,pf,cands in prepared:bypos[pf].extend(x[0] for x in cands)
        for pf,texts in bypos.items():
            if texts:self.sel._static_features_many(texts,pf)

        sentences=[];represented=[];picks=[];recent=[];recent_t=[];paragraphs=[];cur=[]
        induced_selected=0;shape_counts=Counter();focus_shapes=Counter();prev_focus=None;focus_order=[]
        paragraph_sizes=[]
        for gi,(g,target,focus,pf,ded) in enumerate(prepared):
            if pf and cur:
                paragraphs.append(' '.join(cur));paragraph_sizes.append(len(cur));cur=[]
            if focus!=prev_focus:
                focus_shapes=Counter();focus_order.append(focus);prev_focus=focus
            pick=self.sel.choose(ded,recent,recent_t,paragraph_first=pf,shape_counts=shape_counts,focus_shape_counts=focus_shapes)
            text=pick[2]
            if pick[3].get('source')=='induced_wrapper':induced_selected+=1
            sentences.append(text);represented.extend(pick[3]['facts']);picks.append(pick);cur.append(text)
            shape=pick[3]['abstract_shape'];shape_counts[shape]+=1;focus_shapes[shape]+=1
            ws=self.sel.s.tokenize(text);opening=' '.join(w for w in ws[:3] if not self.sel.s.is_slot(w))
            recent=(recent+[opening])[-12:];recent_t=(recent_t+[pick[3]['template']])[-12:]
        if cur:paragraphs.append(' '.join(cur));paragraph_sizes.append(len(cur))
        out={'text':'\n\n'.join(paragraphs),'sentences':sentences,'paragraphs':paragraphs,
             'paragraph_sizes':paragraph_sizes,'represented':represented,'picks':picks,'groups':groups,'targets':targets,
             'induced_candidates':induced_candidates,'induced_selected':induced_selected,
             'compute_backend':'cuda-batched-v14','focus_order':focus_order,'abstract_shapes':len(shape_counts),
             'paragraph_starts':sorted(starts)}
        return attach_semantic_traces(out)


def build_renderer_v14_gpu(root,seed=101,use_hot=False,proposal_weight=.24,position_weight=7.0,
                           diversity_weight=2.6,focus_diversity_weight=1.17,repetition_weight=1.1,
                           template_repetition_weight=None,device=0,memory_limit_mb=4608,lexicon=None):
    root=Path(root);scorer=GpuBagacoSurfaceScorer(root,use_hot=use_hot,device=device,memory_limit_mb=memory_limit_mb)
    grammar=InducedConstructionGrammar(root)
    inducer=SafeWrapperInducer(root,scorer) if (root/'model'/'quality'/'open.jsonl').exists() and (root/'model'/'full'/'open.jsonl').exists() else StoredWrapperInducer(root,scorer)
    proposer=InducedRealizationProposer(inducer)
    selector_cls=LexicalizedDiversitySelectorGPU if lexicon else DiversityAwareSelectorGPU
    selector_kwargs=dict(
        scorer=scorer,seed=seed,grammar=grammar,repetition_weight=float(repetition_weight),
        target_weight=RENDERER_V8_CONFIG['target_weight'],
        template_repetition_weight=(RENDERER_V8_CONFIG['template_repetition_weight'] if template_repetition_weight is None else float(template_repetition_weight)),
        construction_weight=RENDERER_V8_CONFIG['construction_weight'],position_weight=float(position_weight),
        proposal_weight=float(proposal_weight),diversity_weight=float(diversity_weight),focus_diversity_weight=float(focus_diversity_weight))
    if lexicon:
        selector_kwargs['lexicon']=lexicon
    selector=selector_cls(**selector_kwargs)
    planner=GraphDiscoursePlanner(scorer,seed=seed,max_bundle=RENDERER_V8_CONFIG['max_bundle'],q_low=RENDERER_V8_CONFIG['q_low'],q_high=RENDERER_V8_CONFIG['q_high'],target_scale=RENDERER_V8_CONFIG['target_scale'])
    ps=EmpiricalParagraphScheduler(scorer.struct.get('para_sent',[]),seed=seed+7919)
    return scorer,grammar,inducer,RendererV14GPU(selector,planner,proposer,ps)
