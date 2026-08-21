from __future__ import annotations
from pathlib import Path
from collections import Counter, defaultdict
from typing import Any
import math, re

from procedural_runtime_v3 import EmpiricalStructurePlannerV2, focus_bundle_candidates, RENDERER_V7_CONFIG
from procedural_runtime_v4 import InducedConstructionGrammar, RENDERER_V8_CONFIG
from procedural_runtime_v5 import SafeWrapperInducer, StoredWrapperInducer, InducedRealizationProposer
from procedural_runtime_gpu import GpuBagacoSurfaceScorer, LearnedSurfaceSelectorGPU, SemanticTraceVerifier, attach_semantic_traces


def abstract_surface_shape(scorer, text):
    ws=scorer.tokenize(text)
    out=[]
    for w in ws:
        lw=w.lower()
        if re.fullmatch(r'e\d+',lw): out.append('__e__')
        elif re.fullmatch(r'a\d+',lw): out.append('__a__')
        elif re.fullmatch(r'v\d+',lw): out.append('__v__')
        elif re.fullmatch(r'r\d+',lw): out.append('__r__')
        else: out.append(lw)
    return tuple(out)


class GraphDiscoursePlanner(EmpiricalStructurePlannerV2):
    """Generic semantic-graph paragraph ordering.

    Relation names have no semantics here. The planner only observes that two focuses
    are connected in the semantic plan and chooses the ordering that minimizes graph
    jumps. This is a generic compression/coherence mechanism, not a domain rule.
    """
    def _focus_order(self, facts, by, focus_order_hint=None):
        nodes=list(by)
        if len(nodes)<2:
            return nodes
        hint=[]
        if focus_order_hint:
            seen=set()
            for x in focus_order_hint:
                if x in by and x not in seen:
                    hint.append(x);seen.add(x)
        else:
            seen=set()
        adj={x:set() for x in nodes}
        for f in facts:
            if len(f)>=4 and f[0]=='rel':
                a,b=f[1],f[3]
                if a in adj and b in adj and a!=b:
                    adj[a].add(b); adj[b].add(a)

        # Candidate 1: lexical/stable order (baseline fallback).
        cand0=sorted(nodes)

        # Candidate 2: deterministic graph walk, starting in the most connected focus.
        start=max(nodes,key=lambda x:(len(adj[x]),x))
        unvisited=set(nodes); order=[]; cur=start
        while unvisited:
            if cur in unvisited:
                order.append(cur); unvisited.remove(cur)
            linked=[x for x in adj[cur] if x in unvisited]
            if linked:
                cur=max(linked,key=lambda x:(len(adj[x] & unvisited),len(adj[x]),x))
            elif unvisited:
                cur=max(unvisited,key=lambda x:(len(adj[x] & unvisited),len(adj[x]),x))
        cand1=order

        def cost(order):
            # Description-length proxy: an adjacent linked focus costs 0; a jump costs 1.
            return sum(0 if b in adj[a] else 1 for a,b in zip(order,order[1:]))
        chosen=min((cand0,cand1),key=lambda z:(cost(z),tuple(z)))
        if hint:
            return hint+[x for x in chosen if x not in seen]
        return chosen

    def bundle(self,facts,focus_order_hint=None):
        by=defaultdict(list)
        for f in facts: by[f[1]].append(f)
        groups=[];targets=[]
        for focus in self._focus_order(facts,by,focus_order_hint=focus_order_hint):
            fs=by[focus];i=0
            while i<len(fs):
                raw=self.schedule.next();target=max(1,int(round(raw*self.target_scale)))
                remain=fs[i:i+self.max_bundle];best: Any=None;types=[]
                for k,f in enumerate(remain,1):
                    types.append(f[0]);L=self._pattern_len(types)
                    err=abs(L-target)/max(8.0,target)+(0.04 if L<target else 0.0)
                    cand=(err,-k,k,L)
                    if best is None or cand<best: best=cand
                k=best[2];groups.append(remain[:k]);targets.append(target);i+=k
        return groups,targets


class DiversityAwareSelectorGPU(LearnedSurfaceSelectorGPU):
    """Adds document-level structural novelty memory without language-specific rules."""
    def __init__(self,*args,diversity_weight=2.6,focus_diversity_weight=1.17,**kwargs):
        super().__init__(*args,**kwargs)
        self.diversity_weight=float(diversity_weight)
        self.focus_diversity_weight=float(focus_diversity_weight)

    def choose(self,candidates,recent_openings=(),recent_templates=(),paragraph_first=False,
               shape_counts=None,focus_shape_counts=None) -> Any:
        shape_counts=shape_counts or Counter();focus_shape_counts=focus_shape_counts or Counter()
        best: Any=None;ro=Counter(recent_openings);rt=Counter(recent_templates)
        features=self._static_features_many([x[0] for x in candidates],paragraph_first)
        for (text,meta),feat in zip(candidates,features):
            ws,n,lang,lp,opening,support,op,cscore,ccov,chits,pscore,pmeta=feat
            rep=self.repetition_weight*ro.get(opening,0)
            trep=self.template_repetition_weight*rt.get(meta.get('template'),0)
            target=meta.get('target_length')
            tpen=0.0 if target is None else self.target_weight*abs(n-target)/max(8.0,float(target))
            pbonus=self.proposal_weight*float(meta.get('proposal_confidence',0.0))
            shape=abstract_surface_shape(self.s,text)
            dpen=self.diversity_weight*math.log1p(shape_counts.get(shape,0))
            fdpen=self.focus_diversity_weight*math.log1p(focus_shape_counts.get(shape,0))
            score=(lang+lp+support+op-rep-trep-tpen+
                   self.construction_weight*cscore+self.position_weight*pscore+pbonus-dpen-fdpen)
            meta2=dict(meta);meta2.update({
                'construction_score':cscore,'construction_coverage':ccov,'construction_hits':chits,
                'position_evidence':pmeta,'paragraph_first':bool(paragraph_first),
                'proposal_bonus':pbonus,'compute_backend':'cuda','abstract_shape':shape,
                'diversity_penalty':dpen+fdpen})
            row=(score,self.rng.random(),text,meta2,lang,lp,support,rep,op,tpen,trep,cscore,pscore,pbonus)
            if best is None or row[:2]>best[:2]: best=row
        return best


class RendererV12GPU:
    def __init__(self,selector,structure,proposer):
        self.sel=selector;self.structure=structure;self.proposer=proposer

    def render(self,facts,focus_order_hint=None):
        groups,targets=self.structure.bundle(facts,focus_order_hint=focus_order_hint)
        prepared=[];induced_candidates=0;cur_focus=None
        for g,target in zip(groups,targets):
            focus=g[0][1];paragraph_first=(cur_focus is None or focus!=cur_focus)
            cands=[]
            for text,meta in focus_bundle_candidates(g):
                meta=dict(meta);meta['target_length']=target;meta.setdefault('source','verified_v8');cands.append((text,meta))
            new=self.proposer.propose(g,target);induced_candidates+=len(new);cands.extend(new)
            ded=[];seen=set()
            for row in cands:
                if row[0] in seen: continue
                seen.add(row[0]);ded.append(row)
            prepared.append((g,target,focus,paragraph_first,ded));cur_focus=focus

        bypos={False:[],True:[]}
        for _,_,_,pf,cands in prepared: bypos[pf].extend(x[0] for x in cands)
        for pf,texts in bypos.items():
            if texts:self.sel._static_features_many(texts,pf)

        sentences=[];represented=[];picks=[];recent=[];recent_t=[];paragraphs=[];cur_focus=None;cur=[]
        induced_selected=0;shape_counts=Counter();focus_shapes=Counter();prev_focus=None
        focus_order=[]
        for g,target,focus,paragraph_first,ded in prepared:
            if focus!=prev_focus:
                focus_shapes=Counter();focus_order.append(focus);prev_focus=focus
            pick=self.sel.choose(ded,recent,recent_t,paragraph_first=paragraph_first,
                                 shape_counts=shape_counts,focus_shape_counts=focus_shapes)
            text=pick[2]
            if pick[3].get('source')=='induced_wrapper':induced_selected+=1
            sentences.append(text);represented.extend(pick[3]['facts']);picks.append(pick)
            shape=pick[3]['abstract_shape'];shape_counts[shape]+=1;focus_shapes[shape]+=1
            ws=self.sel.s.tokenize(text);opening=' '.join(w for w in ws[:3] if not self.sel.s.is_slot(w))
            recent=(recent+[opening])[-12:];recent_t=(recent_t+[pick[3]['template']])[-12:]
            if cur_focus is None:cur_focus=focus
            if focus!=cur_focus:
                paragraphs.append(' '.join(cur));cur=[];cur_focus=focus
            cur.append(text)
        if cur:paragraphs.append(' '.join(cur))
        out={'text':'\n\n'.join(paragraphs),'sentences':sentences,'paragraphs':paragraphs,
             'represented':represented,'picks':picks,'groups':groups,'targets':targets,
             'induced_candidates':induced_candidates,'induced_selected':induced_selected,
             'compute_backend':'cuda-batched-v12','focus_order':focus_order,
             'abstract_shapes':len(shape_counts)}
        return attach_semantic_traces(out)


def build_renderer_v12_gpu(root,seed=101,use_hot=False,proposal_weight=.24,position_weight=7.0,
                           diversity_weight=2.6,focus_diversity_weight=1.17,repetition_weight=None,
                           template_repetition_weight=None,device=0,memory_limit_mb=4608):
    root=Path(root)
    scorer=GpuBagacoSurfaceScorer(root,use_hot=use_hot,device=device,memory_limit_mb=memory_limit_mb)
    grammar=InducedConstructionGrammar(root)
    if (root/'model'/'quality'/'open.jsonl').exists() and (root/'model'/'full'/'open.jsonl').exists():
        inducer=SafeWrapperInducer(root,scorer)
    else: inducer=StoredWrapperInducer(root,scorer)
    proposer=InducedRealizationProposer(inducer)
    selector=DiversityAwareSelectorGPU(
        scorer,seed=seed,grammar=grammar,
        repetition_weight=(RENDERER_V8_CONFIG['repetition_weight'] if repetition_weight is None else float(repetition_weight)),
        target_weight=RENDERER_V8_CONFIG['target_weight'],
        template_repetition_weight=(RENDERER_V8_CONFIG['template_repetition_weight'] if template_repetition_weight is None else float(template_repetition_weight)),
        construction_weight=RENDERER_V8_CONFIG['construction_weight'],
        position_weight=float(position_weight),proposal_weight=float(proposal_weight),
        diversity_weight=float(diversity_weight),focus_diversity_weight=float(focus_diversity_weight))
    planner=GraphDiscoursePlanner(
        scorer,seed=seed,max_bundle=RENDERER_V8_CONFIG['max_bundle'],
        q_low=RENDERER_V8_CONFIG['q_low'],q_high=RENDERER_V8_CONFIG['q_high'],
        target_scale=RENDERER_V8_CONFIG['target_scale'])
    return scorer,grammar,inducer,RendererV12GPU(selector,planner,proposer)

