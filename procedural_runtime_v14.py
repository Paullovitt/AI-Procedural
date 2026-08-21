from __future__ import annotations

from pathlib import Path
from collections import Counter
from typing import Any
import bisect

from procedural_runtime_v3 import focus_bundle_candidates, _join_items
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


def predicate_relation_candidates(bundle, argument_role='development'):
    """Role-conditioned realizations for already learned relation facts.

    These are generic discourse operators only. They do not change the relation or add
    domain claims; protected semantic IDs remain in every candidate and the CUDA selector
    still chooses the surface using corpus statistics.
    """
    a=bundle[0][1]
    tails=[f'{f[2]} {f[3]}' for f in bundle]
    body=_join_items(tails)
    role=str(argument_role or 'development').lower()
    if role=='opening':
        rows=[
            (f'{a} {body}.','pr_open_direct'),
            (f'De início, {a} {body}.','pr_open_start'),
            (f'Como ponto de partida, {a} {body}.','pr_open_point'),
            (f'Como ponto de partida para o desenvolvimento, {a} {body}.','pr_open_long'),
        ]
    elif role=='synthesis':
        rows=[
            (f'Em síntese, {a} {body}.','pr_syn_summary'),
            (f'Em conjunto, {a} {body}.','pr_syn_together'),
            (f'Por fim, {a} {body}.','pr_syn_final'),
            (f'Ao reunir as associações selecionadas, {a} {body}.','pr_syn_long'),
        ]
    else:
        rows=[
            (f'{a} {body}.','pr_dev_direct'),
            (f'Além disso, {a} {body}.','pr_dev_more'),
            (f'Nesse contexto, {a} {body}.','pr_dev_context'),
            (f'Em complemento, {a} {body}.','pr_dev_complement'),
            (f'Dentro desse recorte temático, {a} {body}.','pr_dev_long_context'),
            (f'Em uma leitura complementar, {a} {body}.','pr_dev_long_reading'),
        ]
    return [(text,{'facts':list(bundle),'template':name,'focus':a,
                   'source':'learned_predicate_v14','argument_role':role}) for text,name in rows]


class RendererV14GPU:
    """V12 surface engine with corpus-learned multi-focus paragraph boundaries."""
    def __init__(self,selector,structure,proposer,paragraph_scheduler,predicate_relations=None,predicate_classes=None,argument_roles=None):
        self.sel=selector;self.structure=structure;self.proposer=proposer;self.paragraph_scheduler=paragraph_scheduler
        self.predicate_relations=set(str(x).lower() for x in (predicate_relations or ()))
        self.predicate_classes={str(k).lower():str(v).lower() for k,v in (predicate_classes or {}).items()}
        self.argument_roles={str(k).lower():str(v).lower() for k,v in (argument_roles or {}).items()}

    def set_prompt_surface(self, lexicon=None, predicate_relations=None, predicate_classes=None, argument_roles=None):
        if lexicon is not None and hasattr(self.sel,'lexicon'):
            self.sel.lexicon={str(k).lower():str(v) for k,v in lexicon.items()}
            if hasattr(self.sel,'_feature_cache'):
                self.sel._feature_cache.clear()
        if predicate_relations is not None:
            self.predicate_relations=set(str(x).lower() for x in predicate_relations)
        if predicate_classes is not None:
            self.predicate_classes={str(k).lower():str(v).lower() for k,v in predicate_classes.items()}
        if argument_roles is not None:
            self.argument_roles={str(k).lower():str(v).lower() for k,v in argument_roles.items()}

    def render(self,facts,focus_order_hint=None):
        base_groups,base_targets=self.structure.bundle(facts,focus_order_hint=focus_order_hint)
        # Prompt predicate facts with different learned meanings should not be fused
        # into one sentence. Split only by lexical predicate class; ordinary V14 facts
        # keep the original corpus-driven grouping unchanged.
        groups=[];targets=[]
        lex=getattr(self.sel,'lexicon',{})
        for g,target in zip(base_groups,base_targets):
            if g and all(f[0]=='rel' and str(f[2]).lower() in self.predicate_relations for f in g):
                buckets=[];where={}
                for f in g:
                    rid=str(f[2]).lower()
                    predicate_label=self.predicate_classes.get(rid,str(lex.get(rid,str(f[2]))).lower())
                    role=self.argument_roles.get(rid,'development')
                    label=(predicate_label,role)
                    if label not in where:
                        where[label]=len(buckets);buckets.append([])
                    buckets[where[label]].append(f)
                for bucket in buckets:
                    groups.append(bucket);targets.append(target)
            else:
                groups.append(g);targets.append(target)
        if self.argument_roles:
            role_order={'opening':0,'development':1,'synthesis':2}
            packed=[]
            for pos,(g,target) in enumerate(zip(groups,targets)):
                if g and all(f[0]=='rel' and str(f[2]).lower() in self.predicate_relations for f in g):
                    role=self.argument_roles.get(str(g[0][2]).lower(),'development')
                else:
                    role='development'
                packed.append((role_order.get(role,1),pos,g,target))
            packed.sort(key=lambda row:(row[0],row[1]))
            groups=[row[2] for row in packed];targets=[row[3] for row in packed]
        starts=self.paragraph_scheduler.starts(len(groups))
        group_roles=[]
        for g in groups:
            if g and all(f[0]=='rel' and str(f[2]).lower() in self.predicate_relations for f in g):
                group_roles.append(self.argument_roles.get(str(g[0][2]).lower(),'development'))
            else:
                group_roles.append('development')
        for gi in range(1,len(group_roles)):
            if group_roles[gi] != group_roles[gi-1]:
                starts.add(gi)
        prepared=[];induced_candidates=0
        for gi,(g,target) in enumerate(zip(groups,targets)):
            focus=g[0][1];paragraph_first=(gi in starts);cands=[]
            predicate_group=(g and all(f[0]=='rel' and str(f[2]).lower() in self.predicate_relations for f in g))
            argument_role=group_roles[gi]
            if predicate_group:
                for text,meta in predicate_relation_candidates(g,argument_role=argument_role):
                    meta=dict(meta);meta['target_length']=target;cands.append((text,meta))
                # The historical wrapper proposer is based on opaque relation microclauses.
                # Skipping it here avoids re-introducing "relation X liga" around verb-phrase slots.
                new=[]
            else:
                for text,meta in focus_bundle_candidates(g):
                    meta=dict(meta);meta['target_length']=target;meta.setdefault('source','verified_v8');cands.append((text,meta))
                new=self.proposer.propose(g,target)
            induced_candidates+=len(new);cands.extend(new)
            ded=[];seen=set()
            for row in cands:
                if row[0] in seen:continue
                seen.add(row[0]);ded.append(row)
            prepared.append((g,target,focus,paragraph_first,argument_role,ded))

        bypos={False:[],True:[]}
        for _,_,_,pf,_,cands in prepared:bypos[pf].extend(x[0] for x in cands)
        for pf,texts in bypos.items():
            if texts:self.sel._static_features_many(texts,pf)

        sentences=[];represented=[];picks=[];recent=[];recent_t=[];paragraphs=[];cur=[]
        induced_selected=0;shape_counts=Counter();focus_shapes=Counter();prev_focus=None;focus_order=[]
        paragraph_sizes=[]
        for gi,(g,target,focus,pf,argument_role,ded) in enumerate(prepared):
            if pf and cur:
                paragraphs.append(' '.join(cur));paragraph_sizes.append(len(cur));cur=[]
            if focus!=prev_focus:
                focus_shapes=Counter();focus_order.append(focus);prev_focus=focus
            if recent_t and len(ded)>1:
                alternatives=[row for row in ded if row[1].get('template')!=recent_t[-1]]
                if alternatives:
                    ded=alternatives
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
             'paragraph_starts':sorted(starts),'predicate_relations':len(self.predicate_relations),
             'argument_role_order':group_roles,'argument_phase_counts':dict(Counter(group_roles))}
        return attach_semantic_traces(out)


def build_renderer_v14_gpu(root,seed=101,use_hot=False,proposal_weight=.24,position_weight=7.0,
                           diversity_weight=2.6,focus_diversity_weight=1.17,repetition_weight=1.1,
                           template_repetition_weight=None,device=0,memory_limit_mb=4608,lexicon=None,
                           lexicalize_entities=False,predicate_relations=None,argument_roles=None,max_bundle=None):
    root=Path(root);scorer=GpuBagacoSurfaceScorer(root,use_hot=use_hot,device=device,memory_limit_mb=memory_limit_mb)
    grammar=InducedConstructionGrammar(root)
    inducer=SafeWrapperInducer(root,scorer) if (root/'model'/'quality'/'open.jsonl').exists() and (root/'model'/'full'/'open.jsonl').exists() else StoredWrapperInducer(root,scorer)
    proposer=InducedRealizationProposer(inducer)
    lexical_mode=lexicon is not None
    selector_cls=LexicalizedDiversitySelectorGPU if lexical_mode else DiversityAwareSelectorGPU
    selector_kwargs: dict[str, Any]=dict(
        scorer=scorer,seed=seed,grammar=grammar,repetition_weight=float(repetition_weight),
        target_weight=RENDERER_V8_CONFIG['target_weight'],
        template_repetition_weight=(RENDERER_V8_CONFIG['template_repetition_weight'] if template_repetition_weight is None else float(template_repetition_weight)),
        construction_weight=RENDERER_V8_CONFIG['construction_weight'],position_weight=float(position_weight),
        proposal_weight=float(proposal_weight),diversity_weight=float(diversity_weight),focus_diversity_weight=float(focus_diversity_weight))
    if lexical_mode:
        selector_kwargs['lexicon']=lexicon or {}
        selector_kwargs['lexicalize_entities']=bool(lexicalize_entities)
    selector=selector_cls(**selector_kwargs)
    planner=GraphDiscoursePlanner(scorer,seed=seed,max_bundle=int(max_bundle or RENDERER_V8_CONFIG['max_bundle']),q_low=RENDERER_V8_CONFIG['q_low'],q_high=RENDERER_V8_CONFIG['q_high'],target_scale=RENDERER_V8_CONFIG['target_scale'])
    ps=EmpiricalParagraphScheduler(scorer.struct.get('para_sent',[]),seed=seed+7919)
    return scorer,grammar,inducer,RendererV14GPU(selector,planner,proposer,ps,predicate_relations=predicate_relations,argument_roles=argument_roles)
