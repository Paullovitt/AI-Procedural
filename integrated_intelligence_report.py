from pathlib import Path
from collections import Counter
import json, random, statistics, time
import numpy as np

from autonomous_reasoning_gpu import HiddenPropagationWorld, GPUTruthTableRuleLearner, path_from_parent
from procedural_runtime_v12 import build_renderer_v12_gpu
from procedural_runtime_gpu import SemanticTraceVerifier
from procedural_runtime_v5 import ProtectedSlotVerifier

ROOT=Path(__file__).resolve().parent
OUT=ROOT/'rigorous_results_v12';OUT.mkdir(exist_ok=True)

class ReasoningReportPlanner:
    """Domain-agnostic report planner over a learned transition model.

    It never reads hidden simulator rules. It ranks interventions by predicted closure
    size and emits only claims backed by a learned proof trace.
    """
    def __init__(self, learner): self.learner=learner

    def analyze(self, attrs, edges, top_k=6):
        n=len(attrs); rows=[]; closures={}; parents={}
        for s in range(n):
            cl,pa=self.learner.closure(attrs,edges,[s]);closures[s]=cl;parents[s]=pa
            rows.append((len(cl),s))
        rows.sort(reverse=True)
        top=rows[:top_k]
        primary=top[0][1]
        # Deepest trace is the most explanatory non-trivial consequence available.
        paths=[]
        for node in closures[primary]:
            p=path_from_parent(parents[primary],node)
            if p: paths.append((len(p),node,p))
        deepest=max(paths,default=(0,primary,[]))
        return {'ranking':top,'primary':primary,'closure':closures[primary],
                'deepest_target':deepest[1],'deepest_path':deepest[2],
                'all_closures':closures}

    @staticmethod
    def semantic_plan(analysis):
        facts=[]
        # Raw numeric consequences; no manually invented risk categories.
        for size,node in analysis['ranking']:
            facts.append(('prop',f'e{node:05d}','a00000',f'v{size:05d}'))
        depth=len(analysis['deepest_path'])
        facts.append(('prop',f'e{analysis["primary"]:05d}','a00001',f'v{depth:05d}'))
        # Proof edges come strictly from the learner's own parent trace.
        for a,r,b in analysis['deepest_path']:
            facts.append(('rel',f'e{a:05d}',f'r{r:05d}',f'e{b:05d}'))
        return facts


def readable(text, analysis):
    # Display-only renaming. No reasoning occurs here.
    repl={
        'a00000':'alcance previsto',
        'a00001':'profundidade da cadeia',
    }
    # Relations stay distinguishable: the learner may have learned different mechanisms.
    for r in range(100): repl[f'r{r:05d}']=f'transi??o {r}'
    # Give opaque objects stable human-readable labels.
    for i in range(1000): repl[f'e{i:05d}']=f'Componente {i}'
    for size,node in analysis['ranking']: repl[f'v{size:05d}']=f'{size} componentes'
    repl[f'v{len(analysis["deepest_path"]):05d}']=f'{len(analysis["deepest_path"])} etapas'
    # longest keys first to avoid accidental prefix collisions
    for a,b in sorted(repl.items(),key=lambda kv:-len(kv[0])): text=text.replace(a,b)
    return text


def main():
    # New hidden law set, separate from the earlier benchmark.
    world=HiddenPropagationWorld(seed=975318,n_attr=8,n_rel=14,max_order=3)
    train=world.observations(300000,seed=91,noise=.02)
    val=world.observations(100000,seed=92,noise=0)
    test=world.observations(160000,seed=93,noise=0)
    t0=time.perf_counter();L=GPUTruthTableRuleLearner(world.n_rel,world.n_feat,max_order=3).fit(train,val);learn_s=time.perf_counter()-t0
    pred=L.predict_batch(test[0],test[1]);transition_acc=float((pred==test[2]).mean())

    attrs,edges=world.random_network(seed=99117,n_nodes=160,n_edges=900)
    planner=ReasoningReportPlanner(L);A=planner.analyze(attrs,edges,top_k=6)
    plan=planner.semantic_plan(A)

    # Audit only: oracle is never passed to planner/learner.
    oracle_sizes=[]
    for s in range(len(attrs)): oracle_sizes.append((len(world.closure(attrs,edges,[s])[0]),s))
    oracle_sizes.sort(reverse=True)
    learned_top=[x[1] for x in A['ranking']];oracle_top=[x[1] for x in oracle_sizes[:6]]
    top_exact=(learned_top==oracle_top)
    opath=world.closure(attrs,edges,[A['primary']])[0]
    closure_exact=(A['closure']==opath)
    proof_valid=all(world.transmit(r,attrs[a],attrs[b]) for a,r,b in A['deepest_path'])

    sc,gr,ind,renderer=build_renderer_v12_gpu(ROOT,seed=777,proposal_weight=.24,position_weight=7.0,
        diversity_weight=2.6,focus_diversity_weight=1.17,device=0,memory_limit_mb=4608)
    proof_focus=[A['primary']]+[b for _,_,b in A['deepest_path']]
    proof_focus += [node for _,node in A['ranking'] if node not in proof_focus]
    out=renderer.render(plan,focus_order_hint=[f'e{x:05d}' for x in proof_focus]);vf=ProtectedSlotVerifier();tv=SemanticTraceVerifier()
    exact=Counter(out['represented'])==Counter(plan);slot_err=len(vf.inspect_render(out));trace_err=len(tv.inspect_render(out))

    raw=out['text'];pretty=readable(raw,A)
    (OUT/'intelligence_report_raw.txt').write_text(raw,encoding='utf8')
    (OUT/'intelligence_report_readable.txt').write_text(pretty,encoding='utf8')
    result={
        'format':'Integrated-Autonomous-Intelligence-Report-V12',
        'train_observations':len(train[0]),'train_noise':.02,'learn_seconds':learn_s,
        'transition_accuracy':transition_acc,'network_nodes':len(attrs),'network_edges':len(edges),
        'top6_exact_order':top_exact,'closure_exact':closure_exact,'proof_valid':proof_valid,
        'primary_node':A['primary'],'primary_cascade_size':len(A['closure']),
        'deepest_target':A['deepest_target'],'proof_depth':len(A['deepest_path']),
        'semantic_plan_facts':len(plan),'render_sentences':len(out['sentences']),
        'semantic_exact':exact,'slot_errors':slot_err,'trace_errors':trace_err,
        'induced_selected':out['induced_selected'],'gpu':sc.gpu_status(),
    }
    (OUT/'integrated_intelligence_report.json').write_text(json.dumps(result,indent=2),encoding='utf8')
    print(json.dumps(result,indent=2));print('---REPORT---');print(pretty)

if __name__=='__main__':main()
