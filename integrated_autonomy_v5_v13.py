from pathlib import Path
from collections import Counter
import json, re, statistics, time
import numpy as np

from autonomous_rule_vm_v2 import OpaqueTransitionWorld
from autonomous_rule_vm_v4 import BoundedMDLRuleInducerGPU
from autonomous_rule_vm_v5 import CertifiedParallelRuleVMGPU
from procedural_runtime_v12 import build_renderer_v12_gpu
from procedural_runtime_v13 import build_renderer_v13_gpu
from procedural_runtime_gpu import SemanticTraceVerifier, GpuLocalOrderVerifier
from procedural_runtime_v5 import ProtectedSlotVerifier

ROOT=Path('.');OUT=ROOT/'rigorous_results_v12';OUT.mkdir(exist_ok=True)
SEED=845219
world=OpaqueTransitionWorld(seed=SEED,n_attr=7,n_rel=12,max_gate_order=3)
train=world.observations(150000,seed=71,noise=.03);val=world.observations(55000,seed=72,noise=0);test=world.observations(90000,seed=73,noise=0)
t0=time.perf_counter();L=BoundedMDLRuleInducerGPU(world.n_rel,world.n_feat,initial_order_cap=3,hard_resource_cap=5,batch_subsets=96).fit(train,val);learn_s=time.perf_counter()-t0
vm=CertifiedParallelRuleVMGPU(L);transition_acc=float((vm.predict_batch(test[0],test[1])==test[2]).mean())

attrs,edges=world.random_network(991811,n_nodes=180,n_edges=1000)
analyses=[]
for s in range(len(attrs)):
    st,pa,meta=vm.execute_fixed_point(attrs,edges,[s]);nodes=set(np.flatnonzero(st));depths=[len(vm.proof(pa,int(n))) for n in nodes]
    analyses.append({'seed':s,'size':len(nodes),'max_depth':max(depths or [0]),'state':st,'parent':pa})
analyses.sort(key=lambda x:(x['size'],x['max_depth'],x['seed']),reverse=True)
top=analyses[:5];primary=top[0]
# deepest learned proof from primary
active=list(np.flatnonzero(primary['state']));target=max(active,key=lambda n:len(vm.proof(primary['parent'],int(n))))
proof=vm.proof(primary['parent'],int(target))

# AUDIT ONLY: hidden oracle never enters learner/planner.
oracle_rank=[]
for s in range(len(attrs)):
    st,_=world.execute_fixed_point(attrs,edges,[s]);oracle_rank.append((int(st.sum()),s))
oracle_rank.sort(reverse=True)
rank_exact=[x['seed'] for x in top]==[s for _,s in oracle_rank[:5]]
ostate,_=world.execute_fixed_point(attrs,edges,[primary['seed']]);closure_exact=bool(np.array_equal(primary['state'],ostate))
proof_valid=True;st=np.zeros(len(attrs),dtype=np.uint8);st[primary['seed']]=1
for a,r,b in proof:
    nxt=world.step_edge(r,int(st[a]),int(st[b]),attrs[a],attrs[b])
    if nxt==st[b]:proof_valid=False;break
    st[b]=nxt

# Semantic report: all claims come from learned closures/proof traces.
facts=[]
for row in top:
    facts.append(('prop',f'e{row["seed"]:05d}','a00000',f'v{row["size"]:05d}'))
    facts.append(('prop',f'e{row["seed"]:05d}','a00001',f'v{row["max_depth"]:05d}'))
for a,r,b in proof:facts.append(('rel',f'e{a:05d}',f'r{r:05d}',f'e{b:05d}'))
focus_hint=[f'e{primary["seed"]:05d}']+[f'e{b:05d}' for _,_,b in proof]+[f'e{x["seed"]:05d}' for x in top[1:]]
# de-duplicate while preserving proof/ranking priority
focus_hint=list(dict.fromkeys(focus_hint))

# Lexicon is only for surface scoring/display; it carries no hidden rule or conclusion.
lex={'a00000':'alcance previsto','a00001':'profundidade da cadeia'}
for r in range(world.n_rel):lex[f'r{r:05d}']='transição'
for row in top:
    lex[f'v{row["size"]:05d}']=str(row['size']);lex[f'v{row["max_depth"]:05d}']=str(row['max_depth'])

sc12,_,_,r12=build_renderer_v12_gpu(ROOT,seed=919,proposal_weight=.24,position_weight=7.0,diversity_weight=2.6,focus_diversity_weight=1.17,repetition_weight=1.1,device=0,memory_limit_mb=4608)
sc13,_,_,r13=build_renderer_v13_gpu(ROOT,lex,seed=919,proposal_weight=.24,position_weight=7.0,diversity_weight=2.6,focus_diversity_weight=1.17,repetition_weight=1.1,device=0,memory_limit_mb=4608)
o12=r12.render(facts,focus_order_hint=focus_hint);o13=r13.render(facts,focus_order_hint=focus_hint)

slot_rx=re.compile(r'\b(?:e\d+|a\d+|v\d+|r\d+)\b',re.I)
def readable(text):
    def repl(m):
        w=m.group(0).lower()
        if w.startswith('e'):return 'Componente '+str(int(w[1:]))
        if w in lex:return lex[w]
        if w.startswith('v'):return str(int(w[1:]))
        if w.startswith('r'):return 'transição '+str(int(w[1:]))
        return w
    return slot_rx.sub(repl,text)

def lexical_metrics(sc,out):
    ss=[readable(x) for x in out['sentences']];langs=[];sups=[]
    if ss:
        l,s=sc.batch_language_support([sc.tokenize(x) for x in ss],max_order=5,slot_aware=False);langs=list(map(float,l));sups=list(map(float,s))
    return {'score':statistics.mean(langs) if langs else 0,'support':statistics.mean(sups) if sups else 0}

vf=ProtectedSlotVerifier();tv=SemanticTraceVerifier();ov=GpuLocalOrderVerifier(sc13,threshold=0.6576928743724331,window=9)
m12=lexical_metrics(sc12,o12);m13=lexical_metrics(sc13,o13)
order_alarms=sum(len(ov.inspect(s)) for s in o13['sentences'])
result={'format':'Integrated-Autonomy-V5-V13','learn_seconds':learn_s,'transition_accuracy':transition_acc,
        'top5_exact_order':rank_exact,'primary_closure_exact':closure_exact,'proof_valid':proof_valid,
        'primary_seed':primary['seed'],'primary_reach':primary['size'],'proof_target':int(target),'proof_depth':len(proof),
        'semantic_facts':len(facts),'v12':{'sentences':len(o12['sentences']),'semantic_exact':Counter(o12['represented'])==Counter(facts),'slot_errors':len(vf.inspect_render(o12)),'trace_errors':len(tv.inspect_render(o12)),**m12},
        'v13':{'sentences':len(o13['sentences']),'semantic_exact':Counter(o13['represented'])==Counter(facts),'slot_errors':len(vf.inspect_render(o13)),'trace_errors':len(tv.inspect_render(o13)),'order_alarms':order_alarms,**m13},
        'selected_rule_orders':[L.models[r]['order'] for r in range(world.n_rel)],'gpu_reasoner':vm.status(),'gpu_renderer':sc13.gpu_status()}
(OUT/'integrated_autonomy_v5_v13.json').write_text(json.dumps(result,indent=2),encoding='utf8')
(OUT/'integrated_autonomy_v5_v13.txt').write_text(readable(o13['text']),encoding='utf8')
print(json.dumps(result,ensure_ascii=False,indent=2));print('---REPORT---');print(readable(o13['text']))
