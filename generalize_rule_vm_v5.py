from pathlib import Path
import json, statistics, time
import numpy as np
import torch
from autonomous_rule_vm_v2 import OpaqueTransitionWorld
from autonomous_rule_vm_v4 import BoundedMDLRuleInducerGPU
from autonomous_rule_vm_v5 import CertifiedParallelRuleVMGPU

OUT=Path('rigorous_results_v12');OUT.mkdir(exist_ok=True)
SEEDS=[120031,220037,320041,420047,520049,620053]
NOISE=[0.0,0.01,0.02,0.03,0.05,0.04]
rows=[]
for wi,(seed,noise) in enumerate(zip(SEEDS,NOISE)):
    # max_gate_order=3 means complete hidden local laws can require interaction order 5
    world=OpaqueTransitionWorld(seed=seed,n_attr=7,n_rel=10,max_gate_order=3)
    train=world.observations(120000,seed=seed+1,noise=noise)
    val=world.observations(45000,seed=seed+2,noise=0.0)
    test=world.observations(80000,seed=seed+3,noise=0.0)
    torch.cuda.reset_peak_memory_stats();t0=time.perf_counter()
    L=BoundedMDLRuleInducerGPU(world.n_rel,world.n_feat,initial_order_cap=3,hard_resource_cap=5,batch_subsets=96).fit(train,val,progress=False)
    learn_s=time.perf_counter()-t0;vm=CertifiedParallelRuleVMGPU(L)
    pred=vm.predict_batch(test[0],test[1]);acc=float((pred==test[2]).mean())
    exact=[];jac=[];proof=[];maxdepth=0
    for ni in range(12):
        attrs,edges=world.random_network(seed+1000+ni,n_nodes=160,n_edges=800);seeds=[ni%160,(ni*19+3)%160]
        os,op=world.execute_fixed_point(attrs,edges,seeds);ps,pp,meta=vm.execute_fixed_point(attrs,edges,seeds)
        O=set(np.flatnonzero(os));P=set(np.flatnonzero(ps));exact.append(O==P);jac.append(len(O&P)/max(1,len(O|P)))
        for node in list(P)[:24]:
            path=vm.proof(pp,node);maxdepth=max(maxdepth,len(path));st=np.zeros(len(attrs),dtype=np.uint8);st[seeds]=1;ok=True
            for a,r,b in path:
                nxt=world.step_edge(r,int(st[a]),int(st[b]),attrs[a],attrs[b])
                if nxt==st[b]:ok=False;break
                st[b]=nxt
            proof.append(ok)
    row={'world':wi,'seed':seed,'train_noise':noise,'transition_accuracy':acc,
         'closure_exact':float(np.mean(exact)),'closure_jaccard':float(np.mean(jac)),
         'proof_validity':float(np.mean(proof)),'max_proof_depth':maxdepth,
         'selected_orders':[L.models[r]['order'] for r in range(world.n_rel)],
         'max_selected_order':max(L.models[r]['order'] for r in range(world.n_rel)),
         'all_val_zero':all(L.models[r]['val_error']==0.0 for r in range(world.n_rel)),
         'parallel_certified':vm.parallel_certified,'learn_seconds':learn_s,
         'peak_allocated_mb':torch.cuda.max_memory_allocated()/2**20,
         'hidden_gate_orders_audit_only':[len(x.gate_features) for x in world.laws]}
    rows.append(row);print('WORLD',json.dumps(row),flush=True)
summary={'format':'Autonomous-RuleVM-v5-Generality','worlds':len(rows),'rows':rows,
         'mean_transition_accuracy':statistics.mean(r['transition_accuracy'] for r in rows),
         'min_transition_accuracy':min(r['transition_accuracy'] for r in rows),
         'mean_closure_exact':statistics.mean(r['closure_exact'] for r in rows),
         'mean_proof_validity':statistics.mean(r['proof_validity'] for r in rows),
         'all_parallel_certified':all(r['parallel_certified'] for r in rows),
         'worlds_requiring_order5':sum(r['max_selected_order']>=5 for r in rows),
         'total_learn_seconds':sum(r['learn_seconds'] for r in rows)}
(OUT/'rule_vm_v5_generality.json').write_text(json.dumps(summary,indent=2),encoding='utf8')
print('FINAL',json.dumps({k:v for k,v in summary.items() if k!='rows'}),flush=True)
