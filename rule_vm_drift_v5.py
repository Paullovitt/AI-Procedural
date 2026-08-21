from pathlib import Path
import json, math, random, time
import numpy as np
import torch
from autonomous_rule_vm_v2 import OpaqueTransitionWorld, _h2, _log2_choose
from autonomous_rule_vm_v4 import BoundedMDLRuleInducerGPU
from autonomous_rule_vm_v5 import CertifiedParallelRuleVMGPU

OUT=Path('rigorous_results_v12');OUT.mkdir(exist_ok=True)
SEED=733991
base=OpaqueTransitionWorld(seed=SEED,n_attr=7,n_rel=12,max_gate_order=3)
pre_train=base.observations(140000,seed=1,noise=.02);pre_val=base.observations(50000,seed=2,noise=0)
L0=BoundedMDLRuleInducerGPU(base.n_rel,base.n_feat,initial_order_cap=3,hard_resource_cap=5,batch_subsets=96).fit(pre_train,pre_val)

# AUDIT-ONLY drift construction. Learner is not told which relation laws changed.
post=OpaqueTransitionWorld(seed=SEED,n_attr=7,n_rel=12,max_gate_order=3)
donor=OpaqueTransitionWorld(seed=991337,n_attr=7,n_rel=12,max_gate_order=3)
changed_ids=[1,5,9]
for r in changed_ids: post.laws[r]=donor.laws[r]

post_train=post.observations(120000,seed=31,noise=.02);post_val=post.observations(50000,seed=32,noise=0);post_test=post.observations(100000,seed=33,noise=0)
L1=BoundedMDLRuleInducerGPU(post.n_rel,post.n_feat,initial_order_cap=3,hard_resource_cap=5,batch_subsets=96).fit(post_train,post_val)

# Generic MDL registry revision: compare old vs newly induced code on recent observations.
def model_bits(m,n_feat=base.n_feat,cap=5):
    k=len(m['features']);return math.log2(cap+1.0)+_log2_choose(n_feat,k)+(1<<k)

def rel_error(model,r,rel,X,y):
    vm=CertifiedParallelRuleVMGPU({r:model})
    mask=(rel==r);pred=vm.predict_batch(rel[mask],X[mask]);return float((pred!=y[mask]).mean()),int(mask.sum())

registry={};decisions=[]
for r in range(base.n_rel):
    old=L0.models[r];new=L1.models[r]
    eo,n=rel_error(old,r,*post_val);en,_=rel_error(new,r,*post_val)
    old_bits=model_bits(old)+n*_h2(eo)+math.log2(n+2.0)
    new_bits=model_bits(new)+n*_h2(en)+math.log2(n+2.0)
    # Minimum-description-length decides; equality keeps existing memory (minimum rewrite cost).
    use_new=(new_bits < old_bits)
    registry[r]=new if use_new else old
    decisions.append({'relation':r,'old_error':eo,'new_error':en,'old_bits':old_bits,'new_bits':new_bits,'revised':use_new,'bit_gain':old_bits-new_bits})

vm_pre=CertifiedParallelRuleVMGPU(L0);vm_post=CertifiedParallelRuleVMGPU(registry)
pre_acc=float((vm_pre.predict_batch(post_test[0],post_test[1])==post_test[2]).mean())
post_acc=float((vm_post.predict_batch(post_test[0],post_test[1])==post_test[2]).mean())
revised={d['relation'] for d in decisions if d['revised']}
closure=[];proof=[]
for wi in range(24):
    attrs,edges=post.random_network(4000+wi,n_nodes=170,n_edges=850);seeds=[wi%170,(wi*23+4)%170]
    os,op=post.execute_fixed_point(attrs,edges,seeds);ps,pp,meta=vm_post.execute_fixed_point(attrs,edges,seeds)
    closure.append(np.array_equal(os,ps))
    for node in list(np.flatnonzero(ps))[:25]:
        path=vm_post.proof(pp,int(node));st=np.zeros(len(attrs),dtype=np.uint8);st[seeds]=1;ok=True
        for a,r,b in path:
            nxt=post.step_edge(r,int(st[a]),int(st[b]),attrs[a],attrs[b])
            if nxt==st[b]:ok=False;break
            st[b]=nxt
        proof.append(ok)
result={'format':'Autonomous-RuleVM-v5-Drift-Revision','hidden_changed_ids_audit_only':changed_ids,
        'pre_revision_post_accuracy':pre_acc,'post_revision_accuracy':post_acc,
        'revised_ids':sorted(revised),'changed_detected':len(revised & set(changed_ids)),
        'changed_total':len(changed_ids),'false_revisions':len(revised-set(changed_ids)),
        'closure_exact_rate':float(np.mean(closure)),'proof_validity':float(np.mean(proof)),
        'decisions':decisions,'gpu':vm_post.status()}
(OUT/'rule_vm_v5_drift.json').write_text(json.dumps(result,indent=2),encoding='utf8')
print(json.dumps({k:v for k,v in result.items() if k!='decisions'},indent=2))
for d in decisions:print('DECISION',json.dumps(d),flush=True)
