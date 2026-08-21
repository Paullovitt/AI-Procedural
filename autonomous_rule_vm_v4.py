from __future__ import annotations

from pathlib import Path
from typing import Any
import json, math, time
import numpy as np
import torch

from autonomous_rule_vm_v2 import OpaqueTransitionWorld, RuleBankVMGPU, _log2_choose
from autonomous_rule_vm_v3 import BatchedMDLRuleInducerGPU


class BoundedMDLRuleInducerGPU(BatchedMDLRuleInducerGPU):
    """Autonomous complexity growth with an MDL lower-bound stopping proof.

    A higher interaction order is explored iff its *best theoretically possible* code
    (zero residual error) could still beat the current rule. No hand-picked error or
    promotion threshold is used.
    """
    def _zero_residual_lower_bound(self,k,n_val):
        model_bits=math.log2(self.hard_resource_cap+1.0)+_log2_choose(self.n_feat,k)+(1<<k)
        return model_bits+math.log2(n_val+2.0)

    def fit(self,train,val,progress=False):
        tr_rel,tr_X,tr_y=train;va_rel,va_X,va_y=val
        trR=torch.as_tensor(tr_rel,device=self.device,dtype=torch.long)
        trX=torch.as_tensor(tr_X,device=self.device,dtype=torch.uint8)
        trY=torch.as_tensor(tr_y,device=self.device,dtype=torch.long)
        vaR=torch.as_tensor(va_rel,device=self.device,dtype=torch.long)
        vaX=torch.as_tensor(va_X,device=self.device,dtype=torch.uint8)
        vaY=torch.as_tensor(va_y,device=self.device,dtype=torch.long)
        registry={};trace={}
        for r in range(self.n_rel):
            X=trX[trR==r];y=trY[trR==r];XV=vaX[vaR==r];yV=vaY[vaR==r]
            best: Any=None;searched=[]
            initial=min(self.initial_order_cap,self.hard_resource_cap)
            for k in range(initial+1):
                best=self._search_order(X,y,XV,yV,k,best);searched.append(k)
            k=initial+1
            while k<=self.hard_resource_cap and self._zero_residual_lower_bound(k,len(yV)) < best[0]:
                best=self._search_order(X,y,XV,yV,k,best);searched.append(k);k+=1
            registry[r]={
                'features':list(best[4]),'table':list(best[5]),'order':int(best[2]),
                'train_error':float(best[3]),'val_error':float(best[1]),
                'description_bits':float(best[0]),'model_bits':float(best[6]),
                'residual_bits':float(best[7]),'val_examples':int((va_rel==r).sum()),
                'searched_orders':searched,
                'next_order_lower_bound':None if k>self.hard_resource_cap else float(self._zero_residual_lower_bound(k,len(yV)))
            }
            trace[r]={'searched_orders':searched,'selected_order':int(best[2]),'val_error':float(best[1]),'description_bits':float(best[0])}
            if progress:print('REL',r,json.dumps(trace[r]),flush=True)
        self.models=registry;self.search_trace=trace;torch.cuda.synchronize(self.device);return self

    def save_rulebank(self,path):
        data={'format':'Autonomous-RuleBank-MDL-v4','n_rel':self.n_rel,'n_feat':self.n_feat,
              'initial_order_cap':self.initial_order_cap,'hard_resource_cap':self.hard_resource_cap,
              'selection':'minimum-description-length; complexity expands only while a zero-residual higher-order model can beat current code length',
              'models':self.models}
        Path(path).write_text(json.dumps(data,indent=2),encoding='utf8');return data


def run_one(seed_world=862771,train_n=150000,val_n=60000,test_n=100000,noise=.015,progress=True):
    world=OpaqueTransitionWorld(seed=seed_world,n_attr=7,n_rel=12,max_gate_order=2)
    train=world.observations(train_n,seed=seed_world+1,noise=noise)
    val=world.observations(val_n,seed=seed_world+2,noise=0.0)
    test=world.observations(test_n,seed=seed_world+3,noise=0.0)
    torch.cuda.reset_peak_memory_stats();t0=time.perf_counter()
    L=BoundedMDLRuleInducerGPU(world.n_rel,world.n_feat,initial_order_cap=3,hard_resource_cap=5,batch_subsets=96).fit(train,val,progress=progress)
    learn_s=time.perf_counter()-t0;vm=RuleBankVMGPU(L)
    pred=vm.predict_batch(test[0],test[1]);acc=float((pred==test[2]).mean())
    exact=[];jacc=[];proof_ok=[];depth=[]
    for wi in range(24):
        attrs,edges=world.random_network(seed_world+10000+wi,n_nodes=130,n_edges=600);seeds=[wi%130,(wi*11+5)%130]
        os,op=world.execute_fixed_point(attrs,edges,seeds);ps,pp=vm.execute_fixed_point(attrs,edges,seeds)
        O=set(np.flatnonzero(os));P=set(np.flatnonzero(ps));exact.append(O==P);jacc.append(len(O&P)/max(1,len(O|P)))
        for node in list(P)[:20]:
            path=vm.proof(pp,node);depth.append(len(path));st=np.zeros(len(attrs),dtype=np.uint8);st[seeds]=1;ok=True
            for a,r,b in path:
                nxt=world.step_edge(r,int(st[a]),int(st[b]),attrs[a],attrs[b])
                if nxt==st[b]:ok=False;break
                st[b]=nxt
            proof_ok.append(ok)
    return world,L,vm,{
        'world_seed':seed_world,'train_examples':train_n,'train_noise':noise,'val_examples':val_n,'test_examples':test_n,
        'learn_seconds':learn_s,'test_transition_accuracy':acc,
        'selected_orders':[L.models[r]['order'] for r in range(world.n_rel)],
        'searched_orders':[L.models[r]['searched_orders'] for r in range(world.n_rel)],
        'val_errors':[L.models[r]['val_error'] for r in range(world.n_rel)],
        'closure_exact_rate':float(np.mean(exact)),'closure_jaccard':float(np.mean(jacc)),
        'proof_validity':float(np.mean(proof_ok)),'max_proof_depth':int(max(depth or [0])),
        'peak_allocated_mb':torch.cuda.max_memory_allocated()/2**20,'gpu':vm.status(),
        'hidden_gate_orders_audit_only':[len(x.gate_features) for x in world.laws]
    }


if __name__=='__main__':
    out=Path('rigorous_results_v12');out.mkdir(exist_ok=True)
    world,L,vm,result=run_one(progress=True)
    L.save_rulebank(out/'AUTONOMOUS_RULEBANK_MDL_V4.json')
    result['format']='Autonomous-RuleVM-MDL-GPU-v4'
    (out/'autonomous_rule_vm_v4.json').write_text(json.dumps(result,indent=2),encoding='utf8')
    print('FINAL',json.dumps(result),flush=True)
