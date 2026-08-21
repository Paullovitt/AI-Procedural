from __future__ import annotations

from pathlib import Path
from typing import Any
import json, time
import numpy as np
import torch

from autonomous_rule_vm_v2 import OpaqueTransitionWorld, RuleBankVMGPU
from autonomous_rule_vm_v4 import BoundedMDLRuleInducerGPU


class CertifiedParallelRuleVMGPU(RuleBankVMGPU):
    """Executes learned rule data in parallel only after proving the rule bank safe.

    The proof is computed from each learned truth table itself. No relation name, domain rule,
    propagation rule, or benchmark answer is encoded here.
    """
    def __init__(self,rulebank,device='cuda:0'):
        super().__init__(rulebank,device=device)
        self.certificates={int(r):self._certify_model(m) for r,m in self.models.items()}
        self.parallel_certified=all(x['safe_parallel_fixed_point'] for x in self.certificates.values())

    @staticmethod
    def _certify_model(m):
        sub=[int(x) for x in m['features']];tab=[int(x) for x in m['table']]
        def monotone_feature(feature):
            if feature not in sub:return True
            p=sub.index(feature);bit=1<<p
            return all(tab[c] <= tab[c|bit] for c in range(len(tab)) if (c&bit)==0)
        def inflationary_dst():
            if 1 not in sub:
                return all(v==1 for v in tab)
            p=sub.index(1);bit=1<<p
            return all(tab[c]==1 for c in range(len(tab)) if (c&bit)!=0)
        ms=monotone_feature(0);md=monotone_feature(1);inf=inflationary_dst()
        return {'monotone_src':ms,'monotone_dst':md,'inflationary_dst':inf,
                'safe_parallel_fixed_point':bool(ms and md and inf)}

    def predict_tensor(self,relT,XT):
        out=torch.zeros(len(relT),device=self.device,dtype=torch.long)
        for r,m in self.models.items():
            mask=(relT==int(r))
            if not bool(mask.any()):continue
            sub=tuple(int(x) for x in m['features']);k=len(sub)
            idx=torch.tensor(sub,device=self.device,dtype=torch.long)
            bits=XT[mask][:,idx].long()
            weights=2**torch.arange(k,device=self.device,dtype=torch.long)
            code=(bits*weights).sum(1) if k else torch.zeros(int(mask.sum().item()),device=self.device,dtype=torch.long)
            tab=torch.tensor(m['table'],device=self.device,dtype=torch.long)
            out[mask]=tab[code]
        return out

    def predict_batch(self,rel,X):
        relT=torch.as_tensor(rel,device=self.device,dtype=torch.long)
        XT=torch.as_tensor(X,device=self.device,dtype=torch.uint8)
        return self.predict_tensor(relT,XT).detach().cpu().numpy().astype(np.uint8)

    def execute_fixed_point(self,attrs,edges,seeds,max_sweeps=None) -> Any:
        if not self.parallel_certified:
            state,parent=super().execute_fixed_point(attrs,edges,seeds,max_sweeps=max_sweeps)
            return state,parent,{'parallel':False,'sweeps':None,'certificates':self.certificates}
        A=torch.as_tensor(attrs,device=self.device,dtype=torch.uint8)
        e=np.asarray(edges,dtype=np.int64);src=torch.as_tensor(e[:,0],device=self.device);rel=torch.as_tensor(e[:,1],device=self.device);dst=torch.as_tensor(e[:,2],device=self.device)
        state=torch.zeros(len(attrs),device=self.device,dtype=torch.long);state[torch.as_tensor(list(seeds),device=self.device,dtype=torch.long)]=1
        parent: dict[int, Any]={int(x):None for x in seeds};cap=int(max_sweeps or (len(attrs)+1));sweeps=0
        src_np=e[:,0];rel_np=e[:,1];dst_np=e[:,2]
        for _ in range(cap):
            sweeps+=1
            X=torch.cat([state[src,None].to(torch.uint8),state[dst,None].to(torch.uint8),A[src],A[dst]],dim=1)
            prop=self.predict_tensor(rel,X)
            new_state=state.clone();new_state.scatter_reduce_(0,dst,prop,reduce='amax',include_self=True)
            changed=(new_state!=state)
            if not bool(changed.any()):break
            changed_nodes=torch.nonzero(changed,as_tuple=False).flatten().detach().cpu().numpy()
            prop_np=prop.detach().cpu().numpy();old_state=state.detach().cpu().numpy()
            for node in changed_nodes:
                cand=np.flatnonzero((dst_np==node)&(prop_np==int(new_state[node].item())))
                if len(cand):
                    # Prefer an event whose source was already active in the previous fixed-point layer;
                    # this is generic provenance, not a domain relation rule.
                    active=cand[old_state[src_np[cand]]==1]
                    j=int(active[0] if len(active) else cand[0]);parent.setdefault(int(node),(int(src_np[j]),int(rel_np[j])))
            state=new_state
        return state.detach().cpu().numpy().astype(np.uint8),parent,{'parallel':True,'sweeps':sweeps,'certificates':self.certificates}

    def status(self):
        d=super().status();d['parallel_certified']=self.parallel_certified;d['certified_relations']=sum(x['safe_parallel_fixed_point'] for x in self.certificates.values());d['total_relations']=len(self.certificates);return d


def run_benchmark(world_seed=862771,train_n=150000,val_n=60000,test_n=100000,noise=.015,worlds=40):
    world=OpaqueTransitionWorld(seed=world_seed,n_attr=7,n_rel=12,max_gate_order=2)
    train=world.observations(train_n,seed=world_seed+1,noise=noise);val=world.observations(val_n,seed=world_seed+2,noise=0.0);test=world.observations(test_n,seed=world_seed+3,noise=0.0)
    torch.cuda.reset_peak_memory_stats();t0=time.perf_counter()
    L=BoundedMDLRuleInducerGPU(world.n_rel,world.n_feat,initial_order_cap=3,hard_resource_cap=5,batch_subsets=96).fit(train,val,progress=True)
    learn_s=time.perf_counter()-t0;vm=CertifiedParallelRuleVMGPU(L)
    pred=vm.predict_batch(test[0],test[1]);acc=float((pred==test[2]).mean())
    closure_exact=[];jacc=[];proof_ok=[];depth=[];sweeps=[]
    for wi in range(worlds):
        attrs,edges=world.random_network(world_seed+10000+wi,n_nodes=180,n_edges=950);seeds=[wi%180,(wi*17+9)%180]
        os,op=world.execute_fixed_point(attrs,edges,seeds);ps,pp,meta=vm.execute_fixed_point(attrs,edges,seeds)
        O=set(np.flatnonzero(os));P=set(np.flatnonzero(ps));closure_exact.append(O==P);jacc.append(len(O&P)/max(1,len(O|P)));sweeps.append(meta.get('sweeps') or 0)
        for node in list(P)[:30]:
            path=vm.proof(pp,node);depth.append(len(path));st=np.zeros(len(attrs),dtype=np.uint8);st[seeds]=1;ok=True
            for a,r,b in path:
                nxt=world.step_edge(r,int(st[a]),int(st[b]),attrs[a],attrs[b])
                if nxt==st[b]:ok=False;break
                st[b]=nxt
            proof_ok.append(ok)
    return world,L,vm,{
        'format':'Autonomous-RuleVM-MDL-Certified-GPU-v5','world_seed':world_seed,'train_examples':train_n,'train_noise':noise,'val_examples':val_n,'test_examples':test_n,
        'learn_seconds':learn_s,'test_transition_accuracy':acc,'selected_orders':[L.models[r]['order'] for r in range(world.n_rel)],
        'searched_orders':[L.models[r]['searched_orders'] for r in range(world.n_rel)],'val_errors':[L.models[r]['val_error'] for r in range(world.n_rel)],
        'closure_worlds':worlds,'closure_exact_rate':float(np.mean(closure_exact)),'closure_jaccard':float(np.mean(jacc)),
        'proof_validity':float(np.mean(proof_ok)),'max_proof_depth':int(max(depth or [0])),'mean_parallel_sweeps':float(np.mean(sweeps)),
        'gpu':vm.status(),'peak_allocated_mb':torch.cuda.max_memory_allocated()/2**20,
        'hidden_gate_orders_audit_only':[len(x.gate_features) for x in world.laws]
    }


if __name__=='__main__':
    out=Path('rigorous_results_v12');out.mkdir(exist_ok=True)
    world,L,vm,result=run_benchmark()
    L.save_rulebank(out/'AUTONOMOUS_RULEBANK_MDL_V5.json')
    (out/'autonomous_rule_vm_v5.json').write_text(json.dumps(result,indent=2),encoding='utf8')
    print('FINAL',json.dumps(result),flush=True)
