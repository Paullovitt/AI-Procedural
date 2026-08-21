from __future__ import annotations

from pathlib import Path
import json, math, time
import numpy as np
import torch

from autonomous_rule_vm_v2 import OpaqueTransitionWorld, MDLRuleInducerGPU, RuleBankVMGPU, _h2, _log2_choose


class BatchedMDLRuleInducerGPU(MDLRuleInducerGPU):
    """Same autonomous MDL learner, but evaluates many hypotheses per CUDA launch.

    Search order expands by itself only when the currently selected model sits on the
    explored boundary. ``hard_resource_cap`` is solely a compute safety ceiling.
    """
    def __init__(self,n_rel,n_feat,initial_order_cap=3,hard_resource_cap=5,batch_subsets=96,device='cuda:0'):
        super().__init__(n_rel,n_feat,search_order_cap=hard_resource_cap,device=device)
        self.initial_order_cap=int(initial_order_cap)
        self.hard_resource_cap=int(hard_resource_cap)
        self.batch_subsets=int(batch_subsets)
        self.search_trace={}

    def _evaluate_chunk(self,X,y,XV,yV,subsets,k):
        B=len(subsets);N=len(X);NV=len(XV);bins=1<<k
        if k:
            S=torch.as_tensor(np.asarray(subsets,dtype=np.int64),device=self.device,dtype=torch.long)
            weights=2**torch.arange(k,device=self.device,dtype=torch.long)
            code=(X[:,S].long()*weights).sum(2)      # N,B
            codev=(XV[:,S].long()*weights).sum(2)    # NV,B
        else:
            code=torch.zeros((N,B),device=self.device,dtype=torch.long)
            codev=torch.zeros((NV,B),device=self.device,dtype=torch.long)
        offs=(torch.arange(B,device=self.device,dtype=torch.long)*bins)[None,:]
        ids=(code+offs).reshape(-1)
        yrep=y[:,None].expand(N,B).reshape(-1).float()
        ones=torch.bincount(ids,weights=yrep,minlength=B*bins).reshape(B,bins)
        tot=torch.bincount(ids,minlength=B*bins).reshape(B,bins).float()
        table=(ones*2>=tot).long()
        pred=table.gather(1,code.T)
        predv=table.gather(1,codev.T)
        trerr=(pred!=y[None,:]).float().mean(1)
        verr=(predv!=yV[None,:]).float().mean(1)
        p=verr.clamp(1e-12,1-1e-12)
        h=-(p*torch.log2(p)+(1-p)*torch.log2(1-p))
        h=torch.where((verr<=0)|(verr>=1),torch.zeros_like(h),h)
        model_bits=math.log2(self.hard_resource_cap+1.0)+_log2_choose(self.n_feat,k)+(1<<k)
        total=model_bits+NV*h+math.log2(NV+2.0)
        return (total.detach().cpu().numpy(),verr.detach().cpu().numpy(),
                trerr.detach().cpu().numpy(),table.detach().cpu().numpy(),float(model_bits))

    def _search_order(self,X,y,XV,yV,k,best):
        subsets=self.subsets_by_order[k]
        for start in range(0,len(subsets),self.batch_subsets):
            chunk=subsets[start:start+self.batch_subsets]
            totals,verrs,trerrs,tables,mb=self._evaluate_chunk(X,y,XV,yV,chunk,k)
            for j,subset in enumerate(chunk):
                rb=float(totals[j]-mb)
                row=(float(totals[j]),float(verrs[j]),k,float(trerrs[j]),subset,
                     tuple(int(z) for z in tables[j].tolist()),mb,rb)
                if best is None or row[:4]<best[:4]:best=row
        return best

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
            best=None;searched=[]
            current=min(self.initial_order_cap,self.hard_resource_cap)
            for k in range(current+1):
                best=self._search_order(X,y,XV,yV,k,best);searched.append(k)
            # Expand only if model selection itself says the best rule is on the boundary.
            while best[2]==searched[-1] and searched[-1]<self.hard_resource_cap:
                k=searched[-1]+1
                best=self._search_order(X,y,XV,yV,k,best);searched.append(k)
            registry[r]={
                'features':list(best[4]),'table':list(best[5]),'order':int(best[2]),
                'train_error':float(best[3]),'val_error':float(best[1]),
                'description_bits':float(best[0]),'model_bits':float(best[6]),
                'residual_bits':float(best[7]),'val_examples':int((va_rel==r).sum()),
                'searched_orders':searched
            }
            trace[r]={'searched_orders':searched,'selected_order':int(best[2]),'val_error':float(best[1])}
            if progress:print('REL',r,json.dumps(trace[r]),flush=True)
        self.models=registry;self.search_trace=trace
        torch.cuda.synchronize(self.device);return self

    def save_rulebank(self,path):
        data={'format':'Autonomous-RuleBank-MDL-v3','n_rel':self.n_rel,'n_feat':self.n_feat,
              'initial_order_cap':self.initial_order_cap,
              'hard_resource_cap':self.hard_resource_cap,
              'selection':'minimum-description-length + boundary-triggered complexity expansion',
              'models':self.models}
        Path(path).write_text(json.dumps(data,indent=2),encoding='utf8');return data


if __name__=='__main__':
    out=Path('rigorous_results_v12');out.mkdir(exist_ok=True)
    world=OpaqueTransitionWorld(seed=862771,n_attr=7,n_rel=12,max_gate_order=2)
    train=world.observations(150000,seed=11,noise=.015)
    val=world.observations(60000,seed=12,noise=0.0)
    test=world.observations(100000,seed=13,noise=0.0)
    torch.cuda.reset_peak_memory_stats();t0=time.perf_counter()
    learner=BatchedMDLRuleInducerGPU(world.n_rel,world.n_feat,initial_order_cap=3,hard_resource_cap=5,batch_subsets=96).fit(train,val,progress=True)
    learn_s=time.perf_counter()-t0
    learner.save_rulebank(out/'AUTONOMOUS_RULEBANK_MDL_V3.json')
    vm=RuleBankVMGPU(learner)
    pred=vm.predict_batch(test[0],test[1]);acc=float((pred==test[2]).mean())
    closure_exact=[];jacc=[];proof_ok=[];depths=[]
    for wi in range(36):
        attrs,edges=world.random_network(5000+wi,n_nodes=140,n_edges=650);seeds=[wi%140,(wi*13+7)%140]
        os,op=world.execute_fixed_point(attrs,edges,seeds);ps,pp=vm.execute_fixed_point(attrs,edges,seeds)
        O=set(np.flatnonzero(os));P=set(np.flatnonzero(ps));closure_exact.append(O==P);jacc.append(len(O&P)/max(1,len(O|P)))
        for node in list(P)[:25]:
            path=vm.proof(pp,node);depths.append(len(path));st=np.zeros(len(attrs),dtype=np.uint8);st[seeds]=1;ok=True
            for a,r,b in path:
                nxt=world.step_edge(r,int(st[a]),int(st[b]),attrs[a],attrs[b])
                if nxt==st[b]:ok=False;break
                st[b]=nxt
            proof_ok.append(ok)
    result={
        'format':'Autonomous-RuleVM-MDL-GPU-v3','train_examples':len(train[0]),'train_noise':.015,
        'val_examples':len(val[0]),'test_examples':len(test[0]),'learn_seconds':learn_s,
        'test_transition_accuracy':acc,'selected_orders':[learner.models[r]['order'] for r in range(world.n_rel)],
        'searched_orders':[learner.models[r]['searched_orders'] for r in range(world.n_rel)],
        'closure_exact_rate':float(np.mean(closure_exact)),'closure_jaccard':float(np.mean(jacc)),
        'proof_validity':float(np.mean(proof_ok)),'max_proof_depth':int(max(depths or [0])),
        'gpu':vm.status(),'peak_allocated_mb':torch.cuda.max_memory_allocated()/2**20,
        'hidden_gate_orders_audit_only':[len(x.gate_features) for x in world.laws]
    }
    (out/'autonomous_rule_vm_v3.json').write_text(json.dumps(result,indent=2),encoding='utf8')
    print('FINAL',json.dumps(result),flush=True)
