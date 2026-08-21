from __future__ import annotations
from dataclasses import dataclass
from itertools import combinations
from pathlib import Path
import json, random, math, statistics, time
import numpy as np
import torch

@dataclass
class HiddenRule:
    features: tuple
    table: tuple

class HiddenPropagationWorld:
    """Audit-only environment. Rules are never exposed to the learner."""
    def __init__(self,seed=1,n_attr=8,n_rel=14,max_order=3):
        self.rng=random.Random(seed);self.n_attr=n_attr;self.n_rel=n_rel;self.n_feat=2*n_attr
        self.rules=[]
        orders=[]
        for r in range(n_rel):
            # Force a broad mix; identities are shuffled so learner cannot use the schedule.
            orders.append(r%(max_order+1))
        self.rng.shuffle(orders)
        for k in orders:
            fs=tuple(sorted(self.rng.sample(range(self.n_feat),k))) if k else tuple()
            if k==0:
                tab=(self.rng.randrange(2),)
            else:
                while True:
                    tab=tuple(self.rng.randrange(2) for _ in range(1<<k))
                    if 0<sum(tab)<len(tab):break
            self.rules.append(HiddenRule(fs,tab))
    def transmit(self,r,src_attr,dst_attr):
        rule=self.rules[r];x=np.concatenate([src_attr,dst_attr])
        code=0
        for j,f in enumerate(rule.features): code|=(int(x[f])<<j)
        return int(rule.table[code])
    def observations(self,n,seed=10,noise=0.0):
        rng=np.random.default_rng(seed)
        rel=rng.integers(0,self.n_rel,size=n,dtype=np.int64)
        src=rng.integers(0,2,size=(n,self.n_attr),dtype=np.uint8)
        dst=rng.integers(0,2,size=(n,self.n_attr),dtype=np.uint8)
        y=np.empty(n,dtype=np.uint8)
        # This y is an observed state change, not a rule label handed to the learner.
        for i in range(n): y[i]=self.transmit(int(rel[i]),src[i],dst[i])
        if noise:
            flip=rng.random(n)<noise;y=np.bitwise_xor(y,flip.astype(np.uint8))
        return rel,np.concatenate([src,dst],axis=1),y
    def random_network(self,seed,n_nodes=220,n_edges=1200):
        rng=np.random.default_rng(seed);attrs=rng.integers(0,2,size=(n_nodes,self.n_attr),dtype=np.uint8)
        edges=[];seen=set()
        while len(edges)<n_edges:
            a=int(rng.integers(n_nodes));b=int(rng.integers(n_nodes));r=int(rng.integers(self.n_rel))
            if a==b or (a,b,r) in seen:continue
            seen.add((a,b,r));edges.append((a,r,b))
        return attrs,edges
    def closure(self,attrs,edges,seeds):
        failed=set(seeds);parent={x:None for x in seeds};changed=True
        while changed:
            changed=False
            for a,r,b in edges:
                if a in failed and b not in failed and self.transmit(r,attrs[a],attrs[b]):
                    failed.add(b);parent[b]=(a,r);changed=True
        return failed,parent

class GPUTruthTableRuleLearner:
    """Generic minimum-interaction truth-table induction on CUDA. No gradients/backprop."""
    def __init__(self,n_rel,n_feat,max_order=3,device='cuda:0',complexity_penalty=1e-5):
        if not torch.cuda.is_available():raise RuntimeError('CUDA required')
        self.n_rel=n_rel;self.n_feat=n_feat;self.max_order=max_order;self.device=torch.device(device);self.penalty=complexity_penalty;self.models={}
        self.subsets=[tuple(c) for k in range(max_order+1) for c in combinations(range(n_feat),k)]
    @staticmethod
    def _fit_subset(X,y,subset):
        dev=X.device;k=len(subset)
        if k==0:
            ones=int(y.sum().item());pred=1 if ones*2>=len(y) else 0
            err=float((y!=pred).float().mean().item());return (pred,),err
        idx=torch.tensor(subset,device=dev,dtype=torch.long);bits=X[:,idx].long();weights=(2**torch.arange(k,device=dev,dtype=torch.long));code=(bits*weights).sum(1);bins=1<<k
        ones=torch.bincount(code,weights=y.float(),minlength=bins);tot=torch.bincount(code,minlength=bins).float();table=(ones*2>=tot).long();pred=table[code];err=float((pred!=y).float().mean().item());return tuple(int(x) for x in table.tolist()),err
    @staticmethod
    def _error_subset(X,y,subset,table):
        k=len(subset)
        if k==0:return float((y!=int(table[0])).float().mean().item())
        idx=torch.tensor(subset,device=X.device,dtype=torch.long);weights=(2**torch.arange(k,device=X.device,dtype=torch.long));code=(X[:,idx].long()*weights).sum(1);tab=torch.tensor(table,device=X.device,dtype=torch.long);return float((tab[code]!=y).float().mean().item())
    def fit(self,train,val):
        tr_rel,tr_X,tr_y=train;va_rel,va_X,va_y=val
        trR=torch.as_tensor(tr_rel,device=self.device,dtype=torch.long);trX=torch.as_tensor(tr_X,device=self.device,dtype=torch.uint8);trY=torch.as_tensor(tr_y,device=self.device,dtype=torch.long)
        vaR=torch.as_tensor(va_rel,device=self.device,dtype=torch.long);vaX=torch.as_tensor(va_X,device=self.device,dtype=torch.uint8);vaY=torch.as_tensor(va_y,device=self.device,dtype=torch.long)
        for r in range(self.n_rel):
            X=trX[trR==r];y=trY[trR==r];XV=vaX[vaR==r];yV=vaY[vaR==r];best=None
            for subset in self.subsets:
                table,trerr=self._fit_subset(X,y,subset);verr=self._error_subset(XV,yV,subset,table)
                score=verr+self.penalty*((1<<len(subset))+len(subset))
                row=(score,len(subset),trerr,subset,table,verr)
                if best is None or row[:3]<best[:3]:best=row
                # Clean validation + minimum-order traversal: no need to test larger orders once exact at this order.
            self.models[r]={'features':best[3],'table':best[4],'train_error':best[2],'val_error':best[5]}
        torch.cuda.synchronize(self.device);return self
    def predict_batch(self,rel,X):
        relT=torch.as_tensor(rel,device=self.device,dtype=torch.long);XT=torch.as_tensor(X,device=self.device,dtype=torch.uint8);out=torch.empty(len(rel),device=self.device,dtype=torch.long)
        for r,m in self.models.items():
            mask=relT==r;sub=m['features'];table=m['table']
            if not mask.any():continue
            if len(sub)==0:out[mask]=int(table[0]);continue
            xx=XT[mask][:,torch.tensor(sub,device=self.device)];weights=2**torch.arange(len(sub),device=self.device);code=(xx.long()*weights).sum(1);tab=torch.tensor(table,device=self.device,dtype=torch.long);out[mask]=tab[code]
        return out.cpu().numpy().astype(np.uint8)
    def transmit(self,r,src,dst):
        x=np.concatenate([src,dst])[None,:];return int(self.predict_batch(np.asarray([r]),x)[0])
    def closure(self,attrs,edges,seeds):
        failed=set(seeds);parent={x:None for x in seeds};changed=True
        while changed:
            changed=False;pending=[];meta=[]
            for a,r,b in edges:
                if a in failed and b not in failed:
                    pending.append(np.concatenate([attrs[a],attrs[b]]));meta.append((a,r,b))
            if not pending:break
            pred=self.predict_batch(np.asarray([x[1] for x in meta]),np.asarray(pending,dtype=np.uint8))
            for (a,r,b),ok in zip(meta,pred):
                if ok and b not in failed:failed.add(b);parent[b]=(a,r);changed=True
        return failed,parent
    def status(self):
        free,total=torch.cuda.mem_get_info(self.device)
        return {'device':str(self.device),'name':torch.cuda.get_device_name(self.device),'allocated_mb':torch.cuda.memory_allocated(self.device)/2**20,'reserved_mb':torch.cuda.memory_reserved(self.device)/2**20,'free_mb':free/2**20,'total_mb':total/2**20,'gradients':False,'neural':False}

def path_from_parent(parent,node):
    p=[];cur=node
    while parent.get(cur) is not None:
        a,r=parent[cur];p.append((a,r,cur));cur=a
    p.reverse();return p

if __name__=='__main__':
    out=Path('rigorous_results');out.mkdir(exist_ok=True)
    world=HiddenPropagationWorld(seed=741852,n_attr=8,n_rel=14,max_order=3)
    train=world.observations(260000,seed=1,noise=.015);val=world.observations(90000,seed=2,noise=0);test=world.observations(140000,seed=3,noise=0)
    torch.cuda.reset_peak_memory_stats();t0=time.perf_counter();learner=GPUTruthTableRuleLearner(world.n_rel,world.n_feat,max_order=3).fit(train,val);learn_s=time.perf_counter()-t0
    pred=learner.predict_batch(test[0],test[1]);acc=float((pred==test[2]).mean())
    exact=0
    for r,m in learner.models.items():
        h=world.rules[r]
        exact+=int(tuple(m['features'])==tuple(h.features) and tuple(m['table'])==tuple(h.table))
    # Unseen networks and multi-hop closure.
    jacc=[];exact_world=[];path_ok=[];max_chain=0;cascade_err=[]
    for wi in range(80):
        attrs,edges=world.random_network(10000+wi,n_nodes=240,n_edges=1350);seeds=[wi%240,(wi*17+3)%240]
        oracle,op=world.closure(attrs,edges,seeds);guess,gp=learner.closure(attrs,edges,seeds)
        jacc.append(len(oracle&guess)/max(1,len(oracle|guess)));exact_world.append(oracle==guess);cascade_err.append(abs(len(oracle)-len(guess)))
        for node in list(guess)[:40]:
            path=path_from_parent(gp,node);max_chain=max(max_chain,len(path));ok=True
            for a,r,b in path:
                if not world.transmit(r,attrs[a],attrs[b]):ok=False;break
            path_ok.append(ok)
    # Intervention intelligence: rank seed nodes by cascade size, learner vs hidden oracle.
    rank_overlap=[];rank_spearman=[]
    for wi in range(12):
        attrs,edges=world.random_network(20000+wi,n_nodes=120,n_edges=650)
        O=[];P=[]
        for s in range(120):
            O.append(len(world.closure(attrs,edges,[s])[0]));P.append(len(learner.closure(attrs,edges,[s])[0]))
        topO=set(np.argsort(O)[-10:]);topP=set(np.argsort(P)[-10:]);rank_overlap.append(len(topO&topP)/10)
        # Pearson on ranks is Spearman.
        ro=np.argsort(np.argsort(O));rp=np.argsort(np.argsort(P));rank_spearman.append(float(np.corrcoef(ro,rp)[0,1]))
    result={'format':'Opaque-Autonomous-Reasoning-GPU-v1','train_observations':len(train[0]),'train_noise':.015,'val_observations':len(val[0]),'test_observations':len(test[0]),'learn_seconds':learn_s,'test_transition_accuracy':acc,'hidden_rules_exact':exact,'hidden_rules_total':world.n_rel,'worlds':80,'closure_exact_rate':statistics.mean(exact_world),'closure_jaccard':statistics.mean(jacc),'mean_cascade_size_error':statistics.mean(cascade_err),'proof_edge_validity':statistics.mean(path_ok),'max_proof_chain':max_chain,'critical_top10_overlap':statistics.mean(rank_overlap),'critical_rank_spearman':statistics.mean(rank_spearman),'learned_models':learner.models,'gpu':learner.status(),'peak_allocated_mb':torch.cuda.max_memory_allocated()/2**20,'hidden_rule_orders_audit':[len(x.features) for x in world.rules]}
    (out/'autonomous_reasoning_gpu.json').write_text(json.dumps(result,indent=2),encoding='utf8');print(json.dumps({k:v for k,v in result.items() if k not in ('learned_models',)},indent=2),flush=True)
