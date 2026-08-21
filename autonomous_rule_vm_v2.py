from __future__ import annotations

from dataclasses import dataclass
from itertools import combinations
from pathlib import Path
import json, math, random, time
import numpy as np
import torch


def _h2(p: float) -> float:
    p=float(min(1.0,max(0.0,p)))
    if p<=0.0 or p>=1.0:
        return 0.0
    return -p*math.log2(p)-(1.0-p)*math.log2(1.0-p)


def _log2_choose(n:int,k:int)->float:
    if k<0 or k>n:return float('inf')
    return (math.lgamma(n+1)-math.lgamma(k+1)-math.lgamma(n-k+1))/math.log(2.0)


@dataclass(frozen=True)
class HiddenLocalLaw:
    gate_features: tuple[int,...]
    gate_table: tuple[int,...]


class OpaqueTransitionWorld:
    """AUDIT-ONLY environment.

    The learner never receives ``laws``.  It only sees local transition episodes:
      (relation id, local before-state features) -> observed destination state after one edge event.

    Hidden dynamics are monotone only so fixed-point evaluation terminates.  The learner is not
    told the monotonic form and must rediscover the complete local transition function.
    """
    def __init__(self,seed=1,n_attr=7,n_rel=12,max_gate_order=2):
        self.rng=random.Random(seed)
        self.n_attr=int(n_attr);self.n_rel=int(n_rel)
        self.n_feat=2+2*self.n_attr  # src_state,dst_state,src attrs,dst attrs
        self.laws=[]
        orders=[i%(max_gate_order+1) for i in range(self.n_rel)]
        self.rng.shuffle(orders)
        attr_positions=list(range(2,self.n_feat))
        for k in orders:
            fs=tuple(sorted(self.rng.sample(attr_positions,k))) if k else tuple()
            if k==0:
                table=(self.rng.randrange(2),)
            else:
                while True:
                    table=tuple(self.rng.randrange(2) for _ in range(1<<k))
                    if 0<sum(table)<len(table):break
            self.laws.append(HiddenLocalLaw(fs,table))

    def local_features(self,src_state,dst_state,src_attr,dst_attr):
        return np.concatenate([
            np.asarray([src_state,dst_state],dtype=np.uint8),
            np.asarray(src_attr,dtype=np.uint8),np.asarray(dst_attr,dtype=np.uint8)
        ])

    def step_edge(self,r,src_state,dst_state,src_attr,dst_attr):
        x=self.local_features(src_state,dst_state,src_attr,dst_attr)
        law=self.laws[int(r)];code=0
        for j,f in enumerate(law.gate_features):code|=(int(x[f])<<j)
        gate=int(law.gate_table[code])
        # Hidden environment law. This exact form is never available to the learner.
        return int(bool(dst_state) or (bool(src_state) and bool(gate)))

    def observations(self,n,seed=10,noise=0.0):
        rng=np.random.default_rng(seed)
        rel=rng.integers(0,self.n_rel,size=n,dtype=np.int64)
        src_state=rng.integers(0,2,size=n,dtype=np.uint8)
        dst_state=rng.integers(0,2,size=n,dtype=np.uint8)
        src=rng.integers(0,2,size=(n,self.n_attr),dtype=np.uint8)
        dst=rng.integers(0,2,size=(n,self.n_attr),dtype=np.uint8)
        X=np.concatenate([src_state[:,None],dst_state[:,None],src,dst],axis=1)
        y=np.empty(n,dtype=np.uint8)
        for i in range(n):
            y[i]=self.step_edge(int(rel[i]),int(src_state[i]),int(dst_state[i]),src[i],dst[i])
        if noise:
            flip=(rng.random(n)<noise).astype(np.uint8);y=np.bitwise_xor(y,flip)
        return rel,X,y

    def random_network(self,seed,n_nodes=180,n_edges=900):
        rng=np.random.default_rng(seed)
        attrs=rng.integers(0,2,size=(n_nodes,self.n_attr),dtype=np.uint8)
        edges=[];seen=set()
        while len(edges)<n_edges:
            a=int(rng.integers(n_nodes));b=int(rng.integers(n_nodes));r=int(rng.integers(self.n_rel))
            if a==b or (a,r,b) in seen:continue
            seen.add((a,r,b));edges.append((a,r,b))
        return attrs,edges

    def execute_fixed_point(self,attrs,edges,seeds,max_sweeps=None):
        state=np.zeros(len(attrs),dtype=np.uint8);state[list(seeds)]=1
        parent={int(x):None for x in seeds}
        cap=int(max_sweeps or (len(attrs)+1))
        for _ in range(cap):
            changed=0
            for a,r,b in edges:
                old=int(state[b]);new=self.step_edge(r,int(state[a]),old,attrs[a],attrs[b])
                if new!=old:
                    state[b]=new;parent.setdefault(int(b),(int(a),int(r)));changed+=1
            if changed==0:break
        return state,parent


class MDLRuleInducerGPU:
    """Generic truth-table induction chosen by description length, not hand-tuned rule thresholds.

    ``search_order_cap`` is a compute/resource bound, not domain knowledge.  Within that bound the
    learner chooses feature subset, interaction order and truth table entirely from observations.
    """
    def __init__(self,n_rel,n_feat,search_order_cap=4,device='cuda:0'):
        if not torch.cuda.is_available():raise RuntimeError('CUDA required')
        self.n_rel=int(n_rel);self.n_feat=int(n_feat);self.search_order_cap=int(search_order_cap)
        self.device=torch.device(device);self.models={}
        self.subsets_by_order={k:list(combinations(range(self.n_feat),k)) for k in range(self.search_order_cap+1)}

    def _fit_table(self,X,y,subset):
        k=len(subset)
        idx=torch.tensor(subset,device=self.device,dtype=torch.long)
        bits=X[:,idx].long() if k else X[:,idx].long()
        weights=2**torch.arange(k,device=self.device,dtype=torch.long)
        code=(bits*weights).sum(1) if k else torch.zeros(len(X),device=self.device,dtype=torch.long)
        bins=1<<k
        ones=torch.bincount(code,weights=y.float(),minlength=bins)
        tot=torch.bincount(code,minlength=bins).float()
        # Empty cells are encoded as 0 but do not affect observed-data likelihood.
        table=(ones*2>=tot).long()
        return table,code

    def _eval_error(self,X,y,subset,table):
        k=len(subset);idx=torch.tensor(subset,device=self.device,dtype=torch.long)
        bits=X[:,idx].long() if k else X[:,idx].long()
        weights=2**torch.arange(k,device=self.device,dtype=torch.long)
        code=(bits*weights).sum(1) if k else torch.zeros(len(X),device=self.device,dtype=torch.long)
        pred=table[code]
        return float((pred!=y).float().mean().item())

    def _description_bits(self,k,val_error,n_val):
        # Universal structural code + deterministic truth table + residual binary error stream.
        model_bits=math.log2(self.search_order_cap+1.0)+_log2_choose(self.n_feat,k)+(1<<k)
        residual_bits=n_val*_h2(val_error)+math.log2(n_val+2.0)
        return model_bits+residual_bits,model_bits,residual_bits

    def fit(self,train,val):
        tr_rel,tr_X,tr_y=train;va_rel,va_X,va_y=val
        trR=torch.as_tensor(tr_rel,device=self.device,dtype=torch.long)
        trX=torch.as_tensor(tr_X,device=self.device,dtype=torch.uint8)
        trY=torch.as_tensor(tr_y,device=self.device,dtype=torch.long)
        vaR=torch.as_tensor(va_rel,device=self.device,dtype=torch.long)
        vaX=torch.as_tensor(va_X,device=self.device,dtype=torch.uint8)
        vaY=torch.as_tensor(va_y,device=self.device,dtype=torch.long)
        registry={}
        for r in range(self.n_rel):
            X=trX[trR==r];y=trY[trR==r];XV=vaX[vaR==r];yV=vaY[vaR==r]
            best=None
            for k in range(self.search_order_cap+1):
                for subset in self.subsets_by_order[k]:
                    table,_=self._fit_table(X,y,subset)
                    trerr=self._eval_error(X,y,subset,table);verr=self._eval_error(XV,yV,subset,table)
                    total,mb,rb=self._description_bits(k,verr,len(yV))
                    row=(total,verr,k,trerr,subset,tuple(int(z) for z in table.tolist()),mb,rb)
                    if best is None or row[:4]<best[:4]:best=row
            registry[r]={
                'features':list(best[4]),'table':list(best[5]),'order':int(best[2]),
                'train_error':float(best[3]),'val_error':float(best[1]),
                'description_bits':float(best[0]),'model_bits':float(best[6]),
                'residual_bits':float(best[7]),'val_examples':int((va_rel==r).sum())
            }
        self.models=registry;torch.cuda.synchronize(self.device);return self

    def save_rulebank(self,path):
        data={'format':'Autonomous-RuleBank-MDL-v2','n_rel':self.n_rel,'n_feat':self.n_feat,
              'search_order_cap_resource_bound':self.search_order_cap,'models':self.models}
        Path(path).write_text(json.dumps(data,indent=2),encoding='utf8');return data

    @classmethod
    def from_rulebank(cls,path,device='cuda:0'):
        d=json.loads(Path(path).read_text(encoding='utf8'))
        obj=cls(d['n_rel'],d['n_feat'],d['search_order_cap_resource_bound'],device=device)
        obj.models={int(k):v for k,v in d['models'].items()};return obj


class RuleBankVMGPU:
    """Executor for learned rules stored as data. It contains no relation-specific/domain rules."""
    def __init__(self,rulebank,device='cuda:0'):
        self.models=rulebank.models if hasattr(rulebank,'models') else rulebank
        self.device=torch.device(device)

    def predict_batch(self,rel,X):
        relT=torch.as_tensor(rel,device=self.device,dtype=torch.long)
        XT=torch.as_tensor(X,device=self.device,dtype=torch.uint8)
        out=torch.zeros(len(rel),device=self.device,dtype=torch.long)
        for r,m in self.models.items():
            mask=(relT==int(r));sub=tuple(int(x) for x in m['features']);k=len(sub)
            idx=torch.tensor(sub,device=self.device,dtype=torch.long)
            bits=XT[mask][:,idx].long()
            weights=2**torch.arange(k,device=self.device,dtype=torch.long)
            code=(bits*weights).sum(1) if k else torch.zeros(int(mask.sum().item()),device=self.device,dtype=torch.long)
            tab=torch.tensor(m['table'],device=self.device,dtype=torch.long)
            out[mask]=tab[code]
        return out.cpu().numpy().astype(np.uint8)

    def execute_fixed_point(self,attrs,edges,seeds,max_sweeps=None):
        state=np.zeros(len(attrs),dtype=np.uint8);state[list(seeds)]=1
        parent={int(x):None for x in seeds};cap=int(max_sweeps or (len(attrs)+1))
        for _ in range(cap):
            changed=0
            # Edge events are generic transition operators; learned rules decide the new dst state.
            for a,r,b in edges:
                x=np.concatenate([np.asarray([state[a],state[b]],dtype=np.uint8),attrs[a],attrs[b]])[None,:]
                new=int(self.predict_batch(np.asarray([r],dtype=np.int64),x)[0]);old=int(state[b])
                if new!=old:
                    state[b]=new;parent.setdefault(int(b),(int(a),int(r)));changed+=1
            if changed==0:break
        return state,parent

    @staticmethod
    def proof(parent,node):
        out=[];cur=int(node);seen=set()
        while parent.get(cur) is not None and cur not in seen:
            seen.add(cur);a,r=parent[cur];out.append((int(a),int(r),cur));cur=int(a)
        out.reverse();return out

    def status(self):
        free,total=torch.cuda.mem_get_info(self.device)
        return {'device':str(self.device),'name':torch.cuda.get_device_name(self.device),
                'allocated_mb':torch.cuda.memory_allocated(self.device)/2**20,
                'reserved_mb':torch.cuda.memory_reserved(self.device)/2**20,
                'free_mb':free/2**20,'total_mb':total/2**20,'neural':False,'gradients':False}


def audit_function(world,vm,n=120000,seed=900):
    rel,X,y=world.observations(n,seed=seed,noise=0.0)
    pred=vm.predict_batch(rel,X)
    return float((pred==y).mean())


if __name__=='__main__':
    out=Path('rigorous_results_v12');out.mkdir(exist_ok=True)
    world=OpaqueTransitionWorld(seed=862771,n_attr=7,n_rel=12,max_gate_order=2)
    train=world.observations(150000,seed=11,noise=.015)
    val=world.observations(60000,seed=12,noise=0.0)
    test=world.observations(100000,seed=13,noise=0.0)
    torch.cuda.reset_peak_memory_stats();t0=time.perf_counter()
    learner=MDLRuleInducerGPU(world.n_rel,world.n_feat,search_order_cap=4).fit(train,val)
    learn_s=time.perf_counter()-t0
    bank=learner.save_rulebank(out/'AUTONOMOUS_RULEBANK_MDL_V2.json')
    vm=RuleBankVMGPU(learner)
    pred=vm.predict_batch(test[0],test[1]);acc=float((pred==test[2]).mean())
    exact=[];proof_ok=[];jacc=[];closure_exact=[];depths=[]
    for wi in range(36):
        attrs,edges=world.random_network(5000+wi,n_nodes=140,n_edges=650);seeds=[wi%140,(wi*13+7)%140]
        os,op=world.execute_fixed_point(attrs,edges,seeds);ps,pp=vm.execute_fixed_point(attrs,edges,seeds)
        O=set(np.flatnonzero(os));P=set(np.flatnonzero(ps));closure_exact.append(O==P);jacc.append(len(O&P)/max(1,len(O|P)))
        for node in list(P)[:25]:
            path=vm.proof(pp,node);depths.append(len(path));ok=True
            # Audit-only oracle checks the learned proof edges; VM never gets this signal.
            st=np.zeros(len(attrs),dtype=np.uint8);st[seeds]=1
            for a,r,b in path:
                nxt=world.step_edge(r,int(st[a]),int(st[b]),attrs[a],attrs[b])
                if nxt==st[b]:ok=False;break
                st[b]=nxt
            proof_ok.append(ok)
    result={
        'format':'Autonomous-RuleVM-MDL-GPU-v2','train_examples':len(train[0]),'train_noise':.015,
        'val_examples':len(val[0]),'test_examples':len(test[0]),'learn_seconds':learn_s,
        'test_transition_accuracy':acc,'selected_orders':[learner.models[r]['order'] for r in range(world.n_rel)],
        'mean_description_bits':float(np.mean([m['description_bits'] for m in learner.models.values()])),
        'closure_exact_rate':float(np.mean(closure_exact)),'closure_jaccard':float(np.mean(jacc)),
        'proof_validity':float(np.mean(proof_ok)),'max_proof_depth':int(max(depths or [0])),
        'gpu':vm.status(),'peak_allocated_mb':torch.cuda.max_memory_allocated()/2**20,
        'rulebank_path':str(out/'AUTONOMOUS_RULEBANK_MDL_V2.json'),
        'hidden_gate_orders_audit_only':[len(x.gate_features) for x in world.laws]
    }
    (out/'autonomous_rule_vm_v2.json').write_text(json.dumps(result,indent=2),encoding='utf8')
    print(json.dumps(result,indent=2),flush=True)
