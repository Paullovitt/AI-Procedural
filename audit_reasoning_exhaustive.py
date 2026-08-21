import json,numpy as np,torch
from pathlib import Path
from autonomous_reasoning_gpu import HiddenPropagationWorld,GPUTruthTableRuleLearner
world=HiddenPropagationWorld(seed=741852,n_attr=8,n_rel=14,max_order=3)
train=world.observations(260000,seed=1,noise=.015);val=world.observations(90000,seed=2,noise=0)
L=GPUTruthTableRuleLearner(world.n_rel,world.n_feat,max_order=3).fit(train,val)
# Exhaustive 16-bit state-pair space.
X=np.unpackbits(np.arange(65536,dtype='>u2').view(np.uint8)).reshape(-1,16).astype(np.uint8)
functional=[];details=[]
for r in range(world.n_rel):
 rel=np.full(len(X),r,dtype=np.int64);p=L.predict_batch(rel,X)
 y=np.empty(len(X),dtype=np.uint8)
 for i,x in enumerate(X):y[i]=world.transmit(r,x[:8],x[8:])
 eq=bool(np.array_equal(p,y));functional.append(eq)
 h=world.rules[r];m=L.models[r]
 details.append({'r':r,'function_equal':eq,'hidden_order':len(h.features),'learned_order':len(m['features']),'same_schema':tuple(h.features)==tuple(m['features']) and tuple(h.table)==tuple(m['table'])})
print(json.dumps({'functional_exact':sum(functional),'total':len(functional),'details':details},indent=2))
p=Path('rigorous_results/autonomous_reasoning_gpu.json');d=json.loads(p.read_text());d['exhaustive_functional_exact']=sum(functional);d['exhaustive_functional_total']=len(functional);d['schema_audit_details']=details;p.write_text(json.dumps(d,indent=2),encoding='utf8')
