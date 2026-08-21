from pathlib import Path
from collections import Counter
import json,math,statistics
from rigorous_gpu_benchmark import stable_world
from procedural_runtime_v12 import build_renderer_v12_gpu,abstract_surface_shape
from procedural_runtime_gpu import SemanticTraceVerifier
from procedural_runtime_v5 import ProtectedSlotVerifier
ROOT=Path(__file__).resolve().parent
prev=json.loads((ROOT/'rigorous_results'/'v12_diversity_calibration.json').read_text())
base=prev['baseline'];rows=list(prev['rows'])
def linked_rate(facts,order):
 edges={tuple(sorted((f[1],f[3]))) for f in facts if f[0]=='rel'}
 return sum(tuple(sorted((a,b))) in edges for a,b in zip(order,order[1:]))/max(1,len(order)-1)
def evalw(w):
 sc,gr,ind,r=build_renderer_v12_gpu(ROOT,seed=9191,proposal_weight=.24,position_weight=7.0,diversity_weight=w,focus_diversity_weight=w*.45,device=0,memory_limit_mb=4608)
 vf=ProtectedSlotVerifier();tv=SemanticTraceVerifier();bad=slots=trace=0;supports=[];uniq=[];links=[];reps=[]
 for i in range(6):
  facts=stable_world(seed=6200000+i,n_entities=180,n_props=52,n_rels=26,n_facts=3200);o=r.render(facts)
  bad+=int(Counter(o['represented'])!=Counter(facts));slots+=len(vf.inspect_render(o));trace+=len(tv.inspect_render(o))
  sh=[abstract_surface_shape(sc,s) for s in o['sentences']];uniq.append(len(set(sh))/len(sh))
  op=[' '.join(x for x in sc.tokenize(s)[:3] if not sc.is_slot(x)) for s in o['sentences']];reps.append(sum(a==b for a,b in zip(op,op[1:]))/max(1,len(op)-1))
  order=[]
  for g in o['groups']:
   f=g[0][1]
   if not order or order[-1]!=f:order.append(f)
  links.append(linked_rate(facts,order))
  for k in range(0,len(o['sentences']),4096):
   _,sp=sc.batch_language_support([sc.tokenize(x) for x in o['sentences'][k:k+4096]],max_order=5,slot_aware=True);supports+=list(map(float,sp))
 return {'kind':'v12','diversity_weight':w,'bad':bad,'slots':slots,'trace':trace,'support':statistics.mean(supports),'within_doc_unique':statistics.mean(uniq),'graph_adjacent':statistics.mean(links),'repeat':statistics.mean(reps)}
for w in [.45,.60,.80,1.05]:
 x=evalw(w);rows.append(x);print(json.dumps(x),flush=True)
# To avoid cross-seed overfitting, choose lowest weight within 1% relative of best uniqueness under hard gates.
valid=[x for x in rows[1:] if x['bad']==x['slots']==x['trace']==0 and x['support']>=base['support']-.006 and x['repeat']<=base['repeat']+.012]
best=max(valid,key=lambda x:x['within_doc_unique'])
target=best['within_doc_unique']*.99
selected=min([x for x in valid if x['within_doc_unique']>=target],key=lambda x:x['diversity_weight'])
print('BEST',json.dumps(best),flush=True);print('SELECTED',json.dumps(selected),flush=True)
(ROOT/'rigorous_results'/'v12_diversity_calibration.json').write_text(json.dumps({'baseline':base,'rows':rows,'best':best,'selected':selected,'rule':'minimum weight within 99% of best heldout uniqueness under semantic/support/repetition gates'},indent=2),encoding='utf8')
