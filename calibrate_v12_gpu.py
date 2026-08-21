from pathlib import Path
from collections import Counter
import json,math,statistics,random
from rigorous_gpu_benchmark import stable_world
from procedural_runtime_gpu import build_renderer_v11_gpu, SemanticTraceVerifier
from procedural_runtime_v12 import build_renderer_v12_gpu, abstract_surface_shape
from procedural_runtime_v5 import ProtectedSlotVerifier
ROOT=Path(__file__).resolve().parent

def entropy(seq):
 c=Counter(seq);N=sum(c.values())
 if len(c)<2:return 0.0
 H=-sum((n/N)*math.log(n/N) for n in c.values())
 return H/math.log(len(c))

def linked_rate(facts,order):
 edges=set()
 for f in facts:
  if f[0]=='rel':edges.add(tuple(sorted((f[1],f[3]))))
 if len(order)<2:return 1.0
 return sum(tuple(sorted((a,b))) in edges for a,b in zip(order,order[1:]))/(len(order)-1)

def eval_model(kind,dw=0.0):
 if kind=='v11':sc,gr,ind,r=build_renderer_v11_gpu(ROOT,seed=8282,proposal_weight=.24,position_weight=7.0,device=0,memory_limit_mb=4608)
 else:sc,gr,ind,r=build_renderer_v12_gpu(ROOT,seed=8282,proposal_weight=.24,position_weight=7.0,diversity_weight=dw,focus_diversity_weight=dw*.45,device=0,memory_limit_mb=4608)
 vf=ProtectedSlotVerifier();tv=SemanticTraceVerifier();bad=slots=trace=0;supports=[];uniq=[];links=[];reps=[];open_entropy=[]
 specs=[dict(seed=6100000+i,n_entities=160,n_props=50,n_rels=24,n_facts=3000) for i in range(8)]
 for di,spec in enumerate(specs):
  facts=stable_world(**spec);o=r.render(facts)
  bad+=int(Counter(o['represented'])!=Counter(facts));slots+=len(vf.inspect_render(o));trace+=len(tv.inspect_render(o))
  shapes=[abstract_surface_shape(sc,x) for x in o['sentences']]
  uniq.append(len(set(shapes))/len(shapes))
  opens=[' '.join(w for w in sc.tokenize(x)[:3] if not sc.is_slot(w)) for x in o['sentences']]
  reps.append(sum(a==b for a,b in zip(opens,opens[1:]))/max(1,len(opens)-1));open_entropy.append(entropy(opens))
  focus_order=[]
  for g in o['groups']:
   f=g[0][1]
   if not focus_order or focus_order[-1]!=f:focus_order.append(f)
  links.append(linked_rate(facts,focus_order))
  sent=o['sentences']
  for i in range(0,len(sent),4096):
   _,sp=sc.batch_language_support([sc.tokenize(x) for x in sent[i:i+4096]],max_order=5,slot_aware=True);supports.extend(map(float,sp))
 return {'kind':kind,'diversity_weight':dw,'bad':bad,'slots':slots,'trace':trace,'support':statistics.mean(supports),'within_doc_unique':statistics.mean(uniq),'graph_adjacent':statistics.mean(links),'repeat':statistics.mean(reps),'opening_entropy':statistics.mean(open_entropy)}
rows=[]
base=eval_model('v11');rows.append(base);print(json.dumps(base),flush=True)
for w in [0.03,0.06,0.10,0.16,0.24,0.36]:
 x=eval_model('v12',w);rows.append(x);print(json.dumps(x),flush=True)
valid=[x for x in rows[1:] if x['bad']==x['slots']==x['trace']==0 and x['support']>=base['support']-.006 and x['repeat']<=base['repeat']+.01]
best=max(valid,key=lambda x:(x['within_doc_unique']+0.35*x['graph_adjacent']+0.10*x['opening_entropy']+0.10*x['support']))
print('SELECTED',json.dumps(best),flush=True)
(ROOT/'rigorous_results'/'v12_diversity_calibration.json').write_text(json.dumps({'baseline':base,'rows':rows,'selected':best},indent=2),encoding='utf8')
