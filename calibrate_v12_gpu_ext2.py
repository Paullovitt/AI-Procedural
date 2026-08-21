from pathlib import Path
from collections import Counter
import json,statistics
from rigorous_gpu_benchmark import stable_world
from procedural_runtime_v12 import build_renderer_v12_gpu,abstract_surface_shape
from procedural_runtime_gpu import SemanticTraceVerifier
from procedural_runtime_v5 import ProtectedSlotVerifier
ROOT=Path(__file__).resolve().parent
D=json.loads((ROOT/'rigorous_results'/'v12_diversity_calibration.json').read_text());base=D['baseline'];rows=list(D['rows'])
def linked(facts,order):
 e={tuple(sorted((f[1],f[3]))) for f in facts if f[0]=='rel'};return sum(tuple(sorted((a,b))) in e for a,b in zip(order,order[1:]))/max(1,len(order)-1)
def ev(w):
 sc,gr,ind,r=build_renderer_v12_gpu(ROOT,seed=9292,proposal_weight=.24,position_weight=7.0,diversity_weight=w,focus_diversity_weight=w*.45,device=0,memory_limit_mb=4608);vf=ProtectedSlotVerifier();tv=SemanticTraceVerifier();bad=slots=trace=0;su=[];uq=[];li=[];rp=[]
 for i in range(5):
  facts=stable_world(seed=6300000+i,n_entities=200,n_props=56,n_rels=28,n_facts=3600);o=r.render(facts);bad+=int(Counter(o['represented'])!=Counter(facts));slots+=len(vf.inspect_render(o));trace+=len(tv.inspect_render(o));sh=[abstract_surface_shape(sc,s) for s in o['sentences']];uq.append(len(set(sh))/len(sh));op=[' '.join(x for x in sc.tokenize(s)[:3] if not sc.is_slot(x)) for s in o['sentences']];rp.append(sum(a==b for a,b in zip(op,op[1:]))/max(1,len(op)-1));order=[]
  for g in o['groups']:
   f=g[0][1]
   if not order or order[-1]!=f:order.append(f)
  li.append(linked(facts,order))
  for k in range(0,len(o['sentences']),4096):
   _,sp=sc.batch_language_support([sc.tokenize(x) for x in o['sentences'][k:k+4096]],max_order=5,slot_aware=True);su+=list(map(float,sp))
 return {'kind':'v12','diversity_weight':w,'bad':bad,'slots':slots,'trace':trace,'support':statistics.mean(su),'within_doc_unique':statistics.mean(uq),'graph_adjacent':statistics.mean(li),'repeat':statistics.mean(rp)}
for w in [1.4,1.9,2.6,3.6]:
 x=ev(w);rows.append(x);print(json.dumps(x),flush=True)
valid=[x for x in rows[1:] if x['bad']==x['slots']==x['trace']==0 and x['support']>=base['support']-.006 and x['repeat']<=base['repeat']+.012]
best=max(valid,key=lambda x:x['within_doc_unique']);target=best['within_doc_unique']*.99;sel=min([x for x in valid if x['within_doc_unique']>=target],key=lambda x:x['diversity_weight']);print('BEST',json.dumps(best),flush=True);print('SELECTED',json.dumps(sel),flush=True);(ROOT/'rigorous_results'/'v12_diversity_calibration.json').write_text(json.dumps({'baseline':base,'rows':rows,'best':best,'selected':sel,'rule':'minimum weight within 99% of best heldout uniqueness under semantic/support/repetition gates'},indent=2),encoding='utf8')
