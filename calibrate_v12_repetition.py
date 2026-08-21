from pathlib import Path
from collections import Counter,defaultdict
import json,statistics,random
from rigorous_gpu_benchmark import stable_world
from procedural_runtime_v12 import build_renderer_v12_gpu,abstract_surface_shape
from procedural_runtime_gpu import SemanticTraceVerifier
from procedural_runtime_v5 import ProtectedSlotVerifier
ROOT=Path(__file__).resolve().parent

def ev(rw):
 sc,gr,ind,r=build_renderer_v12_gpu(ROOT,seed=3333,proposal_weight=.24,position_weight=7.0,diversity_weight=2.6,focus_diversity_weight=1.17,repetition_weight=rw,device=0,memory_limit_mb=4608);vf=ProtectedSlotVerifier();tv=SemanticTraceVerifier();bad=slots=trace=0;su=[];uq=[];rp=[];diff=[];rng=random.Random(7)
 for i in range(7):
  facts=stable_world(seed=9100000+i,n_entities=180,n_props=54,n_rels=28,n_facts=3200);o=r.render(facts);bad+=int(Counter(o['represented'])!=Counter(facts));slots+=len(vf.inspect_render(o));trace+=len(tv.inspect_render(o));sh=[abstract_surface_shape(sc,s) for s in o['sentences']];uq.append(len(set(sh))/len(sh));op=[' '.join(x for x in sc.tokenize(s)[:3] if not sc.is_slot(x)) for s in o['sentences']];rp.append(sum(a==b for a,b in zip(op,op[1:]))/max(1,len(op)-1));by=defaultdict(list)
  for s,g in zip(o['sentences'],o['groups']):by[g[0][1]].append(s)
  for ss in list(by.values())[:120]:
   if len(ss)>1:
    a,_=gr.opening_position_score_tokens(sc.tokenize(ss[0]),True,sc.is_slot);b,_=gr.opening_position_score_tokens(sc.tokenize(rng.choice(ss[1:])),True,sc.is_slot);diff.append(a-b)
  for k in range(0,len(o['sentences']),4096):
   _,sp=sc.batch_language_support([sc.tokenize(x) for x in o['sentences'][k:k+4096]],max_order=5,slot_aware=True);su+=list(map(float,sp))
 return {'repetition_weight':rw,'bad':bad,'slots':slots,'trace':trace,'support':statistics.mean(su),'unique':statistics.mean(uq),'repeat':statistics.mean(rp),'position_win':sum(x>0 for x in diff)/len(diff),'position_mean':statistics.mean(diff)}
rows=[]
for w in [.8,1.1,1.5,2.0,2.7,3.6]:
 x=ev(w);rows.append(x);print(json.dumps(x),flush=True)
base=rows[0];valid=[x for x in rows if x['bad']==x['slots']==x['trace']==0 and x['support']>=base['support']-.006 and x['unique']>=base['unique']-.01 and x['position_win']>=.90]
best=min(valid,key=lambda x:(x['repeat'],x['repetition_weight']))
print('SELECTED',json.dumps(best),flush=True)
(ROOT/'rigorous_results'/'v12_repetition_calibration.json').write_text(json.dumps({'baseline':base,'rows':rows,'selected':best},indent=2),encoding='utf8')
