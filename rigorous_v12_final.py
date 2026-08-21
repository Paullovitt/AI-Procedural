from pathlib import Path
from collections import Counter,defaultdict
import json,statistics,random,time,math,subprocess,threading
import torch
from rigorous_gpu_benchmark import stable_world
from procedural_runtime_v12 import build_renderer_v12_gpu,abstract_surface_shape
from procedural_runtime_gpu import SemanticTraceVerifier,GpuLocalOrderVerifier
from procedural_runtime_v5 import ProtectedSlotVerifier
ROOT=Path(__file__).resolve().parent;OUT=ROOT/'rigorous_results_v12';OUT.mkdir(exist_ok=True)
sc,gr,ind,r=build_renderer_v12_gpu(ROOT,seed=1212,proposal_weight=.24,position_weight=7.0,diversity_weight=2.6,focus_diversity_weight=1.17,device=0,memory_limit_mb=4608)
vf=ProtectedSlotVerifier();tv=SemanticTraceVerifier();ov=GpuLocalOrderVerifier(sc)
# 500k facts, including four much larger documents.
specs=[dict(seed=8000000+i,n_entities=220,n_props=60,n_rels=28,n_facts=4000) for i in range(90)]
specs += [dict(seed=8100000+i,n_entities=512,n_props=80,n_rels=36,n_facts=20000) for i in range(4)]
specs += [dict(seed=8200000,n_entities=768,n_props=96,n_rels=40,n_facts=60000)]
# 90*4000 + 4*20000 + 60000 = 500000
assert sum(x['n_facts'] for x in specs)==500000
mon=[];stop=False
def monitor():
 while not stop:
  try:
   s=subprocess.check_output(['nvidia-smi','--query-gpu=memory.used,utilization.gpu,temperature.gpu,power.draw','--format=csv,noheader,nounits'],text=True).strip().split(', ');mon.append(tuple(map(float,s)))
  except:pass
  time.sleep(.5)
th=threading.Thread(target=monitor,daemon=True);th.start();torch.cuda.reset_peak_memory_stats()
bad=slots=trace=0;facts_n=sent_n=paras=0;uniq=[];links=[];supports=[];opens=[];posdiff=[];all_samples=[];rng=random.Random(919)
t0=time.perf_counter()
for di,spec in enumerate(specs):
 facts=stable_world(**spec);o=r.render(facts);facts_n+=len(facts);sent_n+=len(o['sentences']);paras+=len(o['paragraphs']);bad+=int(Counter(o['represented'])!=Counter(facts));slots+=len(vf.inspect_render(o));trace+=len(tv.inspect_render(o))
 shapes=[abstract_surface_shape(sc,s) for s in o['sentences']];uniq.append(len(set(shapes))/len(shapes));op=[' '.join(w for w in sc.tokenize(s)[:3] if not sc.is_slot(w)) for s in o['sentences']];opens+=op
 order=[]
 for g in o['groups']:
  f=g[0][1]
  if not order or order[-1]!=f:order.append(f)
 edges={tuple(sorted((f[1],f[3]))) for f in facts if f[0]=='rel'};links.append(sum(tuple(sorted((a,b))) in edges for a,b in zip(order,order[1:]))/max(1,len(order)-1))
 by=defaultdict(list)
 for s,g in zip(o['sentences'],o['groups']):by[g[0][1]].append(s)
 for ss in list(by.values())[:64]:
  if len(ss)>1:
   a,_=gr.opening_position_score_tokens(sc.tokenize(ss[0]),True,sc.is_slot);b,_=gr.opening_position_score_tokens(sc.tokenize(rng.choice(ss[1:])),True,sc.is_slot);posdiff.append(a-b)
 for k in range(0,len(o['sentences']),4096):
  _,sp=sc.batch_language_support([sc.tokenize(x) for x in o['sentences'][k:k+4096]],max_order=5,slot_aware=True);supports+=list(map(float,sp))
 if len(all_samples)<5000:all_samples += list(zip(o['sentences'],o['picks'],o['groups']))[:max(0,5000-len(all_samples))]
 if (di+1)%10==0 or di==len(specs)-1:print('PROGRESS',di+1,'/',len(specs),'facts',facts_n,'sent',sent_n,flush=True)
dt=time.perf_counter()-t0;stop=True;th.join(timeout=2)
# adversarial slot-trace and word order checks on heldout samples
role=value=slotn=roleD=valueD=slotD=orderN=orderD=cleanA=0
for s,pick,g in all_samples:
 cleanA+=int(bool(ov.inspect(s)))
 tr=list(SemanticTraceVerifier.trace(s)); ents=[x for x in tr if x.startswith('e')]; vals=[x for x in tr if x.startswith('v')]
 if len(set(ents))>=2:
  a,b=list(dict.fromkeys(ents))[:2];badtext=s.replace(a,'__TMP__').replace(b,a).replace('__TMP__',b);role+=1;roleD+=int(not tv.inspect_sentence(badtext,pick))
 if len(set(vals))>=2:
  a,b=list(dict.fromkeys(vals))[:2];badtext=s.replace(a,'__TMP__').replace(b,a).replace('__TMP__',b);value+=1;valueD+=int(not tv.inspect_sentence(badtext,pick))
 if tr:
  a=tr[0];badtext=s.replace(a,a[0]+'99999',1);slotn+=1;slotD+=int(not tv.inspect_sentence(badtext,pick))
 ws=sc.tokenize(s);poss=[i for i in range(len(ws)-1) if not sc.is_slot(ws[i]) and not sc.is_slot(ws[i+1]) and ws[i]!=ws[i+1]]
 if poss:
  i=rng.choice(poss);x=ws.copy();x[i],x[i+1]=x[i+1],x[i];orderN+=1;orderD+=int(bool(ov.inspect_tokens(x)))
 if role>=2000 and value>=2000 and slotn>=2000 and orderN>=2000:break
rep=sum(a==b for a,b in zip(opens,opens[1:]))/max(1,len(opens)-1)
res={'format':'Renderer-V12-GPU-Final','facts':facts_n,'sentences':sent_n,'paragraphs':paras,'seconds':dt,'facts_per_s':facts_n/dt,'semantic_bad_docs':bad,'slot_errors':slots,'trace_errors':trace,'mean_within_doc_abstract_unique':statistics.mean(uniq),'graph_adjacent_rate':statistics.mean(links),'avg_p2_p5_support':statistics.mean(supports),'immediate_open_repeat':rep,'position_contrast_mean':statistics.mean(posdiff),'position_contrast_win_rate':sum(x>0 for x in posdiff)/len(posdiff),'adversarial':{'role':(roleD,role),'value':(valueD,value),'slot':(slotD,slotn),'order':(orderD,orderN),'clean_order_alarms':cleanA},'gpu':{'peak_torch_allocated_mb':torch.cuda.max_memory_allocated()/2**20,'peak_torch_reserved_mb':torch.cuda.max_memory_reserved()/2**20,'monitor_samples':len(mon),'max_total_vram_mb':max(x[0] for x in mon) if mon else None,'mean_util_pct':statistics.mean(x[1] for x in mon) if mon else None,'max_util_pct':max(x[1] for x in mon) if mon else None,'max_temp_c':max(x[2] for x in mon) if mon else None},'learned':{'proposal_weight':.24,'position_weight':7.0,'diversity_weight':2.6,'focus_diversity_weight':1.17}}
(OUT/'final.json').write_text(json.dumps(res,indent=2),encoding='utf8');print('FINAL',json.dumps(res),flush=True)
