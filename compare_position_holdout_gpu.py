from pathlib import Path
from collections import Counter
import json,random,statistics,time,math
import torch
from rigorous_gpu_benchmark import stable_world,entropy_norm,q,opening,abstract_sentence
from procedural_runtime_gpu import build_renderer_v11_gpu,SemanticTraceVerifier,GpuLocalOrderVerifier
from procedural_runtime_v5 import ProtectedSlotVerifier
ROOT=Path(__file__).resolve().parent
PW=float(json.loads((ROOT/'rigorous_results'/'proposal_auto_calibration_p5.json').read_text())['selected']['proposal_weight'])
TH=float(json.loads((ROOT/'rigorous_results'/'order_verifier_calibration.json').read_text())['train']['threshold'])
specs=[dict(seed=5000000+i,n_entities=256,n_props=80,n_rels=40,n_facts=5000) for i in range(40)]
specs += [dict(seed=5100000+i,n_entities=512,n_props=128,n_rels=64,n_facts=15000) for i in range(4)] # 260k facts/config

def run(pos):
 sc,gr,ind,r=build_renderer_v11_gpu(ROOT,seed=7777,proposal_weight=PW,position_weight=pos,device=0,memory_limit_mb=4608);vf=ProtectedSlotVerifier();tv=SemanticTraceVerifier();facts=sentn=bad=slots=trace=indn=0;sent=[];opens=[];temps=[];absset=set();diff=[];rng=random.Random(222);t0=time.perf_counter()
 for spec in specs:
  p=stable_world(**spec);o=r.render(p);facts+=len(p);sentn+=len(o['sentences']);bad+=int(Counter(o['represented'])!=Counter(p));slots+=len(vf.inspect_render(o));trace+=len(tv.inspect_render(o));indn+=o.get('induced_selected',0);sent+=o['sentences'];opens += [opening(sc,s) for s in o['sentences']];temps += [x[3].get('template','') for x in o['picks']];absset.update(abstract_sentence(s) for s in o['sentences']);by={}
  for s,g in zip(o['sentences'],o['groups']):by.setdefault(g[0][1],[]).append(s)
  for ss in by.values():
   if len(ss)<2:continue
   a,_=gr.opening_position_score_tokens(sc.tokenize(ss[0]),True,sc.is_slot);b,_=gr.opening_position_score_tokens(sc.tokenize(rng.choice(ss[1:])),True,sc.is_slot);diff.append(a-b)
 supports=[];langs=[]
 for i in range(0,len(sent),4096):
  lg,sp=sc.batch_language_support([sc.tokenize(x) for x in sent[i:i+4096]],max_order=5,slot_aware=True);supports.extend(float(x) for x in sp);langs.extend(float(x) for x in lg)
 lens=[len(sc.tokenize(s)) for s in sent]
 return {'position_weight':pos,'facts':facts,'sentences':sentn,'seconds':time.perf_counter()-t0,'bad_docs':bad,'slot_errors':slots,'trace_errors':trace,'support':statistics.mean(supports),'language_score':statistics.mean(langs),'opening_entropy':entropy_norm(opens),'template_entropy':entropy_norm(temps),'repeat':sum(a==b for a,b in zip(opens,opens[1:]))/max(1,len(opens)-1),'abstract_unique_rate':len(absset)/sentn,'contrast_mean':statistics.mean(diff),'contrast_win_rate':sum(x>0 for x in diff)/len(diff),'induced_rate':indn/sentn,'mean_words':statistics.mean(lens),'median_words':statistics.median(lens),'p90':q(lens,.9),'p95':q(lens,.95),'p99':q(lens,.99)}
rows=[run(.15),run(7.0)]
for x in rows:print(json.dumps(x),flush=True)
b=rows[0];n=rows[1]
pass_gate=(n['bad_docs']==0 and n['slot_errors']==0 and n['trace_errors']==0 and n['support']>=b['support']-.005 and n['repeat']<=b['repeat']+.02 and n['opening_entropy']>=b['opening_entropy']-.02 and n['contrast_win_rate']>b['contrast_win_rate'])
print('PROMOTION_GATE',pass_gate,flush=True)
(ROOT/'rigorous_results'/'position_holdout_comparison.json').write_text(json.dumps({'baseline':b,'candidate':n,'promotion_gate':pass_gate},indent=2),encoding='utf8')
