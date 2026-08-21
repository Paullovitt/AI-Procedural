from pathlib import Path
from collections import Counter
import json,random,statistics
from rigorous_gpu_benchmark import stable_world,entropy_norm,opening
from procedural_runtime_gpu import build_renderer_v11_gpu
from procedural_runtime_v5 import ProtectedSlotVerifier
ROOT=Path(__file__).resolve().parent
PW=float(json.loads((ROOT/'rigorous_results'/'proposal_auto_calibration_p5.json').read_text())['selected']['proposal_weight'])
specs=[dict(seed=4000000+i,n_entities=128,n_props=48,n_rels=24,n_facts=1400) for i in range(10)]
weights=[0.0,.05,.10,.15,.25,.40,.60,.90,1.30,1.80]

def evalw(w):
    sc,gr,ind,r=build_renderer_v11_gpu(ROOT,seed=4545,proposal_weight=PW,position_weight=w,device=0,memory_limit_mb=4608)
    vf=ProtectedSlotVerifier();bad=slots=0;sent=[];opens=[];diffs=[];rng=random.Random(99)
    for spec in specs:
        p=stable_world(**spec);o=r.render(p);bad+=int(Counter(o['represented'])!=Counter(p));slots+=len(vf.inspect_render(o));sent+=o['sentences'];opens += [opening(sc,s) for s in o['sentences']]
        by={}
        for s,g in zip(o['sentences'],o['groups']):by.setdefault(g[0][1],[]).append(s)
        for ss in by.values():
            if len(ss)<2:continue
            a,_=gr.opening_position_score_tokens(sc.tokenize(ss[0]),True,sc.is_slot)
            b,_=gr.opening_position_score_tokens(sc.tokenize(rng.choice(ss[1:])),True,sc.is_slot)
            diffs.append(a-b)
    supports=[]
    for i in range(0,len(sent),4096):
        _,sp=sc.batch_language_support([sc.tokenize(x) for x in sent[i:i+4096]],max_order=5,slot_aware=True);supports.extend(float(x) for x in sp)
    rep=sum(a==b for a,b in zip(opens,opens[1:]))/max(1,len(opens)-1)
    return {'position_weight':w,'bad':bad,'slot_errors':slots,'support':statistics.mean(supports),'repeat':rep,'opening_entropy':entropy_norm(opens),
            'contrast_mean':statistics.mean(diffs),'contrast_win_rate':sum(x>0 for x in diffs)/len(diffs),'comparisons':len(diffs)}
rows=[]
for w in weights:
    x=evalw(w);rows.append(x);print(json.dumps(x),flush=True)
base=rows[0]
valid=[x for x in rows if x['bad']==0 and x['slot_errors']==0 and x['support']>=base['support']-.005 and x['repeat']<=base['repeat']+.02]
best=max(valid,key=lambda x:(x['contrast_win_rate'],x['contrast_mean'],x['support'],-x['repeat']))
print('SELECTED',json.dumps(best),flush=True)
(ROOT/'rigorous_results'/'position_auto_calibration.json').write_text(json.dumps({'baseline':base,'rows':rows,'selected':best},indent=2),encoding='utf8')
