from pathlib import Path
from collections import Counter
import json, math, statistics, time
from rigorous_gpu_benchmark import stable_world, entropy_norm, opening
from procedural_runtime_gpu import build_renderer_v9_gpu_batched
from procedural_runtime_v5 import ProtectedSlotVerifier
ROOT=Path(__file__).resolve().parent

specs=[dict(seed=2300000+i,n_entities=128,n_props=48,n_rels=24,n_facts=1200) for i in range(8)]

def evaluate(pw):
    sc,gr,ind,r=build_renderer_v9_gpu_batched(ROOT,seed=6060,use_hot=False,proposal_weight=pw,device=0,memory_limit_mb=4608)
    vf=ProtectedSlotVerifier(); sent=[];opens=[];bad=slots=facts=isel=0
    for spec in specs:
        p=stable_world(**spec);o=r.render(p);facts+=len(p);sent.extend(o['sentences']);isel+=o.get('induced_selected',0)
        bad+=int(Counter(o['represented'])!=Counter(p));slots+=len(vf.inspect_render(o));opens.extend(opening(sc,s) for s in o['sentences'])
    supports=[]
    for i in range(0,len(sent),4096):
        _,sp=sc.batch_language_support([sc.tokenize(x) for x in sent[i:i+4096]],max_order=5,slot_aware=True);supports.extend(float(x) for x in sp)
    rep=sum(a==b for a,b in zip(opens,opens[1:]))/max(1,len(opens)-1)
    ent=entropy_norm(opens);sup=statistics.mean(supports);ir=isel/len(sent)
    valid=(bad==0 and slots==0)
    obj=(-1e9 if not valid else sup+.12*ent+.06*ir-.12*rep)
    return {'proposal_weight':pw,'objective':obj,'support':sup,'opening_entropy':ent,'induced_rate':ir,'repeat':rep,'bad':bad,'slot_errors':slots,'sentences':len(sent)}

rows=[];best=None;w=0.0;step=.03;non_improve_after_best=0
while w<=1.2+1e-9:
    row=evaluate(round(w,2));rows.append(row);print(json.dumps(row),flush=True)
    if best is None or row['objective']>best['objective']+1e-9:
        best=row;non_improve_after_best=0
    elif row['proposal_weight']>best['proposal_weight']:
        non_improve_after_best+=1
    # Stop only after a genuine internal peak with four worse points to the right.
    if best['proposal_weight']<row['proposal_weight'] and non_improve_after_best>=4:break
    w+=step
print('SELECTED',json.dumps(best),flush=True)
(Path(ROOT/'rigorous_results'/'proposal_auto_calibration_p5.json')).write_text(json.dumps({'rows':rows,'selected':best},indent=2),encoding='utf8')
