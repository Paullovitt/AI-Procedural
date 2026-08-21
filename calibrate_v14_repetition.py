from pathlib import Path
from collections import Counter
import json, statistics, math
from rigorous_gpu_benchmark import stable_world
from procedural_runtime_v14 import build_renderer_v14_gpu
from procedural_runtime_v12 import abstract_surface_shape
from procedural_runtime_gpu import SemanticTraceVerifier
from procedural_runtime_v5 import ProtectedSlotVerifier

ROOT=Path('.');OUT=ROOT/'rigorous_results_v12';OUT.mkdir(exist_ok=True)
plans=[stable_world(seed=970000+i,n_entities=160,n_props=56,n_rels=24,n_facts=1800) for i in range(10)]
weights=[1.1,1.4,1.8,2.4,3.2,4.5]

def evalw(w):
    sc,gr,ind,r=build_renderer_v14_gpu(ROOT,seed=6262,proposal_weight=.24,position_weight=7.0,diversity_weight=2.6,focus_diversity_weight=1.17,repetition_weight=w,device=0,memory_limit_mb=4608)
    vf=ProtectedSlotVerifier();tv=SemanticTraceVerifier();bad=slots=trace=0;langs=[];reps=[];margins=[];uniqs=[]
    for p in plans:
        o=r.render(p);bad+=int(Counter(o['represented'])!=Counter(p));slots+=len(vf.inspect_render(o));trace+=len(tv.inspect_render(o))
        opens=[];shapes=set();starts=set(o['paragraph_starts'])
        for i,(s,pick) in enumerate(zip(o['sentences'],o['picks'])):
            ws=sc.tokenize(s);opens.append(' '.join(x for x in ws[:3] if not sc.is_slot(x)));shapes.add(abstract_surface_shape(sc,s));langs.append(float(pick[4]))
            actual=i in starts;a,_=gr.opening_position_score_tokens(ws,actual,sc.is_slot);b,_=gr.opening_position_score_tokens(ws,not actual,sc.is_slot);margins.append(a-b)
        reps.append(sum(a==b for a,b in zip(opens,opens[1:]))/max(1,len(opens)-1));uniqs.append(len(shapes)/max(1,len(o['sentences'])))
    return {'repetition_weight':w,'bad':bad,'slots':slots,'trace':trace,'language_score':statistics.mean(langs),'repeat':statistics.mean(reps),'position_margin':statistics.mean(margins),'unique':statistics.mean(uniqs)}
rows=[]
for w in weights:
    x=evalw(w);rows.append(x);print(json.dumps(x),flush=True)
base=rows[0]
# No weighted objective: a candidate must strictly reduce repetition without degrading
# language score, position evidence, uniqueness, or semantic integrity on the same shadow set.
elig=[x for x in rows if x['bad']==x['slots']==x['trace']==0 and x['repeat']<base['repeat'] and x['language_score']>=base['language_score'] and x['position_margin']>=base['position_margin'] and x['unique']>=base['unique']]
selected=min(elig,key=lambda x:x['repetition_weight']) if elig else base
res={'format':'V14-Repetition-AutoCalibration','baseline':base,'rows':rows,'selected':selected,'selection':'smallest coefficient that strictly reduces opening repetition with no regression in language score, position margin, uniqueness, or semantics'}
(OUT/'v14_repetition_calibration.json').write_text(json.dumps(res,indent=2),encoding='utf8');print('SELECTED',json.dumps(selected),flush=True)
