from __future__ import annotations
from pathlib import Path
from collections import Counter
import json,time,statistics,torch

from procedural_runtime_v5 import build_renderer_v9, make_world, ProtectedSlotVerifier
from procedural_runtime_gpu import build_renderer_v9_gpu_batched

ROOT=Path(__file__).resolve().parent
CFG=json.loads((ROOT/'gpu_config.json').read_text(encoding='utf8'))
PLANS=[make_world(88000+i,n_entities=16,n_props=10,n_rels=6,n_facts=220) for i in range(80)]


def run(renderer):
    vf=ProtectedSlotVerifier();bad=slot_errors=facts=sentences=0
    t0=time.perf_counter()
    for p in PLANS:
        o=renderer.render(p)
        facts+=len(p);sentences+=len(o['sentences'])
        bad+=int(Counter(o['represented'])!=Counter(p))
        slot_errors+=len(vf.inspect_render(o))
    dt=time.perf_counter()-t0
    return {'facts':facts,'sentences':sentences,'seconds':dt,'facts_per_s':facts/dt,
            'bad_docs':bad,'slot_errors':slot_errors}

print('Loading CPU...')
_,_,_,cpu=build_renderer_v9(ROOT,seed=101,use_hot=False,proposal_weight=.06)
print('Loading GPU...')
sc,_,_,gpu=build_renderer_v9_gpu_batched(ROOT,seed=101,use_hot=False,proposal_weight=.06,
                                  device=CFG['device'],memory_limit_mb=CFG['memory_limit_mb'])

# Numerical parity on deliberately varied token sequences.
tests=[['de','acordo','com'],['no','caso','de','e001'],['para','e001','a01','tem','o','valor','v01'],
       ['relativamente','a','e777','r01','liga','e777','a','e002']]
from procedural_runtime_v3 import BagacoSurfaceScorer
cs=BagacoSurfaceScorer(ROOT,use_hot=False)
errs=[]
for ws in tests:
    a=cs.score_tokens(ws,max_order=4,slot_aware=True)
    b=sc.score_tokens(ws,max_order=4,slot_aware=True)
    errs.append(abs(a-b))
print('MAX_SCORE_ABS_ERROR',max(errs))

torch.cuda.reset_peak_memory_stats()
rc=run(cpu)
rg=run(gpu)
rg['peak_allocated_mb']=torch.cuda.max_memory_allocated()/2**20
rg['peak_reserved_mb']=torch.cuda.max_memory_reserved()/2**20
print('CPU',json.dumps(rc,ensure_ascii=False))
print('GPU',json.dumps(rg,ensure_ascii=False))
print('GPU_STATUS',json.dumps(sc.gpu_status(),ensure_ascii=False))

