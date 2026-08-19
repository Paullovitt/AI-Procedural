import json, math, time
from pathlib import Path
import numpy as np
from model_direct_80m import Direct80M32K

ROOT=Path(__file__).resolve().parent
RUN=ROOT/'runs'/'procedural_direct_80m_32k'
VAL=np.memmap(ROOT/'data'/'val.bin',dtype=np.uint8,mode='r')
OFFSETS=[2_000_000,4_000_000,6_000_000]
SEG=32768
model=Direct80M32K(RUN,kappa1=100.,kappa2=300.,kappa4=100.,copy_lambda=.75,copy_key=8,copy_window=32768)
rows=[]
for off in OFFSETS:
    model.reset_context(); ls=0.0; correct=0; n=0
    t0=time.perf_counter()
    data=np.asarray(VAL[off:off+SEG+1],dtype=np.uint8)
    for i in range(SEG):
        p=model.predict_after_observing(int(data[i]))
        y=int(data[i+1]); py=max(float(p[y]),1e-12)
        ls-=math.log(py); correct += int(int(np.argmax(p))==y); n+=1
    sec=time.perf_counter()-t0
    loss=ls/n
    r={'offset':off,'loss':loss,'ppl':math.exp(loss),'accuracy':correct/n,'tokens':n,'sec':sec,'tokens_s':n/sec}
    rows.append(r); print('TEST',json.dumps(r),flush=True)
avg_loss=sum(r['loss'] for r in rows)/len(rows)
summary={'experiment':'Direct-80M-32k exact-sparse PPL','kappa1':100.,'kappa2':300.,'kappa4':100.,'copy_lambda':.75,'copy_window':32768,'offsets':OFFSETS,'tests':rows,'average':{'loss':avg_loss,'ppl':math.exp(avg_loss),'accuracy':sum(r['accuracy'] for r in rows)/len(rows),'tokens_s':sum(r['tokens_s'] for r in rows)/len(rows)}}
(RUN/'ppl_benchmark.json').write_text(json.dumps(summary,indent=2),encoding='utf-8')
print('SUMMARY',json.dumps(summary),flush=True)
