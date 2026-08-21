from __future__ import annotations
from pathlib import Path
import random, json, statistics, math
import numpy as np
from rigorous_gpu_benchmark import stable_world
from procedural_runtime_gpu import build_renderer_v9_gpu_batched

ROOT=Path(__file__).resolve().parent

# General local-order anomaly score: maximum improvement obtainable by one adjacent
# non-slot swap inside a local window. Scoring is entirely from the learned Bagaço
# n-gram model on CUDA; no Portuguese rule or token blacklist is used.
def anomaly_scores(sc, sentences, window=9, max_sentences=None):
    if max_sentences is not None: sentences=sentences[:max_sentences]
    out=[]
    for si,s in enumerate(sentences):
        ws=sc.tokenize(s)
        # Evaluate each valid adjacent swap on a local window to avoid dilution by long sentences.
        variants=[]; bases=[]
        for j in range(len(ws)-1):
            if sc.is_slot(ws[j]) or sc.is_slot(ws[j+1]) or ws[j]==ws[j+1]: continue
            lo=max(0,j-window//2); hi=min(len(ws),lo+window); lo=max(0,hi-window)
            base=ws[lo:hi]
            k=j-lo
            if k<0 or k+1>=len(base): continue
            sw=list(base); sw[k],sw[k+1]=sw[k+1],sw[k]
            bases.append(base); variants.append(sw)
        if not variants:
            out.append(0.0); continue
        # one batched CUDA call per sentence; windows are small.
        langs,_=sc.batch_language_support(bases+variants,max_order=5,slot_aware=True)
        n=len(bases); gains=langs[n:]-langs[:n]
        out.append(float(np.max(gains)))
    return out

def corrupt_adjacent(sc,sentences,seed):
    rng=random.Random(seed); out=[]
    for s in sentences:
        ws=sc.tokenize(s)
        choices=[j for j in range(len(ws)-1) if not sc.is_slot(ws[j]) and not sc.is_slot(ws[j+1]) and ws[j]!=ws[j+1]]
        if not choices: continue
        j=rng.choice(choices); x=list(ws);x[j],x[j+1]=x[j+1],x[j];out.append(' '.join(x))
    return out

def best_threshold(clean,corrupt):
    vals=sorted(set(clean+corrupt))
    if not vals:return 0.0,{}
    candidates=[vals[0]-1e-12]+[(a+b)/2 for a,b in zip(vals,vals[1:])]+[vals[-1]+1e-12]
    best=None
    for t in candidates:
        tpr=sum(x>t for x in corrupt)/max(1,len(corrupt))
        fpr=sum(x>t for x in clean)/max(1,len(clean))
        bal=.5*(tpr+(1-fpr))
        row=(bal,-fpr,tpr,-abs(t),t)
        if best is None or row>best[0]:best=(row,{'threshold':t,'balanced_accuracy':bal,'tpr':tpr,'fpr':fpr})
    return best[1]['threshold'],best[1]

def collect(renderer,seed,n_docs=12,facts=900):
    ss=[]
    for i in range(n_docs):
        p=stable_world(seed=seed+i,n_entities=96,n_props=40,n_rels=20,n_facts=facts)
        ss.extend(renderer.render(p)['sentences'])
    return ss

def main():
    sc,gr,ind,r=build_renderer_v9_gpu_batched(ROOT,seed=31337,use_hot=False,proposal_weight=.12,device=0,memory_limit_mb=4608)
    train=collect(r,2000000,10,700)
    rng=random.Random(123);rng.shuffle(train);train=train[:700]
    corrupt=corrupt_adjacent(sc,train,1234)[:700]
    clean_scores=anomaly_scores(sc,train)
    corrupt_scores=anomaly_scores(sc,corrupt)
    th,fit=best_threshold(clean_scores,corrupt_scores)
    print('TRAIN',json.dumps({'clean_n':len(clean_scores),'corrupt_n':len(corrupt_scores),'clean_mean':statistics.mean(clean_scores),
                             'corrupt_mean':statistics.mean(corrupt_scores),'clean_p95':sorted(clean_scores)[int(.95*(len(clean_scores)-1))],
                             'corrupt_p50':statistics.median(corrupt_scores),**fit}),flush=True)
    # Held-out seeds and independent corruption RNG.
    test=collect(r,2100000,12,800)
    rng=random.Random(456);rng.shuffle(test);test=test[:900]
    testc=corrupt_adjacent(sc,test,9876)[:900]
    cs=anomaly_scores(sc,test); xs=anomaly_scores(sc,testc)
    tpr=sum(x>th for x in xs)/len(xs); fpr=sum(x>th for x in cs)/len(cs)
    res={'threshold':th,'test_clean_n':len(cs),'test_corrupt_n':len(xs),'test_tpr':tpr,'test_fpr':fpr,
         'test_balanced_accuracy':.5*(tpr+1-fpr),'clean_mean':statistics.mean(cs),'corrupt_mean':statistics.mean(xs)}
    print('TEST',json.dumps(res),flush=True)
    (ROOT/'rigorous_results'/'order_verifier_calibration.json').write_text(json.dumps({'train':fit,'test':res},indent=2),encoding='utf8')
if __name__=='__main__':main()
