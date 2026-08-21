from __future__ import annotations
from pathlib import Path
from collections import Counter, defaultdict
import json, time, math, random, re, statistics, hashlib, os
import torch

from procedural_runtime_gpu import build_renderer_v9_gpu_batched
from procedural_runtime_v5 import ProtectedSlotVerifier, LocalFluencyVerifier

ROOT = Path(__file__).resolve().parent
CFG = json.loads((ROOT/'gpu_config.json').read_text(encoding='utf8'))
OUTDIR = ROOT/'rigorous_results'
OUTDIR.mkdir(exist_ok=True)

SLOT_RX = re.compile(r'\b(?:e\d+|a\d+|v\d+|r\d+)\b', re.I)

# Benchmark-only world generator. Properties are functional inside one world:
# one (entity, property) -> one value. This removes contradictions accidentally
# introduced by the historical make_world benchmark. No language rule is given
# to the learner/renderer.
def stable_world(seed:int, n_entities:int, n_props:int, n_rels:int, n_facts:int):
    rng=random.Random(seed)
    ents=[f'e{i:05d}' for i in range(n_entities)]
    props=[f'a{i:04d}' for i in range(n_props)]
    vals=[f'v{i:04d}' for i in range(max(96, n_props*2))]
    rels=[f'r{i:04d}' for i in range(n_rels)]
    max_prop=n_entities*n_props
    max_rel=n_entities*(n_entities-1)*n_rels
    if n_facts > max_prop+max_rel:
        raise ValueError('world capacity too small')
    # Generate a mixed world without duplicate functional properties.
    prop_target=min(max_prop, int(round(n_facts*0.70)))
    rel_target=n_facts-prop_target
    facts=[]
    prop_pairs=list((e,p) for e in ents for p in props)
    rng.shuffle(prop_pairs)
    for e,p in prop_pairs[:prop_target]:
        # Value is deterministic from hidden world state, not exposed as a rule.
        v=rng.choice(vals)
        facts.append(('prop',e,p,v))
    seen_rel=set()
    while len(seen_rel)<rel_target:
        a,b=rng.sample(ents,2); r=rng.choice(rels)
        t=(a,r,b)
        if t in seen_rel: continue
        seen_rel.add(t); facts.append(('rel',a,r,b))
    rng.shuffle(facts)
    return facts

def functional_conflicts(facts):
    seen={}; bad=[]
    for f in facts:
        if f[0]!='prop': continue
        key=(f[1],f[2]); v=f[3]
        old=seen.get(key)
        if old is not None and old!=v: bad.append((key,old,v))
        else: seen[key]=v
    return bad

def q(vals,p):
    if not vals: return 0
    a=sorted(vals); return a[int(p*(len(a)-1))]

def entropy_norm(seq):
    c=Counter(seq); N=sum(c.values())
    if N<=0:return 0.0
    H=-sum((n/N)*math.log(n/N) for n in c.values())
    return H/max(math.log(max(2,len(c))),1e-12)

def abstract_sentence(s):
    return SLOT_RX.sub('__slot__',s.lower())

def opening(sc,s):
    ws=sc.tokenize(s)
    return ' '.join(w for w in ws[:3] if not sc.is_slot(w))

def gpu_support_chunked(sc, sentences, chunk=4096):
    supports=[]; langs=[]
    for i in range(0,len(sentences),chunk):
        words=[sc.tokenize(x) for x in sentences[i:i+chunk]]
        lg,sp=sc.batch_language_support(words,max_order=4,slot_aware=True)
        langs.extend(float(x) for x in lg); supports.extend(float(x) for x in sp)
    return langs,supports

def render_suite(renderer,sc,specs,label):
    vf=ProtectedSlotVerifier()
    torch.cuda.reset_peak_memory_stats()
    before_free,before_total=torch.cuda.mem_get_info()
    t0=time.perf_counter()
    total_facts=total_sent=total_para=total_chars=0
    bad_docs=slot_errors=input_conflicts=output_meta_conflicts=0
    all_lens=[]; all_open=[]; all_abs=[]; all_templates=[]; all_sent=[]
    induced=induced_cands=0; para_focus_bad=0; para_focus_total=0
    doc_rows=[]
    for di,spec in enumerate(specs):
        plan=stable_world(**spec)
        input_conflicts += len(functional_conflicts(plan))
        d0=time.perf_counter(); out=renderer.render(plan); dt=time.perf_counter()-d0
        exact=(Counter(out['represented'])==Counter(plan))
        bad_docs += int(not exact)
        slot_errors += len(vf.inspect_render(out))
        total_facts += len(plan); total_sent += len(out['sentences']); total_para += len(out['paragraphs'])
        total_chars += len(out['text'])
        induced += int(out.get('induced_selected',0)); induced_cands += int(out.get('induced_candidates',0))
        # Every paragraph in this renderer must contain one semantic focus only.
        cur=[]; last=None
        for g in out['groups']:
            focus=g[0][1]
            if last is None: last=focus
            if focus!=last:
                para_focus_total+=1
                if len(set(cur))!=1: para_focus_bad+=1
                cur=[]; last=focus
            cur.append(focus)
        if cur:
            para_focus_total+=1
            if len(set(cur))!=1: para_focus_bad+=1
        for s,pick in zip(out['sentences'],out['picks']):
            ws=sc.tokenize(s); all_lens.append(len(ws)); all_open.append(opening(sc,s)); all_abs.append(abstract_sentence(s))
            all_templates.append(pick[3].get('template','')); all_sent.append(s)
        doc_rows.append({'doc':di,'facts':len(plan),'sentences':len(out['sentences']),'paragraphs':len(out['paragraphs']),
                         'chars':len(out['text']),'seconds':dt,'induced':out.get('induced_selected',0),
                         'semantic_exact':exact,'slot_errors':len(vf.inspect_render(out))})
        if (di+1)%max(1,len(specs)//5)==0:
            print('PROGRESS',label,di+1,'/',len(specs),'facts',total_facts,'sent',total_sent,flush=True)
    elapsed=time.perf_counter()-t0
    langs,supports=gpu_support_chunked(sc,all_sent)
    after_free,after_total=torch.cuda.mem_get_info()
    res={
      'label':label,'documents':len(specs),'facts':total_facts,'sentences':total_sent,'paragraphs':total_para,'chars':total_chars,
      'seconds':elapsed,'facts_per_s':total_facts/elapsed,'chars_per_s':total_chars/elapsed,
      'semantic_bad_docs':bad_docs,'slot_errors':slot_errors,'input_functional_conflicts':input_conflicts,
      'paragraph_focus_bad':para_focus_bad,'paragraph_focus_total':para_focus_total,
      'mean_words':statistics.mean(all_lens),'median_words':statistics.median(all_lens),
      'p90_words':q(all_lens,.90),'p95_words':q(all_lens,.95),'p99_words':q(all_lens,.99),
      'opening_entropy':entropy_norm(all_open),'template_entropy':entropy_norm(all_templates),
      'immediate_open_repeat':sum(a==b for a,b in zip(all_open,all_open[1:]))/max(1,len(all_open)-1),
      'abstract_sentence_unique_rate':len(set(all_abs))/max(1,len(all_abs)),
      'avg_trigram_support':statistics.mean(supports),'avg_gpu_language_score':statistics.mean(langs),
      'induced_selected':induced,'induced_rate':induced/max(1,total_sent),'induced_candidates':induced_cands,
      'peak_allocated_mb':torch.cuda.max_memory_allocated()/2**20,'peak_reserved_mb':torch.cuda.max_memory_reserved()/2**20,
      'vram_total_mb':after_total/2**20,'vram_free_after_mb':after_free/2**20,
      'doc_rows':doc_rows,
    }
    return res, all_sent

def adversarial_suite(sc,renderer,n_docs=24,facts_per_doc=1200,seed=910000):
    vf=ProtectedSlotVerifier(); flu=LocalFluencyVerifier(sc,threshold=.65,window=5)
    rng=random.Random(seed); rows=[]; sentences=[]; groups=[]
    for i in range(n_docs):
        p=stable_world(seed=seed+i,n_entities=128,n_props=48,n_rels=24,n_facts=facts_per_doc)
        o=renderer.render(p)
        sentences.extend(o['sentences']); groups.extend(o['groups'])
    # Semantic slot corruption.
    targets=rng.sample(range(len(sentences)), min(6000,len(sentences)))
    corrupt_n=detected=clean_fp=0
    for idx in targets:
        s=sentences[idx]; g=groups[idx]
        clean_fp += int(not vf.inspect_sentence(s,g))
        slots=SLOT_RX.findall(s)
        if not slots: continue
        old=rng.choice(slots); kind=old[0].lower()
        expected=set(x.lower() for x in vf.expected_slots(g)); j=990000
        while f'{kind}{j}' in expected:j+=1
        cs=re.sub(rf'\b{re.escape(old)}\b',f'{kind}{j}',s,count=1,flags=re.I)
        corrupt_n+=1; detected += int(not vf.inspect_sentence(cs,g))
    # Fluency corruption: adjacent swap of non-slot tokens; evaluate whether local verifier becomes more suspicious.
    flu_trials=flu_detect=flu_clean_alarm=0
    candidates=rng.sample(range(len(sentences)),min(1200,len(sentences)))
    for idx in candidates:
        ws=sc.tokenize(sentences[idx])
        clean=flu.inspect_tokens(ws); flu_clean_alarm += int(bool(clean))
        choices=[i for i in range(len(ws)-1) if not sc.is_slot(ws[i]) and not sc.is_slot(ws[i+1]) and ws[i]!=ws[i+1]]
        if not choices: continue
        i=rng.choice(choices); cw=list(ws); cw[i],cw[i+1]=cw[i+1],cw[i]
        flu_trials+=1; flu_detect += int(bool(flu.inspect_tokens(cw)))
    return {'semantic_corruptions':corrupt_n,'semantic_detected':detected,
            'semantic_detection_rate':detected/max(1,corrupt_n),'clean_slot_false_positive':clean_fp,
            'fluency_swap_trials':flu_trials,'fluency_detected':flu_detect,
            'fluency_detection_rate':flu_detect/max(1,flu_trials),'fluency_clean_alarm_docs':flu_clean_alarm}

def determinism_suite(renderer_builder, plan):
    # Same seed + same plan must be byte-identical after a fresh reload.
    outs=[]
    for _ in range(2):
        sc,gr,ind,r=renderer_builder()
        o=r.render(plan); outs.append(o['text'])
        del r,ind,gr,sc; torch.cuda.empty_cache()
    return {'same_text':outs[0]==outs[1],
            'sha256_1':hashlib.sha256(outs[0].encode('utf8')).hexdigest(),
            'sha256_2':hashlib.sha256(outs[1].encode('utf8')).hexdigest()}

def main():
    print('START',time.strftime('%Y-%m-%d %H:%M:%S'),flush=True)
    print('CUDA',torch.cuda.is_available(),torch.cuda.get_device_name(0),flush=True)
    # Automatic shadow calibration: no hand-picked winning value. All candidates use GPU.
    proposal_grid=[0.00,0.03,0.06,0.09,0.12]
    dev_specs=[dict(seed=800000+i,n_entities=128,n_props=48,n_rels=24,n_facts=1500) for i in range(12)]
    calibration=[]
    for pw in proposal_grid:
        sc,gr,ind,r=build_renderer_v9_gpu_batched(ROOT,seed=31337,use_hot=False,proposal_weight=pw,
                                                   device=CFG['device'],memory_limit_mb=CFG['memory_limit_mb'])
        rr,_=render_suite(r,sc,dev_specs,f'calibration_pw_{pw:.2f}')
        # General objective: hard semantic gate; reward corpus support, diversity, induced generalization; penalize repetition.
        valid=(rr['semantic_bad_docs']==0 and rr['slot_errors']==0 and rr['input_functional_conflicts']==0)
        objective=(-1e9 if not valid else rr['avg_trigram_support'] + 0.12*rr['opening_entropy']
                   +0.06*rr['induced_rate'] -0.12*rr['immediate_open_repeat'])
        calibration.append({'proposal_weight':pw,'objective':objective,
                            **{k:rr[k] for k in ['avg_trigram_support','opening_entropy','induced_rate','immediate_open_repeat','semantic_bad_docs','slot_errors']}})
        print('CAL',json.dumps(calibration[-1]),flush=True)
        del r,ind,gr,sc; torch.cuda.empty_cache()
    best=max(calibration,key=lambda x:x['objective'])
    chosen=float(best['proposal_weight'])
    print('CAL_SELECTED',chosen,flush=True)

    sc,gr,ind,renderer=build_renderer_v9_gpu_batched(ROOT,seed=31337,use_hot=False,proposal_weight=chosen,
                                                       device=CFG['device'],memory_limit_mb=CFG['memory_limit_mb'])
    print('GPU_STATUS',json.dumps(sc.gpu_status()),flush=True)
    print('PROMOTED_WRAPPERS',json.dumps(ind.promoted,ensure_ascii=False),flush=True)

    # Held-out large/complex documents. Seeds are disjoint from calibration.
    specs=[]
    specs += [dict(seed=1000000+i,n_entities=192,n_props=64,n_rels=32,n_facts=2000) for i in range(48)]
    specs += [dict(seed=1100000+i,n_entities=256,n_props=80,n_rels=40,n_facts=5000) for i in range(24)]
    specs += [dict(seed=1200000+i,n_entities=384,n_props=96,n_rels=48,n_facts=12000) for i in range(8)]
    specs += [dict(seed=1300000+i,n_entities=512,n_props=128,n_rels=64,n_facts=20000) for i in range(2)]
    held,sentences=render_suite(renderer,sc,specs,'heldout_large')
    adv=adversarial_suite(sc,renderer)

    det_plan=stable_world(seed=1400000,n_entities=160,n_props=64,n_rels=32,n_facts=3000)
    def builder():
        return build_renderer_v9_gpu_batched(ROOT,seed=424242,use_hot=False,proposal_weight=chosen,
                                              device=CFG['device'],memory_limit_mb=CFG['memory_limit_mb'])
    det=determinism_suite(builder,det_plan)

    # Save representative large text and metrics. The text is generated, not training data.
    sample_plan=stable_world(seed=1500000,n_entities=384,n_props=96,n_rels=48,n_facts=12000)
    sample=renderer.render(sample_plan)
    (OUTDIR/'large_generated_sample.txt').write_text(sample['text'],encoding='utf8')
    result={'format':'AI-Procedural-V9-Rigorous-GPU-v1','chosen_proposal_weight':chosen,
            'calibration':calibration,'heldout':{k:v for k,v in held.items() if k!='doc_rows'},
            'adversarial':adv,'determinism':det,'gpu':sc.gpu_status(),
            'promoted_wrappers':ind.promoted,
            'sample':{'facts':len(sample_plan),'sentences':len(sample['sentences']),'paragraphs':len(sample['paragraphs']),
                      'chars':len(sample['text']),'semantic_exact':Counter(sample['represented'])==Counter(sample_plan),
                      'slot_errors':len(ProtectedSlotVerifier().inspect_render(sample)),
                      'path':str(OUTDIR/'large_generated_sample.txt')}}
    (OUTDIR/'rigorous_gpu_results.json').write_text(json.dumps(result,ensure_ascii=False,indent=2),encoding='utf8')
    (OUTDIR/'heldout_doc_rows.json').write_text(json.dumps(held['doc_rows'],ensure_ascii=False,indent=2),encoding='utf8')
    print('FINAL',json.dumps(result,ensure_ascii=False),flush=True)

if __name__=='__main__':
    main()
