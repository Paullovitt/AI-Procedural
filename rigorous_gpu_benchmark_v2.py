from __future__ import annotations
from pathlib import Path
from collections import Counter
import json,time,math,random,re,statistics,hashlib,threading,subprocess
import torch

from rigorous_gpu_benchmark import stable_world, entropy_norm, q, opening, abstract_sentence, functional_conflicts
from procedural_runtime_gpu import build_renderer_v10_gpu, SemanticTraceVerifier, GpuLocalOrderVerifier
from procedural_runtime_v5 import ProtectedSlotVerifier

ROOT=Path(__file__).resolve().parent
OUT=ROOT/'rigorous_results_v2';OUT.mkdir(exist_ok=True)
cal=json.loads((ROOT/'rigorous_results'/'proposal_auto_calibration_p5.json').read_text(encoding='utf8'))
PW=float(cal['selected']['proposal_weight'])
ordcal=json.loads((ROOT/'rigorous_results'/'order_verifier_calibration.json').read_text(encoding='utf8'))
ORDER_THRESHOLD=float(ordcal['train']['threshold'])

class GpuMonitor:
    def __init__(self):
        self.stop=threading.Event();self.rows=[];self.th=threading.Thread(target=self._run,daemon=True)
    def _run(self):
        while not self.stop.is_set():
            try:
                s=subprocess.check_output(['nvidia-smi','--query-gpu=memory.used,utilization.gpu,temperature.gpu,power.draw','--format=csv,noheader,nounits'],text=True,timeout=2).strip().split(',')
                self.rows.append((time.time(),float(s[0]),float(s[1]),float(s[2]),float(s[3])))
            except Exception:pass
            self.stop.wait(.35)
    def __enter__(self):self.th.start();return self
    def __exit__(self,*a):self.stop.set();self.th.join(timeout=3)
    def stats(self):
        if not self.rows:return {}
        return {'samples':len(self.rows),'max_total_vram_used_mb':max(x[1] for x in self.rows),
                'mean_gpu_util_pct':statistics.mean(x[2] for x in self.rows),'max_gpu_util_pct':max(x[2] for x in self.rows),
                'max_temp_c':max(x[3] for x in self.rows),'max_power_w':max(x[4] for x in self.rows)}

def reservoir_add(res,item,k,rng,seen):
    if len(res)<k:res.append(item)
    else:
        j=rng.randrange(seen)
        if j<k:res[j]=item

def attack_role_swap(text,pick,tracev):
    tr=list(tracev.trace(text)); ents=[]
    for x in tr:
        if x.startswith('e') and x not in ents:ents.append(x)
    if len(ents)<2:return None
    a,b=ents[0],ents[-1]
    return text.replace(a,'__TMP_E__').replace(b,a).replace('__TMP_E__',b)

def attack_value_swap(text,pick,tracev):
    tr=list(tracev.trace(text)); vals=[]
    for x in tr:
        if x.startswith('v') and x not in vals:vals.append(x)
    if len(vals)<2:return None
    a,b=vals[0],vals[-1]
    return text.replace(a,'__TMP_V__').replace(b,a).replace('__TMP_V__',b)

def attack_slot_replace(text,pick,tracev):
    tr=list(tracev.trace(text))
    if not tr:return None
    old=tr[-1];kind=old[0];new=f'{kind}999999'
    return text.replace(old,new,1)

def main():
    print('START_V2',time.strftime('%Y-%m-%d %H:%M:%S'),'proposal_weight',PW,'order_threshold',ORDER_THRESHOLD,flush=True)
    sc,grammar,inducer,renderer=build_renderer_v10_gpu(ROOT,seed=919191,use_hot=False,proposal_weight=PW,device=0,memory_limit_mb=4608)
    tracev=SemanticTraceVerifier();slotv=ProtectedSlotVerifier();ordv=GpuLocalOrderVerifier(sc,threshold=ORDER_THRESHOLD,window=9)
    print('GPU_STATUS',json.dumps(sc.gpu_status(),ensure_ascii=False),flush=True)
    print('LEARNED_WRAPPERS',json.dumps(inducer.promoted,ensure_ascii=False),flush=True)

    specs=[]
    specs += [dict(seed=3000000+i,n_entities=256,n_props=80,n_rels=40,n_facts=5000) for i in range(100)]
    specs += [dict(seed=3100000+i,n_entities=384,n_props=96,n_rels=48,n_facts=12000) for i in range(20)]
    specs += [dict(seed=3200000+i,n_entities=512,n_props=128,n_rels=64,n_facts=20000) for i in range(8)]
    specs += [dict(seed=3300000+i,n_entities=1024,n_props=128,n_rels=64,n_facts=50000) for i in range(2)]
    # total = 1,000,000 facts
    rng=random.Random(20260821); reservoir=[]; seen_sent=0
    facts=sentences=paragraphs=chars=bad_docs=slot_errors=trace_errors=input_conflicts=induced=induced_cands=0
    para_bad=para_total=0
    lens=[];opens=[];templates=[];abs_set=set();support_sum=lang_sum=score_n=0;doc_unique_rates=[];doc_rows=[]
    torch.cuda.reset_peak_memory_stats();t0=time.perf_counter()
    with GpuMonitor() as mon:
      for di,spec in enumerate(specs):
        p=stable_world(**spec);input_conflicts+=len(functional_conflicts(p));d0=time.perf_counter();o=renderer.render(p);dt=time.perf_counter()-d0
        exact=Counter(o['represented'])==Counter(p);bad_docs+=int(not exact);slot_errors+=len(slotv.inspect_render(o));trace_errors+=len(tracev.inspect_render(o))
        facts+=len(p);sentences+=len(o['sentences']);paragraphs+=len(o['paragraphs']);chars+=len(o['text']);induced+=o.get('induced_selected',0);induced_cands+=o.get('induced_candidates',0)
        # paragraph focus purity
        cur=[];last=None
        for g in o['groups']:
            f=g[0][1]
            if last is None:last=f
            if f!=last:
                para_total+=1;para_bad+=int(len(set(cur))!=1);cur=[];last=f
            cur.append(f)
        if cur:para_total+=1;para_bad+=int(len(set(cur))!=1)
        dabs=set()
        for s,pick,g in zip(o['sentences'],o['picks'],o['groups']):
            ws=sc.tokenize(s);lens.append(len(ws));op=opening(sc,s);opens.append(op);templates.append(pick[3].get('template',''))
            ab=abstract_sentence(s);abs_set.add(ab);dabs.add(ab);seen_sent+=1;reservoir_add(reservoir,(s,pick,g),12000,rng,seen_sent)
        doc_unique_rates.append(len(dabs)/max(1,len(o['sentences'])))
        # GPU p2-p5 language score/support in document batches
        for j in range(0,len(o['sentences']),4096):
            words=[sc.tokenize(x) for x in o['sentences'][j:j+4096]];lg,sp=sc.batch_language_support(words,max_order=5,slot_aware=True)
            lang_sum+=sum(float(x) for x in lg);support_sum+=sum(float(x) for x in sp);score_n+=len(words)
        doc_rows.append({'doc':di,'facts':len(p),'sentences':len(o['sentences']),'paragraphs':len(o['paragraphs']),'chars':len(o['text']),'seconds':dt,
                         'semantic_exact':exact,'slot_errors':len(slotv.inspect_render(o)),'trace_errors':len(tracev.inspect_render(o)),
                         'abstract_unique_rate':doc_unique_rates[-1],'induced_rate':o.get('induced_selected',0)/max(1,len(o['sentences']))})
        if (di+1)%10==0 or di+1==len(specs):print('PROGRESS_V2',di+1,'/',len(specs),'facts',facts,'sent',sentences,'chars',chars,flush=True)
      monitor_stats=mon.stats()
    elapsed=time.perf_counter()-t0

    # Rigorous adversarial tests on held-out generated realizations.
    rng.shuffle(reservoir)
    role_total=role_detect=value_total=value_detect=slot_total=slot_detect=0
    for s,pick,g in reservoir[:10000]:
        a=attack_role_swap(s,pick,tracev)
        if a is not None and a!=s:
            role_total+=1;role_detect+=int(not tracev.inspect_sentence(a,pick))
        b=attack_value_swap(s,pick,tracev)
        if b is not None and b!=s:
            value_total+=1;value_detect+=int(not tracev.inspect_sentence(b,pick))
        c=attack_slot_replace(s,pick,tracev)
        if c is not None and c!=s:
            slot_total+=1;slot_detect+=int(not tracev.inspect_sentence(c,pick))

    # Local word-order corruption using a separate sample; verifier is GPU p2-p5.
    clean_alarm=order_total=order_detect=0
    for s,pick,g in reservoir[:2500]:
        clean_alarm+=int(bool(ordv.inspect(s)))
        ws=sc.tokenize(s); choices=[i for i in range(len(ws)-1) if not sc.is_slot(ws[i]) and not sc.is_slot(ws[i+1]) and ws[i]!=ws[i+1]]
        if not choices:continue
        j=rng.choice(choices);x=list(ws);x[j],x[j+1]=x[j+1],x[j];bad=' '.join(x);order_total+=1;order_detect+=int(bool(ordv.inspect(bad)))

    # Learned discourse-position contrast: actual first sentence vs random sentence as paragraph opener.
    # Re-render a held-out medium doc and compare the same learned position score, no labels beyond paragraph boundary.
    cp=stable_world(seed=3400000,n_entities=256,n_props=80,n_rels=40,n_facts=8000);co=renderer.render(cp)
    byfocus={}
    for s,g in zip(co['sentences'],co['groups']):byfocus.setdefault(g[0][1],[]).append(s)
    diffs=[]
    for ss in byfocus.values():
        if len(ss)<2:continue
        actual,_=grammar.opening_position_score_tokens(sc.tokenize(ss[0]),True,sc.is_slot)
        alt_s=rng.choice(ss[1:]);alt,_=grammar.opening_position_score_tokens(sc.tokenize(alt_s),True,sc.is_slot)
        diffs.append(actual-alt)
    discourse={'comparisons':len(diffs),'mean_actual_minus_random':statistics.mean(diffs) if diffs else 0.0,
               'actual_better_rate':sum(x>0 for x in diffs)/max(1,len(diffs))}

    # Fresh deterministic reload check.
    dp=stable_world(seed=3500000,n_entities=192,n_props=64,n_rels=32,n_facts=5000)
    texts=[]
    for _ in range(2):
        s2,g2,i2,r2=build_renderer_v10_gpu(ROOT,seed=123456,use_hot=False,proposal_weight=PW,device=0,memory_limit_mb=4608);texts.append(r2.render(dp)['text'])
        del r2,i2,g2,s2;torch.cuda.empty_cache()
    determinism={'same_text':texts[0]==texts[1],'sha256':hashlib.sha256(texts[0].encode('utf8')).hexdigest()}

    # Largest representative text: 50k facts.
    sp=stable_world(seed=3600000,n_entities=1024,n_props=128,n_rels=64,n_facts=50000);so=renderer.render(sp)
    sample_path=OUT/'large_50000_fact_text.txt';sample_path.write_text(so['text'],encoding='utf8')

    res={'format':'AI-Procedural-V10-GPU-Rigorous-v2','proposal_weight_learned':PW,'order_threshold_learned':ORDER_THRESHOLD,
         'heldout':{'documents':len(specs),'facts':facts,'sentences':sentences,'paragraphs':paragraphs,'chars':chars,'seconds':elapsed,
                    'facts_per_s':facts/elapsed,'chars_per_s':chars/elapsed,'semantic_bad_docs':bad_docs,'slot_errors':slot_errors,
                    'trace_errors':trace_errors,'input_functional_conflicts':input_conflicts,'paragraph_focus_bad':para_bad,'paragraph_focus_total':para_total,
                    'mean_words':statistics.mean(lens),'median_words':statistics.median(lens),'p90_words':q(lens,.90),'p95_words':q(lens,.95),'p99_words':q(lens,.99),
                    'opening_entropy':entropy_norm(opens),'template_entropy':entropy_norm(templates),'immediate_open_repeat':sum(a==b for a,b in zip(opens,opens[1:]))/max(1,len(opens)-1),
                    'global_abstract_unique_rate':len(abs_set)/max(1,sentences),'mean_within_doc_abstract_unique_rate':statistics.mean(doc_unique_rates),
                    'avg_trigram_support':support_sum/max(1,score_n),'avg_p2_p5_language_score':lang_sum/max(1,score_n),
                    'induced_selected':induced,'induced_rate':induced/max(1,sentences),'induced_candidates':induced_cands,
                    'torch_peak_allocated_mb':torch.cuda.max_memory_allocated()/2**20,'torch_peak_reserved_mb':torch.cuda.max_memory_reserved()/2**20,
                    'gpu_monitor':monitor_stats},
         'adversarial':{'role_swaps':role_total,'role_detected':role_detect,'role_detection_rate':role_detect/max(1,role_total),
                        'value_swaps':value_total,'value_detected':value_detect,'value_detection_rate':value_detect/max(1,value_total),
                        'slot_replacements':slot_total,'slot_detected':slot_detect,'slot_detection_rate':slot_detect/max(1,slot_total),
                        'order_swaps':order_total,'order_detected':order_detect,'order_detection_rate':order_detect/max(1,order_total),
                        'order_clean_alarms':clean_alarm,'order_clean_fpr':clean_alarm/max(1,min(2500,len(reservoir)))},
         'discourse_position_contrast':discourse,'determinism':determinism,'gpu':sc.gpu_status(),'learned_wrappers':inducer.promoted,
         'sample':{'facts':len(sp),'sentences':len(so['sentences']),'paragraphs':len(so['paragraphs']),'chars':len(so['text']),
                   'semantic_exact':Counter(so['represented'])==Counter(sp),'slot_errors':len(slotv.inspect_render(so)),'trace_errors':len(tracev.inspect_render(so)),'path':str(sample_path)}}
    (OUT/'rigorous_v2_results.json').write_text(json.dumps(res,ensure_ascii=False,indent=2),encoding='utf8')
    (OUT/'doc_rows.json').write_text(json.dumps(doc_rows,ensure_ascii=False,indent=2),encoding='utf8')
    print('FINAL_V2',json.dumps(res,ensure_ascii=False),flush=True)
if __name__=='__main__':main()
