from pathlib import Path
from collections import Counter
import json, statistics, math, time
from rigorous_gpu_benchmark import stable_world
from procedural_runtime_v12 import build_renderer_v12_gpu, abstract_surface_shape
from procedural_runtime_v14 import build_renderer_v14_gpu
from procedural_runtime_gpu import SemanticTraceVerifier
from procedural_runtime_v5 import ProtectedSlotVerifier

ROOT=Path('.');OUT=ROOT/'rigorous_results_v12';OUT.mkdir(exist_ok=True)
plans=[stable_world(seed=930000+i,n_entities=192,n_props=64,n_rels=28,n_facts=2500) for i in range(24)]

def q(a,p):
    a=sorted(a);return a[int(p*(len(a)-1))] if a else 0

def hist_summary(h):
    t=sum(h);mean=sum(i*n for i,n in enumerate(h))/max(1,t);vals=[]
    for i,n in enumerate(h):vals.extend([i]*min(n,2000))
    # exact quantiles without materializing full histogram
    def hq(p):
        target=p*t;c=0
        for i,n in enumerate(h):
            c+=n
            if c>=target:return i
        return len(h)-1
    return {'mean':mean,'median':hq(.5),'p90':hq(.9),'p95':hq(.95)}

def evaluate(kind):
    if kind=='v12':sc,gr,ind,r=build_renderer_v12_gpu(ROOT,seed=5151,proposal_weight=.24,position_weight=7.0,diversity_weight=2.6,focus_diversity_weight=1.17,repetition_weight=1.1,device=0,memory_limit_mb=4608)
    else:sc,gr,ind,r=build_renderer_v14_gpu(ROOT,seed=5151,proposal_weight=.24,position_weight=7.0,diversity_weight=2.6,focus_diversity_weight=1.17,repetition_weight=1.1,device=0,memory_limit_mb=4608)
    vf=ProtectedSlotVerifier();tv=SemanticTraceVerifier();t0=time.perf_counter();facts=sent=paras=bad=slots=trace=0
    para_sizes=[];position_margins=[];supports=[];lang=[];repeats=[];uniq=[];multi_focus=[];linked_inside=[]
    for di,p in enumerate(plans):
        o=r.render(p);facts+=len(p);sent+=len(o['sentences']);paras+=len(o['paragraphs'])
        bad+=int(Counter(o['represented'])!=Counter(p));slots+=len(vf.inspect_render(o));trace+=len(tv.inspect_render(o))
        ps=o.get('paragraph_sizes')
        if ps is None:
            # V12 paragraphs align to focus changes; reconstruct sizes from text sentence sequence.
            ps=[];cur=None;c=0
            for g in o['groups']:
                f=g[0][1]
                if cur is None:cur=f
                if f!=cur:ps.append(c);c=0;cur=f
                c+=1
            if c:ps.append(c)
        para_sizes+=ps
        # Paragraph starts from sizes.
        starts=set();z=0
        for n in ps:starts.add(z);z+=n
        opens=[];shapes=set();focuses=[g[0][1] for g in o['groups']]
        adj={x:set() for x in focuses}
        for f in p:
            if f[0]=='rel':adj.setdefault(f[1],set()).add(f[3]);adj.setdefault(f[3],set()).add(f[1])
        for i,(s,pick) in enumerate(zip(o['sentences'],o['picks'])):
            ws=sc.tokenize(s);op=' '.join(w for w in ws[:3] if not sc.is_slot(w));opens.append(op);shapes.add(abstract_surface_shape(sc,s))
            actual=(i in starts);a,_=gr.opening_position_score_tokens(ws,actual,sc.is_slot);b,_=gr.opening_position_score_tokens(ws,not actual,sc.is_slot);position_margins.append(a-b)
            supports.append(float(pick[6])/sc.tables.get('p2',{}).__len__() if False else 0.0);lang.append(float(pick[4]))
        repeats.append(sum(a==b for a,b in zip(opens,opens[1:]))/max(1,len(opens)-1));uniq.append(len(shapes)/max(1,len(o['sentences'])))
        # paragraph focus diversity + graph-link rate within paragraphs
        at=0
        for n in ps:
            fs=focuses[at:at+n];multi_focus.append(len(set(fs)))
            for a,b in zip(fs,fs[1:]):linked_inside.append(b in adj.get(a,set()))
            at+=n
    # Proper p2-p5 support in large CUDA batches over all sentence text would be expensive to retain;
    # pick[4] is the GPU language score already used for selection.
    return {'kind':kind,'documents':len(plans),'facts':facts,'sentences':sent,'paragraphs':paras,'seconds':time.perf_counter()-t0,
            'bad_docs':bad,'slot_errors':slots,'trace_errors':trace,'paragraph_mean':statistics.mean(para_sizes),'paragraph_median':statistics.median(para_sizes),'paragraph_p90':q(para_sizes,.9),'paragraph_p95':q(para_sizes,.95),
            'mean_position_margin':statistics.mean(position_margins),'position_margin_positive_rate':sum(x>0 for x in position_margins)/len(position_margins),
            'mean_gpu_language_score':statistics.mean(lang),'immediate_open_repeat':statistics.mean(repeats),'within_doc_abstract_unique':statistics.mean(uniq),
            'mean_focuses_per_paragraph':statistics.mean(multi_focus),'linked_transition_inside_paragraph_rate':statistics.mean(linked_inside) if linked_inside else 0.0,
            'corpus_para_target':hist_summary(sc.struct.get('para_sent',[]))}

r12=evaluate('v12');print('V12',json.dumps(r12),flush=True)
r14=evaluate('v14');print('V14',json.dumps(r14),flush=True)
promote=(r14['bad_docs']==r14['slot_errors']==r14['trace_errors']==0 and r14['mean_position_margin']>r12['mean_position_margin'] and abs(r14['paragraph_median']-r14['corpus_para_target']['median']) < abs(r12['paragraph_median']-r12['corpus_para_target']['median']))
res={'format':'V12-vs-V14-GPU','v12':r12,'v14':r14,'promotion_gate':promote}
(OUT/'v12_v14_comparison.json').write_text(json.dumps(res,indent=2),encoding='utf8');print('PROMOTION',promote,flush=True)
