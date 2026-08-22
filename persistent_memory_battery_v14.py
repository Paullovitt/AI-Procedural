from __future__ import annotations

from pathlib import Path
from tempfile import TemporaryDirectory
import json
import random
import statistics
import time

from persistent_memory_v14 import PersistentDimensionalMemoryV14

ROOT=Path(__file__).resolve().parent
OUT=ROOT/'rigorous_results_v12'/'persistent_memory_v14.json'


def pct(values,p):
    if not values:return 0.0
    rows=sorted(values);pos=(len(rows)-1)*p;lo=int(pos);hi=min(len(rows)-1,lo+1);f=pos-lo
    return rows[lo]*(1-f)+rows[hi]*f


def run_store_battery(n_records=20000,n_queries=500,seed=1414):
    rng=random.Random(seed)
    with TemporaryDirectory() as td:
        path=Path(td)/'episodic.sqlite3'
        mem=PersistentDimensionalMemoryV14(path,candidate_limit=512,associative_per_term=4,min_query_term_coverage=0.30)
        t0=time.perf_counter()
        for i in range(1,n_records+1):
            fluid=f'FL{i%997:03d}'
            interval=100+(i%900)
            code=f'ZX{i:06d}'
            text=f'Registro R{i:06d} equipamento EQ{i:06d} usa fluido {fluid}; código {code}; intervalo {interval} horas.'
            mem.remember(text,source='user')
        build_s=time.perf_counter()-t0

        query_ids=rng.sample(range(1,n_records+1),min(n_queries,n_records))
        lat=[];exact_ok=0;numeric_ok=0
        for i in query_ids:
            q=f'Qual fluido do equipamento EQ{i:06d} e código ZX{i:06d}?'
            rows=mem.search(q,k=3)
            if rows:
                lat.append(float(rows[0]['search_ms']))
                exact_ok += int(f'EQ{i:06d}' in rows[0]['text'] and f'ZX{i:06d}' in rows[0]['text'])
            interval=100+(i%900)
            rows2=mem.search(f'EQ{i:06d} intervalo {interval} horas',k=1)
            numeric_ok += int(bool(rows2) and f'intervalo {interval} horas' in rows2[0]['text'])
            if rows2:lat.append(float(rows2[0]['search_ms']))

        irrelevant=0
        for i in range(50):
            irrelevant += int(bool(mem.search(f'quasar_inexistente_{i} ornitorrinco_{i}',k=3)))

        weak=mem.remember('Exploração espacial, tecnologia, descoberta e futuro.',source='user')
        weak_overlap_false_positive=bool(mem.search('energia solar baterias armazenamento futuro',k=3))
        weak_exact_ok=bool(mem.search('exploração espacial futuro',k=1))

        noisy=mem.remember('meu caroo eh um civic 2015',source='user',index_text='meu caroo eh um civic 2015 carro civic 2015')
        noisy_hit=mem.search('qual carro eu tenho civic?',k=1)
        noisy_ok=bool(noisy_hit and noisy_hit[0]['id']==noisy['id'])

        dup1=mem.remember('A máquina Atlas usa fluido V7.',source='user')
        dup2=mem.remember('A máquina Atlas usa fluido V7.',source='user')
        duplicate_ok=(dup1['id']==dup2['id'] and dup2['recurrence']==2)

        c1=mem.remember('Equipamento CONFLITO estado ativo versão ALFA.',source='user')
        c2=mem.remember('Equipamento CONFLITO estado inativo versão BETA.',source='user')
        conflict=mem.search('CONFLITO versão BETA',k=2)
        contradiction_isolated=bool(conflict and conflict[0]['id']==c2['id'] and conflict[0]['id']!=c1['id'])

        stats_before=mem.stats();last_query=f'ZX{n_records:06d}'
        mem.close()
        t0=time.perf_counter();mem=PersistentDimensionalMemoryV14(path,candidate_limit=512,associative_per_term=4,min_query_term_coverage=0.30)
        reopen_ms=(time.perf_counter()-t0)*1000.0
        persisted=mem.search(last_query,k=1)
        persistence_ok=bool(persisted and f'ZX{n_records:06d}' in persisted[0]['text'])

        forgot=mem.forget(noisy['id']) and not mem.search('civic 2015',k=2)
        final_stats=mem.stats();mem.close()

        exact_acc=exact_ok/max(1,len(query_ids));numeric_acc=numeric_ok/max(1,len(query_ids))
        return {
            'records_requested':n_records,
            'queries':len(query_ids),
            'build_seconds':build_s,
            'exact_top1_accuracy':exact_acc,
            'numeric_top1_accuracy':numeric_acc,
            'irrelevant_false_positives':irrelevant,
            'weak_overlap_false_positive':weak_overlap_false_positive,
            'weak_exact_ok':weak_exact_ok,
            'noisy_semantic_shadow_retrieval':noisy_ok,
            'duplicate_recurrence_ok':duplicate_ok,
            'contradiction_isolated':contradiction_isolated,
            'persistence_reopen_ok':persistence_ok,
            'reopen_ms':reopen_ms,
            'forget_ok':forgot,
            'latency_ms':{
                'mean':statistics.mean(lat) if lat else 0.0,
                'p50':pct(lat,.50),'p95':pct(lat,.95),'p99':pct(lat,.99),'max':max(lat) if lat else 0.0,
            },
            'stats_before_reopen':stats_before,
            'stats_after_forget':final_stats,
        }


def run_live_v14_integration():
    from prompt_session_v14 import PromptSessionV14
    from run_gpu import load_config
    with TemporaryDirectory() as td:
        cfg=load_config();cfg['persistent_memory_path']=str(Path(td)/'model_memory.sqlite3');cfg['default_target_chars']=600
        session=PromptSessionV14(cfg)
        try:
            space,_,_,_=session.generate(
                'Escreva 600 caracteres sobre exploração espacial, tecnologia, descoberta e futuro.',
                target_chars=600,seed=4399)
            energy,_,_,_=session.generate(
                'Escreva 600 caracteres sobre energia solar, baterias, armazenamento e futuro.',
                target_chars=600,seed=4400)
            r1,_,_,_=session.generate('Meu carro é um Civic 2015.',target_chars=600,seed=4401)
            r2,text2,_,reason2=session.generate('Qual carro eu disse que tenho?',target_chars=600,seed=4402)
            r3,text3,_,reason3=session.generate('Qual carro eu disse que tenho?',target_chars=600,seed=4403)
            p2=reason2.get('persistent_memory',{})
            p3=reason3.get('persistent_memory',{})
            return {
                'weak_topic_source_stored':bool(space.get('persistent_memory',{}).get('stored')),
                'weak_topic_retrieved':int(energy.get('persistent_memory',{}).get('retrieved',0)),
                'first_turn_stored':bool(r1.get('persistent_memory',{}).get('stored')),
                'second_turn_retrieved':int(r2.get('persistent_memory',{}).get('retrieved',0)),
                'second_turn_injected_rules':int(p2.get('injected_rules',0)),
                'second_turn_question_stored':bool(r2.get('persistent_memory',{}).get('stored')),
                'second_turn_store_skipped_reason':r2.get('persistent_memory',{}).get('store_skipped_reason'),
                'answer_contains_civic_2015':('civic' in text2.casefold() and '2015' in text2),
                'third_turn_retrieved':int(r3.get('persistent_memory',{}).get('retrieved',0)),
                'third_turn_injected_rules':int(p3.get('injected_rules',0)),
                'third_answer_contains_civic_2015':('civic' in text3.casefold() and '2015' in text3),
                'semantic_verified':bool(r2.get('semantic_verified')),
                'slot_errors':int(r2.get('slot_errors',0)),
                'trace_errors':int(r2.get('trace_errors',0)),
            }
        finally:
            session.close()


def main():
    t0=time.perf_counter()
    store=run_store_battery()
    live=run_live_v14_integration()
    gate=(
        store['exact_top1_accuracy']>=0.99 and store['numeric_top1_accuracy']>=0.99
        and store['irrelevant_false_positives']==0 and not store['weak_overlap_false_positive'] and store['weak_exact_ok']
        and store['noisy_semantic_shadow_retrieval']
        and store['duplicate_recurrence_ok'] and store['contradiction_isolated']
        and store['persistence_reopen_ok'] and store['forget_ok']
        and store['latency_ms']['p95']<10.0
        and live['weak_topic_source_stored'] and live['weak_topic_retrieved']==0
        and live['first_turn_stored'] and live['second_turn_retrieved']>=1
        and live['second_turn_injected_rules']>=1 and not live['second_turn_question_stored']
        and live['second_turn_store_skipped_reason']=='interrogative' and live['answer_contains_civic_2015']
        and live['third_turn_retrieved']>=1 and live['third_turn_injected_rules']>=1 and live['third_answer_contains_civic_2015']
        and live['semantic_verified'] and live['slot_errors']==0 and live['trace_errors']==0
    )
    out={
        'format':'Persistent-Dimensional-Memory-V14-Battery',
        'source_design':'AI-Memory dimensional external memory; V14 integration is incremental SQLite + explicit postings/edges',
        'non_neural':True,
        'store':store,
        'live_v14':live,
        'seconds':time.perf_counter()-t0,
        'gate':'OK' if gate else 'FAIL',
    }
    OUT.parent.mkdir(parents=True,exist_ok=True);OUT.write_text(json.dumps(out,ensure_ascii=False,indent=2)+'\n',encoding='utf8')
    print(json.dumps(out,ensure_ascii=False,indent=2))
    raise SystemExit(0 if gate else 1)


if __name__=='__main__':
    main()
