from __future__ import annotations

from collections import Counter, defaultdict
from pathlib import Path
import argparse
import json
import math
import random
import re
import statistics
import time
import unicodedata

from prompt_session_v14 import PromptSessionV14
from robust_semantic_intake_v14 import semantic_shadow

ROOT = Path(__file__).resolve().parent
OUT_DIR = ROOT / 'rigorous_results_v12'
OUT_PATH = OUT_DIR / 'robust_semantic_v14.json'
FAILURE_ARCHIVE = ROOT / 'robust_semantic_failure_archive_v14.jsonl'
REGRESSIONS = ROOT / 'robust_semantic_regressions_v14.json'

COVERAGE = [
    'simple spelling errors','severe spelling errors','missing letters','duplicated letters','transposed letters',
    'replaced letters','phonetic-like forms','abbreviations/slang exposure','missing accents','missing punctuation',
    'excess punctuation','incomplete/malformed clauses','word reordering','glued words','incorrectly split words',
    'repetition','contradictions preserved as evidence','irrelevant noise','long low-signal text',
    'noise between related entities','number formats','dates','times','values/units','misspelled names',
    'negation','double negation','ambiguity without unsafe rewrite','indirect references/pronouns preserved',
    'abrupt topic shifts','mixed languages','unusual Unicode','emojis','mojibake','HTML','Markdown','JSON fragments',
    'duplicated spans','partial token truncation','nonwords','combined corruptions','minimal meaning-changing edits',
]

EXTRA_CASES = []

WORD_RE = re.compile(r'[^\W\d_]+', re.UNICODE)


def atoms(result):
    out=set()
    for x in result.semantic_atoms:
        raw=x[1:] if x.startswith('#') else x
        sh=semantic_shadow(raw)
        if sh: out.add(sh)
    return out


def strip_accents(text, rng):
    return ''.join(c for c in unicodedata.normalize('NFKD',text) if not unicodedata.combining(c))


def drop_punctuation(text, rng):
    return re.sub(r'[^\w\sÀ-ÖØ-öø-ÿ]', ' ', text)


def excess_punctuation(text, rng):
    return re.sub(r'\s+', lambda m: rng.choice(['!!! ','... ',' ?? ',' ;;; ']), text)


def duplicate_words(text, rng):
    ws=text.split()
    if not ws: return text
    i=rng.randrange(len(ws)); ws[i:i+1]=[ws[i]]*rng.randint(2,4); return ' '.join(ws)


def insert_noise(text, rng):
    ws=text.split(); i=rng.randrange(len(ws)+1); noise=rng.choice(['bla bla tipo assim','xxxxx qqq','[ruido] nada haver','hmm enfim sei la'])
    ws.insert(i,noise); return ' '.join(ws)


def html_markdown(text, rng):
    return f'<div>**{text}**</div>'


def json_mix(text, rng):
    return '{"conteudo": ' + json.dumps(text,ensure_ascii=False) + ', "meta":"ruido"}'


def emoji_unicode(text, rng):
    ws=text.split(); sep=rng.choice(['🔥','🧪','🙂','§','•','\u200b']); return sep.join(ws)


def mixed_case(text, rng):
    return ''.join(c.upper() if rng.random()<.35 else c.lower() if rng.random()<.5 else c for c in text)


def mutate_word(text, rng, mode):
    matches=[m for m in WORD_RE.finditer(text) if len(m.group(0))>=5]
    if not matches: return text
    m=rng.choice(matches); w=m.group(0); i=rng.randrange(1,len(w)-1)
    if mode=='delete': nw=w[:i]+w[i+1:]
    elif mode=='duplicate': nw=w[:i]+w[i]+w[i:]
    elif mode=='transpose' and i+1<len(w): nw=w[:i]+w[i+1]+w[i]+w[i+2:]
    else:
        repl=rng.choice('aeiourstlmn'); nw=w[:i]+repl+w[i+1:]
    return text[:m.start()]+nw+text[m.end():]


def delete_char(text,rng): return mutate_word(text,rng,'delete')
def duplicate_char(text,rng): return mutate_word(text,rng,'duplicate')
def transpose_char(text,rng): return mutate_word(text,rng,'transpose')
def replace_char(text,rng): return mutate_word(text,rng,'replace')


def glue_words(text,rng):
    ws=text.split()
    if len(ws)<2:return text
    i=rng.randrange(len(ws)-1); ws[i:i+2]=[ws[i]+ws[i+1]]; return ' '.join(ws)


def split_word(text,rng):
    matches=[m for m in WORD_RE.finditer(text) if len(m.group(0))>=8]
    if not matches:return text
    m=rng.choice(matches);w=m.group(0);i=rng.randrange(3,len(w)-2);nw=w[:i]+' '+w[i:]
    return text[:m.start()]+nw+text[m.end():]


def partial_truncate(text,rng):
    matches=[m for m in WORD_RE.finditer(text) if len(m.group(0))>=7]
    if not matches:return text
    m=rng.choice(matches);w=m.group(0);cut=rng.choice([1,1,2]);nw=w[:-cut]
    return text[:m.start()]+nw+text[m.end():]


def reorder_local(text,rng):
    ws=text.split()
    if len(ws)<4:return text
    i=rng.randrange(len(ws)-2);ws[i],ws[i+1]=ws[i+1],ws[i];return ' '.join(ws)


def topic_shift(text,rng):
    return text + ' mudando de assunto futebol receita viagem bla bla ' + rng.choice(['agora enfim','qualquer coisa'])


def mixed_language_noise(text,rng):
    return text + ' anyway random stuff pero entonces bla'


def mojibake(text,rng):
    try:return text.encode('utf8').decode('latin1')
    except UnicodeError:return text


MUTATORS = {
    'strip_accents':strip_accents, 'drop_punctuation':drop_punctuation, 'excess_punctuation':excess_punctuation,
    'duplicate_words':duplicate_words, 'insert_noise':insert_noise, 'html_markdown':html_markdown,
    'json_mix':json_mix, 'emoji_unicode':emoji_unicode, 'mixed_case':mixed_case,
    'delete_char':delete_char, 'duplicate_char':duplicate_char, 'transpose_char':transpose_char,
    'replace_char':replace_char, 'glue_words':glue_words, 'split_word':split_word,
    'partial_truncate':partial_truncate, 'reorder_local':reorder_local, 'topic_shift':topic_shift,
    'mixed_language_noise':mixed_language_noise, 'mojibake':mojibake,
}


def bridge_tokens(result):
    return [semantic_shadow(x) for e in result.edges for x in e.bridge]


def evaluate(case, result):
    got=atoms(result); required={semantic_shadow(x) for x in case.get('required',[])}; forbidden={semantic_shadow(x) for x in case.get('forbidden',[])}
    missing=sorted(required-got); included_noise=sorted(forbidden&got)
    numeric_expected={str(x) for x in case.get('numbers',[])}
    numeric_got={str(x['raw']) for x in result.numeric_anchors}
    numeric_missing=sorted(numeric_expected-numeric_got)
    bridges=bridge_tokens(result); bridge_missing=[]; bridge_forbidden=[]
    token=case.get('bridge_token')
    if token and semantic_shadow(token) not in bridges: bridge_missing.append(token)
    for x in case.get('forbidden_bridge',[]):
        if semantic_shadow(x) in bridges: bridge_forbidden.append(x)
    edge=case.get('edge'); edge_missing=False
    if edge:
        edge_missing=not any(e.kind==edge['kind'] and semantic_shadow(e.source)==semantic_shadow(edge['source']) and semantic_shadow(e.target)==semantic_shadow(edge['target']) for e in result.edges)
    recall=1.0-len(missing)/max(1,len(required))
    numeric_recall=1.0-len(numeric_missing)/max(1,len(numeric_expected)) if numeric_expected else 1.0
    meaning_errors=len(bridge_missing)+len(bridge_forbidden)+(1 if edge_missing else 0)
    score=recall - .20*len(included_noise) - .25*(1-numeric_recall) - .20*meaning_errors
    categories=[]
    if missing: categories.append('important_information_loss')
    if included_noise: categories.append('noise_inclusion')
    if numeric_missing: categories.append('numeric_date_quantity_error')
    if meaning_errors: categories.append('relation_or_meaning_error')
    return {'score':score,'recall':recall,'numeric_recall':numeric_recall,'missing':missing,'included_noise':included_noise,
            'numeric_missing':numeric_missing,'bridge_missing':bridge_missing,'bridge_forbidden':bridge_forbidden,
            'edge_missing':edge_missing,'failure_classes':categories,'passed':not categories}


def load_cases():
    fixed=json.loads(REGRESSIONS.read_text(encoding='utf8'))['cases']
    return fixed+EXTRA_CASES


def run_case(intake, case, text, variant, severity):
    t0=time.perf_counter();result=intake.extract(text);ms=(time.perf_counter()-t0)*1000.0;ev=evaluate(case,result)
    return {'case':case['id'],'category':case.get('category'),'variant':variant,'severity':severity,'text':text,
            'latency_ms':ms,'signal_tokens':result.stats['signal_tokens'],'noise_tokens':result.stats['noise_tokens'],
            'fuzzy_queries':result.stats['fuzzy_queries'],**ev}


def adversarial_search(intake, case, rng, rounds=3, width=5):
    current=case['raw'];best=run_case(intake,case,current,'adversarial-base',0)
    history=[]
    names=list(MUTATORS)
    for depth in range(1,rounds+1):
        candidates=[]
        for _ in range(width):
            name=rng.choice(names);candidate=MUTATORS[name](current,rng);row=run_case(intake,case,candidate,'adversarial:'+name,depth);candidates.append(row)
        candidate=min(candidates,key=lambda x:(x['score'], -len(x['failure_classes']), -len(x['missing']), -len(x['numeric_missing']), x['variant'], x['text']))
        history.append(candidate)
        if candidate['score']<=best['score']:
            best=candidate;current=candidate['text']
    return best,history


def archive_failures(rows):
    old={}
    if FAILURE_ARCHIVE.exists():
        for line in FAILURE_ARCHIVE.read_text(encoding='utf8',errors='replace').splitlines():
            if not line.strip():continue
            try:
                obj=json.loads(line);old[(obj.get('case'),obj.get('text'))]=obj
            except json.JSONDecodeError:pass
    for row in rows:
        if row['passed']:continue
        minimal={k:row[k] for k in ('case','category','variant','severity','text','missing','included_noise','numeric_missing','bridge_missing','bridge_forbidden','edge_missing','failure_classes')}
        old[(minimal['case'],minimal['text'])]=minimal
    FAILURE_ARCHIVE.write_text('\n'.join(json.dumps(x,ensure_ascii=False) for x in old.values())+('\n' if old else ''),encoding='utf8')
    return len(old)


def main():
    ap=argparse.ArgumentParser(description='Bateria adversarial V14 para extração semântica robusta.')
    ap.add_argument('--seed',type=int,default=14014)
    ap.add_argument('--quick',action='store_true')
    ap.add_argument('--no-archive',action='store_true')
    args=ap.parse_args();rng=random.Random(args.seed)
    session=PromptSessionV14();intake=session.adapter._intake_for(session.scorer);cases=load_cases();rows=[]
    # Fixed reference cases.
    for case in cases: rows.append(run_case(intake,case,case['raw'],'reference',0))
    # Systematic corruption matrix. One transform at a time keeps the expected meaning observable.
    mutator_items=list(MUTATORS.items())
    if args.quick: mutator_items=mutator_items[:10]
    for case in cases:
        for name,fn in mutator_items:
            text=fn(case['raw'],rng);rows.append(run_case(intake,case,text,name,1))
    # Compound corruptions.
    compound_rounds=1 if args.quick else 3
    for case in cases:
        for ci in range(compound_rounds):
            text=case['raw'];names=rng.sample(list(MUTATORS),k=3)
            for name in names:text=MUTATORS[name](text,rng)
            rows.append(run_case(intake,case,text,'combined:'+','.join(names),3))
    # Automatic failure-directed adversarial search.
    adv=[]
    for case in cases:
        worst,history=adversarial_search(intake,case,rng,rounds=(2 if args.quick else 4),width=(3 if args.quick else 6));adv.append(worst);rows.extend(history)

    latency=[x['latency_ms'] for x in rows]; failures=[x for x in rows if not x['passed']]
    by_class=Counter(c for x in failures for c in x['failure_classes'])
    by_variant=defaultdict(list)
    for x in rows:by_variant[x['variant'].split(':',1)[0]].append(x)
    summary={
        'format':'Robust-Semantic-Battery-V14','seed':args.seed,'cases':len(cases),'evaluations':len(rows),
        'coverage':COVERAGE,'fixed_reference_passed':sum(x['passed'] for x in rows if x['variant']=='reference'),
        'fixed_reference_total':sum(1 for x in rows if x['variant']=='reference'),
        'mean_information_recall':statistics.mean(x['recall'] for x in rows),
        'mean_information_loss':1-statistics.mean(x['recall'] for x in rows),
        'mean_numeric_recall':statistics.mean(x['numeric_recall'] for x in rows),
        'failure_count':len(failures),'failure_classes':dict(by_class),
        'latency_ms':{'p50':statistics.median(latency),'p95':sorted(latency)[max(0,math.ceil(.95*len(latency))-1)],'max':max(latency),'mean':statistics.mean(latency)},
        'intake_status':intake.status(),'model_load_seconds':session.load_seconds,
        'variant_summary':{k:{'n':len(v),'mean_recall':statistics.mean(x['recall'] for x in v),'failures':sum(not x['passed'] for x in v)} for k,v in by_variant.items()},
        'adversarial_worst':adv,
    }
    if not args.no_archive:summary['failure_archive_cases']=archive_failures(failures)
    OUT_DIR.mkdir(exist_ok=True);OUT_PATH.write_text(json.dumps({'summary':summary,'rows':rows},ensure_ascii=False,indent=2),encoding='utf8')
    print(json.dumps(summary,ensure_ascii=False,indent=2))
    # Gates protect known regressions and performance; adversarial failures are recorded instead of hidden.
    gates=[summary['fixed_reference_passed']==summary['fixed_reference_total'],summary['mean_information_recall']>=.80,summary['mean_numeric_recall']>=.90,summary['latency_ms']['p95']<25.0]
    if not all(gates):raise SystemExit('ROBUST SEMANTIC V14 BATTERY GATE: FAIL')
    print('ROBUST SEMANTIC V14 BATTERY GATE: OK')


if __name__=='__main__':main()
