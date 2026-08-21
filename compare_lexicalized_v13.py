from pathlib import Path
from collections import Counter
import re,json,statistics
from procedural_runtime_v12 import build_renderer_v12_gpu
from procedural_runtime_v13 import build_renderer_v13_gpu
from procedural_runtime_gpu import SemanticTraceVerifier
from procedural_runtime_v5 import ProtectedSlotVerifier
ROOT=Path('.')
ents=[f'e{i:03d}' for i in range(12)];facts=[]
for i,e in enumerate(ents):
 facts += [('prop',e,'a00',f'v{(i%4):02d}'),('prop',e,'a01',f'v{(4+(i+1)%4):02d}'),('prop',e,'a02',f'v{(8+(i*2)%4):02d}'),('prop',e,'a03',f'v{(12+(i+2)%4):02d}'),('prop',e,'a04',f'v{((i+2)%4):02d}'),('prop',e,'a05',f'v{(4+(i+3)%4):02d}')]
for i in range(11):facts.append(('rel',ents[i],'r00',ents[i+1]))
for i in range(0,10,2):facts.append(('rel',ents[i],'r01',ents[i+2]))
for i in range(1,11,2):facts.append(('rel',ents[i],'r02',ents[(i+3)%12]))
for i in range(0,12,3):facts.append(('rel',ents[i],'r03',ents[(i+4)%12]))
for i in range(2,12,3):facts.append(('rel',ents[i],'r04',ents[(i+5)%12]))
for i,e in enumerate(ents):facts += [('prop',e,'a06',f'v{(8+(i+1)%4):02d}'),('prop',e,'a07',f'v{(12+(i+1)%4):02d}')]
EM={'e000':'Centro Alfa','e001':'Fábrica Norte','e002':'Armazém Leste','e003':'Usina Solar','e004':'Estação Delta','e005':'Fábrica Sul','e006':'Hub Logístico','e007':'Centro Beta','e008':'Terminal Oeste','e009':'Usina Eólica','e010':'Armazém Central','e011':'Centro Gama','a00':'capacidade','a01':'estabilidade','a02':'prioridade','a03':'risco','a04':'eficiência','a05':'disponibilidade','a06':'demanda','a07':'confiabilidade','v00':'baixa','v01':'moderada','v02':'alta','v03':'muito alta','v04':'instável','v05':'aceitável','v06':'boa','v07':'excelente','v08':'secundária','v09':'normal','v10':'elevada','v11':'crítica','v12':'mínimo','v13':'controlado','v14':'significativo','v15':'severo','r00':'dependência','r01':'fornecimento','r02':'coordenação','r03':'redundância','r04':'contingência'}
lex={k:v for k,v in EM.items() if k[0] in 'arv'}
pat=re.compile(r'\b(?:e\d+|a\d+|v\d+|r\d+)\b',re.I)
def readable(s):return pat.sub(lambda m:EM.get(m.group(0).lower(),m.group(0)),s)
def score(sc,out):
 ss=[readable(x) for x in out['sentences']];langs=[];sups=[]
 for i in range(0,len(ss),1024):
  l,s=sc.batch_language_support([sc.tokenize(x) for x in ss[i:i+1024]],max_order=5,slot_aware=False);langs+=list(map(float,l));sups+=list(map(float,s))
 return statistics.mean(langs),statistics.mean(sups)
sc12,_,_,r12=build_renderer_v12_gpu(ROOT,seed=20260821,proposal_weight=.24,position_weight=7.0,diversity_weight=2.6,focus_diversity_weight=1.17,repetition_weight=1.1,device=0,memory_limit_mb=4608)
o12=r12.render(facts)
sc13,_,_,r13=build_renderer_v13_gpu(ROOT,lex,seed=20260821,device=0,memory_limit_mb=4608)
o13=r13.render(facts)
vf=ProtectedSlotVerifier();tv=SemanticTraceVerifier();m12=score(sc12,o12);m13=score(sc13,o13)
res={'facts':len(facts),'v12':{'sentences':len(o12['sentences']),'semantic_exact':Counter(o12['represented'])==Counter(facts),'slot_errors':len(vf.inspect_render(o12)),'trace_errors':len(tv.inspect_render(o12)),'lexicalized_p2_p5_score':m12[0],'lexicalized_support':m12[1]},'v13':{'sentences':len(o13['sentences']),'semantic_exact':Counter(o13['represented'])==Counter(facts),'slot_errors':len(vf.inspect_render(o13)),'trace_errors':len(tv.inspect_render(o13)),'lexicalized_p2_p5_score':m13[0],'lexicalized_support':m13[1]}}
Path('rigorous_results_v12/naturality_v12.txt').write_text(readable(o12['text']),encoding='utf8');Path('rigorous_results_v12/naturality_v13.txt').write_text(readable(o13['text']),encoding='utf8');Path('rigorous_results_v12/naturality_comparison.json').write_text(json.dumps(res,indent=2),encoding='utf8');print(json.dumps(res,ensure_ascii=False,indent=2));print('---V13---');print(readable(o13['text']))
