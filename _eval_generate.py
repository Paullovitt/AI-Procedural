from pathlib import Path
import json, re, collections, torch
from procedural_runtime_gpu import build_renderer_v9_gpu_batched, SemanticTraceVerifier
from procedural_runtime_v3 import ProtectedSlotVerifier
ROOT=Path('.')
sc,gr,ind,r=build_renderer_v9_gpu_batched(ROOT,seed=20260821,use_hot=False,proposal_weight=.24,device=0,memory_limit_mb=4608)
# Cenário industrial coerente e não-contraditório: fatos fornecidos ao renderer.
ents=[f'e{i:03d}' for i in range(12)]
props=[f'a{i:02d}' for i in range(8)]
vals=[f'v{i:02d}' for i in range(16)]
rels=[f'r{i:02d}' for i in range(5)]
facts=[]
# propriedades funcionais, uma por (entidade, propriedade)
for i,e in enumerate(ents):
    facts += [
      ('prop',e,'a00',f'v{(i%4):02d}'),
      ('prop',e,'a01',f'v{(4+(i+1)%4):02d}'),
      ('prop',e,'a02',f'v{(8+(i*2)%4):02d}'),
      ('prop',e,'a03',f'v{(12+(i+2)%4):02d}'),
      ('prop',e,'a04',f'v{((i+2)%4):02d}'),
      ('prop',e,'a05',f'v{(4+(i+3)%4):02d}'),
    ]
# rede de dependências/fluxos com cadeia e atalhos
for i in range(11): facts.append(('rel',ents[i], 'r00', ents[i+1]))
for i in range(0,10,2): facts.append(('rel',ents[i], 'r01', ents[i+2]))
for i in range(1,11,2): facts.append(('rel',ents[i], 'r02', ents[(i+3)%12]))
for i in range(0,12,3): facts.append(('rel',ents[i], 'r03', ents[(i+4)%12]))
for i in range(2,12,3): facts.append(('rel',ents[i], 'r04', ents[(i+5)%12]))
# mais duas propriedades por entidade para texto mais rico
for i,e in enumerate(ents):
    facts.append(('prop',e,'a06',f'v{(8+(i+1)%4):02d}'))
    facts.append(('prop',e,'a07',f'v{(12+(i+1)%4):02d}'))

out=r.render(facts)
vf=ProtectedSlotVerifier(); tv=SemanticTraceVerifier()
# mapeamento SOMENTE de apresentação; não altera a escolha do modelo.
EM={
'e000':'Centro Alfa','e001':'Fábrica Norte','e002':'Armazém Leste','e003':'Usina Solar','e004':'Estação Delta','e005':'Fábrica Sul',
'e006':'Hub Logístico','e007':'Centro Beta','e008':'Terminal Oeste','e009':'Usina Eólica','e010':'Armazém Central','e011':'Centro Gama',
'a00':'capacidade','a01':'estabilidade','a02':'prioridade','a03':'risco','a04':'eficiência','a05':'disponibilidade','a06':'demanda','a07':'confiabilidade',
'v00':'baixa','v01':'moderada','v02':'alta','v03':'muito alta','v04':'instável','v05':'aceitável','v06':'boa','v07':'excelente',
'v08':'secundária','v09':'normal','v10':'elevada','v11':'crítica','v12':'mínimo','v13':'controlado','v14':'significativo','v15':'severo',
'r00':'dependência','r01':'fornecimento','r02':'coordenação','r03':'redundância','r04':'contingência'
}
pat=re.compile(r'\b(?:e\d+|a\d+|v\d+|r\d+)\b',re.I)
def readable(s): return pat.sub(lambda m: EM.get(m.group(0).lower(),m.group(0)),s)
read=readable(out['text'])
Path('rigorous_results_v2/evaluation_generated_raw.txt').write_text(out['text'],encoding='utf8')
Path('rigorous_results_v2/evaluation_generated_readable.txt').write_text(read,encoding='utf8')
print(json.dumps({
 'facts':len(facts),'sentences':len(out['sentences']),'paragraphs':len(out['paragraphs']),
 'chars_raw':len(out['text']),'induced_selected':out['induced_selected'],
 'induced_rate':out['induced_selected']/len(out['sentences']),
 'semantic_exact':collections.Counter(out['represented'])==collections.Counter(facts),
 'slot_errors':len(vf.inspect_render(out)),'trace_errors':len(tv.inspect_render(out)),
 'gpu':sc.gpu_status()
},ensure_ascii=False,indent=2))
print('---TEXT---')
print(read)
