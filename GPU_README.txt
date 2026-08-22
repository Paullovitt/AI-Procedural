AI-Procedural V14 + RuleVM V6 + Argument Planner + Robust Semantic Intake + Persistent Dimensional Memory - GPU/VRAM

Pasta principal:
  C:\Users\USER\Downloads\CODIGOS\TESTE\AI-Procedural

Runtime padrão:
  Renderer V14 + sessão persistente + RuleVM V6 + Evidence Argument Planner V14
  backend de superfície: cuda-batched-v14

Execução recomendada:
  RUN_GPU.bat

O launcher mantém modelo, tabelas e índices residentes e recebe vários prompts até /sair.

Uma execução isolada:
  python run_gpu.py --prompt "Escreva 2000 caracteres sobre exploração espacial."

Prompt em arquivo:
  python run_gpu.py --prompt-file prompt_example_v14.txt --target-chars 2000 --output saida.txt

Fatos JSON:
  python run_gpu.py --facts meus_fatos.json --output saida.txt

Backend:
  PyTorch CUDA usado como runtime tensorial, sem rede neural, sem gradientes e sem backprop.

Hardware auditado:
  NVIDIA GeForce RTX 3050 6 GB
  PyTorch 2.11.0+cu128
  CUDA runtime 12.8

Na GPU/VRAM:
  - índices ordenados de tokens
  - p2, p3, p4, p5
  - lookups de contagem e backoff
  - reduções de score em lote
  - pontuação das realizações V14
  - PMI/ranking batched do Rule Learner V6

Na CPU, por serem operações pequenas/irregulares:
  - tokenização
  - RuleBank explícito e índice por source
  - dispatch do RuleVM
  - Evidence Argument Planner V14
  - montagem do plano simbólico
  - planejamento estrutural/parágrafos
  - verificadores semânticos/traces
  - cache de contexto de expressões compostas
  - Robust Semantic Intake: tokenização/arestas/fuzzy discreto e auditável
  - Persistent Dimensional Memory V14: SQLite, postings, arestas, provenance e busca episódica

O Argument Planner é propositalmente pequeno: ele não faz aprendizagem neural nem inferência de domínio.
Ele reordena/filtra regras já aprendidas usando confiança, suporte, score, profundidade, cobertura e
alinhamento com o contexto global do prompt. Na bateria final, o maior tempo observado do planner foi
~0,631 ms; o RuleVM permaneceu abaixo de ~0,029 ms.

VRAM:
  gpu_config.json limita o processo a 4608 MB.
  O runtime usa apenas a memória necessária; ocupar VRAM artificialmente não melhora desempenho.

Configuração V14:
  proposal_weight: 0.24
  position_weight: 7.0
  diversity_weight: 2.6
  focus_diversity_weight: 1.17
  repetition_weight: 1.1
  prompt_max_bundle: 4
  argument_planner_enabled: true
  robust_semantic_intake_enabled: true
  robust_semantic_warm_index: true
  persistent_memory_enabled: true
  persistent_memory_path: memory_v14/episodic_v14.sqlite3
  persistent_memory_top_k: 4
  persistent_memory_candidate_limit: 512
  persistent_memory_associative: true
  persistent_memory_auto_store_user: true
  persistent_memory_min_query_term_coverage: 0.30
  persistent_memory_max_associative_document_ratio: 0.20

Resultados finais da bateria multi-tema, alvo 1000 caracteres:
- exploração espacial: 957, OK
- energia solar: 1019, OK
- agricultura sustentável: 949, OK
- música clássica: 953, OK
- segurança digital: 175, evidence_limited=true
- saúde pública: 1032, OK

Gates:
- semantic_verified: 6/6
- slot_errors: 0
- trace_errors: 0
- IDs internos expostos: 0
- cobertura dos conceitos: 100%
- fases argumentativas monotônicas: 6/6
- repetição imediata do mesmo template: 0
- falhas de tamanho não explicadas: 0
- persistent_memory_isolated=true no benchmark RuleVM/Planner

Performance final da bateria:
- carga única GPU/modelo na bateria multi-tema isolada: ~2,927 s
- RuleVM máximo: ~0,029 ms
- Argument Planner máximo: ~0,631 ms
- prompts posteriores: média ~16,98 ms de raciocínio, máximo ~23,0 ms
- sessão persistente evita recarregar ~3-4 s por prompt

Qualidade adicionada sem relaxar evidência:
- conceitos compostos de 2 a 5 palavras usam evidência da expressão inteira
- contextos p3-p5 preservam a expressão observada completa, evitando tokens soltos artificiais
- desambiguação usa o contexto global do prompt para ordenar candidatos já suportados pelo corpus
- associação isolada não corroborada pode ser removida
- se faltarem fatos fortes, evidence_limited é preferido a preencher com ruído
- o refinador pode buscar candidatos fortes adicionais sem baixar thresholds


Robust Semantic Intake V14:
- caminho quente medido: texto limpo ~0,21 ms; quatro typos comuns ~0,25 ms; exemplo longo ruidoso ~0,75 ms
- 225.058 assinaturas rápidas residentes; índice amplo severo somente sob demanda
- bateria extrema final: 448 avaliações, 16/16 referências, recall 90,49%, recall numérico 99,50%
- latência da bateria: p50 ~0,474 ms, p95 ~13,333 ms, máximo ~69,047 ms
- 121 falhas adversariais classificadas; 353 casos distintos preservados e reexecutados pelo failure replay
- replay histórico: 353 casos, 32 resolvidos, 321 ainda falhando; p50 ~2,146 ms, p95 ~25,741 ms
- adversarial determinístico por gravidade semântica; latência é medida, não usada para escolher o pior caso
- sem correção textual intermediária e sem reescrita do dataset

Regressão RuleVM V5 de referência:
- transition accuracy 1.0
- closure exact 1.0
- proof validity 1.0
- generalidade 6/6 mundos: 1.0
- drift: 3/3 mudanças detectadas, 0 falsas revisões

Testes (34/34 automáticos aprovados):
  python -m unittest discover -v
  python robust_semantic_battery_v14.py
  python persistent_memory_battery_v14.py
  python robust_semantic_failure_replay_v14.py
  python benchmark_prompt_rulevm_v6.py
  python architecture_guard.py
  python project_audit.py

Resultado de generalidade/argumentação salvo em:
  rigorous_results_v12/prompt_rulevm_v6_generality.json

Toda implementação nova promovida permanece na V14 e deve atualizar os documentos do projeto.
O architecture_guard.py e project_audit.py devem passar antes de qualquer promoção.
