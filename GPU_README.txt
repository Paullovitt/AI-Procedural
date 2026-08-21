AI-Procedural V14 + Learned RuleVM V6 + Evidence Argument Planner - GPU/VRAM

Pasta principal:
  C:\Users\programacao.cnc01\Downloads\TESTE\AI-Procedural-V9

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
  NVIDIA GeForce GTX 1660 SUPER 6 GB
  PyTorch 2.10.0+cu128
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

O Argument Planner é propositalmente pequeno: ele não faz aprendizagem neural nem inferência de domínio.
Ele reordena/filtra regras já aprendidas usando confiança, suporte, score, profundidade, cobertura e
alinhamento com o contexto global do prompt. Na bateria final, o maior tempo observado do planner foi
~0,861 ms; o RuleVM permaneceu abaixo de ~0,036 ms.

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

Resultados finais da bateria multi-tema, alvo 1000 caracteres:
- exploração espacial: 957, OK
- energia solar: 1019, OK
- agricultura sustentável: 949, OK
- música clássica: 953, OK
- segurança digital: 223, evidence_limited=true
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

Performance final da bateria:
- carga única GPU/modelo: ~3,235 s
- RuleVM máximo: ~0,036 ms
- Argument Planner máximo: ~0,861 ms
- prompts posteriores: média ~17,4 ms de raciocínio, máximo ~23,3 ms
- sessão persistente evita recarregar ~3-4 s por prompt

Qualidade adicionada sem relaxar evidência:
- conceitos compostos de 2 a 5 palavras usam evidência da expressão inteira
- contextos p3-p5 preservam a expressão observada completa, evitando tokens soltos artificiais
- desambiguação usa o contexto global do prompt para ordenar candidatos já suportados pelo corpus
- associação isolada não corroborada pode ser removida
- se faltarem fatos fortes, evidence_limited é preferido a preencher com ruído
- o refinador pode buscar candidatos fortes adicionais sem baixar thresholds

Regressão RuleVM V5 de referência:
- transition accuracy 1.0
- closure exact 1.0
- proof validity 1.0
- generalidade 6/6 mundos: 1.0
- drift: 3/3 mudanças detectadas, 0 falsas revisões

Testes:
  python -m unittest -v test_v14_prompt.py test_rulevm_v6_prompt.py test_argument_planner_v14.py
  python benchmark_prompt_rulevm_v6.py
  python architecture_guard.py
  python project_audit.py

Resultado de generalidade/argumentação salvo em:
  rigorous_results_v12/prompt_rulevm_v6_generality.json

Toda implementação nova promovida permanece na V14 e deve atualizar os documentos do projeto.
O architecture_guard.py e project_audit.py devem passar antes de qualquer promoção.
