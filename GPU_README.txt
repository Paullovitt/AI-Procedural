AI-Procedural V14 + Learned RuleVM V6 - GPU/VRAM

Pasta principal:
  C:\Users\programacao.cnc01\Downloads\TESTE\AI-Procedural-V9

Runtime padrão:
  Renderer V14 + sessão persistente de prompt + RuleVM V6
  backend de superfície: cuda-batched-v14

Execução recomendada:
  RUN_GPU.bat

O launcher mantém modelo e índices residentes e recebe vários prompts até /sair.

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
  - montagem do plano simbólico
  - planejamento estrutural/parágrafos
  - verificadores semânticos/traces
  - cache de contexto de expressões compostas

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

Resultados recentes:
- tabelas CUDA: tokens, p2, p3, p4, p5
- RuleVM V6: dezenas de microssegundos por execução no benchmark multi-tema
- aprendizado de regras V6: tipicamente poucos ms; conceitos compostos de 2 a 5 palavras usam evidência exata p2-p5 e contexto sob demanda
- sessão persistente evita recarregar ~3-4 s de modelo por prompt
- 6/6 prompts de generalidade: sem erros semânticos/slot/trace e sem IDs expostos
- prompts esparsos são marcados evidence_limited em vez de liberar regras fracas

Regressão RuleVM V5 após V6:
- aprendizagem completa: ~1,075 s em 150k treino
- transition accuracy 1.0
- closure exact 1.0
- proof validity 1.0
- generalidade 6/6 mundos: 1.0; tempo total de aprendizado ~5,498 s
- drift: 3/3 mudanças detectadas, 0 falsas revisões

Testes:
  python -m unittest -v test_v14_prompt.py test_rulevm_v6_prompt.py
  python benchmark_prompt_rulevm_v6.py
  python architecture_guard.py
  python project_audit.py

Resultado de generalidade salvo em:
  rigorous_results_v12/prompt_rulevm_v6_generality.json

O architecture_guard.py deve passar antes de qualquer promoção.
