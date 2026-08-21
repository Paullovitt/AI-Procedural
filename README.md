AI-Procedural / Bagaço Renderer

REGRA ARQUITETURAL PERMANENTE
=============================
Este projeto é deliberadamente NÃO NEURAL.

É proibido introduzir:
- neurônios ou redes neurais;
- pesos/parâmetros treináveis como memória do modelo;
- embeddings treináveis;
- backpropagation, gradient descent ou autograd de aprendizagem;
- torch.nn, optimizers, Transformer/LSTM/MLP/CNN ou equivalentes;
- regras de domínio codificadas manualmente para resolver benchmarks.

PyTorch/CUDA é permitido SOMENTE como backend de tensores para computação discreta/estatística, sem gradientes e sem módulos neurais.

O conhecimento deve permanecer explícito e auditável: contagens, fatos, grafos, episódios, regras induzidas, truth tables, índices, sketches, bytecode simbólico, hipóteses e evidências.

O modelo deve aprender regras por conta própria. O código pode conter if/else apenas para controle genérico de execução, integridade, GPU, matching e promoção/rejeição de hipóteses; não pode conter a lei específica que o modelo deveria descobrir.

Código futuro deve ser limpo, simples, otimizado, vetorizado/batched quando apropriado, medido por benchmark e promovido somente após shadow + heldout + adversarial + stress.

Regras completas: PROJECT_RULES.md
Estado técnico consolidado: PROJECT_STATE_2026-08-21.md

Conteúdo principal:
- model/full: modelo amplo Bagaço
- model/quality: camada de qualidade
- bagaco_model_gpu/bagaco_hot_runtime.json.xz: runtime compacto
- LEARNED_CONSTRUCTIONS_V1.json.xz: gramática induzida
- LEARNED_REALIZATION_PROPOSALS_V1.json: realizações promovidas
- procedural_runtime_v3.py, v4.py, v5.py: runtimes históricos
- procedural_runtime_gpu.py: backend CUDA discreto
- procedural_runtime_v12.py: planejamento/discurso avançado
- autonomous_rule_vm_v*.py: indução e execução genérica de regras como dados
- validate_model.py: validação dos artefatos

Princípio central:
"Melhorar o mecanismo que aprende; nunca ensinar a resposta que ele deve aprender."
