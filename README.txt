AI-Procedural / Bagaço Renderer V14

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

O modelo deve aprender regras por conta própria. O código pode conter if/else apenas para controle genérico de execução, integridade, GPU, matching, adaptação de interface e promoção/rejeição de hipóteses; não pode conter a lei específica que o modelo deveria descobrir.

Código futuro deve ser limpo, simples, otimizado, vetorizado/batched quando apropriado, medido por benchmark e promovido somente após shadow + heldout + adversarial + stress.

Regras completas: PROJECT_RULES.md
Estado técnico consolidado: PROJECT_STATE_2026-08-21.md
Guard obrigatório: architecture_guard.py

RUNTIME PADRÃO
==============
O launcher padrão agora usa o Renderer V14 em CUDA/VRAM.

Execução interativa:
  RUN_GPU.bat
ou, a partir da pasta TESTE:
  RUN_AI_GPU.bat

O launcher abre:
  Prompt>

Exemplo por linha de comando:
  python run_gpu.py --prompt "Escreva um texto de 2000 caracteres sobre exploração espacial, tecnologia e futuro."

Prompt em arquivo:
  python run_gpu.py --prompt-file prompt_example_v14.txt --target-chars 2000 --output saida.txt

Fatos simbólicos próprios:
  python run_gpu.py --facts meus_fatos.json --output saida.txt

Teste sintético:
  python run_gpu.py --smoke

Saída padrão:
- os IDs semânticos e000/a000/v000/r000 continuam existindo internamente para auditoria e verificação;
- a tela e os arquivos de saída mostram palavras/frases legíveis por padrão;
- use --raw-slots somente para depuração, quando quiser ver a representação interna.

PROMPTS NO V14
==============
prompt_runtime_v14.py fornece uma camada não neural de prompt:
- interpreta tamanho solicitado e tema;
- extrai material lexical do prompt;
- constrói um plano simbólico auditável;
- entrega esse plano ao V14;
- V14 faz seleção de superfície com p2-p5 em CUDA;
- a saída é lexicalizada somente depois da verificação semântica interna.

O modo de prompt não transforma o sistema em rede neural nem injeta conhecimento oculto. Ele adapta a interface textual ao mecanismo simbólico existente.

Conteúdo principal:
- model/full: modelo amplo Bagaço
- model/quality: camada de qualidade
- bagaco_model_gpu/bagaco_hot_runtime.json.xz: runtime compacto
- LEARNED_CONSTRUCTIONS_V1.json.xz: gramática induzida
- LEARNED_REALIZATION_PROPOSALS_V1.json: realizações promovidas
- procedural_runtime_v3.py, v4.py, v5.py: runtimes históricos
- procedural_runtime_gpu.py: backend CUDA discreto
- procedural_runtime_v12.py: planejamento/discurso avançado
- procedural_runtime_v13.py: lexicalização usada na pontuação de superfície
- procedural_runtime_v14.py: runtime promovido com parágrafos multi-foco e suporte a léxico
- prompt_runtime_v14.py: interface padrão de prompts e apresentação legível
- autonomous_rule_vm_v*.py: indução e execução genérica de regras como dados
- architecture_guard.py: bloqueio arquitetural automático
- test_v14_prompt.py: testes de prompt, lexicalização, semântica e CUDA
- validate_model.py: validação dos artefatos

Princípio central:
"Melhorar o mecanismo que aprende; nunca ensinar a resposta que ele deve aprender."
