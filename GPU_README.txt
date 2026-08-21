AI-Procedural V14 - GPU/VRAM

Pasta principal:
  C:\Users\programacao.cnc01\Downloads\TESTE\AI-Procedural-V9

Runtime padrão:
  Renderer V14
  backend: cuda-batched-v14

Execução interativa por prompt:
  RUN_GPU.bat

Linha de comando:
  python run_gpu.py --prompt "Escreva um texto de 2000 caracteres sobre exploração espacial."

Prompt em arquivo:
  python run_gpu.py --prompt-file prompt_example_v14.txt --target-chars 2000 --output saida.txt

Entrada própria em JSON:
  python run_gpu.py --facts meus_fatos.json --output saida.txt

Teste sintético:
  python run_gpu.py --smoke

Saída legível:
  IDs e000/a000/v000/r000 ficam internos para verificação semântica.
  A tela mostra palavras/frases por padrão.
  --raw-slots habilita a representação interna apenas para depuração.

Backend promovido:
  PyTorch CUDA tensor runtime (sem rede neural, sem gradiente, sem backprop).

Hardware auditado:
  NVIDIA GeForce GTX 1660 SUPER 6 GB
  PyTorch 2.10.0+cu128
  CUDA runtime 12.8

O que fica na GPU/VRAM:
  - índices ordenados de tokens
  - p2
  - p3
  - p4
  - p5
  - busca de contagens
  - log-probabilidade/backoff
  - reduções de score em lote
  - pontuação batched das realizações do V14

O que continua na CPU por projeto:
  - tokenização
  - memória simbólica
  - interpretação genérica do prompt
  - planejamento de fatos/parágrafos
  - gramática induzida baseada em dicionários pequenos
  - verificadores semânticos e traces

Motivo: essas etapas são pequenas/irregulares e movê-las artificialmente para CUDA aumentaria sobrecarga sem ganho. O trabalho tensorial pesado continua na GPU.

VRAM:
  gpu_config.json limita o processo a 4608 MB para deixar margem ao Windows/driver.
  O runtime usa somente a VRAM necessária; ocupar memória artificialmente não melhora desempenho.

Configuração V14 promovida:
  proposal_weight: 0.24
  position_weight: 7.0
  diversity_weight: 2.6
  focus_diversity_weight: 1.17
  repetition_weight: 1.1

Validação atual do modo prompt:
  alvo: 2000 caracteres
  saída de teste: 1985 caracteres
  semantic_verified: true
  slot_errors: 0
  trace_errors: 0
  raw_slot_ids_exposed: false
  backend: cuda-batched-v14
  tabelas em VRAM: tokens, p2, p3, p4, p5

Testes:
  python -m unittest -v test_v14_prompt.py
  python architecture_guard.py

O architecture_guard.py deve passar antes de qualquer promoção de runtime.
