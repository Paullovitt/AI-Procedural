AI-Procedural V9 - GPU/VRAM

Pasta principal:
  C:\Users\programacao.cnc01\Downloads\TESTE\AI-Procedural-V9

Execucao:
  RUN_GPU.bat
ou:
  python run_gpu.py --smoke

Entrada propria em JSON:
  python run_gpu.py --facts meus_fatos.json --output saida.txt

Backend promovido:
  PyTorch CUDA tensor runtime (sem rede neural, sem gradiente, sem backprop).

Hardware auditado:
  NVIDIA GeForce GTX 1660 SUPER 6 GB
  PyTorch 2.10.0+cu128
  CUDA runtime 12.8

O que fica na GPU/VRAM:
  - indices ordenados de tokens
  - p2
  - p3
  - p4
  - busca de contagens
  - log-probabilidade/backoff
  - reducoes de score em lote

O que continua na CPU por projeto:
  - tokenizacao
  - memoria simbolica
  - planejamento de fatos/paragrafos
  - gramática induzida baseada em dicionarios pequenos
  - verificador semantico

Motivo: essas etapas sao pequenas/irregulares e mover tudo para CUDA aumenta a sobrecarga sem ganho.

VRAM:
  gpu_config.json limita o processo a ~4608 MB para deixar margem ao Windows/driver.
  A V9 atual usa poucos MB porque o modelo n-gram comprimido e pequeno; usar VRAM artificialmente nao melhora desempenho.

Benchmark desta maquina:
  CPU: ~4376 fatos/s
  GPU CUDA batched: ~2914 fatos/s
  paridade numerica CPU/GPU: erro maximo 0.0
  erros semanticos: 0
  erros de slots: 0

Conclusao: GPU esta habilitada e correta, mas nesta V9 a CPU ainda e mais rapida para o workload pequeno/irregular. Para aproveitar mais a GPU, o proximo passo arquitetural e acumular lotes maiores de documentos/candidatos ou mover novas operacoes massivas para o backend tensorial.
