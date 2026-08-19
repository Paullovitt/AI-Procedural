# AI-Procedural

Estado oficial atual do projeto: duas linhas de modelo preservadas para comparação no mesmo corpus byte-level em português.

1. **V3 Original 32k** — modelo neural procedural recorrente, treinado com backpropagation.
2. **Direct-80M-32k** — modelo estatístico/associativo explícito, construído em uma passada pelo corpus, sem gradiente.

As variantes Dense 80M, V3 512k, V3+Direct e protótipos intermediários não fazem mais parte do estado atual do repositório.

## Dataset

Fonte: `HuggingFaceFW/fineweb-2`, configuração `por_Latn`, tokenização em bytes UTF-8, vocabulário de 256 valores.

Metadados preservados em `data/meta.json`:

- treino: **943.718.400 bytes**, 282.374 documentos;
- validação: **52.428.800 bytes**, 17.603 documentos;
- teste: **52.428.800 bytes**, 17.549 documentos.

Os arquivos `.bin` não são versionados no GitHub. Para recriar o dataset local:

```powershell
pip install -r requirements.txt
python prepare_portuguese_data.py
```

## V3 Original 32k

Arquivos principais:

- `model_v3.py`
- `train_procedural_v3_512_5000.py`
- `benchmark_v3_original_32k.py`
- `runs/procedural_v3_hybrid32k_512_5000/`

Configuração principal:

- 12 camadas;
- `d_model = 744`;
- 12 heads;
- rank procedural 64;
- chunk de 512 bytes/tokens;
- **4.415.040 parâmetros persistentes treináveis**;
- contexto efetivo de treino: **32.256 bytes/tokens**;
- atenção local exata sobre chunk atual + anterior;
- memória linear recorrente para histórico mais antigo;
- estado recorrente destacado entre chunks, portanto BPTT truncado.

Treino de referência:

- 5.000 passos;
- 10 épocas lógicas de 500 passos;
- 2.560.000 tokens com gradiente;
- tempo efetivo de treino: **970,43 s**;
- wall time: **1.030,51 s**;
- pico de VRAM: **403,28 MiB**;
- GPU: NVIDIA GeForce GTX 1660 SUPER 6 GB.

Documentação completa: `runs/procedural_v3_hybrid32k_512_5000/README.md`.

## Direct-80M-32k

Arquivos principais:

- `model_direct_80m.py`
- `build_direct_80m_32k_exact.py`
- `benchmark_direct_80m_ppl.py`
- `runs/procedural_direct_80m_32k/`

O Direct não é Transformer e não possui pesos neurais. O nome 80M significa exatamente **80.000.000 estados escalares persistentes `uint32`**.

Distribuição dos estados:

| Estrutura | Estados |
|---|---:|
| Unigrama | 256 |
| Bigramas | 65.536 |
| Trigramas | 16.777.216 |
| Order-4 `key_lo` | 21.052.330 |
| Order-4 `key_hi` | 21.052.330 |
| Order-4 `count` | 21.052.330 |
| Metadados | 2 |
| **Total** | **80.000.000** |

Características:

- armazenamento bruto persistente: **320.000.000 bytes** (~305,2 MiB);
- tabela order-4 exata/esparsa: 4 bytes anteriores + byte-alvo → contagem;
- memória associativa online: chave de 8 bytes, janela de **32.768 bytes**;
- gradientes: **0**;
- otimizador: nenhum;
- corpus de treino inteiro lido uma vez;
- tempo de construção: **176,86 s / 2,95 min**;
- throughput de construção: **5,09 MiB/s**;
- entradas order-4 ocupadas: **8.210.951** (39,0% dos slots).

Os 80M estados Direct não devem ser interpretados como equivalentes matematicamente a 80M parâmetros de um Transformer; são formas de capacidade diferentes.

Documentação completa: `runs/procedural_direct_80m_32k/README.md`.

## Benchmark comum atual

Os dois modelos foram reexecutados após a limpeza do projeto nos mesmos três segmentos fixos de 32.768 bytes de `data/val.bin`, offsets 2.000.000, 4.000.000 e 6.000.000.

| Modelo | PPL médio | Accuracy | Throughput medido |
|---|---:|---:|---:|
| V3 Original | **17,55524** | **22,0286%** | ~4.881 tok/s |
| Direct-80M-32k | **4,16538** | **56,5867%** | ~18.577 bytes/s |

Resultados completos:

- `runs/procedural_v3_hybrid32k_512_5000/strict_val_3x32k.json`
- `runs/procedural_direct_80m_32k/ppl_benchmark.json`

### Interpretação correta

- O benchmark é byte-level.
- Os segmentos acima pertencem a `val.bin`, não ao `test.bin` final intocado.
- O V3 foi medido em GPU; o Direct roda em CPU. Throughput não é uma comparação de hardware equivalente.
- O baixo PPL do Direct é fortemente influenciado por regularidades locais de UTF-8, ortografia e sintaxe; isso não prova sozinho compreensão semântica ou qualidade de geração longa.
- O V3 viu 2,56M tokens com gradiente; o Direct percorreu 943,7M bytes uma vez sem backprop. A comparação demonstra diferenças de mecanismo de aquisição, não apenas diferenças de otimizador sob a mesma exposição de dados.

## Arquivos de modelo

Os artefatos `.pt` e `.npy` são versionados com **Git LFS**.

V3 preservado:

- `best.pt`
- `last.pt`
- `step0.pt`
- `summary.json`
- `progress.jsonl`
- `initial_metrics.json`
- benchmark comum.

Direct preservado:

- `unigram.npy`
- `bigram.npy`
- `trigram.npy`
- `order4_key_lo.npy`
- `order4_key_hi.npy`
- `order4_count.npy`
- `meta.npy`
- `summary.json`
- `ppl_benchmark.json`
- amostra de geração seed 42.

## Comandos

Treinar novamente o V3 Original:

```powershell
python train_procedural_v3_512_5000.py
```

Benchmark V3:

```powershell
python benchmark_v3_original_32k.py
```

Reconstruir Direct-80M:

```powershell
python build_direct_80m_32k_exact.py
```

Benchmark Direct:

```powershell
python benchmark_direct_80m_ppl.py
```

Testes de integridade que não exigem os `.bin` completos:

```powershell
python tests/test_metadata.py
python tests/test_v3.py
python tests/test_direct_80m.py
```

## Ambiente registrado

- Windows
- Python 3.10.10
- PyTorch 2.10.0+cu128
- CUDA runtime do PyTorch 12.8
- NumPy 2.2.6
- Numba 0.67.0
- llvmlite 0.49.0
- psutil 7.2.2
- GPU do V3: NVIDIA GeForce GTX 1660 SUPER 6 GB

## Estrutura

```text
AI-Procedural/
├── README.md
├── requirements.txt
├── prepare_portuguese_data.py
├── model_v3.py
├── train_procedural_v3_512_5000.py
├── benchmark_v3_original_32k.py
├── model_direct_80m.py
├── build_direct_80m_32k_exact.py
├── benchmark_direct_80m_ppl.py
├── data/
│   └── meta.json
├── tests/
└── runs/
    ├── procedural_v3_hybrid32k_512_5000/
    └── procedural_direct_80m_32k/
```
