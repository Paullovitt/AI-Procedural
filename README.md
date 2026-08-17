# AI-Procedural

Experimento comparando um Transformer denso de ~80M parâmetros com uma variante procedural de mesma geometria densa equivalente, onde projeções da atenção e FFN são reconstruídas por cálculo sob demanda.

## Arquitetura

Configuração usada no teste principal:

- 12 camadas
- `d_model = 744`
- 12 heads
- FFN = 4x
- contexto = 512 tokens
- tokenizer byte-level UTF-8, vocabulário 256
- baseline: 79.936.848 parâmetros treináveis
- procedural: 4.415.040 parâmetros persistentes, rank 64
- equivalente denso procedural: 79.936.848 parâmetros

## Dataset

O dataset não é versionado no Git por tamanho. `src/prepare_portuguese_data.py` baixa por streaming uma amostra de aproximadamente 1 GiB do FineWeb2 em português (`HuggingFaceFW/fineweb-2`, `por_Latn`):

- treino: 900 MiB
- validação: 50 MiB
- teste: 50 MiB

Para preparar:

```powershell
python src/prepare_portuguese_data.py
```

## Treino principal

Os dois modelos foram treinados do zero, sequencialmente, com:

- contexto: 512
- 5.000 optimizer steps
- 10 épocas lógicas de 500 passos
- batch = 1
- grad accumulation = 1
- 512 tokens por passo
- 2.560.000 tokens vistos por modelo
- baseline LR: `2e-4 -> 2e-5`
- procedural LR: `1.2e-3 -> 1.2e-4`
- warmup: 250 passos

Para reproduzir:

```powershell
powershell -ExecutionPolicy Bypass -File .\scripts\run_80m_ctx512_5000.ps1
```

> As 10 épocas são divisões lógicas dos 5.000 passos, não 10 passagens completas pelo corpus de 900 MiB.

## Resultado medido — GTX 1660 SUPER 6 GB

| Métrica | Baseline 80M | Procedural 80M eq. |
|---|---:|---:|
| Parâmetros persistentes | 79.936.848 | 4.415.040 |
| Equivalente denso | 79.936.848 | 79.936.848 |
| PPL validação final | **11,9318** | 12,5574 |
| Pico VRAM | 1.685 MB | **605 MB** |
| Throughput médio | 858 tok/s | **3.350 tok/s** |
| Tempo total | ~49m51s | **~12m48s** |

No teste, o procedural apresentou aproximadamente 94,5% menos parâmetros persistentes, 64,1% menos pico de VRAM e 3,91x mais throughput, com PPL ~5,2% maior.

Resultados detalhados estão em `results/`.

## Checkpoints

`checkpoints/` contém os melhores checkpoints **model-only**:

- `baseline80m_ctx512_best_model.pt` — ~305 MiB
- `procedural80m_ctx512_best_model.pt` — ~17 MiB

Eles contêm `state_dict`, configuração do modelo, configuração de treino, passo e loss de validação. Arquivos `.pt` são versionados via Git LFS.

## Testes

```powershell
python tests/test_models.py
python tests/test_checkpoints.py
```

`test_models.py` valida contagem de parâmetros e forward de ambas as arquiteturas. `test_checkpoints.py` valida os metadados e estados dos checkpoints publicados.
