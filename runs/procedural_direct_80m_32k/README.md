# Direct-80M-32k — Direct Puro exact-sparse

Este diretório contém o **Direct Puro 80M final**. Ele não é Transformer, não usa backpropagation e não possui parâmetros neurais. O nome 80M significa **80.000.000 estados escalares persistentes `uint32`** usados para armazenar contagens e chaves explícitas.

## Identidade

- Experimento: `Direct-80M-32k exact-sparse`
- Implementação de inferência: `../../model_direct_80m.py`
- Builder final: `../../build_direct_80m_32k_exact.py`
- Benchmark PPL: `../../benchmark_direct_80m_ppl.py`
- Dataset: `../../data/`
- Gradientes: **0**
- Otimizador: **nenhum**
- Contexto associativo de runtime: **32.768 bytes**

## Capacidade persistente

São exatamente 80.000.000 estados escalares `uint32`:

| Estrutura | Estados |
|---|---:|
| Unigramas | 256 |
| Bigramas | 65.536 |
| Trigramas | 16.777.216 |
| Order-4 `key_lo` | 21.052.330 |
| Order-4 `key_hi` | 21.052.330 |
| Order-4 `count` | 21.052.330 |
| Metadados | 2 |
| **Total** | **80.000.000** |

Armazenamento bruto dos estados: **320.000.000 bytes** (~305,2 MiB), sem contar pequenos headers `.npy` e memória temporária de execução.

Isso **não equivale matematicamente a 80M pesos FP32 de um Transformer**. É uma convenção de capacidade baseada no número de estados escalares persistentes.

## Como o modelo prevê

A distribuição do próximo byte combina quatro níveis estatísticos com backoff por confiança:

1. unigrama global;
2. bigrama: 1 byte anterior → próximo byte;
3. trigrama: 2 bytes anteriores → próximo byte;
4. order-4 esparso: **4 bytes anteriores + byte-alvo → contagem**.

Hiperparâmetros atuais de inferência/benchmark:

- `kappa1 = 100`
- `kappa2 = 300`
- `kappa4 = 100`
- `copy_lambda = 0.75`
- `copy_key = 8`
- `copy_window = 32768`
- smoothing `eps = 1e-5`

### Tabela order-4 exata e esparsa

A tabela não cria os 256^5 estados possíveis. Ela usa 21.052.330 slots com open addressing e verificação explícita da chave:

- `order4_key_lo.npy`
- `order4_key_hi.npy`
- `order4_count.npy`

A chave representa os 4 bytes de contexto mais o próximo byte. Colisões de hash são verificadas pela chave armazenada, portanto uma colisão não mistura contagens de padrões diferentes.

Na passada completa foram ocupadas **8.210.951 entradas**, ou **39,00%** dos slots. A sondagem média registrada foi 0,00950 e a máxima 29.

## Memória associativa de 32k

Além das tabelas persistentes, a inferência mantém uma memória online deslizante:

- janela: **32.768 posições/bytes**;
- chave: **8 bytes**;
- associação: contexto de 8 bytes → byte observado em seguida.

Essa memória é estado de runtime e **não entra nos 80M estados persistentes**. Ela é descartada/reinicializada com `reset_context()` e não está gravada nos arquivos do modelo.

Durante a construção completa foram registrados **165.808.585 hits associativos** e 28.866 pares ativos ao final da sequência.

## Construção no dataset completo

Fonte: `data/train.bin`, FineWeb-2 português (`por_Latn`).

Resultado de `summary.json`:

| Métrica | Resultado |
|---|---:|
| Bytes lidos | **943.718.400** |
| Passadas pelo train | **1** |
| Passos de gradiente | **0** |
| Tempo da passada | **176,8566 s** |
| Tempo | **2,9476 min** |
| Throughput | **5,0889 MiB/s** |
| JIT inicial excluído | 0,2958 s |
| RSS antes | 397,09 MiB |
| RSS depois | 1.297,86 MiB |
| Entradas order-4 | 8.210.951 |
| Ocupação order-4 | 39,00% |

O RSS de ~1,30 GiB é memória do processo durante a construção, não o tamanho persistente do modelo.

## Benchmark PPL comum 3 × 32k

Arquivo: `ppl_benchmark.json`.

Foram usados três segmentos fixos de 32.768 bytes em `data/val.bin`, nos mesmos offsets preservados para o V3 Original:

- offset 2.000.000: PPL **4,13291**, accuracy 57,0160%;
- offset 4.000.000: PPL **4,00104**, accuracy 57,4371%;
- offset 6.000.000: PPL **4,37055**, accuracy 55,3070%.

Média:

- loss: **1,426808**;
- PPL: **4,165383**;
- accuracy: **56,5867%**;
- throughput de inferência medido: **18.577 bytes/s**.

Esse benchmark roda a implementação Direct em CPU. Os segmentos são de `val.bin`, e não do `data/test.bin`.

## Geração preservada

- `sample_pt_seed42.txt` — amostra gerada em português;
- `sample_pt_seed42_meta.txt` — parâmetros da geração:
  - 400 bytes gerados;
  - temperature 0,85;
  - top-p 0,95;
  - seed 42;
  - tempo registrado 0,2482 s.

## Arquivos persistentes

- `unigram.npy`
- `bigram.npy`
- `trigram.npy`
- `order4_key_lo.npy`
- `order4_key_hi.npy`
- `order4_count.npy`
- `meta.npy`
- `summary.json`
- `ppl_benchmark.json`
- amostras de geração
- `README.md`

## Reconstruir do zero

A partir da raiz:

```powershell
python build_direct_80m_32k_exact.py
```

O builder lê todo o `data/train.bin` e grava os arquivos neste diretório. Ele usa Numba para compilar os loops críticos.

## Rodar o benchmark PPL

```powershell
python benchmark_direct_80m_ppl.py
```

## Carregar para inferência

```python
from model_direct_80m import Direct80M32K

model = Direct80M32K(
    r"runs\procedural_direct_80m_32k",
    kappa1=100,
    kappa2=300,
    kappa4=100,
    copy_lambda=0.75,
    copy_key=8,
    copy_window=32768,
)
model.reset_context()

# Para cada byte já observado:
p = model.predict_after_observing(byte_value)
# p possui 256 probabilidades para o próximo byte.
```

Não reinicie o contexto entre chunks que pertencem à mesma sequência contínua.

## Limitações de interpretação

- 80M estados Direct **não são 80M parâmetros neurais**.
- O modelo é byte-level e grande parte da força vem de regularidades locais de UTF-8, ortografia e sintaxe.
- A memória persistente order-4 alcança somente quatro bytes anteriores; relações mais longas dependem da memória associativa de 32k e de repetições da chave de 8 bytes.
- Um PPL muito baixo em bytes não prova sozinho compreensão semântica equivalente a um Transformer.
- O benchmark atual usa segmentos de `val.bin`; uma avaliação final em `test.bin` continua recomendada.

## Ambiente registrado

- Python 3.10.10
- NumPy 2.2.6
- Numba 0.67.0
- llvmlite 0.49.0
- Construção/inferência Direct: CPU
