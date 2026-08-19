# V3 Original — Procedural V3 Hybrid 32k

Este diretório contém o **V3 Original** preservado. Ele é um modelo neural procedural recorrente treinado com backpropagation. Não é o Dense 80M e não é o V3+Direct.

## Identidade

- Experimento: `procedural_v3_hybrid32k_512_5000`
- Implementação: `../../model_v3.py`
- Treino/reprodução: `../../train_procedural_v3_512_5000.py`
- Checkpoint principal: `best.pt`
- Checkpoint inicial: `step0.pt`
- Último checkpoint: `last.pt`
- Dataset: `../../data/`
- Hardware do treino registrado: NVIDIA GeForce GTX 1660 SUPER 6 GB

## Arquitetura

Configuração efetiva do checkpoint `best.pt`:

| Campo | Valor |
|---|---:|
| Vocabulário | 256 bytes UTF-8 |
| `d_model` | 744 |
| Camadas | 12 |
| Heads | 12 |
| `seq_len` por chunk | 512 |
| Multiplicador FFN | 4 |
| Dropout | 0,1 |
| Rank procedural | 64 |
| Parâmetros persistentes treináveis | **4.415.040** |
| `fused_qkv` | true |
| `latent_attention` | true |
| `cache_procedural` | true |
| `v3_hybrid` | true |
| beta final de contexto | 0,50 |

O V3 reduz fortemente o número de parâmetros persistentes porque várias projeções são matrizes procedurais determinísticas geradas em runtime, enquanto gates e componentes neurais menores são aprendidos.

### Memória/contexto

Cada camada mantém estado de runtime:

- `_v3_recent_k` / `_v3_recent_v`: bloco recente;
- `_v3_oldS` / `_v3_oldz`: resumo linear recorrente dos blocos mais antigos.

A atenção combina:

1. softmax exata local no chunk atual + chunk anterior;
2. memória linear recorrente para contexto mais antigo;
3. mistura controlada por `v3_beta`.

O estado global é `detach` entre chunks. Portanto existe contexto longo no forward, mas o BPTT é truncado nas fronteiras dos chunks.

No treino original foram usados **63 blocos × 512 = 32.256 tokens/bytes de contexto efetivo por janela de treino**.

## Treino original

- Seed: 1337
- Passos: **5.000**
- Épocas lógicas: **10**
- 500 passos por época lógica
- Tokens com gradiente: **2.560.000**
- Chunk: 512
- LR máximo: 0,0012
- LR mínimo: 0,00012
- Warmup: 250 passos
- AdamW: betas `(0.9, 0.95)`, weight decay `0.1`
- AMP FP16 + GradScaler
- clipping de gradiente: 1,0

Currículo de FFN:

- passos 1–500: 512 dimensões ativas;
- 501–1500: 1024;
- 1501–2500: 1536;
- 2501–5000: FFN completo.

Currículo de memória global (`beta`):

- passos 1–500: 0,05;
- 501–1500: 0,15;
- 1501–2500: 0,30;
- 2501–5000: 0,50.

**Importante:** as 10 épocas são épocas lógicas do experimento, não 10 passadas completas pelo `train.bin`. O V3 recebeu 2,56 milhões de tokens com backprop, enquanto o `train.bin` possui 943.718.400 bytes.

## Resultados registrados no treino

De `summary.json`:

| Métrica | Resultado |
|---|---:|
| Tempo efetivo de treino | **970,430 s** |
| Wall time | **1.030,513 s** |
| Throughput de treino | 2.638 tok/s |
| Throughput wall | 2.484 tok/s |
| Pico VRAM | **403,275 MiB** |
| Val 32k loss | 2,758014 |
| Val 32k PPL | **15,76849** |
| Val 32k accuracy | 23,1957% |
| Val 32k throughput | 5.637,8 tok/s |
| Exact-512 PPL | **16,93522** |

## Benchmark comum 3 × 32k

Arquivo: `strict_val_3x32k.json`.

Usa os mesmos offsets do `data/val.bin` usados para comparar com o Direct-80M:

- offset 2.000.000: PPL 19,14657; accuracy 21,6614%;
- offset 4.000.000: PPL 16,53630; accuracy 22,0306%;
- offset 6.000.000: PPL 17,08802; accuracy 22,3938%.

Média:

- loss: **2,865353**;
- PPL: **17,555240**;
- accuracy: **22,0286%**;
- throughput: **4.881 tok/s**;
- pico do allocator CUDA nessa avaliação: 175,205 MiB.

Esses são segmentos fixos de `val.bin`, e não o `data/test.bin`.

## Arquivos preservados

- `best.pt` — melhor checkpoint do treino;
- `last.pt` — último checkpoint salvo;
- `step0.pt` — inicialização usada pelo experimento;
- `summary.json` — resumo final;
- `progress.jsonl` — histórico das avaliações a cada 500 passos;
- `initial_metrics.json` — métricas antes do treino;
- `strict_val_3x32k.json` — benchmark V3 puro nos offsets comuns;
- `README.md` — este documento.

## Reproduzir o treino

A partir da raiz do projeto:

```powershell
python train_procedural_v3_512_5000.py
```

O script escreve diretamente neste diretório. Para preservar os checkpoints existentes, copie o projeto/run antes de iniciar um novo treino ou altere a pasta `OUT` no script.

## Carregar o checkpoint

```python
import torch
from model_v3 import ModelConfig, TinyTransformer

ckpt = torch.load(
    r"runs\procedural_v3_hybrid32k_512_5000\best.pt",
    map_location="cpu",
    weights_only=False,
)
cfg = ModelConfig(**ckpt["config"]["model_config"])
model = TinyTransformer(cfg)
model.load_state_dict(ckpt["model"], strict=True)
model.set_context_beta(0.5)
model.reset_context()
```

Para contexto contínuo, processe chunks sequencialmente e incremente `position_offset`; não chame `reset_context()` entre chunks pertencentes à mesma sequência.

## Limitações de interpretação

- O V3 possui **4,415M parâmetros persistentes**, não 80M.
- Contexto longo no forward não significa BPTT longo; o estado entre chunks é detached.
- O benchmark 3×32k é em `val.bin`; ainda deve existir um benchmark final separado em `test.bin` para uma avaliação totalmente intocada.
- PPL byte-level mede previsão de bytes e não, sozinho, qualidade semântica ou geração de longo alcance.

## Ambiente registrado

- Python 3.10.10
- PyTorch 2.10.0+cu128
- CUDA runtime do PyTorch 12.8
- NumPy 2.2.6
- GPU: NVIDIA GeForce GTX 1660 SUPER 6 GB
