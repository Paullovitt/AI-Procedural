# AI-Procedural

Este repositório mantém apenas duas referências para comparação:

1. **Dense 80M** — baseline denso de 79.936.848 parâmetros.
2. **Procedural V3 512k** — 4.415.040 parâmetros persistentes, contexto recorrente efetivo de 524.288 tokens.

## Arquitetura comum

- 12 camadas
- `d_model = 744`
- 12 heads
- FFN = 4x
- vocabulário byte-level UTF-8: 256

## Dense 80M

Checkpoint: `checkpoints/baseline80m_ctx512_best_model.pt`

Treino de referência:

- 5.000 passos
- 10 épocas lógicas de 500 passos
- 512 tokens por passo
- 2.560.000 tokens vistos
- melhor PPL de validação registrado: **11,9318**
- PPL em avaliação comum de 512 tokens: **12,0642**

## Procedural V3 512k

Checkpoint: `checkpoints/procedural_v3_512k_best_model.pt`

Implementação: `src/model_v3.py`

O V3 usa duas escalas de atenção:

- **local:** softmax causal exata sobre o chunk atual e o chunk anterior;
- **global:** memória linear recorrente `KV_state + K_state` para o histórico mais antigo.

O estado global tem tamanho fixo em relação ao comprimento do contexto. O forward carrega informação de até **524.288 tokens**; o estado recorrente é destacado entre chunks durante o treino, portanto o backward é truncado entre chunks.

Configuração final:

- parâmetros persistentes: **4.415.040**
- equivalente denso: 79.936.848
- rank procedural: 64
- chunk: 512 tokens
- contexto efetivo: **524.288 tokens**
- `beta` global final: 0,50
- FFN final: largura completa 2.976
- treino: 5.000 passos / 10 épocas lógicas
- tokens vistos: 2.560.000

### Resultado 512k

Em três segmentos independentes de 524.288 tokens:

| Segmento | PPL |
|---|---:|
| 1 | 19,4976 |
| 2 | 20,3336 |
| 3 | 19,7907 |
| **Média** | **19,8710** |

Outras métricas medidas na GTX 1660 SUPER 6 GB:

- tempo puro de treino: **970,53 s**
- throughput de treino: **2.637,7 tok/s**
- pico de VRAM no treino: **403,3 MB**
- forward médio de 524.288 tokens: **96,12 s**
- PPL em avaliação comum de 512 tokens: **19,8343**

O V3 treinado especificamente em 512k obteve PPL **19,50** no segmento canônico; o V3 anteriormente treinado em 32k, apenas extrapolado para 512k, obteve PPL **26,37** no mesmo segmento.

## Carregar o V3

```python
import sys
import torch

sys.path.insert(0, "src")
from model_v3 import ModelConfig, TinyTransformer

ck = torch.load(
    "checkpoints/procedural_v3_512k_best_model.pt",
    map_location="cpu",
    weights_only=False,
)

cfg = ModelConfig(**ck["model_config"])
model = TinyTransformer(cfg)
model.load_state_dict(ck["model"], strict=True)
model.eval()

# O checkpoint usa seq_len=512 porque 512 é o tamanho do chunk.
# Para um fluxo longo, processe chunks consecutivos sem chamar reset_context().
model.reset_context()
model.set_context_beta(0.5)
```

## Dataset

O dataset não é versionado. `src/prepare_portuguese_data.py` baixa por streaming uma amostra do FineWeb2 em português (`HuggingFaceFW/fineweb-2`, `por_Latn`):

- treino: 900 MiB
- validação: 50 MiB
- teste: 50 MiB

```powershell
python src/prepare_portuguese_data.py
```

As 10 épocas citadas são **épocas lógicas** de 500 passos, não 10 passagens completas pelo corpus.

## Resultados

- `results/baseline80m_512_5000_summary.json`
- `results/procedural_v3_512k_5000_summary.json`
- `results/procedural_v3_512k_eval_512k.json`
- `results/comparison_dense_v3_512k.json`

## Testes

```powershell
python tests/test_models.py
python tests/test_checkpoints.py
```

Checkpoints `.pt` são versionados com Git LFS.
