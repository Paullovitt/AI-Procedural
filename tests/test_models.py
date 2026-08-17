import sys
from pathlib import Path
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'src'))
from model import ModelConfig, TinyTransformer

# Contagens exatas do experimento 80M.
base_cfg = ModelConfig(d_model=744, n_layers=12, n_heads=12, seq_len=512, procedural_rank=64, model_type='baseline')
proc_cfg = ModelConfig(d_model=744, n_layers=12, n_heads=12, seq_len=512, procedural_rank=64, model_type='procedural', fused_qkv=True)

base = TinyTransformer(base_cfg)
proc = TinyTransformer(proc_cfg)
assert base.parameter_report()['parameters'] == 79_936_848
assert proc.parameter_report()['parameters'] == 4_415_040

del base, proc

# Forward pequeno para verificar as duas implementações sem exigir muita memória.
for kind in ('baseline', 'procedural'):
    cfg = ModelConfig(d_model=64, n_layers=2, n_heads=4, seq_len=32, procedural_rank=8, model_type=kind, fused_qkv=True)
    model = TinyTransformer(cfg).eval()
    x = torch.randint(0, 256, (2, 32))
    with torch.no_grad():
        logits, loss = model(x, x)
    assert logits.shape == (2, 32, 256)
    assert torch.isfinite(loss)

print('OK: parameter counts + forward baseline/procedural')
