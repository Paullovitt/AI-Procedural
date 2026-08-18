import sys
from pathlib import Path
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'src'))
from model import ModelConfig as DenseConfig, TinyTransformer as DenseTransformer
from model_v3 import ModelConfig as V3Config, TinyTransformer as V3Transformer

# Contagens oficiais.
dense = DenseTransformer(DenseConfig(d_model=744, n_layers=12, n_heads=12, seq_len=512, procedural_rank=64, model_type='baseline'))
v3 = V3Transformer(V3Config(d_model=744, n_layers=12, n_heads=12, seq_len=512, procedural_rank=64, model_type='procedural', fused_qkv=True, latent_attention=True, v3_hybrid=True, v3_beta=0.5))
assert dense.parameter_report()['parameters'] == 79_936_848
assert v3.parameter_report()['parameters'] == 4_415_040

del dense, v3

# Forward pequeno das duas implementações.
dense = DenseTransformer(DenseConfig(d_model=64, n_layers=2, n_heads=4, seq_len=32, procedural_rank=8, model_type='baseline')).eval()
x = torch.randint(0, 256, (2, 32))
with torch.no_grad():
    logits, loss = dense(x, x)
assert logits.shape == (2, 32, 256)
assert torch.isfinite(loss)

v3 = V3Transformer(V3Config(d_model=64, n_layers=2, n_heads=4, seq_len=32, procedural_rank=8, model_type='procedural', fused_qkv=True, latent_attention=True, v3_hybrid=True, v3_beta=0.5)).eval()
v3.reset_context()
with torch.no_grad():
    logits, loss = v3(x, x)
assert logits.shape == (2, 32, 256)
assert torch.isfinite(loss)

print('OK: Dense + Procedural V3 parameter counts and forward')
