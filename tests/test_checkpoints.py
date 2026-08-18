from pathlib import Path
import sys
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'src'))

# Dense
from model import ModelConfig as DenseConfig, TinyTransformer as DenseTransformer
p = ROOT / 'checkpoints' / 'baseline80m_ctx512_best_model.pt'
ck = torch.load(p, map_location='cpu', weights_only=False)
assert ck['step'] == 5000
m = DenseTransformer(DenseConfig(**ck['model_config']))
m.load_state_dict(ck['model'], strict=True)
assert m.parameter_report()['parameters'] == 79_936_848
del m, ck

# Procedural V3 512k
from model_v3 import ModelConfig as V3Config, TinyTransformer as V3Transformer
p = ROOT / 'checkpoints' / 'procedural_v3_512k_best_model.pt'
ck = torch.load(p, map_location='cpu', weights_only=False)
assert ck['step'] == 5000
assert ck['effective_context_tokens'] == 524_288
assert ck['model_config']['v3_beta'] == 0.5
m = V3Transformer(V3Config(**ck['model_config']))
m.load_state_dict(ck['model'], strict=True)
assert m.parameter_report()['parameters'] == 4_415_040
assert ck['training_config']['effective_context'] == 524_288

del m, ck
print('OK: published Dense and Procedural V3 checkpoints load strictly')
