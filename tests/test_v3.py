from pathlib import Path
import torch

ROOT = Path(__file__).resolve().parents[1]
import sys
sys.path.insert(0, str(ROOT))
from model_v3 import ModelConfig, TinyTransformer

run = ROOT / 'runs' / 'procedural_v3_hybrid32k_512_5000'
ck = torch.load(run / 'best.pt', map_location='cpu', weights_only=False)
assert ck['step'] == 5000
cfg = ModelConfig(**ck['config']['model_config'])
model = TinyTransformer(cfg)
model.load_state_dict(ck['model'], strict=True)
assert model.parameter_report()['parameters'] == 4_415_040
assert ck['config']['trainer']['effective_context'] == 32_256
print('OK: V3 Original checkpoint loads strictly')
