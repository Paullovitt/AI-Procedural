from pathlib import Path
import sys
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'src'))
from model import ModelConfig, TinyTransformer

expected = {
    'baseline80m_ctx512_best_model.pt': (79_936_848, 5000),
    'procedural80m_ctx512_best_model.pt': (4_415_040, 5000),
}
for filename, (nparams, step) in expected.items():
    path = ROOT / 'checkpoints' / filename
    assert path.exists(), path
    ck = torch.load(path, map_location='cpu', weights_only=False)
    assert ck['step'] == step
    cfg = ModelConfig(**ck['model_config'])
    model = TinyTransformer(cfg)
    model.load_state_dict(ck['model'], strict=True)
    assert model.parameter_report()['parameters'] == nparams
    assert cfg.seq_len >= 512
    assert ck['training_config']['steps'] == 5000
    del model, ck

print('OK: published checkpoints load strictly and match metadata')
