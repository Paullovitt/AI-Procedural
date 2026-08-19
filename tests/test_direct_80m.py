from pathlib import Path
import json
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
import sys
sys.path.insert(0, str(ROOT))
from model_direct_80m import Direct80M32K

run = ROOT / 'runs' / 'procedural_direct_80m_32k'
summary = json.loads((run / 'summary.json').read_text(encoding='utf-8'))
assert summary['persistent_scalar_states'] == 80_000_000
assert summary['runtime_context_window'] == 32_768
assert summary['gradient_steps'] == 0
assert np.load(run / 'unigram.npy', mmap_mode='r').shape == (256,)
assert np.load(run / 'bigram.npy', mmap_mode='r').shape == (256, 256)
assert np.load(run / 'trigram.npy', mmap_mode='r').shape == (65_536, 256)
model = Direct80M32K(run)
model.reset_context()
p = model.predict_after_observing(ord('a'))
assert p.shape == (256,)
assert abs(float(p.sum()) - 1.0) < 1e-5
print('OK: Direct-80M-32k artifacts load and predict')
