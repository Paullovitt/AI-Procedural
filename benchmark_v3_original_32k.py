import json, math, time
from pathlib import Path
import numpy as np
import torch
import torch.nn.functional as F
from model_v3 import ModelConfig, TinyTransformer

ROOT = Path(__file__).resolve().parent
RUN = ROOT / 'runs' / 'procedural_v3_hybrid32k_512_5000'
VAL = np.memmap(ROOT / 'data' / 'val.bin', dtype=np.uint8, mode='r')
OFFSETS = [2_000_000, 4_000_000, 6_000_000]
CHUNK = 512
SEG = 32768
DEVICE = 'cuda'

ckpt = torch.load(RUN / 'best.pt', map_location='cpu', weights_only=False)
cfgd = dict(ckpt['config']['model_config'])
cfgd['seq_len'] = CHUNK
cfgd['ffn_active_dims'] = 0
model = TinyTransformer(ModelConfig(**cfgd)).to(DEVICE)
model.load_state_dict(ckpt['model'], strict=True)
model.eval()
model.set_context_beta(0.5)

rows = []
torch.cuda.reset_peak_memory_stats()
for off in OFFSETS:
    model.reset_context()
    loss_sum = 0.0
    correct = 0
    n = 0
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    with torch.no_grad(), torch.autocast('cuda', dtype=torch.float16):
        for j in range(SEG // CHUNK):
            p = off + j * CHUNK
            z = np.asarray(VAL[p:p + CHUNK + 1], dtype=np.int64).copy()
            x = torch.from_numpy(z[:-1])[None].to(DEVICE)
            y = torch.from_numpy(z[1:])[None].to(DEVICE)
            logits, _ = model(x, position_offset=j * CHUNK)
            loss_sum += float(F.cross_entropy(logits.reshape(-1, 256), y.reshape(-1), reduction='sum'))
            correct += int((logits.argmax(-1) == y).sum().item())
            n += CHUNK
    torch.cuda.synchronize()
    sec = time.perf_counter() - t0
    loss = loss_sum / n
    row = {
        'offset': off,
        'loss': loss,
        'ppl': math.exp(loss),
        'accuracy': correct / n,
        'tokens': n,
        'sec': sec,
        'tokens_s': n / sec,
        'peak_vram_mb': torch.cuda.max_memory_allocated() / 1048576,
    }
    rows.append(row)
    print('TEST', json.dumps(row), flush=True)

avg_loss = sum(r['loss'] for r in rows) / len(rows)
summary = {
    'experiment': 'V3 Original - strict common 3x32k validation benchmark',
    'checkpoint': str(RUN / 'best.pt'),
    'dataset_split': 'data/val.bin',
    'segment_tokens': SEG,
    'offsets': OFFSETS,
    'tests': rows,
    'average': {
        'loss': avg_loss,
        'ppl': math.exp(avg_loss),
        'accuracy': sum(r['accuracy'] for r in rows) / len(rows),
        'tokens_s_avg': sum(r['tokens_s'] for r in rows) / len(rows),
        'peak_vram_mb_max': max(r['peak_vram_mb'] for r in rows),
    },
    'hardware': torch.cuda.get_device_name(0),
    'note': 'Pure V3 only. Fixed val.bin segments; not data/test.bin.',
}
(RUN / 'strict_val_3x32k.json').write_text(json.dumps(summary, indent=2), encoding='utf-8')
print('SUMMARY', json.dumps(summary), flush=True)
