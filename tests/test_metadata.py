from pathlib import Path
import json
ROOT = Path(__file__).resolve().parents[1]
meta = json.loads((ROOT / 'data' / 'meta.json').read_text(encoding='utf-8'))
assert meta['dataset'] == 'HuggingFaceFW/fineweb-2'
assert meta['config'] == 'por_Latn'
assert meta['tokenizer'] == 'utf8-bytes'
assert meta['actual_bytes']['train'] == 943_718_400
assert meta['actual_bytes']['val'] == 52_428_800
assert meta['actual_bytes']['test'] == 52_428_800
print('OK: dataset metadata')
