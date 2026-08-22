from __future__ import annotations

from pathlib import Path
import argparse
import json
import sys
import time

from procedural_runtime_gpu import GpuBagacoSurfaceScorer
from robust_semantic_intake_v14 import RobustNoiseLearnerV14, RobustSemanticIntakeV14

ROOT = Path(__file__).resolve().parent
DEFAULT_OUT = ROOT / 'ROBUST_SEMANTIC_ALIASES_V14.json'


def iter_texts(paths, text_field='text'):
    for raw_path in paths:
        path = Path(raw_path)
        if path.suffix.lower() in {'.jsonl', '.ndjson'}:
            with path.open('r', encoding='utf8', errors='replace') as fh:
                for line in fh:
                    line=line.strip()
                    if not line: continue
                    obj=json.loads(line)
                    if isinstance(obj, str): yield obj
                    elif isinstance(obj, dict) and obj.get(text_field) is not None: yield str(obj[text_field])
        elif path.suffix.lower() == '.json':
            obj=json.loads(path.read_text(encoding='utf8'))
            rows=obj if isinstance(obj,list) else obj.get('rows',obj.get('data',[])) if isinstance(obj,dict) else []
            for row in rows:
                if isinstance(row,str): yield row
                elif isinstance(row,dict) and row.get(text_field) is not None: yield str(row[text_field])
        else:
            with path.open('r',encoding='utf8',errors='replace') as fh:
                for line in fh:
                    line=line.strip()
                    if line: yield line


def main():
    ap=argparse.ArgumentParser(description='V14 - aprende aliases robustos explícitos a partir de texto bruto; não reescreve o dataset.')
    ap.add_argument('datasets', nargs='+', type=Path)
    ap.add_argument('--text-field', default='text')
    ap.add_argument('--output', type=Path, default=DEFAULT_OUT)
    ap.add_argument('--min-support', type=int, default=3)
    ap.add_argument('--min-dominance', type=float, default=.75)
    ap.add_argument('--min-confidence', type=float, default=.72)
    ap.add_argument('--memory-limit-mb', type=int, default=4608)
    args=ap.parse_args()

    t0=time.perf_counter()
    scorer=GpuBagacoSurfaceScorer(ROOT, memory_limit_mb=args.memory_limit_mb)
    intake=RobustSemanticIntakeV14(scorer)
    intake.warm_index()
    learner=RobustNoiseLearnerV14(intake,args.min_support,args.min_dominance,args.min_confidence)
    rows=0
    for text in iter_texts(args.datasets,args.text_field):
        learner.observe(text); rows+=1
    bank=learner.promote()
    args.output.parent.mkdir(parents=True,exist_ok=True)
    bank.save(args.output)
    report={
        'runtime':'V14', 'format':bank.FORMAT, 'dataset_rows':rows,
        'promoted_aliases':len(bank.aliases), 'seconds':round(time.perf_counter()-t0,4),
        'intake':intake.status(), 'output':str(args.output),
        'rewritten_dataset':False,
    }
    print(json.dumps(report,ensure_ascii=False,indent=2))


if __name__=='__main__':
    main()
