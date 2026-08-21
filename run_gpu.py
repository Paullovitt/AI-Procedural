from __future__ import annotations

from pathlib import Path
from collections import Counter
import argparse, json, time

from procedural_runtime_gpu import build_renderer_v9_gpu_batched, ProtectedSlotVerifier, make_world

ROOT=Path(__file__).resolve().parent


def load_runtime():
    cfg=json.loads((ROOT/'gpu_config.json').read_text(encoding='utf8'))
    scorer,grammar,inducer,renderer=build_renderer_v9_gpu_batched(
        ROOT,seed=int(cfg.get('seed',101)),use_hot=bool(cfg.get('use_hot',False)),
        proposal_weight=float(cfg.get('proposal_weight',.06)),
        device=int(cfg.get('device',0)),memory_limit_mb=int(cfg.get('memory_limit_mb',4608)))
    return cfg,scorer,grammar,inducer,renderer


def verify(plan,out):
    vf=ProtectedSlotVerifier()
    return Counter(out['represented'])==Counter(plan) and len(vf.inspect_render(out))==0


def main():
    ap=argparse.ArgumentParser(description='AI-Procedural V9 - CUDA runtime')
    ap.add_argument('--facts',type=Path,help='JSON contendo uma lista de fatos [tipo, ...]')
    ap.add_argument('--output',type=Path,help='Salvar texto gerado neste arquivo UTF-8')
    ap.add_argument('--smoke',action='store_true',help='Executar teste sintetico local')
    ap.add_argument('--seed',type=int,default=1234)
    args=ap.parse_args()

    t0=time.perf_counter();cfg,s,g,i,r=load_runtime();load_s=time.perf_counter()-t0
    print(json.dumps({'load_seconds':round(load_s,3),'gpu':s.gpu_status()},ensure_ascii=False,indent=2))

    if args.facts:
        plan=[tuple(x) for x in json.loads(args.facts.read_text(encoding='utf8'))]
    else:
        plan=make_world(args.seed,n_entities=12,n_props=8,n_rels=4,n_facts=180)

    t0=time.perf_counter();out=r.render(plan);dt=time.perf_counter()-t0
    ok=verify(plan,out)
    report={'facts':len(plan),'sentences':len(out['sentences']),'induced_selected':out.get('induced_selected',0),
            'render_seconds':round(dt,4),'semantic_verified':ok,'backend':'cuda'}
    print(json.dumps(report,ensure_ascii=False,indent=2))
    if not ok:
        raise SystemExit('ERRO: verificador semantico rejeitou a saida')
    if args.output:
        args.output.write_text(out['text'],encoding='utf8')
        print(f'Salvo: {args.output}')
    else:
        print('\n'+out['text'])


if __name__=='__main__':
    main()

