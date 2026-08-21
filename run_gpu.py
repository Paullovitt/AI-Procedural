from __future__ import annotations

from pathlib import Path
from collections import Counter
import argparse, json, os, sys, time

from procedural_runtime_v3 import make_world
from procedural_runtime_v5 import ProtectedSlotVerifier
from procedural_runtime_gpu import SemanticTraceVerifier
from procedural_runtime_v14 import build_renderer_v14_gpu
from prompt_runtime_v14 import PromptAdapterV14, default_lexicon, lexicalize_text, contains_raw_slots

ROOT = Path(__file__).resolve().parent


def _configure_console():
    # RUN_GPU.bat also switches Windows to code page 65001. This keeps redirected
    # output and terminals that honor Python's stream encoding consistently UTF-8.
    for stream in (sys.stdout, sys.stderr):
        try:
            stream.reconfigure(encoding='utf-8', errors='replace')
        except (AttributeError, ValueError):
            pass


def load_config():
    return json.loads((ROOT / 'gpu_config.json').read_text(encoding='utf8'))


def load_runtime(cfg, lexicon=None):
    return build_renderer_v14_gpu(
        ROOT,
        seed=int(cfg.get('seed', 101)),
        use_hot=bool(cfg.get('use_hot', False)),
        proposal_weight=float(cfg.get('proposal_weight', .24)),
        position_weight=float(cfg.get('position_weight', 7.0)),
        diversity_weight=float(cfg.get('diversity_weight', 2.6)),
        focus_diversity_weight=float(cfg.get('focus_diversity_weight', 1.17)),
        repetition_weight=float(cfg.get('repetition_weight', 1.1)),
        template_repetition_weight=cfg.get('template_repetition_weight'),
        device=int(cfg.get('device', 0)),
        memory_limit_mb=int(cfg.get('memory_limit_mb', 4608)),
        lexicon=lexicon,
    )


def verify(plan, out):
    slots = ProtectedSlotVerifier()
    traces = SemanticTraceVerifier()
    return {
        'semantic_exact': Counter(out['represented']) == Counter(plan),
        'slot_errors': len(slots.inspect_render(out)),
        'trace_errors': len(traces.inspect_render(out)),
    }


def _verified(checks):
    return checks['semantic_exact'] and checks['slot_errors'] == 0 and checks['trace_errors'] == 0


def _read_lexicon(path: Path | None):
    if not path:
        return {}
    obj = json.loads(path.read_text(encoding='utf8'))
    if not isinstance(obj, dict):
        raise ValueError('O léxico deve ser um objeto JSON {slot: palavra/frase}.')
    return {str(k).lower(): str(v) for k, v in obj.items()}


def _load_inputs(args, cfg, custom_lexicon):
    adapter = PromptAdapterV14(int(cfg.get('default_target_chars', 2000)))
    prompt = args.prompt
    if args.prompt_file:
        prompt = args.prompt_file.read_text(encoding='utf8').strip()
    if not args.facts and not args.smoke and not prompt:
        prompt = input('Prompt> ').strip()
        if not prompt:
            raise SystemExit('Prompt vazio.')

    focus_hint = None
    prompt_meta = None
    if args.facts:
        plan = [tuple(x) for x in json.loads(args.facts.read_text(encoding='utf8'))]
        lexicon = default_lexicon(plan)
    elif args.smoke and not prompt:
        plan = make_world(args.seed, n_entities=14, n_props=10, n_rels=5, n_facts=70)
        lexicon = default_lexicon(plan)
    else:
        pp = adapter.build(prompt, target_chars=args.target_chars, seed=args.seed)
        plan, lexicon, focus_hint = pp.facts, pp.lexicon, pp.focus_order
        prompt_meta = {'prompt': pp.prompt, 'topic': pp.topic, 'target_chars': pp.target_chars}

    lexicon.update(custom_lexicon)
    return plan, lexicon, focus_hint, prompt_meta


def _render_checked(renderer, plan, lexicon, focus_hint, cfg):
    t0 = time.perf_counter()
    out = renderer.render(plan, focus_order_hint=focus_hint)
    dt = time.perf_counter() - t0
    checks = verify(plan, out)
    ok = _verified(checks)
    if bool(cfg.get('verify_semantics', True)) and not ok:
        raise SystemExit(f'ERRO: verificação semântica V14 rejeitou a saída: {checks}')
    display = lexicalize_text(out['text'], lexicon)
    if bool(cfg.get('readable_output', True)) and contains_raw_slots(display):
        raise SystemExit('ERRO: a camada de apresentação deixou escapar IDs semânticos.')
    return out, display, checks, ok, dt


def _refine_prompt_length(renderer, adapter, prompt_meta, args, cfg, custom_lexicon,
                          plan, lexicon, focus_hint, out, display, checks, ok, render_s):
    """At most three cheap V14 rerenders; GPU tables stay resident in VRAM."""
    target = int(prompt_meta['target_chars'])
    attempts = 1
    seen_counts = {len(plan)}
    while attempts < 3 and target > 0:
        err = len(display) - target
        tolerance = max(80, int(target * 0.08))
        if abs(err) <= tolerance or not display:
            break
        next_count = max(8, min(260, round(len(plan) * target / max(1, len(display)))))
        if next_count in seen_counts:
            next_count += -1 if err > 0 else 1
            next_count = max(8, min(260, next_count))
        if next_count in seen_counts:
            break
        seen_counts.add(next_count)
        pp = adapter.build(prompt_meta['prompt'], target_chars=target, seed=args.seed, fact_count=next_count)
        plan, lexicon, focus_hint = pp.facts, pp.lexicon, pp.focus_order
        lexicon.update(custom_lexicon)
        if hasattr(renderer.sel, 'lexicon'):
            renderer.sel.lexicon.update({str(k).lower(): str(v) for k, v in lexicon.items()})
        out, display, checks, ok, dt = _render_checked(renderer, plan, lexicon, focus_hint, cfg)
        render_s += dt
        attempts += 1
    return plan, lexicon, focus_hint, out, display, checks, ok, render_s, attempts


def main():
    _configure_console()
    ap = argparse.ArgumentParser(description='AI-Procedural V14 - CUDA/VRAM runtime com prompts e saída lexicalizada')
    ap.add_argument('--prompt', type=str, help='Prompt em linguagem natural')
    ap.add_argument('--prompt-file', type=Path, help='Arquivo UTF-8 contendo o prompt')
    ap.add_argument('--target-chars', type=int, help='Tamanho desejado da saída em caracteres')
    ap.add_argument('--facts', type=Path, help='JSON contendo lista de fatos [tipo, ...]')
    ap.add_argument('--lexicon', type=Path, help='JSON opcional {slot: palavra/frase}')
    ap.add_argument('--output', type=Path, help='Salvar texto legível gerado neste arquivo UTF-8')
    ap.add_argument('--raw-slots', action='store_true', help='Depuração: mostrar também a representação interna com IDs')
    ap.add_argument('--smoke', action='store_true', help='Executar teste sintético local do V14')
    ap.add_argument('--seed', type=int, default=1234)
    args = ap.parse_args()

    cfg = load_config()
    custom_lexicon = _read_lexicon(args.lexicon)
    plan, lexicon, focus_hint, prompt_meta = _load_inputs(args, cfg, custom_lexicon)

    t0 = time.perf_counter()
    scorer, grammar, inducer, renderer = load_runtime(cfg, lexicon=lexicon)
    load_s = time.perf_counter() - t0

    out, display_text, checks, verified, render_s = _render_checked(
        renderer, plan, lexicon, focus_hint, cfg)
    render_attempts = 1

    if prompt_meta:
        adapter = PromptAdapterV14(int(cfg.get('default_target_chars', 2000)))
        plan, lexicon, focus_hint, out, display_text, checks, verified, render_s, render_attempts = _refine_prompt_length(
            renderer, adapter, prompt_meta, args, cfg, custom_lexicon,
            plan, lexicon, focus_hint, out, display_text, checks, verified, render_s)

    gpu = scorer.gpu_status()
    report = {
        'runtime': 'V14',
        'facts': len(plan),
        'sentences': len(out['sentences']),
        'paragraphs': len(out.get('paragraphs', [])),
        'display_chars': len(display_text),
        'induced_selected': out.get('induced_selected', 0),
        'load_seconds': round(load_s, 3),
        'render_seconds_total': round(render_s, 4),
        'render_attempts': render_attempts,
        'semantic_verified': verified,
        'slot_errors': checks['slot_errors'],
        'trace_errors': checks['trace_errors'],
        'raw_slot_ids_exposed': contains_raw_slots(display_text),
        'backend': out.get('compute_backend', 'cuda-batched-v14'),
        'gpu': gpu,
    }
    if prompt_meta:
        report['prompt_mode'] = True
        report['topic'] = prompt_meta['topic']
        report['target_chars'] = prompt_meta['target_chars']
        report['target_error_chars'] = len(display_text) - int(prompt_meta['target_chars'])
    else:
        report['prompt_mode'] = False

    print(json.dumps(report, ensure_ascii=False, indent=2))
    print('\n--- TEXTO V14 ---\n')
    print(display_text)

    if args.raw_slots:
        print('\n--- REPRESENTAÇÃO SEMÂNTICA INTERNA ---\n')
        print(out['text'])
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(display_text, encoding='utf8')
        print(f'\nSalvo: {args.output}')


if __name__ == '__main__':
    main()
