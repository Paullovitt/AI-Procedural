from __future__ import annotations

from pathlib import Path
from collections import Counter
import argparse, json, sys, time

from procedural_runtime_v3 import make_world
from procedural_runtime_v5 import ProtectedSlotVerifier
from procedural_runtime_gpu import SemanticTraceVerifier
from procedural_runtime_v14 import build_renderer_v14_gpu
from prompt_runtime_v14 import PromptAdapterV14, default_lexicon, lexicalize_text, contains_raw_slots

ROOT = Path(__file__).resolve().parent


def _configure_console():
    for stream in (sys.stdout, sys.stderr):
        try:
            stream.reconfigure(encoding='utf-8', errors='replace')
        except (AttributeError, ValueError):
            pass


def load_config():
    return json.loads((ROOT / 'gpu_config.json').read_text(encoding='utf8'))


def load_runtime(cfg, lexicon=None, prompt_mode=False):
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
        lexicalize_entities=bool(prompt_mode),
        max_bundle=(int(cfg.get('prompt_max_bundle', 4)) if prompt_mode else None),
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


def _resolve_prompt(args):
    prompt = args.prompt
    if args.prompt_file:
        prompt = args.prompt_file.read_text(encoding='utf8').strip()
    if not args.facts and not args.smoke and not prompt:
        prompt = input('Prompt> ').strip()
        if not prompt:
            raise SystemExit('Prompt vazio.')
    return prompt


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


def _apply_prompt_plan(renderer, pp, custom_lexicon):
    lexicon = dict(pp.lexicon)
    lexicon.update(custom_lexicon)
    renderer.set_prompt_surface(lexicon=lexicon, predicate_relations=pp.predicate_relations,
                                predicate_classes=getattr(pp, 'predicate_classes', {}),
                                argument_roles=getattr(pp, 'argument_roles', {}))
    return pp.facts, lexicon, pp.focus_order


def _refine_prompt_length(renderer, scorer, adapter, prompt_meta, args, cfg, custom_lexicon,
                          plan, lexicon, focus_hint, out, display, checks, ok, render_s,
                          reasoning_stats):
    """A few cheap rerenders; keep the closest verified length candidate."""
    target = int(prompt_meta['target_chars'])
    attempts = 1
    seen_counts = {len(plan)}
    reasoning_s = 0.0
    best = (plan, lexicon, focus_hint, out, display, checks, ok, reasoning_stats)
    best_error = abs(len(display) - target)
    while attempts < 4 and target > 0:
        err = len(display) - target
        tolerance = max(70, int(target * 0.06))
        if abs(err) <= tolerance or not display:
            break
        next_count = max(8, min(180, round(len(plan) * target / max(1, len(display)))))
        # Contextual disambiguation can reject otherwise strong-but-off-topic rules.
        # Search a wider bank to replace them instead of weakening the quality filter.
        arg_stats=(reasoning_stats or {}).get('argument_planner', {})
        context_filtered=(int(arg_stats.get('context_filtered_rules',0))
                          + int(arg_stats.get('context_filtered_synthesis',0)))
        if err < 0 and context_filtered:
            refill_target=len(plan) + 2*context_filtered
            next_count=max(next_count,min(180,refill_target))
        if next_count in seen_counts:
            next_count += -1 if err > 0 else 1
            next_count = max(8, min(180, next_count))
        if next_count in seen_counts:
            break
        seen_counts.add(next_count)
        t0 = time.perf_counter()
        pp = adapter.build_learned(prompt_meta['prompt'], scorer, target_chars=target,
                                   seed=args.seed, fact_count=next_count)
        reasoning_s += time.perf_counter() - t0
        previous_fact_count = len(plan)
        plan, lexicon, focus_hint = _apply_prompt_plan(renderer, pp, custom_lexicon)
        reasoning_stats = pp.reasoning_stats
        # If stronger rule filtering means there are no additional facts to promote,
        # change only discourse granularity. More/smaller bundles add explanatory
        # sentence context without admitting weak rules merely to hit a length target.
        if hasattr(renderer.structure, 'max_bundle') and len(plan) <= previous_fact_count:
            if err < 0 and int(renderer.structure.max_bundle) > 1:
                renderer.structure.max_bundle = int(renderer.structure.max_bundle) - 1
            elif err > 0:
                renderer.structure.max_bundle = min(int(cfg.get('prompt_max_bundle', 4)), int(renderer.structure.max_bundle) + 1)
        out, display, checks, ok, dt = _render_checked(renderer, plan, lexicon, focus_hint, cfg)
        render_s += dt
        attempts += 1
        candidate_error = abs(len(display) - target)
        if candidate_error < best_error:
            best_error = candidate_error
            best = (plan, lexicon, focus_hint, out, display, checks, ok, reasoning_stats)
    plan, lexicon, focus_hint, out, display, checks, ok, reasoning_stats = best
    return (plan, lexicon, focus_hint, out, display, checks, ok, render_s, attempts,
            reasoning_stats, reasoning_s)


def _reasoning_summary(stats):
    if not stats:
        return {}
    bank = stats.get('rulebank', {})
    return {
        'engine': stats.get('engine'),
        'seed_concepts': stats.get('seed_concepts', []),
        'selected_rules': stats.get('selected_rules', 0),
        'reachable_nodes': stats.get('reachable_nodes', 0),
        'vm_seconds': round(float(stats.get('vm_seconds', 0.0)), 6),
        'indexed_lookups': stats.get('indexed_lookups', 0),
        'rulebank_rules': bank.get('rules', 0),
        'rulebank_kinds': bank.get('kinds', {}),
        'rule_learning_seconds': round(float(bank.get('learn_seconds', 0.0)), 6),
        'context_index_seconds': round(float(bank.get('index_seconds', 0.0)), 6),
        'argument_planner': stats.get('argument_planner', {}),
        'semantic_intake': stats.get('semantic_intake', {}),
        'gpu': bank.get('gpu'),
    }


def main():
    _configure_console()
    ap = argparse.ArgumentParser(description='AI-Procedural V14 - CUDA/VRAM + Learned RuleVM V6')
    source = ap.add_mutually_exclusive_group()
    source.add_argument('--prompt', type=str, help='Prompt em linguagem natural')
    source.add_argument('--prompt-file', type=Path, help='Arquivo UTF-8 contendo o prompt')
    source.add_argument('--facts', type=Path, help='JSON contendo lista de fatos [tipo, ...]')
    source.add_argument('--smoke', action='store_true', help='Executar teste sintético local do V14')
    ap.add_argument('--target-chars', type=int, help='Tamanho desejado da saída em caracteres')
    ap.add_argument('--lexicon', type=Path, help='JSON opcional {slot: palavra/frase}')
    ap.add_argument('--output', type=Path, help='Salvar texto legível gerado neste arquivo UTF-8')
    ap.add_argument('--raw-slots', action='store_true', help='Depuração: mostrar também a representação interna com IDs')
    ap.add_argument('--legacy-prompt', action='store_true', help=argparse.SUPPRESS)
    ap.add_argument('--seed', type=int, default=1234)
    args = ap.parse_args()

    cfg = load_config()
    custom_lexicon = _read_lexicon(args.lexicon)
    prompt = _resolve_prompt(args)
    prompt_mode = bool(prompt and not args.facts and not args.smoke)

    # Non-prompt paths keep their historical explicit fact semantics.
    plan = None; lexicon = None; focus_hint = None; prompt_meta = None
    adapter = PromptAdapterV14(
        int(cfg.get('default_target_chars', 2000)),
        argument_planner_enabled=bool(cfg.get('argument_planner_enabled', True)),
        robust_intake_enabled=bool(cfg.get('robust_semantic_intake_enabled', True)),
        robust_intake_warm_index=bool(cfg.get('robust_semantic_warm_index', True)),
    )
    if args.facts:
        plan = [tuple(x) for x in json.loads(args.facts.read_text(encoding='utf8'))]
        lexicon = default_lexicon(plan);lexicon.update(custom_lexicon)
    elif args.smoke and not prompt:
        plan = make_world(args.seed, n_entities=14, n_props=10, n_rels=5, n_facts=70)
        lexicon = default_lexicon(plan);lexicon.update(custom_lexicon)

    t0 = time.perf_counter()
    scorer, grammar, inducer, renderer = load_runtime(cfg, lexicon=(lexicon if not prompt_mode else {}),
                                                       prompt_mode=prompt_mode)
    if prompt_mode and not args.legacy_prompt and adapter.robust_intake_enabled:
        adapter._intake_for(scorer)
    load_s = time.perf_counter() - t0

    reasoning_stats = {}
    reasoning_s = 0.0
    if prompt_mode:
        t0 = time.perf_counter()
        if args.legacy_prompt:
            pp = adapter.build(prompt, target_chars=args.target_chars, seed=args.seed)
        else:
            pp = adapter.build_learned(prompt, scorer, target_chars=args.target_chars, seed=args.seed)
        reasoning_s = time.perf_counter() - t0
        plan, lexicon, focus_hint = _apply_prompt_plan(renderer, pp, custom_lexicon)
        reasoning_stats = pp.reasoning_stats
        prompt_meta = {'prompt': pp.prompt, 'topic': pp.topic, 'target_chars': pp.target_chars}

    out, display_text, checks, verified, render_s = _render_checked(
        renderer, plan, lexicon, focus_hint, cfg)
    render_attempts = 1

    if prompt_meta and not args.legacy_prompt:
        (plan, lexicon, focus_hint, out, display_text, checks, verified, render_s,
         render_attempts, reasoning_stats, extra_reason_s) = _refine_prompt_length(
            renderer, scorer, adapter, prompt_meta, args, cfg, custom_lexicon,
            plan, lexicon, focus_hint, out, display_text, checks, verified, render_s,
            reasoning_stats)
        reasoning_s += extra_reason_s

    gpu = scorer.gpu_status()
    report = {
        'runtime': 'V14',
        'content_reasoner': ('Learned-Association-RuleVM-v6' if prompt_mode and not args.legacy_prompt else ('legacy' if prompt_mode else None)),
        'facts': len(plan),
        'sentences': len(out['sentences']),
        'paragraphs': len(out.get('paragraphs', [])),
        'display_chars': len(display_text),
        'induced_selected': out.get('induced_selected', 0),
        'load_seconds': round(load_s, 3),
        'reasoning_seconds_total': round(reasoning_s, 4),
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
        target_error = len(display_text) - int(prompt_meta['target_chars'])
        tolerance = max(70, int(int(prompt_meta['target_chars']) * 0.06))
        selected_rules = int(reasoning_stats.get('selected_rules', 0)) if reasoning_stats else 0
        expected_rule_budget = max(8, min(180, round(int(prompt_meta['target_chars']) / 74)))
        report['target_error_chars'] = target_error
        report['target_within_tolerance'] = abs(target_error) <= tolerance
        report['evidence_limited'] = bool(target_error < -tolerance and selected_rules < expected_rule_budget)
        if reasoning_stats:
            report['reasoning'] = _reasoning_summary(reasoning_stats)
    else:
        report['prompt_mode'] = False

    print(json.dumps(report, ensure_ascii=False, indent=2))
    print('\n--- TEXTO V14 ---\n')
    print(display_text)

    if args.raw_slots:
        print('\n--- REPRESENTAÇÃO SEMÂNTICA INTERNA ---\n')
        print(out['text'])
        if reasoning_stats:
            print('\n--- RULEBANK/PROVAS RESUMIDAS ---\n')
            print(json.dumps(_reasoning_summary(reasoning_stats), ensure_ascii=False, indent=2))
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(display_text, encoding='utf8')
        print(f'\nSalvo: {args.output}')


if __name__ == '__main__':
    main()
