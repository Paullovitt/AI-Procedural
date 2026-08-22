from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace
import json
import time

from prompt_runtime_v14 import PromptAdapterV14, contains_raw_slots
from persistent_memory_v14 import should_auto_store_user_memory
from run_gpu import (load_config, load_runtime, _apply_prompt_plan, _configure_console,
                     _reasoning_summary, _refine_prompt_length, _render_checked, _open_persistent_memory,
                     _persistent_memory_projection)

ROOT = Path(__file__).resolve().parent


class PromptSessionV14:
    """Persistent V14 prompt session: GPU tables and learned context index are reused."""
    def __init__(self, cfg=None):
        self.cfg = dict(cfg or load_config())
        self.adapter = PromptAdapterV14(
            int(self.cfg.get('default_target_chars', 2000)),
            argument_planner_enabled=bool(self.cfg.get('argument_planner_enabled', True)),
            robust_intake_enabled=bool(self.cfg.get('robust_semantic_intake_enabled', True)),
            robust_intake_warm_index=bool(self.cfg.get('robust_semantic_warm_index', True)),
            persistent_memory_inject_chars=int(self.cfg.get('persistent_memory_inject_chars',480)),
            persistent_memory_rule_cap=int(self.cfg.get('persistent_memory_rule_cap',4)),
        )
        t0 = time.perf_counter()
        self.scorer, self.grammar, self.inducer, self.renderer = load_runtime(
            self.cfg, lexicon={}, prompt_mode=True)
        if self.adapter.robust_intake_enabled:
            self.adapter._intake_for(self.scorer)
        self.load_seconds = time.perf_counter() - t0
        self.prompt_count = 0
        self.memory = _open_persistent_memory(self.cfg)

    def _reset_document_state(self, seed):
        if hasattr(self.renderer.structure, 'max_bundle'):
            self.renderer.structure.max_bundle = int(self.cfg.get('prompt_max_bundle', 4))
        schedule = getattr(self.renderer.structure, 'schedule', None)
        if schedule is not None and hasattr(schedule, 'i'):
            schedule.i = 0
        if hasattr(self.renderer.paragraph_scheduler, 'i'):
            self.renderer.paragraph_scheduler.i = 0
        rng = getattr(self.renderer.sel, 'rng', None)
        if rng is not None and hasattr(rng, 'seed'):
            rng.seed(int(seed))

    def generate(self, prompt, target_chars=None, seed=1234, custom_lexicon=None):
        prompt = str(prompt or '').strip()
        if not prompt:
            raise ValueError('Prompt vazio.')
        custom_lexicon = dict(custom_lexicon or {})
        self._reset_document_state(seed)

        memory_records=[]
        memory_index_text=prompt
        memory_semantic_keys=[]
        if self.memory is not None:
            memory_index_text,memory_semantic_keys=_persistent_memory_projection(self.adapter,self.scorer,prompt)
            memory_query_text=' '.join(memory_semantic_keys) or prompt
            memory_records=self.memory.search(memory_query_text,k=int(self.cfg.get('persistent_memory_top_k',4)),
                                              associative=bool(self.cfg.get('persistent_memory_associative',True)))
        t0 = time.perf_counter()
        pp = self.adapter.build_learned(prompt, self.scorer, target_chars=target_chars, seed=seed,
                                        memory_records=memory_records)
        reasoning_s = time.perf_counter() - t0
        plan, lexicon, focus_hint = _apply_prompt_plan(self.renderer, pp, custom_lexicon)
        out, display, checks, verified, render_s = _render_checked(
            self.renderer, plan, lexicon, focus_hint, self.cfg)

        args = SimpleNamespace(seed=int(seed))
        prompt_meta = {'prompt': pp.prompt, 'topic': pp.topic, 'target_chars': pp.target_chars,
                       'memory_records': memory_records}
        (plan, lexicon, focus_hint, out, display, checks, verified, render_s,
         attempts, reasoning_stats, extra_reason_s) = _refine_prompt_length(
            self.renderer, self.scorer, self.adapter, prompt_meta, args, self.cfg, custom_lexicon,
            plan, lexicon, focus_hint, out, display, checks, verified, render_s,
            pp.reasoning_stats)
        reasoning_s += extra_reason_s
        self.prompt_count += 1
        memory_store_result=None
        memory_store_skipped_reason=None
        if self.memory is not None and verified and bool(self.cfg.get('persistent_memory_auto_store_user',True)):
            if should_auto_store_user_memory(prompt):
                memory_store_result=self.memory.remember(prompt,source='user',
                    metadata={'runtime':'V14','topic':pp.topic,'session_prompt':self.prompt_count,
                              'semantic_keys':memory_semantic_keys[:32]},index_text=memory_index_text)
            else:
                memory_store_skipped_reason='interrogative'

        target_error = len(display) - int(pp.target_chars)
        tolerance = max(70, int(int(pp.target_chars) * 0.06))
        selected_rules = int(reasoning_stats.get('selected_rules', 0)) if reasoning_stats else 0
        expected_rule_budget = max(8, min(180, round(int(pp.target_chars) / 74)))
        report = {
            'runtime': 'V14',
            'content_reasoner': 'Learned-Association-RuleVM-v6',
            'argument_planner': reasoning_stats.get('argument_planner', {}).get('engine') if reasoning_stats else None,
            'session_prompt': self.prompt_count,
            'model_load_seconds_once': round(self.load_seconds, 3),
            'facts': len(plan),
            'sentences': len(out['sentences']),
            'paragraphs': len(out.get('paragraphs', [])),
            'display_chars': len(display),
            'target_chars': int(pp.target_chars),
            'target_error_chars': target_error,
            'target_within_tolerance': abs(target_error) <= tolerance,
            'evidence_limited': bool(target_error < -tolerance and selected_rules < expected_rule_budget),
            'reasoning_seconds_total': round(reasoning_s, 4),
            'render_seconds_total': round(render_s, 4),
            'render_attempts': attempts,
            'semantic_verified': verified,
            'slot_errors': checks['slot_errors'],
            'trace_errors': checks['trace_errors'],
            'raw_slot_ids_exposed': contains_raw_slots(display),
            'backend': out.get('compute_backend', 'cuda-batched-v14'),
            'reasoning': _reasoning_summary(reasoning_stats),
            'gpu': self.scorer.gpu_status(),
            'persistent_memory': ({
                'retrieved':len(memory_records),
                'retrieved_ids':[x.get('id') for x in memory_records],
                'stored':memory_store_result,
                'store_skipped_reason':memory_store_skipped_reason,
                'stats':self.memory.stats(),
            } if self.memory is not None else {'enabled':False}),
        }
        return report, display, out, reasoning_stats

    def close(self):
        if self.memory is not None:
            self.memory.close()
            self.memory=None

    def memory_status(self):
        return self.memory.stats() if self.memory is not None else {'enabled':False}

    def loop(self):
        _configure_console()
        status = self.scorer.gpu_status()
        print(f"V14 carregado uma vez em {self.load_seconds:.3f}s | {status['name']} | CUDA {status['cuda']}")
        print('Digite um prompt. Use /sair para encerrar.\n')
        while True:
            try:
                prompt = input('Prompt> ').strip()
            except (EOFError, KeyboardInterrupt):
                print()
                break
            if prompt.lower() in {'/sair','sair','exit','quit'}:
                break
            if prompt.lower() in {'/memoria','/memória','/memory'}:
                print(json.dumps(self.memory_status(),ensure_ascii=False,indent=2));print();continue
            if not prompt:
                continue
            try:
                report, text, _, _ = self.generate(prompt)
                print(json.dumps(report, ensure_ascii=False, indent=2))
                print('\n--- TEXTO V14 ---\n')
                print(text)
                print()
            except Exception as exc:
                print(f'ERRO: {exc}')
        self.close()


if __name__ == '__main__':
    PromptSessionV14().loop()
