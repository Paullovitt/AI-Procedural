from __future__ import annotations

from pathlib import Path
from collections import Counter
import json
import re
import statistics
import time

from prompt_session_v14 import PromptSessionV14
from prompt_runtime_v14 import WORD_RX

ROOT = Path(__file__).resolve().parent
OUT = ROOT / 'rigorous_results_v12'
OUT.mkdir(exist_ok=True)

PROMPTS = [
    ('espaco', 'Escreva 1000 caracteres sobre exploração espacial, tecnologia, descoberta e futuro.'),
    ('energia', 'Escreva 1000 caracteres sobre energia solar, baterias, armazenamento e futuro.'),
    ('agricultura', 'Escreva 1000 caracteres sobre agricultura sustentável, água, tecnologia e clima.'),
    ('musica', 'Escreva 1000 caracteres sobre música clássica, composição, orquestra e história.'),
    ('seguranca', 'Escreva 1000 caracteres sobre segurança digital, criptografia, senhas e privacidade.'),
    ('saude_publica', 'Escreva 1000 caracteres sobre saúde pública, prevenção, hospitais e qualidade.'),
]

SURFACE_MARKERS = (
    'compartilha contexto com', 'também surge junto de', 'aparece ainda no mesmo contexto que',
    'mantém proximidade contextual com', 'surge em contextos semelhantes a',
    'aparece em contexto próximo de', 'mantém contexto semelhante a',
    'aparece no mesmo contexto temático que', 'também surge no mesmo pedido que',
    'aparece ainda no pedido ao lado de', 'forma um núcleo contextual com',
)


def sentence_metrics(text: str):
    sentences = [x.strip() for x in re.split(r'(?<=[.!?])\s+', text) if x.strip()]
    normalized = [re.sub(r'\s+', ' ', x.lower()) for x in sentences]
    counts = Counter(normalized)
    marker_counts = {m: text.lower().count(m) for m in SURFACE_MARKERS}
    return {
        'sentences': len(sentences),
        'unique_sentence_ratio': len(counts) / max(1, len(sentences)),
        'max_exact_sentence_repeat': max(counts.values(), default=0),
        'max_surface_marker_repeat': max(marker_counts.values(), default=0),
        'surface_marker_counts': {k: v for k, v in marker_counts.items() if v},
    }


def main():
    session = PromptSessionV14()
    rows = []
    for i, (name, prompt) in enumerate(PROMPTS, 1):
        t0 = time.perf_counter()
        report, text, out, reasoning = session.generate(prompt, target_chars=1000, seed=1200+i)
        wall = time.perf_counter() - t0
        concepts = reasoning.get('seed_concepts', [])
        lower = text.lower()
        covered = [c for c in concepts if str(c).lower() in lower]
        metrics = sentence_metrics(text)
        row = {
            'name': name,
            'prompt': prompt,
            'wall_seconds': wall,
            'display_chars': len(text),
            'target_chars': 1000,
            'target_error_abs': abs(len(text)-1000),
            'target_error_ratio': abs(len(text)-1000)/1000.0,
            'target_within_tolerance': report.get('target_within_tolerance', False),
            'evidence_limited': report.get('evidence_limited', False),
            'semantic_verified': report['semantic_verified'],
            'slot_errors': report['slot_errors'],
            'trace_errors': report['trace_errors'],
            'raw_slot_ids_exposed': report['raw_slot_ids_exposed'],
            'reasoning_seconds': report['reasoning_seconds_total'],
            'render_seconds': report['render_seconds_total'],
            'rules': report['reasoning'].get('selected_rules', 0),
            'rule_learning_seconds': report['reasoning'].get('rule_learning_seconds', 0.0),
            'vm_seconds': report['reasoning'].get('vm_seconds', 0.0),
            'seed_concepts': concepts,
            'seed_concept_coverage': len(covered)/max(1, len(concepts)),
            'covered_concepts': covered,
            **metrics,
            'text': text,
        }
        rows.append(row)
        print(json.dumps({k:v for k,v in row.items() if k not in ('text','surface_marker_counts')}, ensure_ascii=False), flush=True)

    later = rows[1:] if len(rows) > 1 else rows
    result = {
        'format': 'Prompt-RuleVM-V6-Generality',
        'gpu': session.scorer.gpu_status(),
        'model_load_seconds_once': session.load_seconds,
        'prompts': len(rows),
        'all_semantic_verified': all(x['semantic_verified'] for x in rows),
        'total_slot_errors': sum(x['slot_errors'] for x in rows),
        'total_trace_errors': sum(x['trace_errors'] for x in rows),
        'any_raw_slot_ids': any(x['raw_slot_ids_exposed'] for x in rows),
        'mean_target_error_ratio': statistics.mean(x['target_error_ratio'] for x in rows),
        'max_target_error_ratio': max(x['target_error_ratio'] for x in rows),
        'target_within_tolerance_count': sum(bool(x['target_within_tolerance']) for x in rows),
        'evidence_limited_count': sum(bool(x['evidence_limited']) for x in rows),
        'unexplained_length_failures': sum((not x['target_within_tolerance']) and (not x['evidence_limited']) for x in rows),
        'mean_seed_concept_coverage': statistics.mean(x['seed_concept_coverage'] for x in rows),
        'mean_unique_sentence_ratio': statistics.mean(x['unique_sentence_ratio'] for x in rows),
        'mean_later_reasoning_ms': statistics.mean(x['reasoning_seconds'] for x in later)*1000.0,
        'max_later_reasoning_ms': max(x['reasoning_seconds'] for x in later)*1000.0,
        'mean_rule_learning_ms': statistics.mean(x['rule_learning_seconds'] for x in rows)*1000.0,
        'max_vm_ms': max(x['vm_seconds'] for x in rows)*1000.0,
        'rows': rows,
    }
    (OUT/'prompt_rulevm_v6_generality.json').write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding='utf8')
    print('SUMMARY', json.dumps({k:v for k,v in result.items() if k not in ('rows','gpu')}, ensure_ascii=False), flush=True)

    gates = [
        result['all_semantic_verified'], result['total_slot_errors']==0, result['total_trace_errors']==0,
        not result['any_raw_slot_ids'], result['mean_seed_concept_coverage'] >= 0.95,
        result['unexplained_length_failures'] == 0,
        result['mean_unique_sentence_ratio'] >= 0.95, result['max_later_reasoning_ms'] < 100.0,
        result['max_vm_ms'] < 5.0,
    ]
    if not all(gates):
        raise SystemExit('PROMPT RULEVM V6 GENERALITY GATE: FAIL')
    print('PROMPT RULEVM V6 GENERALITY GATE: OK')


if __name__ == '__main__':
    main()
