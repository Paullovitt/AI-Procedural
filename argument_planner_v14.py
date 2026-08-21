from __future__ import annotations

from collections import Counter, defaultdict
import math
import re
import time
import unicodedata


_ROLE_ORDER = {'opening': 0, 'development': 1, 'synthesis': 2}


class EvidenceArgumentPlannerV14:
    """Evidence-driven discourse planner for already learned RuleVM facts.

    The planner never invents facts and never encodes domain laws. It only schedules
    rules that were already learned/promoted by the RuleBank. Ordering uses explicit
    evidence carried by each rule (path confidence, support, score, depth), prompt
    concept coverage and graph source locality.
    """

    SYNTHESIS_KINDS = frozenset({'cluster_summary', 'context_similarity'})

    @staticmethod
    def _evidence_key(row):
        rule, path_conf, depth = row
        return (
            float(path_conf),
            float(rule.confidence),
            math.log1p(max(0, int(rule.support))),
            float(rule.score),
            -int(depth),
        )

    @staticmethod
    def _dedupe(rows):
        seen = set()
        out = []
        for row in rows:
            key = id(row[0])
            if key in seen:
                continue
            seen.add(key)
            out.append(row)
        return out

    @staticmethod
    def _norm_word(word):
        raw=''.join(c for c in unicodedata.normalize('NFKD',str(word).lower()) if not unicodedata.combining(c))
        return ''.join(c for c in raw if c.isalnum())

    @classmethod
    def _trigrams(cls, word):
        word=cls._norm_word(word)
        return {word[i:i+3] for i in range(max(0,len(word)-2))}

    @classmethod
    def _prompt_affinity(cls, row, seeds):
        rule=row[0]
        source_words={cls._norm_word(x) for x in re.findall(r'[^\W_]+',str(rule.source),re.UNICODE)}
        target_words=[cls._norm_word(x) for x in re.findall(r'[^\W_]+',str(rule.target),re.UNICODE)]
        target_words=[x for x in target_words if len(x)>=4 and x not in source_words]
        other_words=[]
        source_key=str(rule.source).lower()
        for seed in seeds:
            if str(seed).lower()==source_key:
                continue
            other_words.extend(cls._norm_word(x) for x in re.findall(r'[^\W_]+',str(seed),re.UNICODE))
        best_shared=0;best_jaccard=0.0
        for a in target_words:
            ga=cls._trigrams(a)
            if not ga: continue
            for b in other_words:
                gb=cls._trigrams(b)
                if not gb: continue
                shared=len(ga & gb)
                jac=shared/max(1,len(ga | gb))
                if (shared,jac)>(best_shared,best_jaccard):
                    best_shared,best_jaccard=shared,jac
        return best_shared,best_jaccard

    def _balanced_development(self, rows, seeds):
        """Round-robin evidence; a dominant lexical prompt signal disambiguates a source."""
        seed_index = {str(x).lower(): i for i, x in enumerate(seeds)}
        by_source = defaultdict(list)
        for row in rows:
            by_source[str(row[0].source).lower()].append(row)
        rejected=set();filtered_sources=set()
        for source,source_rows in list(by_source.items()):
            if len(source_rows)==1 and source_rows[0][0].kind=='corpus_neighbor':
                isolated_affinity=self._prompt_affinity(source_rows[0],seeds)
                if isolated_affinity[0] < 2:
                    rejected.add(id(source_rows[0][0]))
                    by_source[source]=[]
                    continue
            scored=[(self._prompt_affinity(row,seeds),row) for row in source_rows]
            scored.sort(key=lambda item:(item[0][0],item[0][1],self._evidence_key(item[1])),reverse=True)
            # If one morphological/contextual family clearly dominates (for example
            # musica -> musical), keep that prompt-aligned family for realization.
            # The discarded learned rules remain in the RuleBank; only discourse selection changes.
            if scored:
                top_shared=scored[0][0][0]
                second_shared=scored[1][0][0] if len(scored)>1 else 0
                if top_shared>=2 and top_shared>second_shared:
                    keep=[row for affinity,row in scored if affinity[0]==top_shared]
                    keep_ids={id(row[0]) for row in keep}
                    rejected.update(id(row[0]) for _,row in scored if id(row[0]) not in keep_ids)
                    filtered_sources.add(source)
                    source_rows=keep
                else:
                    source_rows=[row for _,row in scored]
            by_source[source]=source_rows

        ordered_sources = [str(x).lower() for x in seeds if str(x).lower() in by_source]
        remaining_sources = [x for x in by_source if x not in seed_index]
        remaining_sources.sort(
            key=lambda x: self._evidence_key(by_source[x][0]) if by_source[x] else (),
            reverse=True,
        )
        ordered_sources.extend(remaining_sources)

        out = []
        cursor = 0
        while True:
            added = False
            for source in ordered_sources:
                rows_for_source = by_source[source]
                if cursor < len(rows_for_source):
                    out.append(rows_for_source[cursor])
                    added = True
            if not added:
                break
            cursor += 1
        return out,rejected,filtered_sources

    def plan(self, fired, seeds, emit_budget):
        t0 = time.perf_counter()
        fired = self._dedupe(list(fired))
        seeds = list(dict.fromkeys(str(x) for x in seeds))
        seed_index = {x.lower(): i for i, x in enumerate(seeds)}
        budget = max(1, int(emit_budget))

        opening = [row for row in fired if row[0].kind == 'prompt_observation']
        synthesis = [row for row in fired if row[0].kind in self.SYNTHESIS_KINDS]
        development = [
            row for row in fired
            if row[0].kind != 'prompt_observation' and row[0].kind not in self.SYNTHESIS_KINDS
        ]

        opening.sort(key=lambda row: (
            -seed_index.get(str(row[0].target).lower(), len(seed_index) + 1),
            self._evidence_key(row),
        ), reverse=True)
        development, context_rejected, filtered_sources = self._balanced_development(development, seeds)
        synthesis_before=len(synthesis)
        synthesis_rejected={id(row[0]) for row in synthesis if str(row[0].source).lower() in filtered_sources}
        synthesis=[row for row in synthesis if id(row[0]) not in synthesis_rejected]
        context_filtered_synthesis=synthesis_before-len(synthesis)
        synthesis.sort(key=lambda row: (
            -seed_index.get(str(row[0].source).lower(), len(seed_index) + 1),
            self._evidence_key(row),
        ), reverse=True)

        # Prompt observations are never discarded. If room remains, reserve a small
        # tail for explicit learned synthesis and spend the rest on balanced evidence.
        effective_budget = max(budget, len(opening))
        slots = max(0, effective_budget - len(opening))
        synth_slots = 0
        if synthesis and slots >= 3:
            synth_slots = min(len(synthesis), max(1, slots // 5))
        dev_slots = max(0, slots - synth_slots)

        selected_opening = opening
        selected_development = development[:dev_slots]
        selected_synthesis = synthesis[:synth_slots]

        # If a phase has unused capacity, fill it with the strongest still-unselected
        # rules while retaining monotonic phase order.
        used = {id(row[0]) for row in selected_opening + selected_development + selected_synthesis}
        remaining = [row for row in fired if id(row[0]) not in used and id(row[0]) not in context_rejected and id(row[0]) not in synthesis_rejected]
        remaining.sort(key=self._evidence_key, reverse=True)
        spare = max(0, effective_budget - len(used))
        if spare:
            for row in remaining[:spare]:
                if row[0].kind in self.SYNTHESIS_KINDS:
                    selected_synthesis.append(row)
                else:
                    selected_development.append(row)

        planned = (
            [(row, 'opening') for row in selected_opening]
            + [(row, 'development') for row in selected_development]
            + [(row, 'synthesis') for row in selected_synthesis]
        )
        planned = planned[:max(effective_budget, len(selected_opening))]

        dev_sources = {
            str(row[0].source).lower()
            for row, role in planned if role == 'development'
        }
        prompt_sources = {x.lower() for x in seeds}
        coverage = len(dev_sources & prompt_sources) / max(1, len(prompt_sources))
        phase_counts = Counter(role for _, role in planned)
        kinds = Counter(row[0].kind for row, _ in planned)
        phase_sequence = [role for _, role in planned]
        monotonic = all(
            _ROLE_ORDER.get(a, 1) <= _ROLE_ORDER.get(b, 1)
            for a, b in zip(phase_sequence, phase_sequence[1:])
        )
        dt = time.perf_counter() - t0
        return {
            'planned': planned,
            'stats': {
                'engine': 'Evidence-Argument-Planner-V14',
                'planner_seconds': dt,
                'input_rules': len(fired),
                'selected_rules': len(planned),
                'dropped_rules': max(0, len(fired) - len(planned)),
                'phase_counts': dict(phase_counts),
                'selected_kinds': dict(kinds),
                'seed_development_coverage': coverage,
                'phase_monotonic': monotonic,
                'synthesis_reserved': synth_slots,
                'context_filtered_rules': len(context_rejected),
                'context_filtered_synthesis': context_filtered_synthesis,
                'context_filtered_sources': sorted(filtered_sources),
            },
        }
