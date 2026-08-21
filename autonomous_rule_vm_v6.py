from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any
import json
import math
import re
import time

import torch


_TOKEN_OK = re.compile(r"^[A-Za-zÀ-ÖØ-öø-ÿ][A-Za-zÀ-ÖØ-öø-ÿ_-]{2,}$", re.UNICODE)


@dataclass(frozen=True)
class AssociationRule:
    source: str
    target: str
    predicate: str
    kind: str
    confidence: float
    support: int
    score: float
    evidence: tuple[str, ...] = ()

    def as_dict(self):
        return {
            'source': self.source,
            'target': self.target,
            'predicate': self.predicate,
            'kind': self.kind,
            'confidence': float(self.confidence),
            'support': int(self.support),
            'score': float(self.score),
            'evidence': list(self.evidence),
        }


class LearnedAssociationRuleBank:
    """Compact explicit rule bank learned from persisted corpus statistics."""
    def __init__(self, rules=(), metadata=None):
        dedup = {}
        for row in rules:
            r = row if isinstance(row, AssociationRule) else AssociationRule(**row)
            key = (r.source.lower(), r.target.lower(), r.kind)
            old = dedup.get(key)
            if old is None or (r.score, r.support) > (old.score, old.support):
                dedup[key] = r
        self.rules = sorted(dedup.values(), key=lambda r: (r.score, r.confidence, r.support), reverse=True)
        self.metadata = dict(metadata or {})
        self.by_source = defaultdict(list)
        for i, rule in enumerate(self.rules):
            self.by_source[rule.source.lower()].append(i)

    def save(self, path):
        data = {
            'format': 'Learned-Association-RuleBank-v6',
            'metadata': self.metadata,
            'rules': [r.as_dict() for r in self.rules],
        }
        Path(path).write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding='utf8')
        return data

    def status(self):
        kinds = defaultdict(int)
        for r in self.rules:
            kinds[r.kind] += 1
        return {
            'format': 'Learned-Association-RuleBank-v6',
            'rules': len(self.rules),
            'indexed_sources': len(self.by_source),
            'kinds': dict(kinds),
            **self.metadata,
        }


class CorpusAssociationRuleLearnerGPU:
    """Learns compact association rules from Bagaço p2 statistics on CUDA.

    No domain relation is encoded in the learner. For arbitrary prompt concepts it
    gathers observed corpus neighborhoods, computes positive-PMI evidence on GPU,
    promotes a bounded set of rules, and uses one CUDA matrix multiplication for
    context-similarity rules.
    """
    PREDICATE_NEIGHBOR = 'compartilha contexto com'
    PREDICATE_SIMILAR = 'surge em contextos semelhantes a'
    PREDICATE_PROMPT = 'aparece no mesmo contexto temático que'
    PREDICATE_CLUSTER = 'forma um núcleo contextual com'

    def __init__(self, scorer, max_unigram_fraction=0.0015, min_bigram_count=4,
                 context_cap=96, max_discourse_ratio=0.08, max_neighbor_degree=144,
                 min_rule_support=40, phrase_min_rule_support=8,
                 relative_weight_floor=0.32, device=None):
        self.s = scorer
        self.device = torch.device(device or getattr(scorer, 'device', 'cuda:0'))
        self.max_unigram_fraction = float(max_unigram_fraction)
        self.min_bigram_count = int(min_bigram_count)
        self.context_cap = int(context_cap)
        self.max_discourse_ratio = float(max_discourse_ratio)
        self.max_neighbor_degree = int(max_neighbor_degree)
        self.min_rule_support = int(min_rule_support)
        self.phrase_min_rule_support = int(phrase_min_rule_support)
        self.relative_weight_floor = float(relative_weight_floor)
        t0 = time.perf_counter()
        self.adj = defaultdict(list)
        for gram, count in self.s.p2.items():
            parts = gram.split('\t')
            if len(parts) != 2:
                continue
            a, b = parts
            c = int(count)
            # direction says where the neighbor occurs relative to the source head.
            self.adj[a].append((b, c, 'right', gram))
            self.adj[b].append((a, c, 'left', gram))
        # Trigram evidence completes terse bigrams without a POS tagger. Highest support wins.
        self.phrase_completion = {}
        self.bridge_completion = {}
        for gram, count in self.s.p3.items():
            w = gram.split('\t')
            if len(w) != 3:
                continue
            a, b, c = w
            count = int(count)
            for pair, added in (((a, b), c), ((b, c), a)):
                previous = self.phrase_completion.get(pair)
                if previous is None or count > previous[0]:
                    self.phrase_completion[pair] = (count, ' '.join(w), added)
            pair = (a, c)
            previous = self.bridge_completion.get(pair)
            if previous is None or count > previous[0]:
                self.bridge_completion[pair] = (count, ' '.join(w), b)

        # Compound contexts are discovered lazily for only the phrases present in a
        # prompt, then cached for the persistent session. This avoids a large global
        # p3-p5 phrase index while preserving exact-phrase disambiguation.
        self._phrase_context_cache = {}
        self.index_seconds = time.perf_counter() - t0

    def _head(self, concept):
        words = self.s.tokenize(str(concept))
        return words[-1] if words else str(concept).lower()

    def _exact_phrase_support(self, words):
        n = len(words)
        if n < 2 or n > 5:
            return 0
        table = self.s.tables.get(f'p{n}', {})
        return int(table.get('\t'.join(words), 0))

    def _phrase_rows(self, concept, words):
        n = len(words)
        if n < 2 or n >= 5:
            return ()
        key = ' '.join(words)
        hit = self._phrase_context_cache.get(key)
        if hit is not None:
            return hit
        phrase_tab = '\t'.join(words)
        best = {}
        for order in range(n + 1, 6):
            table = self.s.tables.get(f'p{order}', {})
            for gram, raw_count in table.items():
                if phrase_tab not in gram:
                    continue
                parts = gram.split('\t')
                count = int(raw_count)
                if count < self.min_bigram_count:
                    continue
                for i in range(len(parts) - n + 1):
                    if parts[i:i+n] != words:
                        continue
                    positions = []
                    for d in range(1, len(parts)):
                        left = i - d
                        right = i + n - 1 + d
                        if left >= 0:
                            positions.append(left)
                        if right < len(parts):
                            positions.append(right)
                    neighbor = next((parts[j] for j in positions if self._candidate_ok(parts[j], words[-1])), None)
                    if neighbor is None:
                        continue
                    old = best.get(neighbor)
                    row = (neighbor, count, 'phrase', gram)
                    if old is None or count > old[1]:
                        best[neighbor] = row
        rows = tuple(sorted(best.values(), key=lambda x: x[1], reverse=True))
        self._phrase_context_cache[key] = rows
        return rows

    def _candidate_ok(self, token, source_head):
        if token == source_head or not _TOKEN_OK.match(token):
            return False
        count = int(self.s.tok.get(token, 0))
        if count <= 0:
            return False
        if (count / max(1.0, float(self.s.total_tok))) > self.max_unigram_fraction:
            return False
        # Data-driven boilerplate/discourse filter: tokens disproportionately used at
        # openings/connections are poor content expansions. No word list is encoded.
        discourse = int(self.s.tables.get('open', {}).get(token, 0)) + int(self.s.tables.get('connect', {}).get(token, 0))
        if discourse / max(1, count) > self.max_discourse_ratio:
            return False
        # Extremely broad context hubs tend to be generic function/discourse words.
        if len(self.adj.get(token, ())) > self.max_neighbor_degree:
            return False
        return True

    def _phrase_target(self, source, head, neighbor, direction):
        pair = (head, neighbor) if direction == 'right' else (neighbor, head)
        phrase = ' '.join(pair)
        bridge = self.bridge_completion.get(pair)
        if bridge is not None and bridge[0] >= self.min_bigram_count:
            phrase = bridge[1]
        else:
            completion = self.phrase_completion.get(pair)
            if completion is not None and self._candidate_ok(completion[2], head):
                phrase = completion[1]
        phrase = ' '.join(phrase.split())
        if phrase.lower() == str(source).lower():
            return None
        return phrase

    def _weighted_neighbors_many(self, concepts, topk=None):
        """Batch all positive-PMI calculations for the requested concepts on CUDA."""
        topk = int(topk or self.context_cap)
        groups = []
        flat_c = []
        flat_src = []
        flat_dst = []
        flat_meta = []
        for concept in concepts:
            words = self.s.tokenize(str(concept))
            head = words[-1] if words else str(concept).lower()
            start = len(flat_c)
            phrase_mode = len(words) >= 2
            phrase_support = self._exact_phrase_support(words) if phrase_mode else 0
            # Multiword concepts never silently collapse to their final token. When the
            # exact phrase exists in p2-p5, only context containing that whole phrase is
            # eligible. Sparse/unseen phrases deliberately yield no corpus expansion.
            if phrase_mode:
                source_rows = self._phrase_rows(concept, words) if phrase_support > 0 else ()
                src_count = max(1, phrase_support)
            else:
                source_rows = self.adj.get(head, ())
                src_count = max(1, int(self.s.tok.get(head, 1)))
            for neighbor, count, direction, evidence in source_rows:
                if count < self.min_bigram_count or not self._candidate_ok(neighbor, head):
                    continue
                target = neighbor if phrase_mode else self._phrase_target(concept, head, neighbor, direction)
                if not target:
                    continue
                flat_c.append(float(count))
                flat_src.append(float(src_count))
                flat_dst.append(float(max(1, int(self.s.tok.get(neighbor, 1)))))
                flat_meta.append((target, neighbor, int(count), evidence, direction))
            groups.append((str(concept), head, start, len(flat_c)))

        if not flat_c:
            return {str(c): [] for c in concepts}

        c = torch.as_tensor(flat_c, dtype=torch.float64, device=self.device)
        sc = torch.as_tensor(flat_src, dtype=torch.float64, device=self.device)
        dc = torch.as_tensor(flat_dst, dtype=torch.float64, device=self.device)
        total = torch.tensor(float(self.s.total_tok), dtype=torch.float64, device=self.device)
        pmi = torch.log(torch.clamp(c * total / (sc * dc), min=1e-30))
        weights = torch.clamp(pmi, min=0.0) * torch.sqrt(c)
        weights_np = weights.detach().cpu().numpy()

        out = {}
        for concept, head, lo, hi in groups:
            rows = []
            for pos in range(lo, hi):
                weight = float(weights_np[pos])
                if weight <= 0:
                    continue
                target, neighbor, support, evidence, direction = flat_meta[pos]
                rows.append((weight, support, target, neighbor, evidence, direction))
            rows.sort(reverse=True)
            out[concept] = rows[:topk]
        return out

    def _similarity_rules(self, concepts, contexts, max_rules):
        concepts = list(dict.fromkeys(str(x) for x in concepts))
        if len(concepts) < 2 or max_rules <= 0:
            return []
        vocab = {}
        for concept in concepts:
            for _, _, _, neighbor, _, _ in contexts.get(concept, ()):
                if neighbor not in vocab:
                    vocab[neighbor] = len(vocab)
        if not vocab:
            return []
        mat = torch.zeros((len(concepts), len(vocab)), dtype=torch.float32, device=self.device)
        support_maps = []
        for i, concept in enumerate(concepts):
            sm = {}
            for weight, support, _, neighbor, _, _ in contexts.get(concept, ()):
                mat[i, vocab[neighbor]] = float(weight)
                sm[neighbor] = int(support)
            support_maps.append(sm)
        norm = torch.linalg.vector_norm(mat, dim=1, keepdim=True).clamp_min(1e-12)
        sim = ((mat / norm) @ (mat / norm).T).detach().cpu().numpy()
        rows = []
        for i in range(len(concepts)):
            for j in range(i + 1, len(concepts)):
                score = float(sim[i, j])
                if score <= 0:
                    continue
                shared = set(support_maps[i]) & set(support_maps[j])
                if not shared:
                    continue
                support = sum(min(support_maps[i][x], support_maps[j][x]) for x in shared)
                evidence = tuple(sorted(shared, key=lambda x: min(support_maps[i][x], support_maps[j][x]), reverse=True)[:4])
                rows.append((score, support, i, j, evidence))
        rows.sort(reverse=True)
        out = []
        for score, support, i, j, evidence in rows[:max_rules]:
            conf = min(0.99, max(0.05, score))
            out.append(AssociationRule(concepts[i], concepts[j], self.PREDICATE_SIMILAR,
                                       'context_similarity', conf, support, score, evidence))
        return out

    def fit(self, concepts, rule_budget=28, expansion_depth=1, expansion_per_node=None):
        t0 = time.perf_counter()
        seeds = list(dict.fromkeys(str(x).strip() for x in concepts if str(x).strip()))
        budget = max(4, min(256, int(rule_budget)))
        if not seeds:
            return LearnedAssociationRuleBank(metadata={'learn_seconds': 0.0, 'gpu': str(self.device)})

        max_pairs = len(seeds) * (len(seeds) - 1) // 2
        pair_reserve = min(max_pairs, max(1, budget // 5)) if max_pairs else 0
        neighbor_budget = max(1, budget - pair_reserve)
        per_node = int(expansion_per_node or max(2, min(8, math.ceil(neighbor_budget / max(1, len(seeds))))) )
        rules = []
        seen_targets = defaultdict(set)
        frontier = list(seeds)
        context_cache = {}

        # One-hop expansion is the promoted default: it keeps corpus evidence anchored
        # to prompt concepts and avoids topic drift. Depth remains configurable for tests.
        for _depth in range(max(1, int(expansion_depth))):
            if not frontier or len(rules) >= neighbor_budget:
                break
            contexts = self._weighted_neighbors_many(frontier, topk=self.context_cap)
            context_cache.update(contexts)
            next_frontier = []
            for source in frontier:
                rows = contexts.get(source, ())
                if not rows:
                    continue
                maxw = max((x[0] for x in rows), default=1.0)
                chosen = 0
                for weight, support, target, neighbor, evidence, _direction in rows:
                    if len(rules) >= neighbor_budget or chosen >= per_node:
                        break
                    if weight < maxw * self.relative_weight_floor:
                        break
                    required_support = self.phrase_min_rule_support if _direction == 'phrase' else self.min_rule_support
                    if support < required_support:
                        continue
                    if target.lower() in seen_targets[source.lower()]:
                        continue
                    seen_targets[source.lower()].add(target.lower())
                    conf = min(0.995, max(0.10, float(weight / max(maxw, 1e-12))))
                    rules.append(AssociationRule(source, target, self.PREDICATE_NEIGHBOR,
                                                 'corpus_neighbor', conf, support, float(weight), (evidence,)))
                    chosen += 1
                    next_frontier.append(target)
            frontier = next_frontier[:neighbor_budget]

        missing = [x for x in seeds if x not in context_cache]
        if missing:
            context_cache.update(self._weighted_neighbors_many(missing, topk=self.context_cap))
        pair_rules = self._similarity_rules(seeds, context_cache, pair_reserve)
        rules.extend(pair_rules)

        # Induce one higher-level cluster rule from multiple promoted neighbors. This is
        # a generic aggregation of learned evidence, not a domain conclusion.
        by_source = defaultdict(list)
        for rule in rules:
            if rule.kind == 'corpus_neighbor':
                by_source[rule.source].append(rule)
        for source, source_rules in by_source.items():
            source_rules.sort(key=lambda r: (r.confidence, r.score, r.support), reverse=True)
            if len(source_rules) < 2:
                continue
            top = source_rules[:3]
            target = ', '.join(r.target for r in top[:-1]) + (' e ' + top[-1].target if len(top) > 1 else top[0].target)
            rules.append(AssociationRule(source, target, self.PREDICATE_CLUSTER, 'cluster_summary',
                                         sum(r.confidence for r in top)/len(top),
                                         sum(r.support for r in top),
                                         sum(r.score for r in top)/len(top),
                                         tuple(r.evidence[0] for r in top if r.evidence)))

        # Prompt co-occurrence is an observed fact, not a fabricated domain rule. Keep
        # these anchors even when the learned-neighbor candidate pool is saturated.
        prompt_rules = []
        if len(seeds) > 1:
            root = seeds[0]
            for target in seeds[1:]:
                prompt_rules.append(AssociationRule(root, target, self.PREDICATE_PROMPT, 'prompt_observation',
                                                    1.0, 1, 1.0, ('same_prompt',)))

        budget = min(256, max(budget, len(prompt_rules)))
        learned_ranked = LearnedAssociationRuleBank(rules).rules
        learned_slots = max(0, budget - len(prompt_rules))
        selected_rules = list(learned_ranked[:learned_slots]) + prompt_rules

        torch.cuda.synchronize(self.device)
        dt = time.perf_counter() - t0
        return LearnedAssociationRuleBank(selected_rules, metadata={
            'learn_seconds': dt,
            'index_seconds': self.index_seconds,
            'gpu': str(self.device),
            'rule_budget': budget,
            'seed_concepts': len(seeds),
            'expansion_depth': int(expansion_depth),
            'expansion_per_node': per_node,
            'pair_rule_reserve': pair_reserve,
            'prompt_rule_reserve': len(prompt_rules),
        })


class IndexedAssociationRuleVM:
    """Simple indexed VM for learned association rules with provenance."""
    def __init__(self, rulebank: LearnedAssociationRuleBank):
        self.bank = rulebank

    def execute(self, seeds, max_depth=2, max_rules=64, min_path_confidence=0.01):
        t0 = time.perf_counter()
        seeds = list(dict.fromkeys(str(x) for x in seeds))
        best = {x.lower(): 1.0 for x in seeds}
        labels = {x.lower(): x for x in seeds}
        parent: dict[str, Any] = {x.lower(): None for x in seeds}
        frontier = [(x, 0) for x in seeds]
        fired = []
        fired_ids = set()

        while frontier and len(fired) < int(max_rules):
            source, depth = frontier.pop(0)
            if depth >= int(max_depth):
                continue
            source_key = source.lower()
            base_conf = best.get(source_key, 0.0)
            for ridx in self.bank.by_source.get(source_key, ()):
                if len(fired) >= int(max_rules):
                    break
                rule = self.bank.rules[ridx]
                path_conf = base_conf * float(rule.confidence)
                if path_conf < float(min_path_confidence):
                    continue
                if ridx not in fired_ids:
                    fired_ids.add(ridx)
                    fired.append((rule, path_conf, depth + 1))
                target_key = rule.target.lower()
                old = best.get(target_key, -1.0)
                if path_conf > old + 1e-12:
                    best[target_key] = path_conf
                    labels[target_key] = rule.target
                    parent[target_key] = (source, ridx, path_conf)
                    frontier.append((rule.target, depth + 1))

        fired.sort(key=lambda row: (row[1], row[0].score, row[0].support), reverse=True)
        dt = time.perf_counter() - t0
        return {
            'nodes': [labels[k] for k in sorted(labels, key=lambda x: best[x], reverse=True)],
            'confidence': {labels[k]: float(v) for k, v in best.items()},
            'parent': parent,
            'fired': fired[:int(max_rules)],
            'vm_seconds': dt,
            'indexed_lookups': len(fired),
        }

    def proof(self, execution, target):
        key = str(target).lower()
        out = []
        seen = set()
        parent = execution.get('parent', {})
        while key in parent and parent[key] is not None and key not in seen:
            seen.add(key)
            source, ridx, conf = parent[key]
            rule = self.bank.rules[int(ridx)]
            out.append({'source': source, 'rule': rule.as_dict(), 'path_confidence': float(conf)})
            key = str(source).lower()
        out.reverse()
        return out


def benchmark_with_scorer(scorer, concepts, rule_budget=32, rounds=200):
    learner = CorpusAssociationRuleLearnerGPU(scorer)
    t0 = time.perf_counter();bank = learner.fit(concepts, rule_budget=rule_budget);learn_s = time.perf_counter() - t0
    vm = IndexedAssociationRuleVM(bank);seeds = list(dict.fromkeys(str(x) for x in concepts))
    t0 = time.perf_counter();last = None
    for _ in range(int(rounds)):
        last = vm.execute(seeds, max_depth=2, max_rules=rule_budget)
    infer_s = time.perf_counter() - t0
    return {
        'format': 'Association-RuleVM-GPU-v6-benchmark',
        'rules': len(bank.rules),
        'learn_seconds': learn_s,
        'vm_rounds': int(rounds),
        'vm_total_seconds': infer_s,
        'vm_mean_ms': (infer_s / max(1, int(rounds))) * 1000.0,
        'reachable_nodes': len(last['nodes']) if last else 0,
        'rulebank': bank.status(),
    }
