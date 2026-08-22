from __future__ import annotations

from pathlib import Path
import json
import statistics
import time
import unittest

import torch

from prompt_session_v14 import PromptSessionV14
from robust_semantic_intake_v14 import RobustNoiseLearnerV14, semantic_shadow

ROOT = Path(__file__).resolve().parent


def atoms(result):
    out = set()
    for x in result.semantic_atoms:
        raw = x[1:] if x.startswith('#') else x
        key = semantic_shadow(raw)
        if key:
            out.add(key)
    return out


@unittest.skipUnless(torch.cuda.is_available(), 'CUDA required by V14 scorer')
class RobustSemanticIntakeV14Tests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.session = PromptSessionV14()
        cls.intake = cls.session.adapter._intake_for(cls.session.scorer)
        cls.regressions = json.loads((ROOT/'robust_semantic_regressions_v14.json').read_text(encoding='utf8'))['cases']

    def test_permanent_regression_cases(self):
        for case in self.regressions:
            with self.subTest(case=case['id']):
                out = self.intake.extract(case['raw'])
                got = atoms(out)
                for item in case.get('required', []):
                    self.assertIn(semantic_shadow(item), got)
                for item in case.get('forbidden', []):
                    self.assertNotIn(semantic_shadow(item), got)
                if case.get('numbers'):
                    raw_numbers = {str(x['raw']) for x in out.numeric_anchors}
                    for number in case['numbers']:
                        self.assertIn(str(number), raw_numbers)
                edge = case.get('edge')
                if edge:
                    self.assertTrue(any(e.kind == edge['kind'] and semantic_shadow(e.source) == semantic_shadow(edge['source'])
                                        and semantic_shadow(e.target) == semantic_shadow(edge['target']) for e in out.edges))
                bridge = case.get('bridge')
                if bridge:
                    candidates = [e for e in out.edges if semantic_shadow(e.source) == semantic_shadow(bridge['source'])
                                  and semantic_shadow(e.target) == semantic_shadow(bridge['target'])]
                    self.assertTrue(candidates)
                    joined = [semantic_shadow(x) for x in candidates[0].bridge]
                    for token in bridge['contains']:
                        self.assertGreaterEqual(joined.count(semantic_shadow(token)), int(bridge.get('count', 1)))

    def test_clean_and_common_typo_paths_are_sub_5ms_hot(self):
        cases = [
            'energia solar baterias armazenamento futuro',
            'enerjia solar bateras armazenamnto futoro',
            'joao comprou 3 caro ontem e paguo 20 mil cada mas acho que ele tava falando de carros usados bla bla',
        ]
        for text in cases:
            self.intake.extract(text)
            samples = []
            for _ in range(30):
                t0 = time.perf_counter(); self.intake.extract(text); samples.append((time.perf_counter()-t0)*1000.0)
            self.assertLess(statistics.median(samples), 3.0)
            self.assertLess(sorted(samples)[28], 5.0)

    def test_noisy_and_clean_prompt_have_same_core_seeds(self):
        clean = self.session.adapter.build_learned(
            'Escreva 800 caracteres sobre energia solar, baterias, armazenamento e futuro.',
            self.session.scorer, target_chars=800, seed=1201)
        noisy = self.session.adapter.build_learned(
            'Escreva 800 caracteres sobre enerjia solar bateras armazenamnto e futoro!!!',
            self.session.scorer, target_chars=800, seed=1202)
        self.assertEqual(clean.reasoning_stats['seed_concepts'][:4], noisy.reasoning_stats['seed_concepts'][:4])
        self.assertEqual(noisy.reasoning_stats['seed_concepts'][:4], ['energia solar','baterias','armazenamento','futuro'])

    def test_raw_text_is_evidence_not_rewritten_output(self):
        raw = 'enerjia solar bateras'
        out = self.intake.extract(raw)
        self.assertEqual(out.raw_text, raw)
        projection = out.to_training_projection()
        self.assertNotIn('clean_text', projection)
        self.assertNotIn('corrected_text', projection)
        self.assertEqual(projection['format'], 'Robust-Semantic-Projection-V14')

    def test_valid_word_is_not_silently_corrected(self):
        out = self.intake.extract('joao comprou 3 caro e depois falou de carros usados')
        by_raw = {x.raw.casefold(): x for x in out.tokens if x.kind == 'word'}
        self.assertEqual(by_raw['caro'].canonical, 'caro')
        self.assertTrue(any(e.kind == 'local_echo_variant' and e.source == 'caro' and e.target == 'carros' for e in out.edges))

    def test_negation_and_double_negation_survive_as_bridge_evidence(self):
        one = self.intake.extract('joao nao comprou carro ontem')
        two = self.intake.extract('joao nao nao comprou carro ontem')
        def neg_count(result):
            edge = next(e for e in result.edges if e.source == 'joao' and e.target == 'comprou')
            return sum(semantic_shadow(x) == 'nao' for x in edge.bridge)
        self.assertEqual(neg_count(one), 1)
        self.assertEqual(neg_count(two), 2)

    def test_markup_mojibake_and_fragment_boundaries(self):
        html = self.intake.extract('<div> energia solar </div> **baterias**')
        self.assertNotIn('div', html.spine)
        self.assertTrue({'energia','solar','baterias'} <= atoms(html))
        mojibake = self.intake.extract('saÃºde pÃºblica prevencao hospitais')
        self.assertTrue({semantic_shadow(x) for x in ('saúde','pública','prevenção','hospitais')} <= atoms(mojibake))
        split = self.intake.extract('ener giasolar baterias')
        self.assertTrue({'energia','solar','baterias'} <= atoms(split))
        self.assertTrue(any(t.source == 'joined' and t.atoms == ('energia','solar') for t in split.tokens))

    def test_numeric_date_time_anchors_keep_raw_values(self):
        out = self.intake.extract('Maria comprou 3 carros em 21/08/2026 às 14:30 por 20.000 reais')
        kinds = {(x['raw'], x['kind']) for x in out.numeric_anchors}
        self.assertIn(('3','number'), kinds)
        self.assertIn(('21/08/2026','date'), kinds)
        self.assertIn(('14:30','time'), kinds)
        self.assertIn(('20.000','number'), kinds)

    def test_failure_archive_is_structurally_replayable(self):
        from robust_semantic_battery_v14 import FAILURE_ARCHIVE, load_cases
        valid = {case["id"] for case in load_cases()}
        rows = [json.loads(line) for line in FAILURE_ARCHIVE.read_text(encoding="utf8").splitlines() if line.strip()]
        self.assertGreater(len(rows), 0)
        for row in rows:
            self.assertIn(row.get("case"), valid)
            self.assertTrue(str(row.get("text", "")).strip())
            self.assertIsInstance(row.get("failure_classes", []), list)

    def test_noise_learner_promotes_recurring_oov_alias_only(self):
        learner = RobustNoiseLearnerV14(self.intake, min_support=3, min_dominance=.75, min_confidence=.72)
        learner.observe('joao comprou carro ontem e paguo 20 mil')
        learner.observe('maria paguo 10 reais ontem')
        learner.observe('cliente paguo 30 reais por unidade')
        learner.observe('pedro paguo 40 cada item')
        learner.observe('caro carro usado')  # valid vocabulary token must never become a global alias
        bank = learner.promote()
        alias = bank.get('paguo')
        self.assertIsNotNone(alias)
        self.assertEqual(alias.canonical, 'pagou')
        self.assertIsNone(bank.get('caro'))


if __name__ == '__main__':
    unittest.main(verbosity=2)
