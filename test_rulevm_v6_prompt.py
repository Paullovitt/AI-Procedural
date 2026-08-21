from __future__ import annotations

import unittest

import torch

from prompt_session_v14 import PromptSessionV14


@unittest.skipUnless(torch.cuda.is_available(), 'CUDA required by V14/RuleVM V6')
class RuleVMV6PromptTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.session = PromptSessionV14()

    def test_rulebank_is_learned_indexed_and_fast(self):
        pp = self.session.adapter.build_learned(
            'Escreva 800 caracteres sobre exploração espacial, tecnologia, descoberta e futuro.',
            self.session.scorer,
            target_chars=800,
            seed=901,
        )
        stats = pp.reasoning_stats
        self.assertEqual(stats.get('engine'), 'Learned-Association-RuleVM-v6')
        self.assertGreater(stats.get('selected_rules', 0), 0)
        self.assertGreater(stats.get('indexed_lookups', 0), 0)
        self.assertLess(float(stats.get('vm_seconds', 1.0)), 0.01)
        bank = stats.get('rulebank', {})
        self.assertEqual(bank.get('gpu'), 'cuda:0')
        self.assertLess(float(bank.get('learn_seconds', 1.0)), 0.25)
        self.assertIn('prompt_observation', bank.get('kinds', {}))

    def test_compound_topic_does_not_fallback_to_ambiguous_head(self):
        pp = self.session.adapter.build_learned(
            'Escreva 800 caracteres sobre música clássica, composição, orquestra e história.',
            self.session.scorer,
            target_chars=800,
            seed=902,
        )
        rows = pp.reasoning_stats.get('rules', [])
        root_rows = [r for r in rows if r.get('source', '').lower() == 'música clássica']
        targets = ' | '.join(r.get('target', '').lower() for r in root_rows)
        # Regression: the old head-only path treated "clássica" as the topic and
        # promoted unrelated phrases such as "antiguidade clássica".
        self.assertNotIn('antiguidade clássica', targets)
        self.assertNotIn('guitarra clássica', targets)

    def test_persistent_session_reuses_context_index(self):
        self.session.generate(
            'Escreva 700 caracteres sobre exploração espacial, tecnologia e futuro.',
            target_chars=700,
            seed=903,
        )
        report, _, _, _ = self.session.generate(
            'Escreva 700 caracteres sobre agricultura sustentável, água e clima.',
            target_chars=700,
            seed=904,
        )
        self.assertTrue(report['semantic_verified'])
        self.assertEqual(report['slot_errors'], 0)
        self.assertEqual(report['trace_errors'], 0)
        self.assertFalse(report['raw_slot_ids_exposed'])
        self.assertLess(float(report['reasoning_seconds_total']), 0.10)
        self.assertLess(float(report['reasoning']['vm_seconds']), 0.01)

    def test_sparse_prompt_reports_evidence_limit_instead_of_weak_rules(self):
        report, text, _, stats = self.session.generate(
            'Escreva 1000 caracteres sobre segurança digital, criptografia, senhas e privacidade.',
            target_chars=1000,
            seed=905,
        )
        self.assertTrue(report['semantic_verified'])
        self.assertFalse(report['raw_slot_ids_exposed'])
        self.assertTrue(report['evidence_limited'])
        self.assertFalse(report['target_within_tolerance'])
        self.assertGreater(len(text), 0)
        self.assertLess(stats.get('selected_rules', 999), 14)


    def test_three_word_compound_uses_exact_phrase_context_only(self):
        learner = self.session.adapter._learner_for(self.session.scorer)
        concept = 'estação espacial internacional'
        words = self.session.scorer.tokenize(concept)
        self.assertEqual(len(words), 3)
        self.assertGreater(learner._exact_phrase_support(words), 0)
        self.assertGreater(len(learner.adj.get(words[-1], ())), 0)
        rows = learner._weighted_neighbors_many([concept]).get(concept, [])
        phrase_tab = '\t'.join(words)
        for row in rows:
            self.assertIn(phrase_tab, row[4])

    def test_prompt_observations_survive_small_emit_budget(self):
        pp = self.session.adapter.build_learned(
            'Escreva 200 caracteres sobre tecnologia, futuro, ciência, energia, espaço, música, história, saúde, água, clima, segurança e educação.',
            self.session.scorer,
            target_chars=200,
            seed=906,
        )
        seeds = pp.reasoning_stats.get('seed_concepts', [])
        prompt_rows = [r for r in pp.reasoning_stats.get('rules', []) if r.get('kind') == 'prompt_observation']
        self.assertGreater(len(seeds), 8)
        self.assertEqual(len(prompt_rows), len(seeds) - 1)
        self.assertGreaterEqual(pp.reasoning_stats.get('selected_rules', 0), len(prompt_rows))


if __name__ == '__main__':
    unittest.main(verbosity=2)
