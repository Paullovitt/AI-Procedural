from __future__ import annotations

import unittest

import torch

from prompt_session_v14 import PromptSessionV14


ROLE_ORDER = {'opening': 0, 'development': 1, 'synthesis': 2}


@unittest.skipUnless(torch.cuda.is_available(), 'CUDA required by V14 argument planner')
class ArgumentPlannerV14Tests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.session = PromptSessionV14()

    def test_argument_phases_are_monotonic_and_fast(self):
        report, _, out, stats = self.session.generate(
            'Escreva 1000 caracteres sobre exploração espacial, tecnologia, descoberta e futuro.',
            target_chars=1000,
            seed=1701,
        )
        arg = stats.get('argument_planner', {})
        self.assertEqual(arg.get('engine'), 'Evidence-Argument-Planner-V14')
        self.assertTrue(arg.get('phase_monotonic'))
        self.assertLess(float(arg.get('planner_seconds', 1.0)), 0.005)
        roles = out.get('argument_role_order', [])
        self.assertTrue(roles)
        self.assertTrue(all(ROLE_ORDER[a] <= ROLE_ORDER[b] for a, b in zip(roles, roles[1:])))
        self.assertTrue(report['semantic_verified'])
        self.assertEqual(report['slot_errors'], 0)
        self.assertEqual(report['trace_errors'], 0)

    def test_context_disambiguation_prefers_prompt_aligned_composition(self):
        report, text, _, stats = self.session.generate(
            'Escreva 1000 caracteres sobre música clássica, composição, orquestra e história.',
            target_chars=1000,
            seed=1702,
        )
        lower = text.lower()
        self.assertIn('composição musical', lower)
        self.assertNotIn('composição química', lower)
        self.assertNotIn('composição corporal', lower)
        self.assertNotIn('composição nutricional', lower)
        arg = stats.get('argument_planner', {})
        self.assertGreater(arg.get('context_filtered_rules', 0), 0)
        self.assertGreater(arg.get('context_filtered_synthesis', 0), 0)
        self.assertTrue(report['target_within_tolerance'])

    def test_compound_phrase_context_preserves_complete_observed_expression(self):
        pp = self.session.adapter.build_learned(
            'Escreva 1000 caracteres sobre energia solar, baterias, armazenamento e futuro.',
            self.session.scorer,
            target_chars=1000,
            seed=1703,
        )
        rows = [r for r in pp.reasoning_stats.get('rules', []) if r.get('kind') == 'phrase_context']
        self.assertTrue(rows)
        for row in rows:
            source = row['source'].lower()
            target = row['target'].lower()
            self.assertIn(source, target)
            self.assertGreater(len(target.split()), len(source.split()))

    def test_sparse_isolated_neighbor_is_not_realized(self):
        report, text, _, stats = self.session.generate(
            'Escreva 1000 caracteres sobre segurança digital, criptografia, senhas e privacidade.',
            target_chars=1000,
            seed=1704,
        )
        self.assertTrue(report['evidence_limited'])
        self.assertNotIn('cofina', text.lower())
        arg = stats.get('argument_planner', {})
        self.assertGreaterEqual(arg.get('context_filtered_rules', 0), 1)
        kinds = arg.get('selected_kinds', {})
        self.assertEqual(kinds.get('corpus_neighbor', 0), 0)

    def test_no_immediate_template_repeat_when_alternative_exists(self):
        _, _, out, _ = self.session.generate(
            'Escreva 1000 caracteres sobre música clássica, composição, orquestra e história.',
            target_chars=1000,
            seed=1705,
        )
        templates = [pick[3].get('template') for pick in out.get('picks', [])]
        self.assertFalse(any(a == b for a, b in zip(templates, templates[1:])))


if __name__ == '__main__':
    unittest.main(verbosity=2)
