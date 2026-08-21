from __future__ import annotations

from collections import Counter
from pathlib import Path
import json
import unittest

import torch

from procedural_runtime_v5 import ProtectedSlotVerifier
from procedural_runtime_gpu import SemanticTraceVerifier
from procedural_runtime_v14 import build_renderer_v14_gpu
from prompt_runtime_v14 import PromptAdapterV14, default_lexicon, lexicalize_text, contains_raw_slots

ROOT = Path(__file__).resolve().parent


class PromptAdapterTests(unittest.TestCase):
    def test_prompt_constraints_and_topic(self):
        p = PromptAdapterV14(2000).build(
            'Escreva um texto de 2000 caracteres sobre exploração espacial, tecnologia e futuro, com linguagem clara.'
        )
        self.assertEqual(p.target_chars, 2000)
        self.assertEqual(p.topic.lower(), 'exploração espacial')
        self.assertGreaterEqual(len(p.facts), 8)

    def test_default_lexicalization_hides_slots(self):
        facts = [('prop','e000','a000','v000'), ('rel','e000','r000','e001')]
        lex = default_lexicon(facts)
        text = lexicalize_text('e000 apresenta a000 como v000; r000 liga e000 a e001.', lex)
        self.assertFalse(contains_raw_slots(text))
        self.assertNotIn('e000', text.lower())


@unittest.skipUnless(torch.cuda.is_available(), 'CUDA required by V14 runtime')
class PromptV14GpuIntegrationTests(unittest.TestCase):
    def test_prompt_render_cuda_semantics_and_display(self):
        cfg = json.loads((ROOT / 'gpu_config.json').read_text(encoding='utf8'))
        pp = PromptAdapterV14(900).build(
            'Escreva um texto de 900 caracteres sobre exploração espacial, tecnologia, descoberta e futuro.',
            target_chars=900,
            seed=77,
        )
        scorer, _, _, renderer = build_renderer_v14_gpu(
            ROOT,
            seed=int(cfg.get('seed',101)),
            use_hot=bool(cfg.get('use_hot',False)),
            proposal_weight=float(cfg.get('proposal_weight',.24)),
            position_weight=float(cfg.get('position_weight',7.0)),
            diversity_weight=float(cfg.get('diversity_weight',2.6)),
            focus_diversity_weight=float(cfg.get('focus_diversity_weight',1.17)),
            repetition_weight=float(cfg.get('repetition_weight',1.1)),
            device=int(cfg.get('device',0)),
            memory_limit_mb=int(cfg.get('memory_limit_mb',4608)),
            lexicon=pp.lexicon,
        )
        out = renderer.render(pp.facts, focus_order_hint=pp.focus_order)
        self.assertEqual(out.get('compute_backend'), 'cuda-batched-v14')
        self.assertEqual(Counter(out['represented']), Counter(pp.facts))
        self.assertEqual(ProtectedSlotVerifier().inspect_render(out), [])
        self.assertEqual(SemanticTraceVerifier().inspect_render(out), [])
        display = lexicalize_text(out['text'], pp.lexicon)
        self.assertFalse(contains_raw_slots(display))
        status = scorer.gpu_status()
        self.assertEqual(status['backend'], 'pytorch-cuda-tensors')
        self.assertIn('p5', status['tables_on_gpu'])
        self.assertFalse(status['neural_network'])
        self.assertFalse(status['gradients'])


if __name__ == '__main__':
    unittest.main(verbosity=2)
