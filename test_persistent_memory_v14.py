from __future__ import annotations

from pathlib import Path
from tempfile import TemporaryDirectory
import unittest

from persistent_memory_v14 import PersistentDimensionalMemoryV14, should_auto_store_user_memory
from prompt_runtime_v14 import PromptAdapterV14


class PersistentDimensionalMemoryV14Tests(unittest.TestCase):
    def test_persists_across_reopen_and_retrieves_exact_fact(self):
        with TemporaryDirectory() as td:
            path=Path(td)/'memory.sqlite3'
            mem=PersistentDimensionalMemoryV14(path)
            saved=mem.remember('Meu carro é um Civic 2015.',source='user')
            self.assertTrue(saved['new'])
            self.assertEqual(mem.search('Qual carro eu disse que tenho?',k=2)[0]['id'],saved['id'])
            mem.close()

            mem=PersistentDimensionalMemoryV14(path)
            hit=mem.search('Civic 2015',k=2)[0]
            self.assertEqual(hit['text'],'Meu carro é um Civic 2015.')
            self.assertEqual(hit['source'],'user')
            mem.close()

    def test_duplicate_observation_strengthens_recurrence_without_duplicate_episode(self):
        with TemporaryDirectory() as td:
            mem=PersistentDimensionalMemoryV14(Path(td)/'memory.sqlite3')
            first=mem.remember('A máquina Atlas usa fluido V7.')
            second=mem.remember('A máquina Atlas usa fluido V7.')
            self.assertEqual(first['id'],second['id'])
            self.assertEqual(second['recurrence'],2)
            stats=mem.stats()
            self.assertEqual(stats['episodes'],1)
            self.assertEqual(stats['observations'],2)
            self.assertEqual(mem.search('Atlas fluido',k=1)[0]['recurrence'],2)
            mem.close()

    def test_semantic_index_shadow_can_recover_noisy_raw_memory(self):
        with TemporaryDirectory() as td:
            mem=PersistentDimensionalMemoryV14(Path(td)/'memory.sqlite3')
            mem.remember('meu caroo eh civic 2015',index_text='meu caroo eh civic 2015 carro civic 2015')
            hit=mem.search('qual carro eu tenho?',k=1)[0]
            self.assertEqual(hit['text'],'meu caroo eh civic 2015')
            self.assertIn('carro',hit['text'].replace('caroo','carro'))
            mem.close()

    def test_raw_numbers_and_dates_are_preserved(self):
        with TemporaryDirectory() as td:
            mem=PersistentDimensionalMemoryV14(Path(td)/'memory.sqlite3')
            text='Revisão Atlas em 21/08/2026 às 14:30; custo 20.000 reais.'
            mem.remember(text)
            hit=mem.search('Atlas 2026 custo',k=1)[0]['text']
            self.assertEqual(hit,text)
            self.assertIn('21/08/2026',hit)
            self.assertIn('20.000',hit)
            mem.close()

    def test_weak_single_term_overlap_does_not_recall_unrelated_episode(self):
        with TemporaryDirectory() as td:
            mem=PersistentDimensionalMemoryV14(Path(td)/'memory.sqlite3',min_query_term_coverage=0.30)
            mem.remember('Exploração espacial, tecnologia, descoberta e futuro.')
            self.assertEqual(mem.search('energia solar baterias armazenamento futuro',k=3),[])
            self.assertTrue(mem.search('exploração espacial futuro',k=1))
            mem.close()

    def test_irrelevant_query_returns_no_memory(self):
        with TemporaryDirectory() as td:
            mem=PersistentDimensionalMemoryV14(Path(td)/'memory.sqlite3')
            mem.remember('A máquina Atlas usa fluido V7.')
            self.assertEqual(mem.search('ornitorrinco quasar',k=3),[])
            mem.close()

    def test_forget_removes_episode_and_updates_explicit_index_incrementally(self):
        with TemporaryDirectory() as td:
            mem=PersistentDimensionalMemoryV14(Path(td)/'memory.sqlite3')
            a=mem.remember('Atlas usa fluido V7.')
            mem.remember('Orion usa fluido A2.')
            self.assertTrue(mem.forget(a['id']))
            self.assertEqual(mem.search('Atlas V7',k=2),[])
            self.assertTrue(mem.search('Orion A2',k=2))
            self.assertFalse(mem.forget(999999))
            mem.close()

    def test_prompt_adapter_turns_retrieval_into_auditable_rule(self):
        adapter=PromptAdapterV14(persistent_memory_rule_cap=2,persistent_memory_inject_chars=200)
        rows=adapter._memory_rows([{
            'id':7,'text':'Meu carro é um Civic 2015.','source':'user','recurrence':3,
            'score':4.2,'coverage':1.0,'match_kind':'exact',
        }])
        self.assertEqual(len(rows),1)
        rule,path_conf,depth=rows[0]
        self.assertEqual(rule.kind,'memory_retrieval')
        self.assertEqual(rule.target,'Meu carro é um Civic 2015')
        self.assertIn('memory:7',rule.evidence)
        self.assertEqual(path_conf,0.995)
        self.assertEqual(depth,1)

    def test_interrogative_user_text_is_not_auto_promotable(self):
        self.assertFalse(should_auto_store_user_memory('Qual carro eu disse que tenho?'))
        self.assertFalse(should_auto_store_user_memory('¿Qué coche tengo?'))
        self.assertTrue(should_auto_store_user_memory('Meu carro é um Civic 2015.'))

    def test_associative_hub_is_filtered_by_document_frequency(self):
        with TemporaryDirectory() as td:
            mem=PersistentDimensionalMemoryV14(
                Path(td)/'memory.sqlite3',associative_per_term=8,max_associative_document_ratio=0.20
            )
            for i in range(20):
                mem.remember(f'comum item{i} valor{i}')
            rare=mem.remember('atlas raro fluidoV7')
            self.assertEqual(mem._associative_term_ids([mem._query_terms('comum')[0]['id']],20),{})
            hit=mem.search('atlas raro',k=1)[0]
            self.assertEqual(hit['id'],rare['id'])
            mem.close()


if __name__ == '__main__':
    unittest.main(verbosity=2)
