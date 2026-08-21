from __future__ import annotations

from dataclasses import dataclass, field
import re

from autonomous_rule_vm_v6 import CorpusAssociationRuleLearnerGPU, IndexedAssociationRuleVM
from argument_planner_v14 import EvidenceArgumentPlannerV14

SLOT_RX = re.compile(r"\b(?:e\d+|a\d+|v\d+|r\d+)\b", re.I)
WORD_RX = re.compile(r"[A-Za-zÀ-ÖØ-öø-ÿ0-9]+", re.UNICODE)

_STOPWORDS = {
    'a','o','as','os','um','uma','uns','umas','de','da','do','das','dos','e','ou','em','no','na','nos','nas',
    'para','por','com','sem','sobre','acerca','respeito','que','se','ao','aos','à','às','como','mais','menos','muito',
    'texto','escreva','escrever','gere','gerar','crie','criar','faça','faca','produza','produzir','quero','preciso',
    'caracteres','caráter','carateres','chars','palavras','frases','parágrafos','paragrafos','coerente','coerência',
    'criativo','criativa','criativos','criativas','criatividade','inteligente','inteligência','inteligencia',
    'claro','clara','claros','claras','natural','naturais','português','portugues','linguagem','tom','estilo','foco',
    'aproximadamente','cerca','aprox','tema','assunto','prompt','v14','gpu','vram'
}

# Legacy fallback only. The promoted prompt path uses learned association rules instead.
_GENERIC_ASPECTS = (
    'ideia central','contexto principal','ponto de atenção','perspectiva','desenvolvimento',
    'conexão conceitual','possibilidade','consequência','contraste','continuidade','síntese','direção'
)
_GENERIC_RELATIONS = ('associação temática','conexão contextual','continuidade conceitual','relação de contexto')
_GENERIC_VALUES = ('visão inicial','desenvolvimento possível','perspectiva complementar','leitura contextual','síntese do tema')
_ENTITY_FALLBACKS = ('contexto do tema','desenvolvimento do tema','perspectiva do tema','síntese do tema','horizonte do tema')

_PREDICATE_SURFACE_VARIANTS = {
    'compartilha contexto com': (
        'se relaciona tematicamente com', 'surge no mesmo contexto que',
        'mantém proximidade temática com', 'se conecta contextualmente a'
    ),
    'aparece em expressões como': (
        'aparece em expressões como', 'surge em construções como',
        'é recorrente em expressões como'
    ),
    'surge em contextos semelhantes a': (
        'compartilha padrões de contexto com', 'mantém semelhança contextual com',
        'surge em contexto próximo de'
    ),
    'aparece no mesmo contexto temático que': (
        'se conecta a', 'integra o mesmo recorte temático que', 'surge no mesmo eixo temático que'
    ),
    'forma um núcleo contextual com': ('reúne associações com',),
}


def _clean_space(text: str) -> str:
    return re.sub(r'\s+', ' ', str(text or '')).strip()


def _sentence_case(text: str) -> str:
    chars=list(str(text or ''))
    capitalize=True
    for i,ch in enumerate(chars):
        if capitalize and ch.isalpha():
            chars[i]=ch.upper()
            capitalize=False
        elif ch in '.!?\n':
            capitalize=True
        elif capitalize and not ch.isspace() and not ch in '"\'([{':
            capitalize=False
    return ''.join(chars)


def alpha_label(index: int) -> str:
    alphabet = (
        'alfa','beta','gama','delta','épsilon','zeta','eta','teta','iota','capa','lambda','mi','ni',
        'csi','ômicron','pi','rô','sigma','tau','ípsilon','fi','qui','psi','ômega','aurora','horizonte'
    )
    i = max(0, int(index))
    if i < len(alphabet):
        return alphabet[i]
    out = []
    n = i + 1
    while n:
        n, r = divmod(n - 1, 26)
        out.append(chr(ord('a') + r))
    return 'grupo ' + ''.join(reversed(out))


def default_lexicon(facts) -> dict[str, str]:
    lex: dict[str, str] = {}
    for fact in facts:
        for slot in fact[1:]:
            s = str(slot).lower()
            m = re.fullmatch(r'([eavr])(\d+)', s, re.I)
            if not m or s in lex:
                continue
            kind, raw = m.group(1).lower(), int(m.group(2))
            tag = alpha_label(raw)
            if kind == 'e': lex[s] = f'elemento {tag}'
            elif kind == 'a': lex[s] = f'aspecto {tag}'
            elif kind == 'v': lex[s] = f'valor {tag}'
            else: lex[s] = f'relação {tag}'
    return lex


def lexicalize_text(text: str, lexicon: dict[str, str] | None = None) -> str:
    lex = {str(k).lower(): _clean_space(v) for k, v in (lexicon or {}).items()}

    def repl(match):
        key = match.group(0).lower()
        if key in lex:
            return lex[key]
        m = re.fullmatch(r'([eavr])(\d+)', key)
        if not m:
            return key
        tag = alpha_label(int(m.group(2)))
        return {'e':'elemento ','a':'aspecto ','v':'valor ','r':'relação '}[m.group(1)] + tag

    return _sentence_case(SLOT_RX.sub(repl, text))


def contains_raw_slots(text: str) -> bool:
    return bool(SLOT_RX.search(text or ''))


@dataclass
class PromptPlan:
    prompt: str
    topic: str
    target_chars: int
    facts: list[tuple]
    lexicon: dict[str, str]
    focus_order: list[str]
    predicate_relations: set[str] = field(default_factory=set)
    predicate_classes: dict[str, str] = field(default_factory=dict)
    argument_roles: dict[str, str] = field(default_factory=dict)
    reasoning_stats: dict = field(default_factory=dict)


class PromptAdapterV14:
    """Non-neural prompt adapter with a learned RuleBank/RuleVM content layer.

    The promoted path extracts concepts from the prompt, learns corpus-supported
    association rules from Bagaço statistics on CUDA, executes those rules in an
    indexed VM, and gives the resulting explicit fact graph to V14.
    """
    TARGET_RX = re.compile(r'\b(\d{2,6})\s*(?:caracteres|caráter(?:es)?|carateres|chars)\b', re.I)
    TOPIC_RX = re.compile(r'\b(?:sobre|acerca\s+de|a\s+respeito\s+de|tema\s*[:=-])\s+(.+)', re.I)

    def __init__(self, default_target_chars: int = 2000, argument_planner_enabled: bool = True):
        self.default_target_chars = max(200, int(default_target_chars))
        self.argument_planner_enabled = bool(argument_planner_enabled)
        self.argument_planner = EvidenceArgumentPlannerV14()
        self._learner_cache = {}

    def _learner_for(self, scorer):
        key = id(scorer)
        learner = self._learner_cache.get(key)
        if learner is None or getattr(learner, 's', None) is not scorer:
            learner = CorpusAssociationRuleLearnerGPU(scorer)
            self._learner_cache = {key: learner}
        return learner

    def target_from_prompt(self, prompt: str, fallback: int | None = None) -> int:
        if fallback is not None:
            return max(200, min(50000, int(fallback)))
        m = self.TARGET_RX.search(prompt or '')
        if m:
            return max(200, min(50000, int(m.group(1))))
        return self.default_target_chars

    def topic_from_prompt(self, prompt: str) -> str:
        text = _clean_space(prompt)
        m = self.TOPIC_RX.search(text)
        topic = m.group(1) if m else text
        topic = self.TARGET_RX.sub('', topic)
        topic = re.split(r'[,;]\s*(?:com|em)\s+(?:tom|estilo|linguagem|foco)\b', topic, maxsplit=1, flags=re.I)[0]
        if ',' in topic:
            topic = topic.split(',', 1)[0]
        topic = topic.strip(' .,:;-')
        words = WORD_RX.findall(topic)
        if len(words) > 8:
            words = words[:8]
        return ' '.join(words) or 'tema solicitado'

    def keywords(self, prompt: str, topic: str) -> list[str]:
        source = f'{topic} {prompt}'
        out = []
        seen = set()
        for w in WORD_RX.findall(source):
            lw = w.lower()
            if lw.isdigit() or len(lw) < 3 or lw in _STOPWORDS or lw in seen:
                continue
            seen.add(lw)
            out.append(w)
        return out[:24] or ['conteúdo','contexto','desenvolvimento']

    def concepts(self, prompt: str, topic: str) -> list[str]:
        """Keep the topic phrase intact, then add non-duplicate prompt concepts."""
        topic_tokens = {x.lower() for x in WORD_RX.findall(topic)}
        concepts = [topic]
        for key in self.keywords(prompt, topic):
            if key.lower() not in topic_tokens and key.lower() not in {x.lower() for x in concepts}:
                concepts.append(key)
        return concepts[:12]

    def build_learned(self, prompt: str, scorer, target_chars: int | None = None,
                      seed: int = 101, fact_count: int | None = None) -> PromptPlan:
        prompt = _clean_space(prompt)
        target = self.target_from_prompt(prompt, target_chars)
        topic = self.topic_from_prompt(prompt)
        concepts = self.concepts(prompt, topic)
        rule_budget = int(fact_count) if fact_count is not None else round(target / 74)
        rule_budget = max(8, min(180, rule_budget))

        learner = self._learner_for(scorer)
        emit_budget = max(rule_budget, max(0, len(concepts) - 1))
        initial_bank_budget=min(256, max(rule_budget + 8, int(rule_budget * 1.35)))

        def select_bank(candidate_bank):
            vm = IndexedAssociationRuleVM(candidate_bank)
            execution_local = vm.execute(concepts, max_depth=2, max_rules=max(1, len(candidate_bank.rules)))
            fired_all_local = execution_local['fired']
            prompt_rules_local = [row for row in fired_all_local if row[0].kind == 'prompt_observation']
            if self.argument_planner_enabled:
                argument_plan_local = self.argument_planner.plan(fired_all_local, concepts, emit_budget)
                planned_rows_local = argument_plan_local['planned']
                fired_local = [row for row, _role in planned_rows_local]
                roles_local = {id(row[0]): role for row, role in planned_rows_local}
                argument_stats_local = argument_plan_local['stats']
            else:
                prompt_ids = {id(row[0]) for row in prompt_rules_local}
                others = [row for row in fired_all_local if id(row[0]) not in prompt_ids]
                fired_local = (prompt_rules_local + others)[:emit_budget]
                roles_local = {id(row[0]): ('opening' if row[0].kind == 'prompt_observation' else 'development') for row in fired_local}
                argument_stats_local = {
                    'engine': 'disabled', 'planner_seconds': 0.0, 'input_rules': len(fired_all_local),
                    'selected_rules': len(fired_local), 'phase_counts': {}, 'phase_monotonic': True,
                }
            return execution_local, fired_local, roles_local, argument_stats_local

        bank = learner.fit(concepts, rule_budget=initial_bank_budget, expansion_depth=1)
        execution, fired, role_by_rule, argument_stats = select_bank(bank)

        # If contextual quality filtering removed realizations even though the bank had
        # enough candidates, widen the learned bank once before rendering. This is a
        # replacement search, not a relaxation of promotion thresholds.
        refill_passes=0
        if self.argument_planner_enabled and len(fired) < emit_budget and len(bank.rules) >= emit_budget:
            filtered=(int(argument_stats.get('context_filtered_rules',0))
                      + int(argument_stats.get('context_filtered_synthesis',0)))
            if filtered:
                wider_budget=min(256,max(initial_bank_budget + 2*filtered,int(initial_bank_budget*1.5)))
                if wider_budget > initial_bank_budget:
                    wider_bank=learner.fit(concepts,rule_budget=wider_budget,expansion_depth=1)
                    candidate=select_bank(wider_bank)
                    if len(candidate[1]) > len(fired):
                        bank=wider_bank
                        execution,fired,role_by_rule,argument_stats=candidate
                        refill_passes=1
        argument_stats['refill_passes']=refill_passes
        argument_stats['emit_budget']=emit_budget

        if not fired:
            # Sparse/OOV fallback remains auditable and uses only prompt observations.
            return self.build(prompt, target_chars=target, seed=seed, fact_count=rule_budget)

        entity_ids: dict[str, str] = {}
        lex: dict[str, str] = {}

        def entity_id(label: str):
            key = str(label).lower()
            if key not in entity_ids:
                eid = f'e{len(entity_ids):03d}'
                entity_ids[key] = eid
                lex[eid] = str(label)
            return entity_ids[key]

        # Preserve prompt concepts first in the entity/focus order.
        for concept in concepts:
            entity_id(concept)

        facts = []
        predicate_relations = set()
        predicate_classes = {}
        argument_roles = {}
        surface_counts = {}
        rule_rows = []
        for rule, path_conf, depth in fired:
            src = entity_id(rule.source)
            dst = entity_id(rule.target)
            rid = f'r{len(facts):03d}'
            canonical = str(rule.predicate)
            surface_key = (src, canonical)
            surface_i = surface_counts.get(surface_key, 0)
            surface_counts[surface_key] = surface_i + 1
            variants = _PREDICATE_SURFACE_VARIANTS.get(canonical, (canonical,))
            lex[rid] = variants[surface_i % len(variants)]
            predicate_relations.add(rid)
            predicate_classes[rid] = canonical
            argument_roles[rid] = role_by_rule.get(id(rule), 'development')
            facts.append(('rel', src, rid, dst))
            rule_rows.append({
                'source': rule.source,
                'target': rule.target,
                'predicate': rule.predicate,
                'kind': rule.kind,
                'confidence': rule.confidence,
                'path_confidence': path_conf,
                'support': rule.support,
                'score': rule.score,
                'depth': depth,
                'argument_role': argument_roles[rid],
                'evidence': list(rule.evidence),
            })

        focus_labels = list(dict.fromkeys(concepts + execution.get('nodes', [])))
        focus = [entity_ids[x.lower()] for x in focus_labels if x.lower() in entity_ids]
        stats = {
            'engine': 'Learned-Association-RuleVM-v6',
            'seed_concepts': concepts,
            'rulebank': bank.status(),
            'selected_rules': len(facts),
            'reachable_nodes': len(execution.get('nodes', [])),
            'vm_seconds': execution.get('vm_seconds', 0.0),
            'indexed_lookups': execution.get('indexed_lookups', 0),
            'argument_planner': argument_stats,
            'rules': rule_rows,
        }
        return PromptPlan(prompt, topic, target, facts, lex, focus, predicate_relations, predicate_classes, argument_roles, stats)

    def build(self, prompt: str, target_chars: int | None = None, seed: int = 101,
              fact_count: int | None = None) -> PromptPlan:
        """Legacy deterministic fallback retained for sparse/OOV prompts and A/B tests."""
        prompt = _clean_space(prompt)
        target = self.target_from_prompt(prompt, target_chars)
        topic = self.topic_from_prompt(prompt)
        keys = self.keywords(prompt, topic)
        n_facts = int(fact_count) if fact_count is not None else round(target / 88)
        n_facts = max(8, min(260, n_facts))
        n_entities = max(4, min(10, len(keys) + 1))
        ents = [f'e{i:03d}' for i in range(n_entities)]
        lex: dict[str, str] = {ents[0]: topic}
        for i, e in enumerate(ents[1:], 1):
            lex[e] = keys[i - 1] if i - 1 < len(keys) else _ENTITY_FALLBACKS[(i - 1) % len(_ENTITY_FALLBACKS)]

        facts: list[tuple] = []
        rel_count = 0
        prop_count = 0
        local_prop_count = {e: 0 for e in ents}
        for i in range(n_facts):
            if i % 5 == 4:
                src_i = (i // 5) % n_entities
                dst_i = (src_i + 1) % n_entities
                src, dst = ents[src_i], ents[dst_i]
                rid = f'r{rel_count:03d}'
                lex[rid] = _GENERIC_RELATIONS[rel_count % len(_GENERIC_RELATIONS)]
                rel_count += 1
                facts.append(('rel', src, rid, dst))
                continue

            src_i = (i + i // 5) % n_entities
            src = ents[src_i]
            local_index = local_prop_count[src]
            local_prop_count[src] += 1
            aid = f'a{prop_count:03d}'
            vid = f'v{prop_count:03d}'
            lex[aid] = _GENERIC_ASPECTS[local_index % len(_GENERIC_ASPECTS)]
            if keys:
                start = src_i % len(keys)
                value_index = (start + local_index + 1) % len(keys)
                candidate = keys[value_index]
                source_label = lex.get(src, '').lower()
                if candidate.lower() == source_label and len(keys) > 1:
                    candidate = keys[(value_index + 1) % len(keys)]
                lex[vid] = candidate
            else:
                lex[vid] = _GENERIC_VALUES[prop_count % len(_GENERIC_VALUES)] + f' sobre {topic}'
            prop_count += 1
            facts.append(('prop', src, aid, vid))

        focus = list(dict.fromkeys(f[1] for f in facts))
        return PromptPlan(prompt, topic, target, facts, lex, focus)
