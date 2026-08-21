from __future__ import annotations

from dataclasses import dataclass
import re

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

_GENERIC_ASPECTS = (
    'ideia central','contexto principal','ponto de atenção','perspectiva','desenvolvimento',
    'conexão conceitual','possibilidade','consequência','contraste','continuidade','síntese','direção'
)
_GENERIC_RELATIONS = ('associação temática','conexão contextual','continuidade conceitual','relação de contexto')
_GENERIC_VALUES = ('visão inicial','desenvolvimento possível','perspectiva complementar','leitura contextual','síntese do tema')
_ENTITY_FALLBACKS = ('contexto do tema','desenvolvimento do tema','perspectiva do tema','síntese do tema','horizonte do tema')


def _clean_space(text: str) -> str:
    return re.sub(r'\s+', ' ', str(text or '')).strip()


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
    return SLOT_RX.sub(repl, text)


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


class PromptAdapterV14:
    """Non-neural prompt adapter that builds an auditable symbolic plan for V14."""
    TARGET_RX = re.compile(r'\b(\d{2,6})\s*(?:caracteres|caráter(?:es)?|carateres|chars)\b', re.I)
    TOPIC_RX = re.compile(r'\b(?:sobre|acerca\s+de|a\s+respeito\s+de|tema\s*[:=-])\s+(.+)', re.I)

    def __init__(self, default_target_chars: int = 2000):
        self.default_target_chars = max(200, int(default_target_chars))

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

    def build(self, prompt: str, target_chars: int | None = None, seed: int = 101,
              fact_count: int | None = None) -> PromptPlan:
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
