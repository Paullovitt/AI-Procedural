from __future__ import annotations

from collections import Counter, defaultdict
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Iterable
import json
import math
import re
import time
import unicodedata

TOKEN_RX = re.compile(r"\d{1,4}(?:[./:\-]\d{1,4}){1,3}|\d+(?:[.,]\d+)*|[^\W\d_]+(?:['’\-][^\W\d_]+)*|[^\s]", re.UNICODE)
DATE_RX = re.compile(r"^(?:\d{1,2}[./-]\d{1,2}(?:[./-]\d{2,4})?|\d{4}[./-]\d{1,2}[./-]\d{1,2})$")
TIME_RX = re.compile(r"^\d{1,2}:\d{2}(?::\d{2})?$")
NUMBER_RX = re.compile(r"^\d+(?:[.,]\d+)*$")
STRONG_BREAKS = frozenset({'.', '!', '?', ';'})


def semantic_shadow(text: str) -> str:
    s = unicodedata.normalize('NFKC', str(text or '')).casefold()
    s = ''.join(c for c in unicodedata.normalize('NFKD', s) if not unicodedata.combining(c))
    return ''.join(c for c in s if c.isalnum() or c in "'-")


def bounded_damerau(a: str, b: str, max_dist: int) -> int:
    if a == b:
        return 0
    if abs(len(a) - len(b)) > max_dist:
        return max_dist + 1
    if not a or not b:
        return max(len(a), len(b))
    prev2 = None
    prev = list(range(len(b) + 1))
    for i, ca in enumerate(a, 1):
        cur = [i]
        row_min = i
        for j, cb in enumerate(b, 1):
            cost = 0 if ca == cb else 1
            v = min(cur[j - 1] + 1, prev[j] + 1, prev[j - 1] + cost)
            if prev2 is not None and i > 1 and j > 1 and ca == b[j - 2] and a[i - 2] == cb:
                v = min(v, prev2[j - 2] + 1)
            cur.append(v)
            row_min = min(row_min, v)
        if row_min > max_dist:
            return max_dist + 1
        prev2, prev = prev, cur
    return prev[-1]


def qgrams(text: str) -> set[str]:
    return {text[i:i+2] for i in range(max(0, len(text) - 1))} or ({text} if text else set())


def numeric_normalized(raw: str) -> str:
    s = str(raw).strip()
    if NUMBER_RX.match(s) and ',' in s and '.' not in s:
        p = s.split(',')
        if len(p) == 2 and 0 < len(p[1]) <= 3:
            return p[0] + '.' + p[1]
    return s


@dataclass(frozen=True)
class SemanticTokenV14:
    raw: str
    canonical: str
    kind: str
    source: str
    confidence: float
    signal: float
    index: int
    start: int
    end: int
    segment: int
    redundant: bool = False
    atoms: tuple[str, ...] = ()


@dataclass(frozen=True)
class SemanticEdgeV14:
    source: str
    target: str
    source_index: int
    target_index: int
    raw_distance: int
    strength: float
    support: int
    bridge: tuple[str, ...] = ()
    kind: str = 'ordered_context'


@dataclass(frozen=True)
class SemanticPhraseV14:
    canonical: str
    token_indices: tuple[int, ...]
    support: int
    confidence: float


@dataclass
class SemanticIntakeResultV14:
    raw_text: str
    tokens: list[SemanticTokenV14]
    signal_indices: list[int]
    edges: list[SemanticEdgeV14]
    phrases: list[SemanticPhraseV14]
    numeric_anchors: list[dict]
    stats: dict = field(default_factory=dict)

    @property
    def spine(self) -> list[str]:
        out = []
        for i in self.signal_indices:
            x = self.tokens[i].canonical
            if not out or out[-1].casefold() != x.casefold():
                out.append(x)
        return out

    @property
    def semantic_atoms(self) -> list[str]:
        out = []; seen = set()
        for i in self.signal_indices:
            tok = self.tokens[i]
            for atom in tok.atoms or (tok.canonical,):
                key = semantic_shadow(atom)
                if key and key not in seen:
                    seen.add(key); out.append(atom)
        for anchor in self.numeric_anchors:
            key = '#' + str(anchor.get('normalized', anchor.get('raw', '')))
            if key not in seen:
                seen.add(key); out.append(key)
        return out

    def fingerprint(self) -> set[str]:
        return {semantic_shadow(x[1:] if x.startswith('#') else x) for x in self.semantic_atoms if x}

    def to_training_projection(self) -> dict:
        return {
            'format': 'Robust-Semantic-Projection-V14',
            'nodes': [asdict(self.tokens[i]) for i in self.signal_indices],
            'edges': [asdict(x) for x in self.edges],
            'phrases': [asdict(x) for x in self.phrases],
            'numeric_anchors': list(self.numeric_anchors),
            'stats': dict(self.stats),
        }


@dataclass(frozen=True)
class AliasEvidenceV14:
    canonical: str
    support: int
    confidence: float
    context_support: int = 0


class LearnedAliasBankV14:
    FORMAT = 'Learned-Robust-AliasBank-V14'

    def __init__(self, aliases: dict[str, AliasEvidenceV14] | None = None, metadata: dict | None = None):
        self.aliases = dict(aliases or {})
        self.metadata = dict(metadata or {})

    def get(self, raw: str) -> AliasEvidenceV14 | None:
        return self.aliases.get(semantic_shadow(raw))

    def save(self, path: str | Path):
        obj = {'format': self.FORMAT, 'metadata': self.metadata,
               'aliases': {k: asdict(v) for k, v in sorted(self.aliases.items())}}
        Path(path).write_text(json.dumps(obj, ensure_ascii=False, indent=2), encoding='utf8')

    @classmethod
    def load(cls, path: str | Path):
        p = Path(path)
        if not p.exists():
            return cls()
        obj = json.loads(p.read_text(encoding='utf8'))
        if obj.get('format') != cls.FORMAT:
            raise ValueError(f'AliasBank V14 inválido: {p}')
        return cls({k: AliasEvidenceV14(**v) for k, v in obj.get('aliases', {}).items()}, obj.get('metadata', {}))

    def status(self) -> dict:
        return {'format': self.FORMAT, 'aliases': len(self.aliases), **self.metadata}


class RobustSemanticIntakeV14:
    """Low-latency non-neural semantic intake over raw imperfect text.

    The input is never rewritten. Canonical labels are internal hypotheses; every node
    keeps its original evidence span. Clean tokens use O(1) vocabulary lookup. Fuzzy
    structures are built lazily only after a suspicious token is observed.
    """

    def __init__(self, scorer, alias_bank: LearnedAliasBankV14 | None = None,
                 signal_threshold: float = 0.46, min_vocab_support: int = 4,
                 context_window: int = 5):
        self.s = scorer
        self.alias_bank = alias_bank or LearnedAliasBankV14()
        self.signal_threshold = float(signal_threshold)
        self.min_vocab_support = int(min_vocab_support)
        self.context_window = int(context_window)
        self.max_tok_count = max(self.s.tok.values(), default=1)
        self.shadow_best = None
        self.len_buckets = None
        self.first_buckets = None
        self.last_buckets = None
        self.delete_buckets = None
        self.fast_fuzzy_min_support = 64
        self.fuzzy_index_build_seconds = 0.0
        self.fuzzy_queries = 0
        self.extract_calls = 0
        self.candidate_cache = {}

    def _ensure_fuzzy_index(self):
        if self.shadow_best is not None:
            return
        t0 = time.perf_counter(); best = {}; db = defaultdict(list)
        for token, raw_count in self.s.tok.items():
            count = int(raw_count)
            if count < self.min_vocab_support:
                continue
            sh = semantic_shadow(token)
            old = best.get(sh)
            if sh and (old is None or count > old[1]):
                best[sh] = (str(token), count)
        for sh, (_canonical, count) in best.items():
            if count >= self.fast_fuzzy_min_support and len(sh) >= 3:
                for i in range(len(sh)):
                    db[sh[:i] + sh[i+1:]].append(sh)
        self.shadow_best = best
        self.delete_buckets = {k: tuple(v) for k, v in db.items()}
        self.fuzzy_index_build_seconds = time.perf_counter() - t0

    def _ensure_broad_index(self):
        if self.len_buckets is not None:
            return
        self._ensure_fuzzy_index(); lb = defaultdict(list); fb = defaultdict(list); rb = defaultdict(list)
        for sh in self.shadow_best:
            n = len(sh); lb[n].append(sh); fb[(n, sh[0])].append(sh); rb[(n, sh[-1])].append(sh)
        self.len_buckets = {k: tuple(v) for k, v in lb.items()}
        self.first_buckets = {k: tuple(v) for k, v in fb.items()}
        self.last_buckets = {k: tuple(v) for k, v in rb.items()}

    def _pair_support(self, a: str, b: str) -> int:
        if not a or not b:
            return 0
        return int(self.s.p2.get(f'{a}\t{b}', 0)) + int(self.s.p2.get(f'{b}\t{a}', 0))

    def _context_support(self, candidate: str, context) -> int:
        return sum(self._pair_support(candidate, x) for x in context if x and x != candidate)

    def _shortlist(self, shadow: str, limit: int = 96) -> list[str]:
        self._ensure_fuzzy_index(); n = len(shadow)
        max_dist = 1 if n <= 4 else (2 if n <= 10 else 3)
        pool = set(self.delete_buckets.get(shadow, ())) if self.delete_buckets is not None else set()
        # Exact one-deletion and adjacent-transposition probes are O(length) and can
        # recover low-frequency proper names that are deliberately absent from the
        # support>=64 resident signature index.
        for i in range(n):
            sig=shadow[:i]+shadow[i+1:]
            if sig in self.shadow_best:
                pool.add(sig)
        for i in range(n-1):
            swapped=shadow[:i]+shadow[i+1]+shadow[i]+shadow[i+2:]
            if swapped in self.shadow_best:
                pool.add(swapped)
        for i in range(n):
            sig = shadow[:i] + shadow[i+1:]
            hit = self.shadow_best.get(sig)
            if hit is not None and hit[1] >= self.fast_fuzzy_min_support:
                pool.add(sig)
            if self.delete_buckets is not None:
                pool.update(self.delete_buckets.get(sig, ()))
        if pool:
            q = qgrams(shadow); scored = []
            for cand in pool:
                if abs(len(cand) - n) > max_dist:
                    continue
                scored.append((len(q & qgrams(cand)), int(cand[0] == shadow[0]),
                               int(cand[-1] == shadow[-1]), -abs(len(cand) - n), cand))
            scored.sort(reverse=True)
            return [x[-1] for x in scored[:limit]]

        # Severe/rare corruption fallback. It is intentionally cold: broad buckets are
        # built and visited only when the compact one-deletion index finds no candidate.
        self._ensure_broad_index()
        for ln in range(max(1, n - 2), n + 3):
            pool.update(self.first_buckets.get((ln, shadow[0]), ()))
            pool.update(self.last_buckets.get((ln, shadow[-1]), ()))
        if not pool:
            for ln in range(max(1, n - 1), n + 2):
                pool.update(self.len_buckets.get(ln, ()))
        q = qgrams(shadow); chars = set(shadow); scored = []
        for cand in pool:
            shared_q = len(q & qgrams(cand)); shared_c = len(chars & set(cand))
            if n >= 5 and shared_q == 0 and shared_c < max(2, len(chars) // 2):
                continue
            scored.append((shared_q, shared_c, -abs(n - len(cand)), cand))
        scored.sort(reverse=True)
        return [x[-1] for x in scored[:limit]]

    def _best_fuzzy(self, raw: str, context: tuple[str, ...]):
        shadow = semantic_shadow(raw); key = (shadow, context)
        if key in self.candidate_cache:
            return self.candidate_cache[key]
        self.fuzzy_queries += 1; self._ensure_fuzzy_index()
        direct = self.shadow_best.get(shadow)
        if direct is not None:
            canonical, _ = direct
            out = (canonical, 0, self._context_support(canonical, context), 0.995)
            self.candidate_cache[key] = out; return out
        max_dist = 1 if len(shadow) <= 4 else (2 if len(shadow) <= 10 else 3)
        best = None
        for cshadow in self._shortlist(shadow):
            dist = bounded_damerau(shadow, cshadow, max_dist)
            if dist > max_dist:
                continue
            canonical, count = self.shadow_best[cshadow]
            ctx = self._context_support(canonical, context)
            char_conf = 1.0 - dist / max(2.0, float(max(len(shadow), len(cshadow))))
            transposed = 0
            if len(shadow) == len(cshadow) and dist == 1:
                for k in range(len(shadow) - 1):
                    if shadow[:k] + shadow[k+1] + shadow[k] + shadow[k+2:] == cshadow:
                        transposed = 1; break
            # Edit distance is primary evidence. Corpus context breaks ties but cannot
            # turn a much less similar token into the winner. Exact transpositions are
            # preferred because they are a common single-keystroke corruption.
            row = (-dist, transposed, char_conf, int(ctx > 0), min(ctx, 10000), math.log1p(count), canonical, ctx)
            if best is None or row[:6] > best[:6]:
                best = row
        if best is None:
            self.candidate_cache[key] = None; return None
        neg_dist, _, char_conf, _, _, _, canonical, ctx = best
        dist = -neg_dist
        if dist > 1 and ctx <= 0 and char_conf < 0.84:
            self.candidate_cache[key] = None; return None
        conf = min(0.99, max(0.45, 0.58 * char_conf + 0.42 * min(1.0, math.log1p(ctx) / 6.0)))
        out = (canonical, int(dist), int(ctx), float(conf)); self.candidate_cache[key] = out
        return out

    def _best_split(self, raw: str):
        sh = semantic_shadow(raw)
        if len(sh) < 6:
            return None
        self._ensure_fuzzy_index(); best = None
        for i in range(3, len(sh) - 2):
            left = self.shadow_best.get(sh[:i]); right = self.shadow_best.get(sh[i:])
            if left is None or right is None:
                continue
            a, ac = left; b, bc = right; support = self._pair_support(a, b)
            row = (support, min(ac, bc), a, b)
            if support > 0 and (best is None or row[:2] > best[:2]):
                best = row
        if best is None:
            return None
        support, _, a, b = best
        return f'{a} {b}', (a, b), int(support)

    def _best_phrase_segmentation(self, raw: str):
        """Recover a concatenated 2-token phrase only when corpus p2 supports it."""
        sh=semantic_shadow(raw)
        if len(sh)<6:
            return None
        self._ensure_fuzzy_index(); best=None
        for i in range(3,len(sh)-2):
            left=self.shadow_best.get(sh[:i]); right=self.shadow_best.get(sh[i:])
            if left is None or right is None:
                continue
            a,ac=left; b,bc=right; support=int(self.s.p2.get(f'{a}\t{b}',0))
            if support<=0:
                continue
            row=(support,min(ac,bc),a,b)
            if best is None or row[:2]>best[:2]:
                best=row
        if best is None:
            return None
        support,_,a,b=best
        return f'{a} {b}',(a,b),int(support)

    def _repair_mojibake_shadow(self, text: str):
        raw = str(text or '')
        if not any(x in raw for x in ('Ã', 'Â', 'â€', 'ðŸ')):
            return raw, False
        candidates = [raw]
        for enc in ('latin1', 'cp1252'):
            try: candidates.append(raw.encode(enc).decode('utf8'))
            except (UnicodeEncodeError, UnicodeDecodeError): pass
        def hits(s): return sum(1 for m in TOKEN_RX.finditer(s) if m.group(0).casefold() in self.s.tok)
        best = max(candidates, key=hits)
        return best, best != raw

    def _raw_tokens(self, text: str):
        rows = []; segment = 0
        markup_ranges=[(m.start(),m.end()) for m in re.finditer(r'<[^>]{1,256}>', text)]
        markup_ranges += [(m.start(),m.end()) for m in re.finditer(r'"(?:[^"\\]|\\.){1,80}"\s*:', text)]
        def in_markup(start, end):
            return any(start >= a and end <= b for a,b in markup_ranges)
        for i, m in enumerate(TOKEN_RX.finditer(text)):
            raw = m.group(0)
            if in_markup(m.start(),m.end()): kind = 'markup'
            elif DATE_RX.match(raw): kind = 'date'
            elif TIME_RX.match(raw): kind = 'time'
            elif NUMBER_RX.match(raw): kind = 'number'
            elif any(c.isalpha() for c in raw): kind = 'word'
            else: kind = 'syntax'
            rows.append({'raw':raw,'start':m.start(),'end':m.end(),'index':i,'segment':segment,'kind':kind})
            if raw in STRONG_BREAKS or (kind=='markup' and raw==':'): segment += 1
        return rows

    def _initial_exact(self, rows):
        out = []
        for row in rows:
            if row['kind'] != 'word': out.append(None); continue
            low = row['raw'].casefold(); alias = self.alias_bank.get(row['raw'])
            out.append(low if low in self.s.tok else (alias.canonical if alias else None))
        return out

    def _nearest_context(self, exact, pos):
        out = []
        for j in range(max(0,pos-self.context_window), min(len(exact),pos+self.context_window+1)):
            if j != pos and exact[j] is not None: out.append(str(exact[j]))
        return tuple(dict.fromkeys(out))

    def _resolve_words(self, rows):
        exact = self._initial_exact(rows); resolved = []
        for i, row in enumerate(rows):
            raw = row['raw']; kind = row['kind']
            if kind != 'word':
                resolved.append({'canonical':raw,'atoms':(raw,),'source':kind,'confidence':1.0,'context_support':0}); continue
            context = self._nearest_context(exact, i); alias = self.alias_bank.get(raw); low = raw.casefold()
            if alias is not None:
                resolved.append({'canonical':alias.canonical,'atoms':(alias.canonical,),'source':'learned_alias','confidence':alias.confidence,'context_support':alias.context_support}); exact[i]=alias.canonical; continue
            if low in self.s.tok:
                # A valid vocabulary word is never silently rewritten here. Contextual
                # ambiguity is handled by later graph/rule selection, keeping the clean
                # path O(1) and preventing caro->carro style meaning corruption.
                ctx = self._context_support(low, context)
                resolved.append({'canonical':low,'atoms':(low,),'source':'exact','confidence':1.0,'context_support':ctx}); exact[i]=low; continue
            fuzzy = self._best_fuzzy(raw, context)
            if fuzzy:
                canonical, dist, ctx, conf = fuzzy; source='accent_shadow' if dist == 0 else 'fuzzy'
                resolved.append({'canonical':canonical,'atoms':(canonical,),'source':source,'confidence':conf,'context_support':ctx}); exact[i]=canonical; continue
            split = self._best_split(raw)
            if split:
                phrase, atoms, support = split; resolved.append({'canonical':phrase,'atoms':atoms,'source':'split','confidence':0.92,'context_support':support}); exact[i]=atoms[-1]; continue
            resolved.append({'canonical':low,'atoms':(low,),'source':'unresolved','confidence':0.42,'context_support':0})
        return resolved

    def _merge_neighbors(self, rows, resolved):
        if not any(x['source'] in ('unresolved','fuzzy') for x in resolved):
            return {}, set()
        self._ensure_fuzzy_index(); skip=set(); merged={}
        for i in range(len(rows)):
            if i in skip or rows[i]['kind']!='word' or resolved[i]['source'] not in ('unresolved','fuzzy'):
                continue
            for width in (3,2):
                ids=list(range(i,min(len(rows),i+width)))
                if len(ids)!=width or any(rows[j]['kind']!='word' for j in ids):
                    continue
                joined=''.join(semantic_shadow(rows[j]['raw']) for j in ids)
                hit=self.shadow_best.get(joined)
                if hit:
                    merged[i]=(ids,hit[0],hit[1],(hit[0],)); skip.update(ids[1:]); break
                seg=self._best_phrase_segmentation(joined)
                if seg:
                    phrase,atoms,support=seg
                    merged[i]=(ids,phrase,support,atoms); skip.update(ids[1:]); break
        return merged,skip

    def _base_signal(self, canonical, source, confidence, context_support, kind):
        if kind in ('number','date','time'): return 1.0
        if kind != 'word': return 0.0
        if source == 'unresolved': return 0.38
        count=max((int(self.s.tok.get(a,0)) for a in canonical.split()),default=0)
        rarity=1.0-min(1.0,math.log1p(count)/max(1e-9,math.log1p(self.max_tok_count)))
        context=min(1.0,math.log1p(max(0,context_support))/7.0)
        score=0.36+0.28*rarity+0.20*context+0.10*confidence
        if count/max(1,self.s.total_tok)>0.0015: score*=0.62
        discourse=int(self.s.tables.get('open',{}).get(canonical,0))+int(self.s.tables.get('connect',{}).get(canonical,0))
        discourse_ratio=discourse/max(1,count)
        if discourse_ratio>0.25:
            score*=max(0.34,1.0-min(0.75,discourse_ratio))
        if 0<count<100:
            score*=0.72
        return max(0.0,min(1.0,score))

    def _build_tokens(self, rows, resolved, merged, skip):
        occ=Counter()
        for i,row in enumerate(rows):
            if i in skip: continue
            canonical=merged[i][1] if i in merged else resolved[i]['canonical']; occ[semantic_shadow(canonical)]+=1
        tokens=[]; first_segments=set(); emitted_counts=Counter(); numeric_positions={i for i,row in enumerate(rows) if row['kind'] in ('number','date','time')}
        for i,row in enumerate(rows):
            if i in skip: continue
            r=resolved[i]; raw=row['raw']; start=row['start']; end=row['end']; kind=row['kind']
            if i in merged:
                ids,canonical,count,atoms=merged[i]; raw=' '.join(rows[j]['raw'] for j in ids); end=rows[ids[-1]]['end']; source='joined'; conf=0.94; ctx=count
            else:
                canonical=r['canonical']; atoms=tuple(r.get('atoms') or (canonical,)); source=r['source']; conf=float(r['confidence']); ctx=int(r.get('context_support',0))
            signal=self._base_signal(canonical,source,conf,ctx,kind); key=semantic_shadow(canonical); redundant=False
            if kind=='word' and len(semantic_shadow(canonical))>=3 and any(abs(i-j)<=3 for j in numeric_positions):
                count=max((int(self.s.tok.get(a,0)) for a in canonical.split()),default=0)
                discourse=int(self.s.tables.get('open',{}).get(canonical,0))+int(self.s.tables.get('connect',{}).get(canonical,0))
                if count>0 and count/max(1,self.s.total_tok)<=0.0015 and discourse/max(1,count)<0.60:
                    signal=max(signal,0.54)
            if key and occ[key]>1 and emitted_counts[key]>0:
                prev_same=any(semantic_shadow(t.canonical)==key for t in tokens[-2:])
                if prev_same or source=='unresolved':
                    redundant=True; signal*=0.24 if source=='unresolved' else 0.55
            if kind=='word' and row['segment'] not in first_segments:
                first_segments.add(row['segment'])
                count=max((int(self.s.tok.get(a,0)) for a in canonical.split()),default=0)
                # Segment-initial low-frequency words are often entities/topics. Keep
                # them as nodes without declaring a linguistic POS or rewriting them.
                if source!='unresolved' and occ[key]<=2 and count/max(1,self.s.total_tok)<=0.0015:
                    signal=max(signal,0.52)
                elif source=='unresolved' and i+1<len(resolved) and resolved[i+1]['source']!='unresolved':
                    signal=max(signal,0.52)
            tokens.append(SemanticTokenV14(raw,canonical,kind,source,round(conf,6),round(signal,6),len(tokens),start,end,row['segment'],redundant,atoms))
            if key: emitted_counts[key]+=1
        return tokens

    def _signal_indices(self,tokens):
        return [t.index for t in tokens if t.kind in ('number','date','time') or (t.signal>=self.signal_threshold and not t.redundant)]

    def _local_echo_pairs(self,tokens,signal_indices):
        word_ids=[i for i in signal_indices if tokens[i].kind=='word']
        out=[]
        for p,ai in enumerate(word_ids):
            a=semantic_shadow(tokens[ai].canonical)
            if len(a)<4: continue
            for bi in word_ids[p+1:p+9]:
                if tokens[bi].index-tokens[ai].index>16: break
                b=semantic_shadow(tokens[bi].canonical)
                if len(b)<4 or a==b or a[0]!=b[0] or abs(len(a)-len(b))>2: continue
                d=bounded_damerau(a,b,2)
                if d<=2:
                    conf=1.0-d/max(len(a),len(b))
                    if conf>=0.60: out.append((ai,bi,conf,d))
        return out

    def _numeric_anchors(self,tokens,signal_indices):
        keep=set(signal_indices); echoes=self._local_echo_pairs(tokens,signal_indices); echo_map={}
        for a,b,conf,_ in echoes:
            if conf>=0.60: echo_map.setdefault(a,[]).append(tokens[b].canonical)
        out=[]
        for t in tokens:
            if t.kind not in ('number','date','time'): continue
            left_tokens=[x for x in tokens[max(0,t.index-4):t.index] if x.index in keep and x.kind=='word'][-2:]
            right_tokens=[x for x in tokens[t.index+1:t.index+5] if x.index in keep and x.kind=='word'][:2]
            inferred=[]
            for x in left_tokens+right_tokens:
                inferred.extend(echo_map.get(x.index,()))
            out.append({'raw':t.raw,'normalized':numeric_normalized(t.raw),'kind':t.kind,
                        'left_context':[x.canonical for x in left_tokens],
                        'right_context':[x.canonical for x in right_tokens],
                        'local_echo_context':list(dict.fromkeys(inferred)),
                        'token_index':t.index})
        return out

    def _phrases(self,tokens,signal_indices):
        word_ids=[i for i in signal_indices if tokens[i].kind=='word']; rows=[]; used=set()
        for n in range(5,1,-1):
            tab=self.s.tables.get(f'p{n}',{})
            for p in range(max(0,len(word_ids)-n+1)):
                ids=word_ids[p:p+n]
                if any(x in used for x in ids): continue
                seq=[tokens[x] for x in ids]
                if len({x.segment for x in seq})!=1 or any(seq[j+1].index-seq[j].index>2 for j in range(len(seq)-1)): continue
                atoms=[x.atoms[0] for x in seq if len(x.atoms)==1]
                if len(atoms)!=n: continue
                support=int(tab.get('\t'.join(atoms),0))
                if support>0:
                    rows.append(SemanticPhraseV14(' '.join(atoms),tuple(ids),support,round(sum(x.confidence for x in seq)/n,6))); used.update(ids)
        rows.sort(key=lambda x:(len(x.token_indices),x.support,x.confidence),reverse=True); return rows

    def _edges(self,tokens,signal_indices):
        out=[]
        for ap,ai in enumerate(signal_indices):
            a=tokens[ai]
            for bp in range(ap+1,min(len(signal_indices),ap+5)):
                bi=signal_indices[bp]; b=tokens[bi]; dist=max(1,b.index-a.index)
                if b.segment-a.segment>1 or dist>10: break
                support=self._pair_support(a.canonical.split()[-1],b.canonical.split()[0])
                if bp>ap+1 and support<=0 and dist>6: continue
                bridge=tuple(tokens[j].canonical for j in range(a.index+1,b.index) if tokens[j].kind=='word')[:4]
                strength=math.sqrt(max(.01,a.signal)*max(.01,b.signal))*(1+min(.7,math.log1p(support)/12.0))/(1+.08*max(0,dist-1))
                out.append(SemanticEdgeV14(a.canonical,b.canonical,a.index,b.index,dist,round(strength,6),support,bridge,'supported_skip' if support>0 and bp>ap+1 else 'ordered_context'))
        for ai,bi,conf,_ in self._local_echo_pairs(tokens,signal_indices):
            a=tokens[ai]; b=tokens[bi]
            out.append(SemanticEdgeV14(a.canonical,b.canonical,a.index,b.index,b.index-a.index,round(conf,6),0,(), 'local_echo_variant'))
        return out

    def extract(self,text: str) -> SemanticIntakeResultV14:
        t0=time.perf_counter(); self.extract_calls+=1; raw=str(text or ''); work,moji=self._repair_mojibake_shadow(raw); rows=self._raw_tokens(work); fuzzy0=self.fuzzy_queries
        resolved=self._resolve_words(rows); merged,skip=self._merge_neighbors(rows,resolved); tokens=self._build_tokens(rows,resolved,merged,skip); signal=self._signal_indices(tokens); phrases=self._phrases(tokens,signal); numeric=self._numeric_anchors(tokens,signal); edges=self._edges(tokens,signal); dt=time.perf_counter()-t0
        stats={'engine':'Robust-Semantic-Intake-V14','seconds':dt,'tokens':len(tokens),'signal_tokens':len(signal),'noise_tokens':max(0,len(tokens)-len(signal)),'signal_ratio':len(signal)/max(1,len(tokens)),'edges':len(edges),'phrases':len(phrases),'numeric_anchors':len(numeric),'sources':dict(Counter(x.source for x in tokens)),'fuzzy_queries':self.fuzzy_queries-fuzzy0,'fuzzy_index_built':self.shadow_best is not None,'fuzzy_index_build_seconds':self.fuzzy_index_build_seconds,'alias_bank_size':len(self.alias_bank.aliases),'mojibake_shadow_used':moji}
        return SemanticIntakeResultV14(raw,tokens,signal,edges,phrases,numeric,stats)

    def warm_index(self):
        self._ensure_fuzzy_index()
        return self.status()

    def status(self):
        return {'engine':'Robust-Semantic-Intake-V14','extract_calls':self.extract_calls,'fuzzy_queries_total':self.fuzzy_queries,'fuzzy_index_built':self.shadow_best is not None,'fuzzy_index_build_seconds':self.fuzzy_index_build_seconds,'candidate_cache':len(self.candidate_cache),'fast_fuzzy_min_support':self.fast_fuzzy_min_support,'fast_delete_signatures':len(self.delete_buckets or {}),'broad_index_built':self.len_buckets is not None,'alias_bank':self.alias_bank.status()}


class RobustNoiseLearnerV14:
    """Promotes recurring non-exact variants as explicit aliases during dataset ingestion."""
    def __init__(self,intake: RobustSemanticIntakeV14,min_support=3,min_dominance=.75,min_confidence=.72):
        self.intake=intake; self.min_support=int(min_support); self.min_dominance=float(min_dominance); self.min_confidence=float(min_confidence); self.votes=defaultdict(Counter); self.conf=defaultdict(lambda:defaultdict(float)); self.ctx=defaultdict(lambda:defaultdict(int)); self.contexts=defaultdict(lambda:defaultdict(set)); self.observations=0

    def observe(self,text: str):
        result=self.intake.extract(text); self.observations+=1
        for tok in result.tokens:
            if tok.kind!='word' or tok.source not in ('fuzzy','accent_shadow','split','joined'): continue
            if tok.raw.casefold() in self.intake.s.tok: continue
            raw=semantic_shadow(tok.raw); can=tok.canonical; self.votes[raw][can]+=1; self.conf[raw][can]+=tok.confidence
            self.ctx[raw][can]+=sum(e.support for e in result.edges if e.source_index==tok.index or e.target_index==tok.index)
            neighbors=[]
            for other in result.tokens[max(0,tok.index-3):tok.index+4]:
                if other.index!=tok.index and other.kind in ('word','number','date','time'):
                    neighbors.append(semantic_shadow(other.canonical))
            self.contexts[raw][can].add(tuple(x for x in neighbors if x))
        return result

    def promote(self):
        aliases={}
        for raw,votes in self.votes.items():
            total=sum(votes.values()); canonical,support=votes.most_common(1)[0]; dominance=support/max(1,total); mean=self.conf[raw][canonical]/max(1,support)
            diverse_contexts=len(self.contexts[raw][canonical]); combined=1.0-(1.0-max(0.0,min(1.0,mean)))**min(6,support)
            if support>=self.min_support and dominance>=self.min_dominance and diverse_contexts>=2 and combined>=self.min_confidence:
                aliases[raw]=AliasEvidenceV14(canonical,int(support),round(float(combined),6),int(self.ctx[raw][canonical]))
        return LearnedAliasBankV14(aliases,{'learner':'Robust-Noise-Learner-V14','observations':self.observations,'min_support':self.min_support,'min_dominance':self.min_dominance,'min_confidence':self.min_confidence,'requires_diverse_contexts':2})


def iter_training_projections(texts: Iterable[str], intake: RobustSemanticIntakeV14):
    for text in texts:
        yield intake.extract(text).to_training_projection()
