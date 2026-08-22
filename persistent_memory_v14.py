from __future__ import annotations

from collections import Counter
from pathlib import Path
import hashlib
import json
import math
import re
import sqlite3
import time
import unicodedata

TOKEN_RE = re.compile(r"[^\W_]+", re.UNICODE)
SCHEMA_VERSION = 2


def should_auto_store_user_memory(text: str) -> bool:
    """Promote user text only when it is not explicitly interrogative.

    This is deliberately domain/language agnostic: question punctuation is a
    conversational control signal, not a semantic rule about any subject.
    """
    raw = str(text or "").strip()
    return bool(raw) and "?" not in raw and "¿" not in raw


def memory_norm(text: str) -> str:
    raw = unicodedata.normalize("NFKD", str(text or "").casefold())
    return "".join(ch for ch in raw if not unicodedata.combining(ch))


def memory_tokens(text: str) -> list[str]:
    return [memory_norm(x) for x in TOKEN_RE.findall(str(text or "")) if x]


class PersistentDimensionalMemoryV14:
    """Persistent, explicit episodic memory for V14.

    This is a non-neural second memory layer inspired by the user's AI-Memory
    dimensional index. Knowledge stays auditable in SQLite tables: episodes, terms,
    postings and directed adjacency counts. It never replaces the Bagaço corpus.

    The database is incrementally writable and directly reopenable between processes;
    no full ``fit()``/index rebuild is required at startup.
    """

    def __init__(self, path: str | Path, *, candidate_limit: int = 512,
                 associative_per_term: int = 4, min_query_term_coverage: float = 0.0,
                 max_associative_document_ratio: float = 0.20,
                 timeout: float = 10.0):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.candidate_limit = max(32, int(candidate_limit))
        self.associative_per_term = max(0, int(associative_per_term))
        self.min_query_term_coverage = max(0.0, min(1.0, float(min_query_term_coverage)))
        self.max_associative_document_ratio = max(0.0, min(1.0, float(max_associative_document_ratio)))
        self.db = sqlite3.connect(str(self.path), timeout=float(timeout))
        self.db.row_factory = sqlite3.Row
        self.db.execute("PRAGMA journal_mode=WAL")
        self.db.execute("PRAGMA synchronous=NORMAL")
        self.db.execute("PRAGMA temp_store=MEMORY")
        self._ensure_schema()

    def _ensure_schema(self):
        self.db.executescript(
            """
            CREATE TABLE IF NOT EXISTS meta(
                key TEXT PRIMARY KEY,
                value TEXT NOT NULL
            );
            CREATE TABLE IF NOT EXISTS episodes(
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                created_ns INTEGER NOT NULL,
                last_seen_ns INTEGER NOT NULL,
                source TEXT NOT NULL,
                text TEXT NOT NULL,
                normalized TEXT NOT NULL,
                index_text TEXT NOT NULL,
                fingerprint TEXT NOT NULL UNIQUE,
                recurrence INTEGER NOT NULL DEFAULT 1,
                metadata_json TEXT NOT NULL DEFAULT '{}'
            );
            CREATE TABLE IF NOT EXISTS terms(
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                term TEXT NOT NULL UNIQUE,
                frequency INTEGER NOT NULL DEFAULT 0,
                document_frequency INTEGER NOT NULL DEFAULT 0
            );
            CREATE TABLE IF NOT EXISTS postings(
                term_id INTEGER NOT NULL,
                episode_id INTEGER NOT NULL,
                PRIMARY KEY(term_id, episode_id),
                FOREIGN KEY(term_id) REFERENCES terms(id) ON DELETE CASCADE,
                FOREIGN KEY(episode_id) REFERENCES episodes(id) ON DELETE CASCADE
            );
            CREATE INDEX IF NOT EXISTS idx_postings_episode ON postings(episode_id);
            CREATE TABLE IF NOT EXISTS edges(
                src_term_id INTEGER NOT NULL,
                dst_term_id INTEGER NOT NULL,
                count INTEGER NOT NULL DEFAULT 0,
                PRIMARY KEY(src_term_id, dst_term_id),
                FOREIGN KEY(src_term_id) REFERENCES terms(id) ON DELETE CASCADE,
                FOREIGN KEY(dst_term_id) REFERENCES terms(id) ON DELETE CASCADE
            );
            CREATE INDEX IF NOT EXISTS idx_edges_dst ON edges(dst_term_id);
            """
        )
        columns={str(r[1]) for r in self.db.execute("PRAGMA table_info(episodes)")}
        if 'index_text' not in columns:
            self.db.execute("ALTER TABLE episodes ADD COLUMN index_text TEXT NOT NULL DEFAULT ''")
            self.db.execute("UPDATE episodes SET index_text=text WHERE index_text='' ")
        self.db.execute(
            "INSERT INTO meta(key,value) VALUES('schema_version',?) "
            "ON CONFLICT(key) DO UPDATE SET value=excluded.value",
            (str(SCHEMA_VERSION),),
        )
        self.db.commit()

    @staticmethod
    def _fingerprint(source: str, normalized: str) -> str:
        raw = f"{source}\0{normalized}".encode("utf-8", "replace")
        return hashlib.sha256(raw).hexdigest()

    def _term_ids_for_write(self, counts: Counter[str], is_new_document: bool) -> dict[str, int]:
        out: dict[str, int] = {}
        for term, count in counts.items():
            row = self.db.execute("SELECT id FROM terms WHERE term=?", (term,)).fetchone()
            if row is None:
                cur = self.db.execute(
                    "INSERT INTO terms(term,frequency,document_frequency) VALUES(?,?,?)",
                    (term, int(count), 1 if is_new_document else 0),
                )
                tid = int(cur.lastrowid)
            else:
                tid = int(row["id"])
                self.db.execute(
                    "UPDATE terms SET frequency=frequency+?, "
                    "document_frequency=document_frequency+? WHERE id=?",
                    (int(count), 1 if is_new_document else 0, tid),
                )
            out[term] = tid
        return out

    def remember(self, text: str, *, source: str = "user", metadata: dict | None = None,
                 index_text: str | None = None) -> dict:
        text = str(text or "").strip()
        if not text:
            raise ValueError("Memória vazia.")
        source = str(source or "user")
        raw_ts = memory_tokens(text)
        ts = memory_tokens(index_text if index_text is not None else text)
        if not raw_ts or not ts:
            raise ValueError("Memória sem termos indexáveis.")
        normalized = " ".join(raw_ts)
        index_text = str(index_text if index_text is not None else text)
        fingerprint = self._fingerprint(source, normalized)
        now = time.time_ns()
        metadata_json = json.dumps(dict(metadata or {}), ensure_ascii=False, separators=(",", ":"))
        counts = Counter(ts)

        with self.db:
            existing = self.db.execute(
                "SELECT id,recurrence FROM episodes WHERE fingerprint=?", (fingerprint,)
            ).fetchone()
            is_new = existing is None
            if is_new:
                cur = self.db.execute(
                    "INSERT INTO episodes(created_ns,last_seen_ns,source,text,normalized,index_text,fingerprint,recurrence,metadata_json) "
                    "VALUES(?,?,?,?,?,?,?,1,?)",
                    (now, now, source, text, normalized, index_text, fingerprint, metadata_json),
                )
                episode_id = int(cur.lastrowid)
                recurrence = 1
            else:
                episode_id = int(existing["id"])
                recurrence = int(existing["recurrence"]) + 1
                self.db.execute(
                    "UPDATE episodes SET recurrence=?,last_seen_ns=?,metadata_json=?,index_text=? WHERE id=?",
                    (recurrence, now, metadata_json, index_text, episode_id),
                )

            term_ids = self._term_ids_for_write(counts, is_new_document=is_new)
            if is_new:
                self.db.executemany(
                    "INSERT OR IGNORE INTO postings(term_id,episode_id) VALUES(?,?)",
                    ((term_ids[t], episode_id) for t in counts),
                )
            ids = [term_ids[t] for t in ts]
            edge_counts = Counter(zip(ids, ids[1:]))
            for (src, dst), count in edge_counts.items():
                self.db.execute(
                    "INSERT INTO edges(src_term_id,dst_term_id,count) VALUES(?,?,?) "
                    "ON CONFLICT(src_term_id,dst_term_id) DO UPDATE SET count=count+excluded.count",
                    (int(src), int(dst), int(count)),
                )

        return {
            "id": episode_id,
            "source": source,
            "recurrence": recurrence,
            "new": bool(is_new),
            "terms": len(counts),
        }

    def _query_terms(self, query: str):
        q = list(dict.fromkeys(memory_tokens(query)))
        if not q:
            return []
        placeholders = ",".join("?" for _ in q)
        rows = self.db.execute(
            f"SELECT id,term,document_frequency FROM terms WHERE term IN ({placeholders})", q
        ).fetchall()
        by_term = {str(r["term"]): r for r in rows}
        return [by_term[t] for t in q if t in by_term]

    def _posting_docs(self, term_id: int, limit: int) -> list[int]:
        return [int(r[0]) for r in self.db.execute(
            "SELECT episode_id FROM postings WHERE term_id=? ORDER BY episode_id DESC LIMIT ?",
            (int(term_id), int(limit)),
        )]

    def _associative_term_ids(self, qids: list[int], total_docs: int) -> dict[int, int]:
        if not qids or self.associative_per_term <= 0:
            return {}
        n=max(1,int(total_docs))
        max_df=max(1,int(n*self.max_associative_document_ratio)) if self.max_associative_document_ratio>0 else n
        q_placeholders=",".join("?" for _ in qids)
        q_df={int(r["id"]):int(r["document_frequency"]) for r in self.db.execute(
            f"SELECT id,document_frequency FROM terms WHERE id IN ({q_placeholders})", qids
        )}
        best: dict[int, int] = {}
        for qid in qids:
            # Very frequent query terms are graph hubs. They remain valid for exact
            # postings but are not allowed to fan out into associative candidates.
            if q_df.get(qid,n)>max_df:
                continue
            rows = self.db.execute(
                "SELECT e.dst_term_id AS tid,e.count,t.document_frequency AS df "
                "FROM edges e JOIN terms t ON t.id=e.dst_term_id WHERE e.src_term_id=? "
                "UNION ALL "
                "SELECT e.src_term_id AS tid,e.count,t.document_frequency AS df "
                "FROM edges e JOIN terms t ON t.id=e.src_term_id WHERE e.dst_term_id=?",
                (qid, qid),
            ).fetchall()
            rows=[r for r in rows if int(r["df"])<=max_df]
            rows = sorted(rows, key=lambda r: (int(r["count"]),-int(r["df"])), reverse=True)[:self.associative_per_term]
            for row in rows:
                tid = int(row["tid"])
                if tid in qids:
                    continue
                best[tid] = max(best.get(tid, 0), int(row["count"]))
        return best

    def _episode_rows(self, ids: list[int]):
        if not ids:
            return []
        placeholders = ",".join("?" for _ in ids)
        return self.db.execute(
            f"SELECT id,created_ns,last_seen_ns,source,text,recurrence,metadata_json "
            f"FROM episodes WHERE id IN ({placeholders})",
            ids,
        ).fetchall()

    def search(self, query: str, *, k: int = 4, associative: bool = True) -> list[dict]:
        k = max(1, int(k))
        t0 = time.perf_counter()
        query_terms=list(dict.fromkeys(memory_tokens(query)))
        term_rows = self._query_terms(query)
        if not term_rows:
            return []
        # Generic relevance gate: a single shared word must not pull a long, otherwise
        # unrelated query into an old episode. Short queries remain permissive.
        recognized_ratio=len(term_rows)/max(1,len(query_terms))
        if len(query_terms)>=3 and recognized_ratio < self.min_query_term_coverage:
            return []
        n = max(1, int(self.db.execute("SELECT COUNT(*) FROM episodes").fetchone()[0]))
        term_rows = sorted(term_rows, key=lambda r: (int(r["document_frequency"]), int(r["id"])))
        qids = [int(r["id"]) for r in term_rows]
        df = {int(r["id"]): int(r["document_frequency"]) for r in term_rows}

        candidate_ids: list[int] = []
        seen = set()
        for row in term_rows:
            for doc in self._posting_docs(int(row["id"]), self.candidate_limit):
                if doc not in seen:
                    seen.add(doc)
                    candidate_ids.append(doc)
                    if len(candidate_ids) >= self.candidate_limit:
                        break
            if len(candidate_ids) >= self.candidate_limit:
                break

        exact_ids = set(candidate_ids)
        assoc_strength: dict[int, int] = {}
        if associative and len(candidate_ids) < max(k, 2):
            assoc_terms = self._associative_term_ids(qids,n)
            for tid, strength in sorted(assoc_terms.items(), key=lambda x: x[1], reverse=True):
                for doc in self._posting_docs(tid, max(16, self.candidate_limit // 8)):
                    if doc not in seen:
                        seen.add(doc)
                        candidate_ids.append(doc)
                        assoc_strength[doc] = max(assoc_strength.get(doc, 0), strength)
                        if len(candidate_ids) >= self.candidate_limit:
                            break
                if len(candidate_ids) >= self.candidate_limit:
                    break

        if not candidate_ids:
            return []
        docs = self._episode_rows(candidate_ids)
        placeholders_docs = ",".join("?" for _ in candidate_ids)
        placeholders_q = ",".join("?" for _ in qids)
        matched: dict[int, set[int]] = {}
        if qids:
            for row in self.db.execute(
                f"SELECT episode_id,term_id FROM postings WHERE episode_id IN ({placeholders_docs}) "
                f"AND term_id IN ({placeholders_q})",
                candidate_ids + qids,
            ):
                matched.setdefault(int(row["episode_id"]), set()).add(int(row["term_id"]))

        ranked = []
        for row in docs:
            doc_id = int(row["id"])
            mids = matched.get(doc_id, set())
            exact = bool(mids)
            lexical_score = sum((math.log((1.0 + n) / (1.0 + df[tid])) + 1.0) ** 2 for tid in mids)
            coverage = len(mids) / max(1, len(qids))
            recurrence = int(row["recurrence"])
            association = int(assoc_strength.get(doc_id, 0))
            rank_key = (
                1 if exact else 0,
                float(lexical_score),
                float(coverage),
                association,
                math.log1p(recurrence),
                int(row["last_seen_ns"]),
            )
            ranked.append((rank_key, row, lexical_score, coverage, association, exact))
        ranked.sort(key=lambda x: x[0], reverse=True)
        latency_ms = (time.perf_counter() - t0) * 1000.0

        out = []
        for _, row, score, coverage, association, exact in ranked[:k]:
            try:
                metadata = json.loads(str(row["metadata_json"] or "{}"))
            except json.JSONDecodeError:
                metadata = {}
            out.append({
                "id": int(row["id"]),
                "text": str(row["text"]),
                "source": str(row["source"]),
                "recurrence": int(row["recurrence"]),
                "created_ns": int(row["created_ns"]),
                "last_seen_ns": int(row["last_seen_ns"]),
                "score": round(float(score), 6),
                "coverage": round(float(coverage), 6),
                "association_strength": association,
                "match_kind": "exact" if exact else "associative",
                "metadata": metadata,
                "search_ms": round(latency_ms, 6),
            })
        return out

    def recent(self, k: int = 4, *, source: str | None = None) -> list[dict]:
        k = max(1, int(k))
        if source is None:
            rows = self.db.execute(
                "SELECT id,created_ns,last_seen_ns,source,text,recurrence,metadata_json "
                "FROM episodes ORDER BY last_seen_ns DESC LIMIT ?", (k,)
            ).fetchall()
        else:
            rows = self.db.execute(
                "SELECT id,created_ns,last_seen_ns,source,text,recurrence,metadata_json "
                "FROM episodes WHERE source=? ORDER BY last_seen_ns DESC LIMIT ?", (str(source), k)
            ).fetchall()
        return [dict(r) for r in rows]

    def forget(self, episode_id: int) -> bool:
        row = self.db.execute(
            "SELECT index_text,text,recurrence FROM episodes WHERE id=?", (int(episode_id),)
        ).fetchone()
        if row is None:
            return False
        ts=memory_tokens(str(row["index_text"] or row["text"]))
        counts=Counter(ts)
        recurrence=max(1,int(row["recurrence"]))
        term_ids={}
        for term in counts:
            hit=self.db.execute("SELECT id FROM terms WHERE term=?",(term,)).fetchone()
            if hit is not None:
                term_ids[term]=int(hit["id"])
        ids=[term_ids[t] for t in ts if t in term_ids]
        edge_counts=Counter(zip(ids,ids[1:]))

        with self.db:
            self.db.execute("DELETE FROM postings WHERE episode_id=?", (int(episode_id),))
            self.db.execute("DELETE FROM episodes WHERE id=?", (int(episode_id),))
            for term,count in counts.items():
                tid=term_ids.get(term)
                if tid is None:
                    continue
                self.db.execute(
                    "UPDATE terms SET frequency=MAX(0,frequency-?), "
                    "document_frequency=MAX(0,document_frequency-1) WHERE id=?",
                    (int(count)*recurrence,tid),
                )
            for (src,dst),count in edge_counts.items():
                self.db.execute(
                    "UPDATE edges SET count=MAX(0,count-?) WHERE src_term_id=? AND dst_term_id=?",
                    (int(count)*recurrence,int(src),int(dst)),
                )
            self.db.execute("DELETE FROM edges WHERE count<=0")
            self.db.execute("DELETE FROM terms WHERE document_frequency<=0")
        return True

    def clear(self):
        with self.db:
            self.db.execute("DELETE FROM postings")
            self.db.execute("DELETE FROM edges")
            self.db.execute("DELETE FROM terms")
            self.db.execute("DELETE FROM episodes")

    def stats(self) -> dict:
        episodes = int(self.db.execute("SELECT COUNT(*) FROM episodes").fetchone()[0])
        dimensions = int(self.db.execute("SELECT COUNT(*) FROM terms").fetchone()[0])
        edges = int(self.db.execute("SELECT COUNT(*) FROM edges").fetchone()[0])
        recurrences = int(self.db.execute("SELECT COALESCE(SUM(recurrence),0) FROM episodes").fetchone()[0])
        size = self.path.stat().st_size if self.path.exists() else 0
        wal = self.path.with_name(self.path.name + "-wal")
        wal_size = wal.stat().st_size if wal.exists() else 0
        return {
            "engine": "Persistent-Dimensional-Memory-V14",
            "schema_version": SCHEMA_VERSION,
            "episodes": episodes,
            "observations": recurrences,
            "dimensions": dimensions,
            "directed_edges": edges,
            "database_bytes": int(size + wal_size),
            "candidate_limit": self.candidate_limit,
            "associative_per_term": self.associative_per_term,
            "min_query_term_coverage": self.min_query_term_coverage,
            "max_associative_document_ratio": self.max_associative_document_ratio,
        }

    def close(self):
        if self.db is not None:
            self.db.commit()
            self.db.close()
            self.db = None

    def __del__(self):
        try:
            self.close()
        except Exception:
            pass

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, tb):
        self.close()


__all__ = ["PersistentDimensionalMemoryV14", "memory_norm", "memory_tokens", "should_auto_store_user_memory"]
