from __future__ import annotations
from pathlib import Path
import hashlib, json, sys

ROOT = Path(__file__).resolve().parent
MODELS = [ROOT / "model" / "full", ROOT / "model" / "quality"]
REQUIRED = ["tokens.jsonl","p2.jsonl","p3.jsonl","p4.jsonl","p5.jsonl","open.jsonl","close.jsonl","connect.jsonl","structural.json","stats.json","MANIFEST.json"]

EXPECTED_FULL = {
    "tokens.jsonl": 120_000,
    "p2.jsonl": 120_000,
    "p3.jsonl": 100_000,
    "p4.jsonl": 70_000,
    "p5.jsonl": 50_000,
    "open.jsonl": 100_000,
    "close.jsonl": 100_000,
    "connect.jsonl": 100_000,
}


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def count_and_check_jsonl(path: Path) -> tuple[int, int]:
    n = 0
    previous = None
    bad_order = 0
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            obj = json.loads(line)
            if not isinstance(obj.get("k"), str) or not isinstance(obj.get("n"), int) or obj["n"] < 0:
                raise AssertionError(f"registro invalido em {path}: {obj!r}")
            if previous is not None and obj["n"] > previous:
                bad_order += 1
            previous = obj["n"]
            n += 1
    return n, bad_order


def lookup(path: Path, phrase: str) -> int | None:
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            obj = json.loads(line)
            if obj["k"] == phrase:
                return obj["n"]
    return None


def validate_one(model_dir: Path, exact_full: bool) -> dict:
    missing = [x for x in REQUIRED if not (model_dir / x).is_file()]
    if missing:
        raise AssertionError(f"{model_dir.name}: arquivos ausentes: {missing}")

    manifest = json.loads((model_dir / "MANIFEST.json").read_text(encoding="utf-8"))
    manifest_files = {x["name"]: x for x in manifest.get("files", [])}
    hash_bad = []
    for name, meta in manifest_files.items():
        p = model_dir / name
        if not p.is_file() or sha256(p) != meta["sha256"]:
            hash_bad.append(name)
    if hash_bad:
        raise AssertionError(f"{model_dir.name}: SHA-256 invalido: {hash_bad}")

    counts = {}
    order_violations = {}
    for name in REQUIRED:
        if name.endswith(".jsonl"):
            n, bad = count_and_check_jsonl(model_dir / name)
            counts[name] = n
            order_violations[name] = bad

    if exact_full:
        for name, expected in EXPECTED_FULL.items():
            if counts[name] != expected:
                raise AssertionError(f"full: {name}={counts[name]}, esperado={expected}")

    stats = json.loads((model_dir / "stats.json").read_text(encoding="utf-8"))
    if exact_full:
        assert stats["processed_shards"] == 459
        assert stats["missing_shards"] == [109]
        assert stats["deep_tokens"] > 900_000_000

    return {
        "model": model_dir.name,
        "counts": counts,
        "order_violations": order_violations,
        "stats": stats,
        "de_acordo_com": lookup(model_dir / "p3.jsonl", "de\tacordo\tcom"),
    }


def main() -> int:
    results = []
    for model_dir in MODELS:
        results.append(validate_one(model_dir, exact_full=(model_dir.name == "full")))
    print(json.dumps({"ok": True, "models": results}, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
