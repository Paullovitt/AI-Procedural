from __future__ import annotations

from pathlib import Path
import hashlib
import json
import lzma

from architecture_guard import SKIP_DIRS, scan_file

ROOT = Path(__file__).resolve().parent
TEMP_PATTERNS = (
    "_patch_*",
    "_final_patch_*",
    "rulevm_v6_output*.txt",
    "final_v6_preview.txt",
    "legacy_output_compare.txt",
)
REQUIRED = (
    "gpu_config.json",
    "RUN_GPU.bat",
    "run_gpu.py",
    "prompt_runtime_v14.py",
    "prompt_session_v14.py",
    "procedural_runtime_v14.py",
    "argument_planner_v14.py",
    "autonomous_rule_vm_v6.py",
    "test_v14_prompt.py",
    "test_rulevm_v6_prompt.py",
    "test_argument_planner_v14.py",
    "benchmark_prompt_rulevm_v6.py",
    "robust_semantic_intake_v14.py",
    "robust_semantic_battery_v14.py",
    "robust_semantic_regressions_v14.json",
    "test_robust_semantic_v14.py",
    "train_robust_semantic_v14.py",
    "robust_semantic_failure_archive_v14.jsonl",
    "robust_semantic_failure_replay_v14.py",
    "persistent_memory_v14.py",
    "persistent_memory_battery_v14.py",
    "test_persistent_memory_v14.py",
    "rigorous_results_v12/robust_semantic_v14.json",
    "rigorous_results_v12/robust_semantic_failure_replay_v14.json",
    "rigorous_results_v12/persistent_memory_v14.json",
    "PROJECT_RULES.md",
    "PROJECT_STATE_2026-08-21.md",
    "GPU_README.txt",
    "README.txt",
    "README.md",
    "requirements-gpu.txt",
    "SHA256SUMS.txt",
)


def audit_python(problems: list[str]) -> int:
    checked = 0
    for path in ROOT.rglob("*.py"):
        rel = path.relative_to(ROOT)
        if any(part in SKIP_DIRS for part in rel.parts[:-1]):
            continue
        checked += 1
        try:
            source = path.read_text(encoding="utf-8-sig")
            compile(source, str(path), "exec")
        except Exception as exc:
            problems.append(f"python parse: {rel}: {exc}")
        for row in scan_file(path):
            problems.append(f"architecture: {row}")
    return checked


def audit_json(problems: list[str]) -> int:
    count = 0
    for path in ROOT.rglob("*.json"):
        if ".git" in path.parts or "__pycache__" in path.parts:
            continue
        count += 1
        try:
            json.loads(path.read_text(encoding="utf8"))
        except Exception as exc:
            problems.append(f"json: {path.relative_to(ROOT)}: {exc}")
    for path in ROOT.rglob("*.xz"):
        if ".git" in path.parts:
            continue
        count += 1
        try:
            with lzma.open(path, "rt", encoding="utf8") as fh:
                json.load(fh)
        except Exception as exc:
            problems.append(f"xz-json: {path.relative_to(ROOT)}: {exc}")
    return count


def audit_manifest(problems: list[str]) -> int:
    manifest = ROOT / "SHA256SUMS.txt"
    seen: set[str] = set()
    for line in manifest.read_text(encoding="utf8").splitlines():
        if not line.strip():
            continue
        try:
            expected, rel = line.split(None, 1)
        except ValueError:
            problems.append(f"manifest malformed: {line}")
            continue
        rel = rel.strip()
        if rel.startswith("./"):
            rel = rel[2:]
        if rel in seen:
            problems.append(f"manifest duplicate: {rel}")
            continue
        seen.add(rel)
        path = ROOT / rel
        if not path.is_file():
            problems.append(f"manifest missing: {rel}")
            continue
        got = hashlib.sha256(path.read_bytes()).hexdigest()
        if got != expected:
            problems.append(f"manifest hash mismatch: {rel}")
    return len(seen)


def audit_consistency(problems: list[str]) -> None:
    for rel in REQUIRED:
        if not (ROOT / rel).is_file():
            problems.append(f"required file missing: {rel}")
    if (ROOT / "README.md").read_bytes() != (ROOT / "README.txt").read_bytes():
        problems.append("README.md and README.txt differ")

    cfg = json.loads((ROOT / "gpu_config.json").read_text(encoding="utf8"))
    if cfg.get("runtime_version") != "V14":
        problems.append(f"runtime_version is not V14: {cfg.get('runtime_version')}")
    if cfg.get("content_reasoner") != "Learned-Association-RuleVM-v6":
        problems.append(f"unexpected content_reasoner: {cfg.get('content_reasoner')}")
    if cfg.get("argument_planner_enabled") is not True:
        problems.append("argument_planner_enabled is not true")
    if cfg.get("robust_semantic_intake_enabled") is not True:
        problems.append("robust_semantic_intake_enabled is not true")
    if cfg.get("robust_semantic_warm_index") is not True:
        problems.append("robust_semantic_warm_index is not true")
    if cfg.get("persistent_memory_enabled") is not True:
        problems.append("persistent_memory_enabled is not true")
    if int(cfg.get("persistent_memory_top_k",0)) <= 0:
        problems.append("persistent_memory_top_k is not positive")
    if not (0.0 < float(cfg.get("persistent_memory_min_query_term_coverage",0.0)) <= 1.0):
        problems.append("persistent_memory_min_query_term_coverage is invalid")
    if not (0.0 < float(cfg.get("persistent_memory_max_associative_document_ratio",0.0)) <= 1.0):
        problems.append("persistent_memory_max_associative_document_ratio is invalid")

    launcher = (ROOT / "RUN_GPU.bat").read_text(encoding="utf8", errors="replace")
    if "prompt_session_v14.py" not in launcher:
        problems.append("RUN_GPU.bat does not launch prompt_session_v14.py")

    req = (ROOT / "requirements-gpu.txt").read_text(encoding="utf8")
    if "V9 GPU" in req:
        problems.append("requirements-gpu.txt still references V9 GPU as current runtime")

    docs = "\n".join(
        (ROOT / rel).read_text(encoding="utf8", errors="replace")
        for rel in ("README.md", "GPU_README.txt", "PROJECT_STATE_2026-08-21.md")
    )
    if "Argument Planner" not in docs:
        problems.append("project documentation does not mention Argument Planner")
    if "Robust Semantic Intake" not in docs:
        problems.append("project documentation does not mention Robust Semantic Intake")
    if "Persistent Dimensional Memory" not in docs:
        problems.append("project documentation does not mention Persistent Dimensional Memory")

    regressions = json.loads((ROOT / "robust_semantic_regressions_v14.json").read_text(encoding="utf8"))
    if regressions.get("format") != "Robust-Semantic-Regressions-V14":
        problems.append("unexpected robust semantic regression format")
    if len(regressions.get("cases", [])) < 16:
        problems.append("robust semantic permanent regressions below promoted floor of 16")

    prompt_benchmark=json.loads((ROOT / "rigorous_results_v12/prompt_rulevm_v6_generality.json").read_text(encoding="utf8"))
    if prompt_benchmark.get("persistent_memory_isolated") is not True:
        problems.append("prompt RuleVM benchmark is not isolated from persistent episodic memory")
    prompt_benchmark_text="\n".join(str(x.get("text","")) for x in prompt_benchmark.get("rows",[])).casefold()
    if "memória recuperada" in prompt_benchmark_text or "memory_retrieval" in json.dumps(prompt_benchmark,ensure_ascii=False):
        problems.append("prompt RuleVM benchmark was contaminated by persistent memory retrieval")

    robust_result = json.loads((ROOT / "rigorous_results_v12/robust_semantic_v14.json").read_text(encoding="utf8"))
    robust_summary = robust_result.get("summary", {})
    if robust_summary.get("fixed_reference_passed") != robust_summary.get("fixed_reference_total"):
        problems.append("robust semantic fixed-reference gate is not clean")
    if int(robust_summary.get("fixed_reference_total", 0)) < 16:
        problems.append("robust semantic result does not cover all 16 permanent references")

    replay = json.loads((ROOT / "rigorous_results_v12/robust_semantic_failure_replay_v14.json").read_text(encoding="utf8"))
    replay_summary = replay.get("summary", {})
    if replay_summary.get("format") != "Robust-Semantic-Failure-Replay-V14":
        problems.append("unexpected robust semantic failure replay format")
    archive_cases = int(replay_summary.get("archive_cases", 0))
    resolved = int(replay_summary.get("resolved", 0))
    unresolved = int(replay_summary.get("unresolved", 0))
    archive_lines = [x for x in (ROOT / "robust_semantic_failure_archive_v14.jsonl").read_text(encoding="utf8").splitlines() if x.strip()]
    if archive_cases <= 0 or resolved + unresolved != archive_cases:
        problems.append("robust semantic failure replay counts are inconsistent")
    if archive_cases != len(archive_lines):
        problems.append("failure replay does not cover the complete current archive")
    if int(robust_summary.get("failure_archive_cases", -1)) != len(archive_lines):
        problems.append("robust battery result and cumulative failure archive are out of sync")
    if "Failure Replay" not in docs and "failure replay" not in docs:
        problems.append("project documentation does not mention failure replay")

    memory_result=json.loads((ROOT / "rigorous_results_v12/persistent_memory_v14.json").read_text(encoding="utf8"))
    if memory_result.get("format") != "Persistent-Dimensional-Memory-V14-Battery":
        problems.append("unexpected persistent memory battery format")
    if memory_result.get("gate") != "OK":
        problems.append("persistent memory battery gate is not OK")
    store=memory_result.get("store",{})
    live=memory_result.get("live_v14",{})
    if int(store.get("records_requested",0)) < 20000:
        problems.append("persistent memory promoted battery below 20k episodes")
    if float(store.get("exact_top1_accuracy",0.0)) < 0.99 or float(store.get("numeric_top1_accuracy",0.0)) < 0.99:
        problems.append("persistent memory retrieval accuracy below gate")
    if int(store.get("irrelevant_false_positives",1)) != 0:
        problems.append("persistent memory irrelevant false-positive gate failed")
    if store.get("weak_overlap_false_positive") is not False or store.get("weak_exact_ok") is not True:
        problems.append("persistent memory weak-overlap relevance gate failed")
    if int(live.get("weak_topic_retrieved",1)) != 0:
        problems.append("persistent memory live weak-topic isolation failed")
    if live.get("second_turn_question_stored") is not False or live.get("second_turn_store_skipped_reason") != "interrogative":
        problems.append("persistent memory interrogative auto-store gate failed")
    if int(live.get("second_turn_retrieved",0)) != 1 or int(live.get("third_turn_retrieved",0)) != 1:
        problems.append("persistent memory repeated-question retrieval is not stable")
    if int(live.get("second_turn_injected_rules",0)) < 1 or int(live.get("third_turn_injected_rules",0)) < 1:
        problems.append("persistent memory repeated-question rule injection failed")
    if live.get("answer_contains_civic_2015") is not True or live.get("third_answer_contains_civic_2015") is not True:
        problems.append("persistent memory repeated-question factual recall failed")
    if not (live.get("semantic_verified") and int(live.get("slot_errors",1))==0 and int(live.get("trace_errors",1))==0):
        problems.append("persistent memory live V14 semantic gate failed")
    gitignore=(ROOT/'.gitignore').read_text(encoding='utf8')
    if 'memory_v14/' not in gitignore:
        problems.append("memory_v14 runtime database is not ignored by Git")

    for pattern in TEMP_PATTERNS:
        for path in ROOT.glob(pattern):
            problems.append(f"temporary artifact present: {path.name}")


def main() -> None:
    problems: list[str] = []
    py_count = audit_python(problems)
    json_count = audit_json(problems)
    manifest_count = audit_manifest(problems)
    audit_consistency(problems)

    if problems:
        print("PROJECT AUDIT: FAIL")
        for row in problems:
            print(" -", row)
        raise SystemExit(1)

    print(
        f"PROJECT AUDIT: OK; python={py_count}; "
        f"json/xz={json_count}; manifest={manifest_count}"
    )


if __name__ == "__main__":
    main()
