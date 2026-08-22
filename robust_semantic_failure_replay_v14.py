from __future__ import annotations

from collections import Counter
from pathlib import Path
import argparse, json, statistics, time

from prompt_session_v14 import PromptSessionV14
from robust_semantic_battery_v14 import FAILURE_ARCHIVE, OUT_DIR, evaluate, load_cases

OUT_PATH = OUT_DIR / "robust_semantic_failure_replay_v14.json"

def main():
    ap=argparse.ArgumentParser(description="Reexecuta permanentemente todas as falhas adversariais arquivadas da V14.")
    ap.add_argument("--limit",type=int,default=0)
    args=ap.parse_args()
    base={c["id"]:c for c in load_cases()}
    archived=[]
    if FAILURE_ARCHIVE.exists():
        for line_no,line in enumerate(FAILURE_ARCHIVE.read_text(encoding="utf8",errors="replace").splitlines(),1):
            if not line.strip(): continue
            row=json.loads(line); cid=row.get("case")
            if cid not in base: raise SystemExit(f"archive case desconhecido na linha {line_no}: {cid}")
            archived.append(row)
    if args.limit>0: archived=archived[:args.limit]
    session=PromptSessionV14(); intake=session.adapter._intake_for(session.scorer)
    rows=[]; t0=time.perf_counter()
    for old in archived:
        cid=old["case"]; text=str(old.get("text", "")); start=time.perf_counter(); result=intake.extract(text); latency=(time.perf_counter()-start)*1000
        ev=evaluate(base[cid],result)
        rows.append({"case":cid,"text":text,"latency_ms":latency,"previous_failure_classes":old.get("failure_classes",[]),**ev})
    resolved=[r for r in rows if r["passed"]]; unresolved=[r for r in rows if not r["passed"]]; lat=[r["latency_ms"] for r in rows]
    summary={"format":"Robust-Semantic-Failure-Replay-V14","archive_cases":len(rows),"resolved":len(resolved),"unresolved":len(unresolved),
             "resolved_ratio":len(resolved)/max(1,len(rows)),"unresolved_classes":dict(Counter(x for r in unresolved for x in r["failure_classes"])),
             "latency_ms":{"p50":statistics.median(lat) if lat else 0.0,"p95":sorted(lat)[max(0,int(.95*len(lat))-1)] if lat else 0.0,"max":max(lat) if lat else 0.0},
             "seconds":time.perf_counter()-t0,"model_load_seconds":session.load_seconds}
    OUT_DIR.mkdir(exist_ok=True); OUT_PATH.write_text(json.dumps({"summary":summary,"rows":rows},ensure_ascii=False,indent=2),encoding="utf8")
    print(json.dumps(summary,ensure_ascii=False,indent=2)); print("ROBUST SEMANTIC V14 FAILURE REPLAY: OK")

if __name__=="__main__": main()
