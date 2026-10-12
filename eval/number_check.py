"""Deterministic audit of llm_eval.json (no LLM): does each answer contain every number in the reference?

Strict lower bound: references sometimes carry extra context numbers the question did not ask for.
"""

import json
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
ZEN = str.maketrans("０１２３４５６７８９．", "0123456789.")


def numbers(s: str):
    return re.findall(r"\d+(?:\.\d+)?", s.translate(ZEN))


def main(path="results/llm_eval.json"):
    gold = {json.loads(x)["id"]: json.loads(x) for x in (ROOT / "data/gold.jsonl").read_text(encoding="utf-8").splitlines()}
    rows = [r for r in json.loads(Path(path).read_text(encoding="utf-8"))["rows"] if "answer" in r]
    misses = []
    for r in rows:
        ans = set(numbers(r["answer"]))
        missing = [n for n in numbers(gold[r["id"]]["answer"]) if n not in ans]
        if missing:
            misses.append({"id": r["id"], "missing": missing})
    out = {"n": len(rows), "all_reference_numbers_present": round(1 - len(misses) / max(len(rows), 1), 4),
           "misses": misses}
    print(json.dumps(out, ensure_ascii=False, indent=1))
    Path("results/number_check.json").write_text(json.dumps(out, ensure_ascii=False, indent=1), encoding="utf-8")


if __name__ == "__main__":
    main(*sys.argv[1:])
