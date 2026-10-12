"""Question / evidence wording overlap (why BM25 does well). NO LLM calls, no PDFs needed.

overlap = share of the question's character bigrams (punctuation removed) that also appear in
its evidence sentence(s). 1.0 = every bigram of the question is copied from the source text.

usage: python eval/overlap.py
"""

import json
import re
import statistics
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from rag_eval.bm25 import bigrams  # noqa: E402
from rag_eval.data import ROOT, load_gold  # noqa: E402
from rag_eval.text import normalize  # noqa: E402

PUNCT = re.compile(r"[、。？！?!「」『』()（）・,，:：]")


def overlap(question: str, evidence: str) -> float:
    q = set(bigrams(PUNCT.sub("", normalize(question))))
    e = set(bigrams(PUNCT.sub("", normalize(evidence))))
    return len(q & e) / max(len(q), 1)


def stats(xs):
    return {"n": len(xs), "mean": round(statistics.mean(xs), 4), "median": round(statistics.median(xs), 4),
            "min": round(min(xs), 4), "max": round(max(xs), 4)}


def main():
    gold = load_gold()
    hard = [json.loads(x) for x in (ROOT / "data" / "gold_hard.jsonl").read_text(encoding="utf-8").splitlines() if x]
    o42 = {g["id"]: round(overlap(g["question"], g["evidence"]), 4) for g in gold}
    oh = {h["id"]: round(overlap(h["question"], "".join(h["evidence"])), 4) for h in hard}
    out = {"metric": "share of question char bigrams found in evidence", "gold42": stats(list(o42.values())),
           "hard": stats(list(oh.values())), "per_question": {**o42, **oh}}
    Path(ROOT / "results" / "overlap.json").write_text(json.dumps(out, ensure_ascii=False, indent=1), encoding="utf-8")
    print(json.dumps({k: out[k] for k in ("gold42", "hard")}))


if __name__ == "__main__":
    main()
