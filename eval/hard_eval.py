"""Harder question set (paraphrased + multi-chunk), scored SEPARATELY. NO LLM calls.

Each question has one or more evidence "groups"; a group is a list of acceptable (doc, page).
- page-level hit@k / MRR@5: rank of the first chunk from ANY acceptable page.
- all@5 (multi-chunk questions): every group has at least one chunk in the top 5.

usage: python eval/hard_eval.py --methods bm25 dense --chunk-sizes 256 512 1024
"""

import argparse
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from rag_eval.data import PDF_DIR, ROOT, download  # noqa: E402
from rag_eval.metrics import summarize  # noqa: E402
from rag_eval.sources import SOURCES  # noqa: E402
from rag_eval.text import chunk_pages, load_pdf_pages, normalize  # noqa: E402
from retrieval_eval import DENSE_MODEL, build_retriever  # noqa: E402

HARD = ROOT / "data" / "gold_hard.jsonl"


def load_hard(path=HARD):
    return [json.loads(x) for x in Path(path).read_text(encoding="utf-8").splitlines() if x.strip()]


def in_group(chunk, group):
    return any(chunk["doc"] == g["doc"] and chunk["page"] == g["page"] for g in group)


def any_rank(top, q):
    for r, c in enumerate(top, 1):
        if any(in_group(c, g) for g in q["groups"]):
            return r
    return 0


def all_groups_hit(top, q):
    return all(any(in_group(c, g) for c in top) for g in q["groups"])


def check_evidence(q, pages):
    """Every evidence string must appear on one of the acceptable pages (guards typos)."""
    text = {(p["doc"], p["page"]): p["text"] for p in pages}
    acc = {(g["doc"], g["page"]) for grp in q["groups"] for g in grp}
    return all(any(normalize(e) in text[k] for k in acc) for e in q["evidence"])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--methods", nargs="+", default=["bm25"])
    ap.add_argument("--chunk-sizes", nargs="+", type=int, default=[512])
    ap.add_argument("--overlap", type=int, default=50)
    ap.add_argument("--out", default="results/retrieval_hard.json")
    a = ap.parse_args()
    hashes = download()
    hard = load_hard()
    pages = [p for name in SOURCES for p in load_pdf_pages(PDF_DIR / name)]
    bad = [q["id"] for q in hard if not check_evidence(q, pages)]
    if bad:
        raise SystemExit(f"evidence not found on stated pages: {bad}")
    cache, runs = {}, []
    for size in a.chunk_sizes:
        chunks = chunk_pages(pages, size, a.overlap)
        for method in a.methods:
            t0 = time.time()
            search = build_retriever(method, chunks, cache)
            ranks, per_q, multi_all = {"all": [], "paraphrase": [], "multi": []}, [], []
            for q in hard:
                top = [chunks[i] for i in search(q["question"], 5)]
                r = any_rank(top, q)
                ranks["all"].append(r)
                ranks[q["type"]].append(r)
                row = {"id": q["id"], "type": q["type"], "page_rank": r}
                if q["type"] == "multi":
                    row["all_groups@5"] = all_groups_hit(top, q)
                    multi_all.append(row["all_groups@5"])
                per_q.append(row)
            run = {"method": method, "chunk_size": size, "overlap": a.overlap, "n_chunks": len(chunks),
                   "n_questions": len(hard),
                   "page_level": {k: summarize(v) | {"n": len(v)} for k, v in ranks.items()},
                   "multi_all_groups@5": round(sum(multi_all) / max(len(multi_all), 1), 4),
                   "seconds": round(time.time() - t0, 1), "per_question": per_q}
            runs.append(run)
            print(method, size, run["page_level"]["all"], "multi all@5", run["multi_all_groups@5"])
    out = {"pdf_sha256": hashes, "dense_model": DENSE_MODEL, "llm_calls": 0, "runs": runs}
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    Path(a.out).write_text(json.dumps(out, ensure_ascii=False, indent=1), encoding="utf-8")


if __name__ == "__main__":
    main()
