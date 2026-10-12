"""Retrieval eval: hit@k and MRR on the frozen gold set. NO LLM calls.

usage: python eval/retrieval_eval.py --methods bm25 dense --chunk-sizes 256 512 1024
"""

import argparse
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from rag_eval.bm25 import BM25  # noqa: E402
from rag_eval.data import PDF_DIR, download, load_gold  # noqa: E402
from rag_eval.metrics import evidence_rank, first_hit_rank, summarize  # noqa: E402
from rag_eval.sources import SOURCES  # noqa: E402
from rag_eval.text import chunk_pages, load_pdf_pages  # noqa: E402

DENSE_MODEL = "BAAI/bge-m3"


def build_retriever(method, chunks, model_cache):
    texts = [c["text"] for c in chunks]
    if method == "bm25":
        bm = BM25(texts)
        return lambda q, k: [i for i, _ in bm.search(q, k)]
    if method == "dense":
        import numpy as np
        from sentence_transformers import SentenceTransformer

        if "m" not in model_cache:
            model_cache["m"] = SentenceTransformer(DENSE_MODEL, device="cpu")
        m = model_cache["m"]
        emb = m.encode(texts, batch_size=16, normalize_embeddings=True, show_progress_bar=False)

        def search(q, k):
            qv = m.encode([q], normalize_embeddings=True)[0]
            return list(np.argsort(-emb @ qv)[:k])

        return search
    raise ValueError(method)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--methods", nargs="+", default=["bm25"])
    ap.add_argument("--chunk-sizes", nargs="+", type=int, default=[512])
    ap.add_argument("--overlap", type=int, default=50)
    ap.add_argument("--out", default="results/retrieval.json")
    a = ap.parse_args()

    hashes = download()
    gold = load_gold()
    pages = [p for name in SOURCES for p in load_pdf_pages(PDF_DIR / name)]
    cache, runs = {}, []
    for size in a.chunk_sizes:
        chunks = chunk_pages(pages, size, a.overlap)
        for method in a.methods:
            t0 = time.time()
            search = build_retriever(method, chunks, cache)
            page_ranks, ev_ranks, per_q = [], [], []
            for g in gold:
                top = [chunks[i] for i in search(g["question"], 5)]
                pr, er = first_hit_rank(top, g), evidence_rank(top, g)
                page_ranks.append(pr)
                ev_ranks.append(er)
                per_q.append({"id": g["id"], "page_rank": pr, "evidence_rank": er})
            run = {
                "method": method, "chunk_size": size, "overlap": a.overlap, "n_chunks": len(chunks),
                "n_questions": len(gold), "page_level": summarize(page_ranks),
                "evidence_level": summarize(ev_ranks), "seconds": round(time.time() - t0, 1),
                "per_question": per_q,
            }
            runs.append(run)
            print(method, size, len(chunks), "page", run["page_level"], "evidence", run["evidence_level"])
    out = {"pdf_sha256": hashes, "dense_model": DENSE_MODEL, "llm_calls": 0, "runs": runs}
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    Path(a.out).write_text(json.dumps(out, ensure_ascii=False, indent=1), encoding="utf-8")


if __name__ == "__main__":
    main()
