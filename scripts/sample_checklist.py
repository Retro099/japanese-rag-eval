"""Pick 15 gold questions for the author's hand check. Fixed seed: random.Random(42).sample.

Writes results/checklist_sample.json with question, answer, page and an excerpt of the
512-char chunk that holds the evidence (80 chars either side). NO LLM calls.
"""

import json
import random
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from rag_eval.data import PDF_DIR, ROOT, download, load_gold  # noqa: E402
from rag_eval.sources import SOURCES  # noqa: E402
from rag_eval.text import chunk_pages, load_pdf_pages  # noqa: E402

SEED, N = 42, 15


def main():
    download()
    gold = load_gold()
    ids = sorted(random.Random(SEED).sample(range(len(gold)), N))
    pages = [p for name in SOURCES for p in load_pdf_pages(PDF_DIR / name)]
    chunks = chunk_pages(pages, 512, 50)
    rows = []
    for i in ids:
        q = gold[i]
        text = next(c["text"] for c in chunks
                    if c["doc"] == q["doc"] and c["page"] == q["page"] and q["evidence"] in c["text"])
        j = text.find(q["evidence"])
        rows.append({"id": q["id"], "question": q["question"], "answer": q["answer"], "doc": q["doc"],
                     "page": q["page"], "evidence": q["evidence"],
                     "chunk_excerpt": text[max(0, j - 80): j + len(q["evidence"]) + 80]})
    out = {"seed": SEED, "method": "random.Random(42).sample(range(42), 15)", "rows": rows}
    (ROOT / "results" / "checklist_sample.json").write_text(json.dumps(out, ensure_ascii=False, indent=1), encoding="utf-8")
    print([r["id"] for r in rows])


if __name__ == "__main__":
    main()
