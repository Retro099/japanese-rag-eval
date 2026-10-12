# japanese-rag-eval

**One line:** a Japanese RAG pipeline (bge-m3 + ChromaDB + FastAPI/Streamlit) with a **reproducible evaluation**: a frozen 42-question gold set on a CC BY 4.0 government white paper, retrieval metrics that need no LLM, and a call-capped LLM judge.

[![CI](https://github.com/Retro099/japanese-rag-eval/actions/workflows/ci.yml/badge.svg)](https://github.com/Retro099/japanese-rag-eval/actions/workflows/ci.yml)

## Results (single run, 2026-10-12, GitHub Actions CPU, [run 38246449992](https://github.com/Retro099/japanese-rag-eval/actions/runs/38246449992))

Corpus: 3 PDFs from 令和7年版 情報通信白書 (総務省), 33 pages. Gold set: 42 questions with answer + page (`data/gold.jsonl`). Raw outputs: `results/`.

### Retrieval (no LLM calls) — "hit" = a top-k chunk comes from the gold page

| Retriever | Chunk size (chars) | Chunks | hit@1 | hit@3 | hit@5 | MRR@5 |
|---|---|---|---|---|---|---|
| BM25 (char bigrams) | 256 | 264 | 0.79 | 0.93 | 0.95 | 0.85 |
| bge-m3 dense | 256 | 264 | 0.79 | 0.95 | 0.98 | 0.87 |
| BM25 (char bigrams) | 512 | 130 | 0.74 | 0.93 | 0.98 | 0.84 |
| **bge-m3 dense (default)** | **512** | 130 | **0.81** | **0.93** | **0.98** | **0.87** |
| BM25 (char bigrams) | 1024 | 68 | 0.86 | 0.95 | 1.00 | 0.90 |
| bge-m3 dense | 1024 | 68 | 0.83 | 0.95 | 0.95 | 0.89 |

Stricter "evidence-level" scores (the chunk must contain the exact evidence sentence) are in `results/retrieval.json`; e.g. dense@512 MRR 0.78, BM25@1024 MRR 0.91.

### Answers (dense@512, top-5 → qwen/qwen-2.5-72b-instruct, temperature 0)

| Metric | Score | How measured |
|---|---|---|
| Answer correct (LLM judge) | 42/42 = 1.00 | same Qwen model as judge |
| Faithful to context (LLM judge) | 42/42 = 1.00 | same Qwen model as judge |
| All reference numbers present | 34/42 = 0.81 | `eval/number_check.py`, deterministic, strict lower bound |

**Read the judge score with care.** The deterministic check found a real error the judge accepted: q12 answered 約220億 parameters where the source says 約225億. q33 left out half the answer. Most other "misses" are extra context numbers in the reference that the question did not ask for (e.g. q17's 1925). A reasonable reading is **~40/42 correct**, and the judge is lenient because it is the same model grading itself.

LLM calls used: **84** (42 generate + 42 judge, 0 retries; hard cap set to 100).

## Quickstart

```bash
pip install -r requirements-dev.txt && pytest -q            # 10 unit tests, no downloads
pip install -r requirements-eval.txt                         # pymupdf + sentence-transformers (CPU OK)
python eval/retrieval_eval.py --methods bm25 --chunk-sizes 512   # downloads the 3 PDFs, ~seconds, no LLM
python eval/retrieval_eval.py --methods bm25 dense --chunk-sizes 256 512 1024   # bge-m3, ~2 GB model, ~7 min on CPU
AICREDITS_API_KEY=... python eval/llm_eval.py --max-calls 100   # OpenAI-compatible endpoint, hard call cap
python eval/number_check.py
```

Or run the **Eval** workflow from the Actions tab (retrieval always; LLM part only if `run_llm` is ticked and the secret exists).

The original demo app (FastAPI `api.py` + Streamlit `app/`, ChromaDB, Docker) is kept as-is; see `requirements-app.txt` and `.env.example`.

## How it works

1. **Ingest:** PyMuPDF reads each page; text is NFKC-normalised and whitespace removed (PDF line breaks split Japanese words).
2. **Chunk:** split on 。！？ into chunks of ≤ N characters with 50-character overlap; every chunk keeps its document and page number.
3. **Retrieve:** either BM25 over character bigrams (no tokenizer, no model) or `BAAI/bge-m3` embeddings with cosine similarity.
4. **Score retrieval:** for each gold question, the rank of the first chunk from the gold page → hit@k and MRR. No LLM involved, so it is cheap and repeatable.
5. **Generate + judge:** top-5 chunks → Qwen answers in 1–2 sentences → a judge prompt returns `{correct, faithful}`. Every HTTP attempt counts against a hard budget.

## Limitations

- **Gold set is AI-drafted, author spot-check pending.** Questions and answers were drafted by an AI assistant from the PDF text; every evidence quote was checked by script to appear on the stated page, but a human has not reviewed all 42 yet.
- Questions were written while reading the text, so they share wording with it. That favours keyword search (BM25) and makes scores optimistic versus real user questions.
- Small corpus (33 pages, 3 PDFs from one white paper); one run, no confidence intervals. With 42 questions, one question = 2.4 points.
- The judge is the same model as the generator; see the q12 example above.
- Facts appear in both the summary and the chapter, so page-level hit can miss an equally valid page.
- No reranker in this run (earlier reranker trials in the old repo were rejected).

## Data and license

See [DATA_SOURCES.md](DATA_SOURCES.md). Source: 「令和7年版 情報通信白書」（総務省）, CC BY 4.0. PDFs are downloaded at run time, not committed.

## History

Split from [Retro099/ML-Projects](https://github.com/Retro099/ML-Projects) (archived) on 2026-10-12. The old "0.96 faithfulness" figure there came from a different, unreproducible setup (curated contexts, 70B judge) and is **not** claimed here; the numbers above replace it.

Code license: MIT.
