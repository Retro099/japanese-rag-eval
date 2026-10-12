# japanese-rag-eval — retrieval evaluation for Japanese government documents

[![CI](https://github.com/Retro099/japanese-rag-eval/actions/workflows/ci.yml/badge.svg)](https://github.com/Retro099/japanese-rag-eval/actions/workflows/ci.yml) [![License: MIT](https://img.shields.io/badge/license-MIT-blue.svg)](LICENSE) [![Python 3.11](https://img.shields.io/badge/python-3.11-blue.svg)](.github/workflows/ci.yml)

**TL;DR**
- **Problem:** how well does retrieval find the right page in a Japanese government white paper (令和7年版 情報通信白書, 33 pages), and is keyword search (BM25) or dense embeddings (bge-m3) better?
- **Approach:** a reproducible **retrieval evaluation** (hit@k, MRR; no LLM calls) on a frozen 42-question gold set (**AI-drafted, 15/42 hand-checked by the author, 15/15 correct**) plus 12 harder paraphrased questions scored separately.
- **Result:** dense@512 hit@5 0.98 / MRR 0.87 on the gold set (single run; questions reuse source wording, so this is optimistic); on the 12 harder questions dense hit@1 0.83 vs BM25 0.50–0.67. **Answer quality is not the focus:** the small LLM answer check uses the same model as judge and is lenient.

| Metric (single run, 2026-10-12/13) | Result | Caveat |
|---|---|---|
| Gold 42, dense@512 hit@5 / MRR@5 | **0.98 / 0.87** | AI-drafted questions, high wording overlap (favours BM25) |
| Harder 12, dense@512 vs BM25 hit@1 | **0.83** vs 0.50–0.67 | only 12 questions, 1 question = 8.3 points |
| Hand check of gold questions | **15/15 correct** | random 15 of 42 (seed 42); 27 not hand-checked |

**About the questions:** all 42 gold questions and answers were **drafted by an AI assistant** from the PDF text. A script checked that every evidence quote appears on the stated page. **Author hand-checked 15/42 (random seed 42): 15 correct, 0 flagged.** Sample IDs and excerpts: `results/checklist_sample.json`. The other 27 questions are not hand-checked yet.

## BM25 vs dense at a glance

Page-level retrieval, from `results/retrieval.json` (42 questions) and `results/retrieval_hard.json` (12 harder questions). No LLM calls.

| Question set | Retriever (chunk chars) | hit@1 | hit@5 | MRR@5 |
|---|---|---|---|---|
| Gold 42 (wording close to the source) | **bge-m3 dense (512, default)** | 0.81 | 0.98 | 0.87 |
| Gold 42 | BM25 char bigrams (1024, best BM25) | 0.86 | 1.00 | 0.90 |
| Gold 42 | BM25 char bigrams (512) | 0.74 | 0.98 | 0.84 |
| Harder 12 (paraphrased + multi-chunk) | **bge-m3 dense (512)** | 0.83 | 1.00 | 0.92 |
| Harder 12 | BM25 char bigrams (1024) | 0.67 | 0.83 | 0.73 |
| Harder 12 | BM25 char bigrams (512) | 0.50 | 0.83 | 0.64 |

On the gold set BM25 can beat dense retrieval because the questions reuse the source wording (see the overlap note below). When the question is paraphrased, BM25 drops and dense retrieval holds up. With only 12 hard questions, one question = 8.3 points, so treat this as a direction, not a precise gap.

## Full results (single run, 2026-10-12, GitHub Actions CPU, [run 38246449992](https://github.com/Retro099/japanese-rag-eval/actions/runs/38246449992))

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

### Harder questions (scored separately, no LLM calls)

`data/gold_hard.jsonl`: 12 questions written on 2026-10-13 **by an AI assistant**. They deliberately avoid the source wording and were written before any retriever was run on them. 9 are paraphrased single-fact questions; 3 are multi-chunk questions that need facts from two different pages. A question can list more than one acceptable page (e.g. a fact that appears in both the summary and the chapter). Script: `eval/hard_eval.py`. Output: `results/retrieval_hard.json`. Run on the bot's CPU box on 2026-10-13; the 42-question run was repeated in the same session and reproduced `results/retrieval.json` exactly.

| Retriever | Chunk size | hit@1 | hit@3 | hit@5 | MRR@5 | paraphrase (9) MRR@5 | multi (3): all pages in top 5 |
|---|---|---|---|---|---|---|---|
| BM25 | 256 | 0.50 | 0.75 | 0.75 | 0.63 | 0.56 | 3/3 |
| bge-m3 dense | 256 | 0.92 | 1.00 | 1.00 | 0.94 | 0.93 | 3/3 |
| BM25 | 512 | 0.50 | 0.75 | 0.83 | 0.64 | 0.58 | 2/3 |
| bge-m3 dense | 512 | 0.83 | 1.00 | 1.00 | 0.92 | 0.89 | 3/3 |
| BM25 | 1024 | 0.67 | 0.75 | 0.83 | 0.73 | 0.69 | 2/3 |
| bge-m3 dense | 1024 | 0.83 | 1.00 | 1.00 | 0.92 | 0.89 | 3/3 |

BM25 misses h02 (キャッシュレス決済 asked as 「現金以外で支払われた割合」) and h03 (SNS asked as 「ソーシャルメディア」) at every chunk size. These are word-substitution paraphrases that bigram matching cannot bridge.

### Question / source wording overlap (why BM25 looks strong on the gold set)

`eval/overlap.py` → `results/overlap.json`: the share of a question's character bigrams that also appear in its evidence sentence.

| Set | n | mean | median | max |
|---|---|---|---|---|
| Gold 42 | 42 | 0.36 | 0.35 | 0.85 |
| Harder 12 | 12 | 0.09 | 0.07 | 0.29 |

The gold questions were drafted while reading the text, so they copy many of its phrases. Keyword search (BM25) benefits, and gold-set scores are optimistic compared with real user questions. The harder set was written to cut that overlap.

### Secondary: answer check (not the focus; dense@512, top-5 → qwen/qwen-2.5-72b-instruct, temperature 0)

| Metric | Score | How measured |
|---|---|---|
| Answer correct (LLM judge) | 42/42 = 1.00 | same Qwen model as judge |
| Faithful to context (LLM judge) | 42/42 = 1.00 | same Qwen model as judge |
| All reference numbers present | 34/42 = 0.81 | `eval/number_check.py`, deterministic, strict lower bound |

**Read the judge score with care.** The deterministic check found a real error the judge accepted: q12 answered 約220億 parameters where the source says 約225億. q33 left out half the answer. Most other "misses" are extra context numbers in the reference that the question did not ask for (e.g. q17's 1925). A reasonable reading is **~40/42 correct**, and the judge is lenient because it is the same model grading itself.

LLM calls used: **84** (42 generate + 42 judge, 0 retries; hard cap set to 100).

## How it works

1. **Ingest:** PyMuPDF reads each page; text is NFKC-normalised and whitespace removed (PDF line breaks split Japanese words).
2. **Chunk:** split on 。！？ into chunks of ≤ N characters with 50-character overlap; every chunk keeps its document and page number.
3. **Retrieve:** either BM25 over character bigrams (no tokenizer, no model) or `BAAI/bge-m3` embeddings with cosine similarity.
4. **Score retrieval:** for each gold question, the rank of the first chunk from the gold page → hit@k and MRR. No LLM involved, so it is cheap and repeatable.
5. **Generate + judge:** top-5 chunks → Qwen answers in 1–2 sentences → a judge prompt returns `{correct, faithful}`. Every HTTP attempt counts against a hard budget.

## Quickstart

```bash
pip install -r requirements-dev.txt && pytest -q            # 12 unit tests, no downloads
pip install -r requirements-eval.txt
python eval/retrieval_eval.py --methods bm25 --chunk-sizes 512   # downloads the 3 PDFs, ~seconds, no LLM
```

## Reproduce the full evaluation

```bash
pip install -r requirements-eval.txt                         # pymupdf + sentence-transformers (CPU OK)
python eval/retrieval_eval.py --methods bm25 dense --chunk-sizes 256 512 1024   # bge-m3, ~2 GB model, ~7 min on CPU
AICREDITS_API_KEY=... python eval/llm_eval.py --max-calls 100   # OpenAI-compatible endpoint, hard call cap
python eval/number_check.py
python eval/hard_eval.py --methods bm25 dense --chunk-sizes 256 512 1024   # 12 harder questions
python eval/overlap.py                                       # question/evidence wording overlap
python scripts/sample_checklist.py                           # 15-question hand-check sample (seed 42)
```

Or run the **Eval** workflow from the Actions tab (retrieval always; LLM part only if `run_llm` is ticked and the secret exists).

The original demo app (FastAPI `api.py` + Streamlit `app/`, ChromaDB, Docker) is kept as-is; see `requirements-app.txt` and `.env.example`.

## Limitations

- **Gold set is AI-drafted; author hand-checked 15/42 (random seed 42): 15 correct, 0 flagged.** The other 27 are not hand-checked. Questions and answers were drafted by an AI assistant from the PDF text. Every evidence quote was checked by script to appear on the stated page. The 12 harder questions are also AI-written.
- Gold questions share wording with the text (mean bigram overlap 0.36 vs 0.09 for the harder set). That favours BM25 and makes gold-set scores optimistic. The harder set is small (12) and was written to avoid source wording, which works against BM25 by design.
- Small corpus (33 pages, 3 PDFs from one white paper); one run, no confidence intervals. With 42 questions, one question = 2.4 points.
- The judge is the same model as the generator; see the q12 example above.
- Facts appear in both the summary and the chapter, so page-level hit can miss an equally valid page.
- No reranker in this run (earlier reranker trials in the old repo were rejected).

## Data source

See [DATA_SOURCES.md](DATA_SOURCES.md). Source: 「令和7年版 情報通信白書」（総務省）, CC BY 4.0. PDFs are downloaded at run time, not committed.

## History

Split from [Retro099/ML-Projects](https://github.com/Retro099/ML-Projects) (archived) on 2026-10-12. The old "0.96 faithfulness" figure there came from a different, unreproducible setup (curated contexts, 70B judge) and is **not** claimed here; the numbers above replace it.

## Repository structure

```text
rag_eval/   # evaluation library: BM25, metrics, text normalisation, PDF sources, capped LLM client
eval/       # retrieval_eval.py, hard_eval.py, llm_eval.py, number_check.py, overlap.py
data/       # gold.jsonl (42), gold_hard.jsonl (12)
results/    # committed raw outputs behind every number above
scripts/    # sample_checklist.py (15-question hand check), retrieve_smoke.py
src/, api.py, app/   # original RAG demo (ChromaDB + FastAPI + Streamlit), kept as-is
tests/      # 12 unit tests (no downloads)
```

## License

Code: MIT ([LICENSE](LICENSE)). Data: 総務省, CC BY 4.0 ([DATA_SOURCES.md](DATA_SOURCES.md)).
