"""Answer eval: retrieve top-5 -> generate with Qwen -> LLM judge (correct / faithful).

Needs AICREDITS_API_KEY. Every HTTP attempt counts against --max-calls (hard cap).
usage: python eval/llm_eval.py --retriever dense --chunk-size 512 --max-calls 140
"""

import argparse
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from eval.retrieval_eval import build_retriever  # noqa: E402
from rag_eval.data import PDF_DIR, download, load_gold  # noqa: E402
from rag_eval.llm import LLM, BudgetExceeded, parse_judge  # noqa: E402
from rag_eval.sources import SOURCES  # noqa: E402
from rag_eval.text import chunk_pages, load_pdf_pages  # noqa: E402

GEN_SYS = ("あなたは日本語の資料に基づいて答えるアシスタントです。与えられたコンテキストだけを使い、"
           "質問に1〜2文で簡潔に答えてください。コンテキストに答えがなければ「資料に記載がありません」と答えてください。")
JUDGE_SYS = ("You are a strict evaluator. Given a question, a reference answer, retrieved context and a model answer, "
             "return ONLY JSON: {\"correct\": 0 or 1, \"faithful\": 0 or 1}. correct=1 if the model answer contains the "
             "key facts of the reference answer (numbers must match; extra correct detail is fine). faithful=1 if every "
             "claim in the model answer is supported by the context.")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--retriever", default="dense")
    ap.add_argument("--chunk-size", type=int, default=512)
    ap.add_argument("--max-calls", type=int, default=140)
    ap.add_argument("--base-url", default="https://api.aicredits.in/v1")
    ap.add_argument("--model", default="qwen/qwen-2.5-72b-instruct")
    ap.add_argument("--out", default="results/llm_eval.json")
    a = ap.parse_args()

    download()
    gold = load_gold()
    pages = [p for n in SOURCES for p in load_pdf_pages(PDF_DIR / n)]
    chunks = chunk_pages(pages, a.chunk_size, 50)
    search = build_retriever(a.retriever, chunks, {})
    llm = LLM(a.base_url, a.model, a.max_calls)
    rows, stop = [], None
    for g in gold:
        row = {"id": g["id"]}
        try:
            ctx = "\n---\n".join(chunks[i]["text"] for i in search(g["question"], 5))
            ans = llm.chat([{"role": "system", "content": GEN_SYS},
                            {"role": "user", "content": f"コンテキスト:\n{ctx}\n\n質問: {g['question']}"}], 200)
            row["answer"] = ans.strip()
            jr = llm.chat([{"role": "system", "content": JUDGE_SYS},
                           {"role": "user", "content": json.dumps({"question": g["question"], "reference": g["answer"],
                                                                   "context": ctx, "model_answer": row["answer"]},
                                                                  ensure_ascii=False)}], 60)
            row.update(parse_judge(jr))
        except BudgetExceeded as e:
            stop = str(e)
            row["skipped"] = stop
            rows.append(row)
            break
        except Exception as e:  # noqa: BLE001  (2 tries already used inside chat)
            row["skipped"] = f"{type(e).__name__}: {str(e)[:120]}"
        rows.append(row)
        time.sleep(1)
    judged = [r for r in rows if "correct" in r]
    n = len(judged) or 1
    out = {
        "generator": a.model, "judge": a.model, "temperature": 0, "retriever": a.retriever,
        "chunk_size": a.chunk_size, "top_k": 5, "n_gold": len(gold), "n_judged": len(judged),
        "n_skipped": sum(1 for r in rows if "skipped" in r), "stopped": stop,
        "answer_correct": round(sum(r["correct"] for r in judged) / n, 4),
        "faithful": round(sum(r["faithful"] for r in judged) / n, 4),
        "llm_calls_used": llm.calls, "rows": rows,
    }
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    Path(a.out).write_text(json.dumps(out, ensure_ascii=False, indent=1), encoding="utf-8")
    print({k: v for k, v in out.items() if k != "rows"})


if __name__ == "__main__":
    main()
