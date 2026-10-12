import json
from pathlib import Path

import pytest

from rag_eval.bm25 import BM25, bigrams
from rag_eval.llm import LLM, BudgetExceeded, parse_judge
from rag_eval.metrics import evidence_rank, first_hit_rank, summarize
from rag_eval.text import chunk_page, normalize

ROOT = Path(__file__).resolve().parents[1]


def test_normalize_removes_spaces_and_widths():
    assert normalize("２０２５年 ７月\nＸ") == "2025年7月X"


def test_chunk_page_respects_size_and_keeps_page():
    page = {"doc": "a.pdf", "page": 3, "text": "これはテストです。" * 100}
    chunks = chunk_page(page, size=100, overlap=10)
    assert len(chunks) > 1
    assert all(len(c["text"]) <= 100 for c in chunks)
    assert all(c["page"] == 3 and c["doc"] == "a.pdf" for c in chunks)


def test_chunk_page_hard_splits_long_sentence():
    chunks = chunk_page({"doc": "a", "page": 1, "text": "あ" * 1000}, size=200, overlap=20)
    assert all(len(c["text"]) <= 200 for c in chunks)


def test_chunk_page_rejects_bad_overlap():
    with pytest.raises(ValueError):
        chunk_page({"doc": "a", "page": 1, "text": "x"}, size=10, overlap=10)


def test_bigrams():
    assert bigrams("日本語") == ["日本", "本語"]
    assert bigrams("日") == ["日"]


def test_bm25_ranks_matching_doc_first():
    bm = BM25(["猫が好きです。", "クラウドサービスの利用率は倍増した。", "犬の散歩"])
    assert bm.search("クラウドの利用率", 1)[0][0] == 1


def test_metrics():
    gold = {"doc": "d", "page": 2, "evidence": "倍増"}
    ret = [{"doc": "d", "page": 1, "text": "x"}, {"doc": "d", "page": 2, "text": "倍増した"}]
    assert first_hit_rank(ret, gold) == 2
    assert evidence_rank(ret, gold) == 2
    s = summarize([1, 2, 0, 6])
    assert s["hit@1"] == 0.25 and s["hit@5"] == 0.5 and s["mrr@5"] == 0.375


def test_gold_file_is_valid():
    rows = [json.loads(x) for x in (ROOT / "data/gold.jsonl").read_text(encoding="utf-8").splitlines()]
    assert 30 <= len(rows) <= 50
    assert len({r["id"] for r in rows}) == len(rows)
    for r in rows:
        assert r["question"] and r["answer"] and r["evidence"] and r["page"] >= 1
        assert r["doc"] in {"r07_summary01.pdf", "r07_n1110000.pdf", "r07_n1120000.pdf"}


def test_parse_judge():
    assert parse_judge('ok {"correct": 1, "faithful": 0}') == {"correct": 1, "faithful": 0}
    with pytest.raises(ValueError):
        parse_judge("no json")


def test_llm_budget_is_hard_cap():
    llm = LLM("http://127.0.0.1:9", "m", max_calls=0)
    with pytest.raises(BudgetExceeded):
        llm.chat([{"role": "user", "content": "hi"}])
