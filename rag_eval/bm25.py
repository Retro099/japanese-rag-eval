"""Dependency-free BM25 over character bigrams (works for Japanese without a tokenizer)."""

import math
from collections import Counter
from typing import Dict, List, Tuple


def bigrams(text: str) -> List[str]:
    return [text[i:i + 2] for i in range(len(text) - 1)] or ([text] if text else [])


class BM25:
    def __init__(self, docs: List[str], k1: float = 1.5, b: float = 0.75):
        self.k1, self.b = k1, b
        self.tfs = [Counter(bigrams(d)) for d in docs]
        self.lens = [sum(tf.values()) for tf in self.tfs]
        self.avgdl = sum(self.lens) / max(len(self.lens), 1)
        df: Dict[str, int] = Counter()
        for tf in self.tfs:
            df.update(tf.keys())
        n = len(docs)
        self.idf = {t: math.log(1 + (n - c + 0.5) / (c + 0.5)) for t, c in df.items()}

    def search(self, query: str, k: int = 5) -> List[Tuple[int, float]]:
        q = Counter(bigrams(query))
        scores = []
        for i, tf in enumerate(self.tfs):
            s = 0.0
            norm = self.k1 * (1 - self.b + self.b * self.lens[i] / self.avgdl)
            for t in q:
                f = tf.get(t)
                if f:
                    s += self.idf[t] * f * (self.k1 + 1) / (f + norm)
            scores.append((i, s))
        scores.sort(key=lambda x: -x[1])
        return scores[:k]
