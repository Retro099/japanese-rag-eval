"""Minimal OpenAI-compatible chat client with a hard call budget (no SDK needed)."""

import json
import os
import time
import urllib.request


class BudgetExceeded(RuntimeError):
    pass


class LLM:
    def __init__(self, base_url: str, model: str, max_calls: int, api_key_env: str = "AICREDITS_API_KEY"):
        self.base_url = base_url.rstrip("/")
        self.model = model
        self.max_calls = max_calls
        self.calls = 0  # every HTTP attempt counts, including failures
        self._key = os.environ.get(api_key_env, "")

    def chat(self, messages, max_tokens: int = 300, tries: int = 2) -> str:
        last = None
        for attempt in range(tries):
            if self.calls >= self.max_calls:
                raise BudgetExceeded(f"call budget {self.max_calls} reached")
            self.calls += 1
            body = json.dumps({"model": self.model, "messages": messages, "temperature": 0,
                               "max_tokens": max_tokens}).encode()
            req = urllib.request.Request(f"{self.base_url}/chat/completions", data=body, headers={
                "Authorization": f"Bearer {self._key}", "Content-Type": "application/json"})
            try:
                with urllib.request.urlopen(req, timeout=90) as r:
                    return json.loads(r.read())["choices"][0]["message"]["content"]
            except Exception as e:  # noqa: BLE001
                last = f"{type(e).__name__}: {str(e)[:120]}"
                time.sleep(5 * (attempt + 1))
        raise RuntimeError(last)


def parse_judge(text: str) -> dict:
    """Pull the first {...} JSON object out of the judge reply."""
    s, e = text.find("{"), text.rfind("}")
    if s < 0 or e < s:
        raise ValueError("no JSON in judge reply")
    d = json.loads(text[s:e + 1])
    return {"correct": int(bool(d.get("correct"))), "faithful": int(bool(d.get("faithful")))}
