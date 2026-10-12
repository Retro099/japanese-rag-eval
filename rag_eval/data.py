import hashlib
import json
import urllib.request
from pathlib import Path
from typing import Dict, List

from rag_eval.sources import SOURCES
from rag_eval.text import normalize

ROOT = Path(__file__).resolve().parents[1]
PDF_DIR = ROOT / "data" / "pdfs"
GOLD = ROOT / "data" / "gold.jsonl"


def load_gold(path: Path = GOLD) -> List[Dict]:
    rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
    for r in rows:
        r["evidence"] = normalize(r["evidence"])
    return rows


def download(dest: Path = PDF_DIR) -> Dict[str, str]:
    """Download the source PDFs (not committed) and return their sha256."""
    dest.mkdir(parents=True, exist_ok=True)
    hashes = {}
    for name, url in SOURCES.items():
        p = dest / name
        if not p.exists():
            req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0 (research eval)"})
            with urllib.request.urlopen(req, timeout=120) as r:
                p.write_bytes(r.read())
        hashes[name] = hashlib.sha256(p.read_bytes()).hexdigest()
    return hashes
