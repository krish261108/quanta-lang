"""Gzip every per-task JSONL record file under results/ for committing.

The raw .jsonl files stay on disk (git ignores them); `run.read_jsonl` reads the .gz
copy when the raw file is absent, so every analysis reproduces from a fresh clone.

    python experiments/capability_acquisition/pack_results.py
"""
import gzip
from pathlib import Path

ROOT = Path(__file__).resolve().parent / "results"
n = 0
for path in sorted(ROOT.rglob("*.jsonl")):
    if path.name in ("events.jsonl", "registry.jsonl"):
        continue                                  # small, kept as plain text
    gz = path.with_name(path.name + ".gz")
    if gz.exists() and gz.stat().st_mtime >= path.stat().st_mtime:
        continue
    gz.write_bytes(gzip.compress(path.read_bytes(), 9, mtime=0))
    n += 1
print(f"packed {n} files")
