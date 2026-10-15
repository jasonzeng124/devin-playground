"""CPU-only data preparation for the track D baseline (untimed; docs/RULES_v3.md "Clock and hardware").

    python scripts/prepare_track_d_data.py [--out data/track_d] [--workers 16]

Downloads the pinned FineWeb-Edu shards (`benchmark.V3_TRACK_D_DATASET`, `V3_TRACK_D_SHARDS`) and
the pinned TinyLlama tokenizer (`V3_TRACK_D_TOKENIZER`), then tokenizes the shards with
`scripts/pretrain_scratch.py tokenize`. Writes <out>/tokenizer/ and <out>/fineweb-edu-tinyllama.bin
(uint16 ids, EOS after each document). Skips steps whose outputs already exist.
"""

from __future__ import annotations

import argparse
import os
import subprocess
import sys
from pathlib import Path

from huggingface_hub import hf_hub_download
from transformers import AutoTokenizer

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from rlvr_speedrun import benchmark  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--out", type=Path, default=benchmark.V3_TRACK_D_DATA_DIR)
    parser.add_argument("--workers", type=int, default=min(16, os.cpu_count() or 1))
    args = parser.parse_args()
    out = args.out
    parquet = out / "parquet"
    repo, revision = benchmark.V3_TRACK_D_DATASET
    for shard in benchmark.V3_TRACK_D_SHARDS:
        hf_hub_download(repo, shard, repo_type="dataset", revision=revision, local_dir=parquet)
    tok_dir = out / "tokenizer"
    if not (tok_dir / "tokenizer.json").exists():
        name, tok_revision = benchmark.V3_TRACK_D_TOKENIZER
        AutoTokenizer.from_pretrained(name, revision=tok_revision).save_pretrained(tok_dir)
    tokens = out / "fineweb-edu-tinyllama.bin"
    if not tokens.exists():
        partial = tokens.with_suffix(".partial")
        subprocess.run([sys.executable, str(Path(__file__).with_name("pretrain_scratch.py")), "tokenize", "--parquet", str(parquet),
                        "--tokenizer", str(tok_dir), "--out", str(partial), "--workers", str(args.workers)], check=True)
        partial.rename(tokens)
    print(f"track D data ready: {tokens} ({tokens.stat().st_size / 2 / 1e9:.3f}B tokens), tokenizer {tok_dir}")


if __name__ == "__main__":
    main()
