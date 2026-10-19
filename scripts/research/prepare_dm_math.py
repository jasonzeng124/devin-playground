"""CPU-only data preparation (untimed): DeepMind's mathematics_dataset arithmetic modules as track D
pretraining text, plus mixes with the baseline's FineWeb-Edu tokens.

    python scripts/research/prepare_dm_math.py --out ~/orcd/pool/data/dm_math --tokenizer <dir> \
        [--base ~/orcd/pool/data/fineweb-edu-tinyllama.bin --fracs 0.2,0.5] [--workers 16]

The source is the published v1.0 release (Saxton et al. 2019, "Analysing Mathematical Reasoning
Abilities of Neural Models"; the file that HF `deepmind/math_dataset` loads), checked against its
md5. From the train-easy/medium/hard splits of each module in --modules, the question/answer
lines are kept exactly as published, --pairs-per-doc consecutive pairs per document, and
tokenized with `scripts/pretrain_scratch.py tokenize` (uint16 ids, EOS after each document).
Nothing is generated: RULES_v3 ("Both tracks: external data") admits public expression-value
data such as these modules, but not arithmetic text made for a submission. `--fracs` then writes
<out>/mix_<frac>.bin with `scripts/math_mix.py mix` (the math share of the tokens is ~frac).
Deterministic; skips outputs that exist.
"""

from __future__ import annotations

import argparse
import hashlib
import os
import subprocess
import sys
import tarfile
import urllib.request
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq

URL = "https://storage.googleapis.com/mathematics-dataset/mathematics_dataset-v1.0.tar.gz"
MD5 = "c5f9590beaf2d2445b55240ce0dab9a5"
ARITHMETIC = ("arithmetic__add_or_sub", "arithmetic__add_sub_multiple", "arithmetic__div", "arithmetic__mixed",
              "arithmetic__mul", "arithmetic__mul_div_multiple")
SPLITS = ("train-easy", "train-medium", "train-hard")
SCRIPTS = Path(__file__).resolve().parents[1]


def download(path: Path) -> None:
    if path.exists():
        return
    partial = path.with_suffix(".partial")
    md5 = hashlib.md5()
    with urllib.request.urlopen(URL, timeout=120) as response, open(partial, "wb") as out:
        while chunk := response.read(16 << 20):
            md5.update(chunk)
            out.write(chunk)
    if md5.hexdigest() != MD5:
        raise SystemExit(f"md5 {md5.hexdigest()} != {MD5}; left {partial}")
    partial.rename(path)


def extract(tarball: Path, parquet_dir: Path, modules: list[str], pairs_per_doc: int) -> None:
    """One parquet file per (split, module): documents of consecutive question/answer pairs."""
    wanted = {f"mathematics_dataset-v1.0/{split}/{module}.txt": f"{split}__{module}" for split in SPLITS for module in modules}
    if all((parquet_dir / f"{name}.parquet").exists() for name in wanted.values()):
        return
    parquet_dir.mkdir(parents=True, exist_ok=True)
    seen = []
    with tarfile.open(tarball, "r|gz") as tar:
        for member in tar:
            seen.append(member.name)
            if member.name not in wanted:
                continue
            lines = tar.extractfile(member).read().decode("utf-8").splitlines()
            if len(lines) % 2:
                raise SystemExit(f"{member.name}: odd number of lines ({len(lines)})")
            pairs = [f"{q}\n{a}" for q, a in zip(lines[0::2], lines[1::2])]
            docs = ["\n".join(pairs[i : i + pairs_per_doc]) for i in range(0, len(pairs), pairs_per_doc)]
            pq.write_table(pa.table({"text": docs}), parquet_dir / f"{wanted[member.name]}.parquet", row_group_size=20_000)
            print(f"{member.name}: {len(pairs)} pairs, {len(docs)} documents; e.g. {pairs[0]!r}", flush=True)
    missing = [name for name in wanted.values() if not (parquet_dir / f"{name}.parquet").exists()]
    if missing:
        raise SystemExit(f"not in the tarball: {missing}; it holds e.g. {seen[:20]}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--tokenizer", required=True, help="the pinned TinyLlama tokenizer (benchmark.V3_TRACK_D_TOKENIZER)")
    parser.add_argument("--modules", default=",".join(ARITHMETIC))
    parser.add_argument("--pairs-per-doc", type=int, default=8)
    parser.add_argument("--base", help="FineWeb-Edu token file to mix with")
    parser.add_argument("--fracs", default="", help="comma-separated math shares for mix_<frac>.bin")
    parser.add_argument("--workers", type=int, default=min(16, os.cpu_count() or 1))
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    tarball = args.out / "mathematics_dataset-v1.0.tar.gz"
    download(tarball)
    extract(tarball, args.out / "parquet", args.modules.split(","), args.pairs_per_doc)
    tokens = args.out / "dm_math.bin"
    if not tokens.exists():
        partial = tokens.with_suffix(".partial")
        subprocess.run([sys.executable, str(SCRIPTS / "pretrain_scratch.py"), "tokenize", "--parquet", str(args.out / "parquet"),
                        "--tokenizer", args.tokenizer, "--out", str(partial), "--workers", str(args.workers)], check=True)
        partial.rename(tokens)
    print(f"{tokens}: {tokens.stat().st_size / 2e9:.3f}B tokens", flush=True)
    for frac in filter(None, args.fracs.split(",")):
        mix = args.out / f"mix_{frac}.bin"
        if not mix.exists():
            partial = mix.with_suffix(".partial")
            subprocess.run([sys.executable, str(SCRIPTS / "math_mix.py"), "mix", "--base", args.base, "--extra", str(tokens),
                            "--frac", frac, "--max-repeat", "64", "--out", str(partial)], check=True)
            partial.rename(mix)


if __name__ == "__main__":
    main()
