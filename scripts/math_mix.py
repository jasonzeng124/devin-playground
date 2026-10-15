"""Math-heavy pretraining mixes for the track D math-data pilot (calibration/sweep_d15, CPU only).

    python scripts/math_mix.py filter --parquet <fineweb-edu dir> --out <dir>/math.parquet [--min-eq 3]
    python scripts/math_mix.py mix --base base.bin --extra extra.bin --frac 0.2 [--max-repeat 8] --out mix.bin

`filter` keeps FineWeb-Edu documents with at least --min-eq written equations ("12 * 3 = 36",
"4.5 + 2 - 1 = 5.5") and prints how math-heavy the corpus is. It is a hand-written rule, so the
result is a legal track D subset (RULES_v3: filtering that does not use an external model).
`mix` appends whole copies of extra.bin (uint16 token ids, EOS-separated, from
`pretrain_scratch.py tokenize`) to base.bin until extra is ~--frac of the tokens; pretraining
samples uniformly from the file, so that is the mixing weight.
"""

from __future__ import annotations

import argparse
import math
import re
import shutil
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq

NUM = r"\d+(?:\.\d+)?"
OP = r"\s*[-+*/×÷]\s*"
EQUATION = re.compile(rf"(?<![\w.]){NUM}(?:{OP}{NUM})+\s*=\s*-?{NUM}")


def filter_math(args: argparse.Namespace) -> None:
    files = sorted(Path(args.parquet).rglob("*.parquet"))
    writer = None
    docs = kept = chars = kept_chars = equations = 0
    examples: list[str] = []
    for path in files:
        pf = pq.ParquetFile(path)
        for group in range(pf.num_row_groups):
            texts = pf.read_row_group(group, columns=["text"]).column("text").to_pylist()
            keep = []
            for text in texts:
                n = len(EQUATION.findall(text))
                docs += 1
                chars += len(text)
                equations += n
                if n >= args.min_eq:
                    keep.append(text)
                    kept_chars += len(text)
                    if len(examples) < 5:
                        examples.append(EQUATION.search(text).group(0))
            kept += len(keep)
            if keep:
                table = pa.table({"text": keep})
                writer = writer or pq.ParquetWriter(args.out, table.schema)
                writer.write_table(table)
    if writer:
        writer.close()
    print(f"{docs} documents, {equations} equations ({equations / max(docs, 1):.3f} per document)")
    print(f"kept {kept} documents with >= {args.min_eq} equations: {kept / max(docs, 1):.2%} of documents, "
          f"{kept_chars / max(chars, 1):.2%} of characters")
    print("examples:", examples)


def mix(args: argparse.Namespace) -> None:
    base, extra = Path(args.base).stat().st_size, Path(args.extra).stat().st_size
    repeats = min(args.max_repeat, max(1, math.ceil(args.frac * base / ((1 - args.frac) * extra))))
    with open(args.out, "wb") as out:
        for path in [args.base] + [args.extra] * repeats:
            with open(path, "rb") as f:
                shutil.copyfileobj(f, out, 64 << 20)
    total = base + repeats * extra
    print(f"{args.out}: base {base / 2e9:.3f}B + {repeats} x extra {extra / 2e9:.3f}B tokens "
          f"= {total / 2e9:.3f}B, extra share {repeats * extra / total:.1%}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = parser.add_subparsers(dest="cmd", required=True)
    f = sub.add_parser("filter")
    f.add_argument("--parquet", required=True)
    f.add_argument("--out", required=True)
    f.add_argument("--min-eq", type=int, default=3)
    m = sub.add_parser("mix")
    m.add_argument("--base", required=True)
    m.add_argument("--extra", required=True)
    m.add_argument("--frac", type=float, default=0.2)
    m.add_argument("--max-repeat", type=int, default=8)
    m.add_argument("--out", required=True)
    args = parser.parse_args()
    (filter_math if args.cmd == "filter" else mix)(args)


if __name__ == "__main__":
    main()
