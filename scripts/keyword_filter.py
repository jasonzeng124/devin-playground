"""Keyword filter and reasoning-cue counts for public pretraining data (track D data prep, CPU only, no model).

    python scripts/keyword_filter.py drop --parquet <dir> --out <dir>/kept.parquet
    python scripts/keyword_filter.py cues --bin tokens.bin --tokenizer <dir> [--windows 2000] [--length 1024] [--seed 0]

`drop` removes every document that matches COUNTDOWN, a hand-written keyword rule for Countdown, the 24 game and
other numbers-game puzzles (RULES_v3, external data: "a general dataset that contains some such items may be used
once those items are removed (a keyword filter is enough)"), writes the rest as parquet with one `text` column, and
prints how many documents each pattern matched, with a few matches. The rule over-drops on purpose (a "countdown
timer" page goes too); nothing is generated or rewritten.

`cues` decodes random windows of a token file (`pretrain_scratch.py tokenize` output, uint16) and counts phrases
that open or carry written reasoning (CUES) per million characters, to compare how much worked reasoning a corpus
holds. It reads only; deterministic for a seed.
"""

from __future__ import annotations

import argparse
import random
import re
from collections import Counter
from pathlib import Path

COUNTDOWN_PATTERNS = {
    "countdown": r"count\s*-?\s*down",
    "24 game": r"\b24[\s-]*(?:point\s+)?(?:game|puzzle|challenge)s?\b|\bgames?\s+of\s+24\b",
    "make 24": r"\b(?:make|makes|making|reach|reaches)\s+(?:exactly\s+)?(?:24|twenty[\s-]*four)\b",
    "numbers game": r"\bnumbers?\s+(?:game|round|puzzle)s?\b",
    "krypto": r"\bkrypto\b",
    "target number": r"\btarget\s+(?:number|value|total)\b",
    "each number once": r"\b(?:each|every|all)\s+(?:of\s+the\s+)?(?:four\s+|given\s+)?numbers?\s+(?:\w+\s+){0,3}(?:exactly\s+)?once\b",
}
COUNTDOWN = {name: re.compile(p, re.IGNORECASE) for name, p in COUNTDOWN_PATTERNS.items()}

CUES = {
    "step by step": r"\bstep[\s-]+by[\s-]+step\b",
    "let's / let us": r"\blet(?:'|’)?s\b|\blet\s+us\b",
    "let me": r"\blet\s+me\b",
    "first,": r"\bfirst(?:ly)?,",
    "therefore / thus": r"\b(?:therefore|thus|hence)\b",
    "so the answer": r"\bso\s+the\s+(?:answer|result|solution)\b",
    "the answer is": r"\bthe\s+(?:final\s+)?answer\s+is\b",
    "check / verify": r"\b(?:check|verify|verifying|checking)\b",
    "wait / hmm / okay": r"\b(?:wait|hmm+|okay|alright)\b",
    "we get / we have": r"\bwe\s+(?:get|have|obtain|find)\b",
    "solution:": r"\bsolution\s*:",
}
CUE = {name: re.compile(p, re.IGNORECASE) for name, p in CUES.items()}


def countdown_matches(text: str) -> list[str]:
    """Names of the COUNTDOWN patterns a document matches (empty: the document is kept)."""
    return [name for name, rx in COUNTDOWN.items() if rx.search(text)]


def cue_counts(text: str) -> Counter:
    return Counter({name: len(rx.findall(text)) for name, rx in CUE.items()})


def drop(args: argparse.Namespace) -> None:
    import pyarrow as pa
    import pyarrow.parquet as pq

    files = sorted(Path(args.parquet).rglob("*.parquet"))
    writer = None
    docs = kept = chars = kept_chars = 0
    hits: Counter = Counter()
    examples: dict[str, list[str]] = {}
    for path in files:
        pf = pq.ParquetFile(path)
        for group in range(pf.num_row_groups):
            texts = pf.read_row_group(group, columns=["text"]).column("text").to_pylist()
            keep = []
            for text in texts:
                docs += 1
                chars += len(text)
                names = countdown_matches(text)
                if not names:
                    keep.append(text)
                    kept_chars += len(text)
                    continue
                hits.update(names)
                for name in names:
                    if len(examples.setdefault(name, [])) < args.examples:
                        m = COUNTDOWN[name].search(text)
                        examples[name].append(text[max(0, m.start() - 60):m.end() + 60].replace("\n", " "))
            kept += len(keep)
            if keep:
                table = pa.table({"text": keep})
                writer = writer or pq.ParquetWriter(args.out, table.schema)
                writer.write_table(table)
    if writer:
        writer.close()
    print(f"{len(files)} files, {docs} documents; dropped {docs - kept} ({(docs - kept) / max(docs, 1):.2%} of documents, "
          f"{1 - kept_chars / max(chars, 1):.2%} of characters); kept {kept}")
    for name in COUNTDOWN:
        print(f"  {name}: {hits[name]} documents; e.g. {examples.get(name, [])}")


def cues(args: argparse.Namespace) -> None:
    import numpy as np
    from transformers import AutoTokenizer

    tok = AutoTokenizer.from_pretrained(args.tokenizer)
    ids = np.memmap(args.bin, dtype=np.uint16, mode="r")
    rng = random.Random(args.seed)
    total: Counter = Counter()
    chars = 0
    for _ in range(args.windows):
        start = rng.randrange(0, len(ids) - args.length)
        text = tok.decode(ids[start:start + args.length].tolist(), skip_special_tokens=True)
        chars += len(text)
        total.update(cue_counts(text))
    print(f"{args.bin}: {len(ids) / 1e9:.3f}B tokens; {args.windows} windows of {args.length} tokens, {chars} characters")
    for name in CUE:
        print(f"  {name}: {total[name]} ({1e6 * total[name] / max(chars, 1):.1f} per million characters)")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = parser.add_subparsers(dest="cmd", required=True)
    d = sub.add_parser("drop")
    d.add_argument("--parquet", required=True)
    d.add_argument("--out", required=True)
    d.add_argument("--examples", type=int, default=3)
    c = sub.add_parser("cues")
    c.add_argument("--bin", required=True)
    c.add_argument("--tokenizer", required=True)
    c.add_argument("--windows", type=int, default=2000)
    c.add_argument("--length", type=int, default=1024)
    c.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()
    {"drop": drop, "cues": cues}[args.cmd](args)


if __name__ == "__main__":
    main()
