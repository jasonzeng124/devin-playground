#!/usr/bin/env bash
# More keyword-filtered FineMath for U25's 6x rung (unranked; CPU only, no model). FineMath-4plus
# (HuggingFaceTB/finemath, revision e92b25a616738fe95dc186b64dfb19f9c8525594) shards 3-31 of 64, downloaded by
# agent/dl_finemath_1122.py, with the Countdown keyword rule of scripts/keyword_filter.py applied (as r116 did
# for shards 0-2), tokenized with the pinned TinyLlama tokenizer, then appended to r116's finemath_kw.bin
# (shards 0-2), so the 6x rung's FineMath share repeats less than once instead of about 6 times. It also counts
# written-reasoning phrases (keyword_filter.py cues) and prints md5 sums.
#   PYTHON=<venv python> bash scripts/research/prep_u25_finemath.sh <shard dir> <r116 out dir> <out dir>
set -euo pipefail
src=${1:?usage: prep_u25_finemath.sh <shard dir> <r116 out dir> <out dir>}
r116=${2:?r116 out dir}
out=${3:?out dir}
data=${DATA:-$HOME/orcd/pool/data}
PY=${PYTHON:-python}
tok="$data/tinyllama-tokenizer"
mkdir -p "$out/finemath_kw_3_31"
kept="$out/finemath_kw_3_31/kept.parquet"
[[ -e "$kept" ]] || $PY scripts/keyword_filter.py drop --parquet "$src" --out "$kept.tmp"
[[ -e "$kept" ]] || mv "$kept.tmp" "$kept"
[[ -e "$out/finemath_kw_3_31.bin" ]] || {
  $PY scripts/pretrain_scratch.py tokenize --parquet "$out/finemath_kw_3_31" --tokenizer "$tok" \
    --out "$out/finemath_kw_3_31.bin.tmp" --workers "${WORKERS:-16}"
  mv "$out/finemath_kw_3_31.bin.tmp" "$out/finemath_kw_3_31.bin"
}
# Both are raw uint16 ids with EOS after each document, so concatenation is a valid token file.
cat "$r116/finemath_kw.bin" "$out/finemath_kw_3_31.bin" > "$out/finemath_kw_0_31.bin.tmp"
mv "$out/finemath_kw_0_31.bin.tmp" "$out/finemath_kw_0_31.bin"
$PY scripts/keyword_filter.py cues --bin "$out/finemath_kw_3_31.bin" --tokenizer "$tok"
ls -l "$out"/*.bin
md5sum "$out/finemath_kw_0_31.bin"
echo "=== done $(date -u +%FT%TZ)"
