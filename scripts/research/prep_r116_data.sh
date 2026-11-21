#!/usr/bin/env bash
# Track D pretraining data for research screen r116 (CPU only, no model; untimed preparation under RULES_v3):
# FineMath-4plus (HuggingFaceTB/finemath, revision e92b25a616738fe95dc186b64dfb19f9c8525594, shards 0-2 of 64,
# as scripts/modal_v3.py downloaded them) with the Countdown keyword rule of scripts/keyword_filter.py applied,
# tokenized with the pinned TinyLlama tokenizer, and two copies appended to mix_0.2.bin (D 007's mix). It also
# counts written-reasoning phrases in each corpus (keyword_filter.py cues) and prints md5 sums.
#   PYTHON=<venv python> bash scripts/research/prep_r116_data.sh <out dir>
set -euo pipefail
out=${1:?usage: prep_r116_data.sh <out dir>}
data=${DATA:-$HOME/orcd/pool/data}
PY=${PYTHON:-python}
tok="$data/tinyllama-tokenizer"
mkdir -p "$out/finemath_kw"
[[ -e "$out/finemath_kw/kept.parquet" ]] || $PY scripts/keyword_filter.py drop --parquet "$data/finemath/finemath-4plus" --out "$out/finemath_kw/kept.parquet.tmp"
[[ -e "$out/finemath_kw/kept.parquet" ]] || mv "$out/finemath_kw/kept.parquet.tmp" "$out/finemath_kw/kept.parquet"
$PY scripts/pretrain_scratch.py tokenize --parquet "$out/finemath_kw" --tokenizer "$tok" --out "$out/finemath_kw.bin" --workers 16
$PY scripts/math_mix.py mix --base "$data/dm_math/mix_0.2.bin" --extra "$out/finemath_kw.bin" --frac 0.3 --max-repeat 2 \
  --out "$out/mix_0.2_finemath_kw2.bin"
for b in "$data/fineweb-edu-tinyllama.bin" "$data/dm_math/dm_math.bin" "$data/finemath.bin" "$out/finemath_kw.bin"; do
  $PY scripts/keyword_filter.py cues --bin "$b" --tokenizer "$tok"
done
md5sum "$out/finemath_kw.bin" "$out/mix_0.2_finemath_kw2.bin"
echo "=== done $(date -u +%FT%TZ)"
