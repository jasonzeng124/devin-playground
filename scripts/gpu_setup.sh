#!/usr/bin/env bash
set -euo pipefail

pip install -U "transformers<5" numpy pytest
python -m pytest -q tests
