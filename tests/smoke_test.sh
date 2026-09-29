#!/usr/bin/env bash
# Runs the whole pipeline on MOCK data in a few minutes. The numbers are meaningless.
set -euo pipefail
OUT=${1:-results_mock}
python tests/make_mock_data.py --out data/mock
python -m src.experiment_static --quick --data-dir data/mock --out "$OUT/static_quick.csv"
python -m src.experiment_temporal --quick --data-dir data/mock --out "$OUT/temporal_quick.csv"
python -m src.analysis --static "$OUT/static_quick.csv" --temporal "$OUT/temporal_quick.csv" --out "$OUT"
echo "Smoke test finished: outputs in $OUT/ (mock data, do not report)."
