#!/usr/bin/env bash
# Full experiment from the raw datasets in data/raw/ (about 8 CPU-hours on 4 cores).
# Results are checkpointed in results/parts/, so an interrupted run can simply be restarted.
#   bash run_all.sh            full experiment
#   bash run_all.sh --quick    quick test on the real data (3 folds, 2 classifiers)
set -euo pipefail
MODE=${1:-}
python -m src.experiment_static $MODE
python -m src.experiment_temporal $MODE
if [[ "$MODE" == "--quick" ]]; then
  python -m src.analysis --static results/static_quick.csv --temporal results/temporal_quick.csv --out results/analysis
else
  python -m src.analysis --static results/static_full.csv --temporal results/temporal_full.csv --out results/analysis
  python -m src.paper --raw results --out results/paper_rerun
fi
