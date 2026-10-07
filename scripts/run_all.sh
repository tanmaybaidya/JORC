#!/usr/bin/env bash
# End-to-end pipeline. Usage: bash scripts/run_all.sh [smoke|reduced|paper]
set -euo pipefail
SCALE="${1:-smoke}"
cd "$(dirname "$0")/.."
python -m pytest -q tests
python experiments/reproduce_figures.py --scale "$SCALE"
python experiments/reproduce_tables.py --scale "$SCALE"
python experiments/sensitivity_analysis.py
python experiments/cache_capacity_diagnostic.py --units GB
python experiments/cache_capacity_diagnostic.py --units MB
echo "Done. See results/$SCALE/{figures,tables}, results/sensitivity, results/cache_diagnostic"
