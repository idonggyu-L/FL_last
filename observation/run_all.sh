#!/usr/bin/env bash
# Generate every figure and the backing table.
#   bash run_all.sh [config.json] [out_dir]
set -eu
cd "$(dirname "$0")"
CFG=${1:-config.json}
OUT=${2:-figs}
PY=${PYTHON:-python3}
CELLS=$($PY -c "import json,sys;print(' '.join(json.load(open('$CFG'))['cells']))")
for c in $CELLS; do
  $PY plot_plane.py --config "$CFG" --cell "$c" --out "$OUT"
  for ax in rep par; do
    $PY plot_axis1d.py --config "$CFG" --cell "$c" --axis "$ax" --out "$OUT"
  done
done
$PY report_table.py --config "$CFG" --out "$OUT"
echo "-> $OUT"
