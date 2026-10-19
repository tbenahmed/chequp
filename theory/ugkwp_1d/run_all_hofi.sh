#!/bin/bash
# UGKWP (4 seeds), full-particle reference (4 seeds) and two-fluid Euler for a given grid
cd "$(dirname "$0")"
N=${1:-200}
for kind in ugkwp ref; do
  for s in 1 2 3 4; do
    python3 run_hofi.py $kind $s $N > log_${kind}_n${N}_$s.txt 2>&1 &
  done
  wait
done
python3 run_hofi.py fluid 1 $N > log_fluid_n${N}_1.txt 2>&1
