#!/usr/bin/env bash
# Post-processing of one finished run (all settings frozen; see DEVELOPMENT_LOG.md):
# cached predictions + metrics, risk analysis on the test pool, criterion ablation, and for seed-0 runs the
# corruption predictions and the shift analysis. Each step leaves a marker so the script is resumable.
cd "$(dirname "$0")/.."
r=$1
b=${r#gpu_}   # GPU-trained runs (scripts/pc_queue.py) are analysed like their CPU counterparts
M=/home/user/obj1data/markers/$r; mkdir -p $M
L=/home/user/obj1data/logs/postproc_$r.log
run() { OMP_NUM_THREADS=1 nice -n 15 "$@" >> $L 2>&1; }
[ -f $M/predict ] || { run python3 scripts/predict.py $r && touch $M/predict; }
FA=""; [[ $b == mt_* ]] && FA="--fa 0.05"
[ -f $M/risk ] || { run python3 scripts/analyze_risk.py $r --pool test --reps 500 $FA && touch $M/risk; }
[ -f $M/strict ] || { run python3 scripts/analyze_risk.py $r --pool test --reps 200 --strict $FA && touch $M/strict; }
if [[ $b == neu_s0_yolo11n_256_seed* || $b == mt_s0_yolo11n_320_seed* ]]; then
  for cf in 0.25 0.75; do
    [ -f $M/cal$cf ] || { run python3 scripts/analyze_risk.py $r --pool test --reps 200 --cal-frac $cf $FA && touch $M/cal$cf; }
  done
fi
if [[ $b == *seed0 && $b != gc10r_* ]]; then
  for k in noise blur motion contrast dark bright jpeg; do for s in 1 2 3; do
    [ -f $M/c_$k$s ] || { run python3 scripts/predict.py $r --splits test --corrupt $k:$s && touch $M/c_$k$s; }
  done; done
  [ -f $M/shift ] || { run python3 scripts/analyze_shift.py $r && touch $M/shift; }
fi
touch $M/all
