#!/usr/bin/env bash
# Background watcher: runs scripts/process_run.sh for every finished run (run_record.json present).
cd "$(dirname "$0")/.."
while true; do
  for d in /home/user/obj1data/runs/*/; do
    r=$(basename "$d")
    [ -f "$d/run_record.json" ] || continue
    [ -f "/home/user/obj1data/markers/$r/all" ] && continue
    echo "$(date -u) process $r" >> /home/user/obj1data/logs/postproc.log
    bash scripts/process_run.sh "$r"
  done
  [ -f /home/user/obj1data/logs/postproc.stop ] && break
  sleep 300
done
