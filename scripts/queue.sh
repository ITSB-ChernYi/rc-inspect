#!/usr/bin/env bash
# Resumable job queue: each line of the job file is "<name>|<args for train.py>".
# A job is skipped when /home/user/obj1data/runs/<name>/weights/last.pt and run_record.json exist.
# Usage: bash scripts/queue.sh configs/jobs_neu.txt
cd "$(dirname "$0")/.."
JOBS=$1
while IFS='|' read -r name args; do
  [[ -z "$name" || "$name" == \#* ]] && continue
  if [ -f "/home/user/obj1data/runs/$name/run_record.json" ]; then continue; fi
  echo "$(date -u) START $name" >> /home/user/obj1data/logs/queue.log
  # a partially trained run restarts from scratch (deterministic, same seed)
  rm -rf "/home/user/obj1data/runs/$name"
  eval OMP_NUM_THREADS=4 python3 scripts/train.py --name "$name" $args > "/home/user/obj1data/logs/$name.log" 2>&1
  echo "$(date -u) END $name rc=$?" >> /home/user/obj1data/logs/queue.log
done < "$JOBS"
echo "$(date -u) QUEUE DONE $JOBS" >> /home/user/obj1data/logs/queue.log
