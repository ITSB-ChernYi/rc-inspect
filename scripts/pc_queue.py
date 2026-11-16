"""Resumable training queue for a GPU machine (any OS); counterpart of scripts/queue.sh.

usage: python objective1/scripts/pc_queue.py objective1/configs/jobs_pc.txt [--device 0] [--workers 6]

Runs every job of the file whose run_record.json does not exist yet (a partially trained run restarts
from scratch with the same seed). After each job the files needed for evaluation are copied to
objective1/pc_runs/<run>/ (best.pt, results.csv, args.yaml, run_record.json), ready to commit and push.
Stop with Ctrl+C at any time and start again later; finished jobs are skipped.
"""
import argparse
import os
import shlex
import shutil
import subprocess
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, ".."))
from sdd.paths import DATA  # noqa: E402

OUT = os.path.join(HERE, "..", "pc_runs")


def package(name):
    src = f"{DATA}/runs/{name}"
    dst = os.path.join(OUT, name)
    os.makedirs(dst, exist_ok=True)
    shutil.copy(f"{src}/weights/best.pt", dst)
    for f in ("results.csv", "args.yaml", "run_record.json"):
        shutil.copy(f"{src}/{f}", dst)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("jobs")
    ap.add_argument("--device", default="0")
    ap.add_argument("--workers", type=int, default=6)
    a = ap.parse_args()
    os.makedirs(f"{DATA}/logs", exist_ok=True)
    qlog = open(f"{DATA}/logs/pc_queue.log", "a")
    jobs = [l.strip().split("|", 1) for l in open(a.jobs) if l.strip() and not l.startswith("#")]
    for k, (name, args) in enumerate(jobs, 1):
        if os.path.exists(f"{DATA}/runs/{name}/run_record.json"):
            if not os.path.exists(os.path.join(OUT, name, "best.pt")):
                package(name)
            continue
        shutil.rmtree(f"{DATA}/runs/{name}", ignore_errors=True)
        argv = [x.replace("/home/user/obj1data", DATA) for x in shlex.split(args)]
        cmd = [sys.executable, os.path.join(HERE, "train.py"), "--name", name, *argv,
               "--device", a.device, "--workers", str(a.workers)]
        print(f"[{k}/{len(jobs)}] {time.strftime('%H:%M:%S')} start {name}", flush=True)
        qlog.write(f"{time.strftime('%Y-%m-%d %H:%M:%S')} START {name}\n")
        qlog.flush()
        t = time.time()
        with open(f"{DATA}/logs/{name}.log", "w") as log:
            rc = subprocess.run(cmd, stdout=log, stderr=subprocess.STDOUT).returncode
        qlog.write(f"{time.strftime('%Y-%m-%d %H:%M:%S')} END {name} rc={rc}\n")
        qlog.flush()
        if rc != 0:
            print(f"  FAILED (exit {rc}); see {DATA}/logs/{name}.log", flush=True)
            continue
        package(name)
        print(f"  done in {(time.time() - t) / 60:.1f} min -> objective1/pc_runs/{name}", flush=True)
    print("queue finished")


if __name__ == "__main__":
    main()
