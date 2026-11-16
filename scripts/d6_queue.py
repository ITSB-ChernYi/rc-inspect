"""Resumable queue for the D6 detector comparison: train, cache predictions, run the risk analysis, one job at a time.

usage: python scripts/d6_queue.py configs/jobs_d6.txt [--device 0] [--workers 2]

Per job, three steps with a marker each ($OBJ1DATA/markers/<run>/): training (scripts/train.py, or
scripts/frcnn.py for --model fasterrcnn_r50fpn), predictions on val and test (scripts/predict.py or frcnn.py
predict) and the main risk analysis on the test pool (scripts/analyze_risk.py --reps 500), with the same settings
as scripts/process_run.sh. Finished runs are packaged in pc_runs/<run>/ as by scripts/pc_queue.py.
A failed step is logged and the queue moves to the next job.
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
PY = sys.executable


def package(name):
    src, dst = f"{DATA}/runs/{name}", os.path.join(OUT, name)
    os.makedirs(dst, exist_ok=True)
    shutil.copy(f"{src}/weights/best.pt", dst)
    for f in ("results.csv", "args.yaml", "run_record.json"):
        if os.path.exists(f"{src}/{f}"):
            shutil.copy(f"{src}/{f}", dst)


def step(name, tag, cmd, qlog, env=None):
    mk = f"{DATA}/markers/{name}/{tag}"
    if os.path.exists(mk):
        return True
    t = time.time()
    qlog.write(f"{time.strftime('%Y-%m-%d %H:%M:%S')} START {name} {tag}\n")
    qlog.flush()
    with open(f"{DATA}/logs/{name}_{tag}.log", "w") as log:
        rc = subprocess.run(cmd, stdout=log, stderr=subprocess.STDOUT, cwd=os.path.join(HERE, ".."), env=env).returncode
    qlog.write(f"{time.strftime('%Y-%m-%d %H:%M:%S')} END {name} {tag} rc={rc} {(time.time() - t) / 60:.1f} min\n")
    qlog.flush()
    if rc != 0:
        print(f"  {tag} FAILED (exit {rc}); see {DATA}/logs/{name}_{tag}.log", flush=True)
        return False
    os.makedirs(os.path.dirname(mk), exist_ok=True)
    open(mk, "w").close()
    print(f"  {tag} done in {(time.time() - t) / 60:.1f} min", flush=True)
    return True


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("jobs")
    ap.add_argument("--device", default="0")
    ap.add_argument("--workers", type=int, default=2)
    a = ap.parse_args()
    os.makedirs(f"{DATA}/logs", exist_ok=True)
    qlog = open(f"{DATA}/logs/d6_queue.log", "a")
    jobs = [l.strip().split("|", 1) for l in open(a.jobs) if l.strip() and not l.startswith("#")]
    for k, (name, args) in enumerate(jobs, 1):
        print(f"[{k}/{len(jobs)}] {time.strftime('%H:%M:%S')} {name}", flush=True)
        argv = [x.replace("/home/user/obj1data", DATA) for x in shlex.split(args)]
        frcnn = "fasterrcnn_r50fpn" in argv
        if not os.path.exists(f"{DATA}/runs/{name}/run_record.json"):
            shutil.rmtree(f"{DATA}/runs/{name}", ignore_errors=True)
            shutil.rmtree(f"{DATA}/markers/{name}", ignore_errors=True)
            script = ["frcnn.py", "train"] if frcnn else ["train.py"]
            cmd = [PY, os.path.join(HERE, script[0]), *script[1:], "--name", name, *argv,
                   "--device", a.device, "--workers", str(a.workers)]
            if not step(name, "train", cmd, qlog):
                continue
        package(name)
        pred = [PY, os.path.join(HERE, "frcnn.py"), "predict", name] if frcnn else [PY, os.path.join(HERE, "predict.py"), name]
        if not step(name, "predict", pred, qlog):
            continue
        env = {**os.environ, "OMP_NUM_THREADS": "1"}
        step(name, "risk", [PY, os.path.join(HERE, "analyze_risk.py"), name, "--pool", "test", "--reps", "500"], qlog, env)
    print("queue finished", flush=True)


if __name__ == "__main__":
    main()
