"""Import runs trained on the GPU PC (objective1/pc_runs/<run>/) into the cloud evaluation pipeline.

usage: git fetch origin pc-results && git checkout origin/pc-results -- objective1/pc_runs
       python scripts/import_pc_runs.py

Each run is copied to /home/user/obj1data/runs/<run>/ (weights/best.pt, results.csv, args.yaml,
run_record.json). The data path recorded on the PC is mapped to the cloud copy of the same split
(identical membership and labels, verified by scripts/pc_setup.py against results/audit/pc_reference.json),
and the original path is kept as data_trained. scripts/postproc_watch.sh then evaluates the runs on the
cloud CPU like every other run. Runs already imported are skipped.
"""
import glob
import json
import os
import shutil

HERE = os.path.dirname(os.path.abspath(__file__))
SRC = os.path.join(HERE, "..", "pc_runs")
RUNS = "/home/user/obj1data/runs"


def main():
    for d in sorted(glob.glob(f"{SRC}/*/")):
        name = os.path.basename(d.rstrip("/"))
        dst = f"{RUNS}/{name}"
        if os.path.exists(f"{dst}/run_record.json"):
            continue
        rec = json.load(open(f"{d}/run_record.json"))
        data = rec["args"]["data"].replace("\\", "/")
        ds = data.rstrip("/").split("/")[-2]           # e.g. neu_s0
        rec["args"]["data_trained"] = data
        rec["args"]["data"] = f"/home/user/obj1data/{ds}/data.yaml"
        assert os.path.exists(rec["args"]["data"]), rec["args"]["data"]
        os.makedirs(f"{dst}/weights", exist_ok=True)
        shutil.copy(f"{d}/best.pt", f"{dst}/weights/best.pt")
        for f in ("results.csv", "args.yaml"):
            shutil.copy(f"{d}/{f}", dst)
        json.dump(rec, open(f"{dst}/run_record.json", "w"), indent=1)   # written last: marks the run complete
        print("imported", name, rec.get("hardware", {}).get("gpu"))


if __name__ == "__main__":
    main()
