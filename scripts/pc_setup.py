"""Prepare a second training machine (e.g. a Windows PC with a CUDA GPU).

usage: python objective1/scripts/pc_setup.py            (data root: $OBJ1DATA, e.g. C:/obj1data)

1. clones the three public dataset repositories at the commits used in the cloud workspace,
2. rebuilds the NEU-DET, GC10-DET (grouped and random) and Magnetic Tile splits with the same code,
3. checks split membership and every label file against results/audit/pc_reference.json,
4. downloads the COCO-pretrained weights and checks their sha256.
Re-running is safe: finished steps are skipped. The script stops at the first mismatch.
"""
import hashlib
import json
import os
import shutil
import subprocess
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, ".."))
from sdd import gc10, mt, neu  # noqa: E402
from sdd.paths import DATA, label_digest  # noqa: E402

REF = json.load(open(os.path.join(HERE, "..", "results", "audit", "pc_reference.json")))
MT_GROUPS = json.load(open(os.path.join(HERE, "..", "results", "audit", "mt_tile_groups.json")))["groups"]


def step(msg):
    print(f"\n== {msg}", flush=True)


def clone():
    os.makedirs(f"{DATA}/raw", exist_ok=True)
    for name, r in REF["repos"].items():
        d = f"{DATA}/raw/{name}"
        if not os.path.isdir(f"{d}/.git"):
            step(f"cloning {r['url']} (GC10 is about 2 GB)")
            subprocess.run(["git", "clone", r["url"], d], check=True)
        subprocess.run(["git", "-C", d, "-c", "advice.detachedHead=false", "checkout", r["commit"]], check=True)
        head = subprocess.run(["git", "-C", d, "rev-parse", "HEAD"], capture_output=True, text=True).stdout.strip()
        assert head == r["commit"], f"{name}: HEAD {head} != {r['commit']}"
        print(f"{name}: at {head[:12]}")


def verify(ds):
    root = f"{DATA}/{ds}"
    m = json.load(open(f"{root}/split_meta.json"))["splits"]
    ok = True
    for s, want in REF["datasets"][ds].items():
        n = m[s].get("n", m[s].get("images"))
        got = {"n": n, "stems_sha256": m[s]["sha256"], "labels_sha256": label_digest(root, s)}
        for k in want:
            if got[k] != want[k]:
                print(f"MISMATCH {ds}/{s} {k}: {got[k]} != {want[k]}")
                ok = False
    print(f"{ds}: {'OK' if ok else 'FAILED'}")
    return ok


def build():
    jobs = (("neu_s0", lambda r: neu.build(r, 0)),
            ("gc10_grouped_s0", lambda r: gc10.build(r, 0, "grouped")),
            ("gc10_random_s0", lambda r: gc10.build(r, 0, "random")),
            ("mt_s0", lambda r: mt.build(r, 0, groups=MT_GROUPS)))
    for ds, fn in jobs:
        root = f"{DATA}/{ds}"
        if os.path.exists(f"{root}/split_meta.json"):
            if verify(ds):
                continue
            step(f"existing {ds} does not match; rebuilding it")
            shutil.rmtree(root)
        step(f"building {ds}")
        fn(root)
        if not verify(ds):
            sys.exit(f"{ds} does not match the cloud split; do not train on it")


def weights():
    from ultralytics.utils.downloads import attempt_download_asset
    os.makedirs(f"{DATA}/weights", exist_ok=True)
    for w, want in REF["weights"].items():
        p = f"{DATA}/weights/{w}.pt"
        if not os.path.exists(p):
            cwd = os.getcwd()
            os.chdir(f"{DATA}/weights")
            attempt_download_asset(f"{w}.pt")
            os.chdir(cwd)
        got = hashlib.sha256(open(p, "rb").read()).hexdigest()
        print(f"{w}.pt: {'OK' if got == want else 'DIFFERENT from the cloud copy'}")
        if got != want:
            sys.exit(f"{w}.pt differs from the weights used in the cloud runs")


def main():
    print(f"data root: {DATA}")
    step("dataset repositories")
    clone()
    step("splits")
    build()
    step("pretrained weights")
    weights()
    import torch
    step("GPU")
    print("CUDA available:", torch.cuda.is_available(),
          "|", torch.cuda.get_device_name(0) if torch.cuda.is_available() else "no GPU visible to PyTorch")
    print("\nSetup complete.")


if __name__ == "__main__":
    main()
