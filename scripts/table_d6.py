"""D6 comparison table: test accuracy and guaranteed operating point of every detector on NEU-DET and DeepPCB.

Means (and standard deviations) over the seeds that exist for each detector. RC-Inspect = Learn-then-Test with the
union review rule, alpha = 0.10, false-alarm budget 0.5 (row ltt_review_union@fp0.5 of the risk summaries).
Writes results/tables/d6_comparison.csv and prints it. Latency columns are added from results/latency_d9.json if present.
usage: python scripts/table_d6.py
"""
import csv
import json
import os

import numpy as np

R = os.path.join(os.path.dirname(__file__), "..", "results")
DETS = [("Faster R-CNN (R50-FPN)", "frcnn"), ("RT-DETR-L", "rtdetrl"), ("YOLOv5nu", "yolov5nu"), ("YOLOv5su", "yolov5su"),
        ("YOLOv8n", "yolov8n"), ("YOLOv8s", "yolov8s"), ("YOLOv10n", "yolov10n"), ("YOLO11n", "yolo11n"),
        ("YOLO11s", "yolo11s"), ("YOLOv12n", "yolo12n")]
SETS = [("NEU-DET", "gpu_neu_s0_{}_256_seed{}"), ("DeepPCB (grouped)", "gpu_pcbg_s0_{}_640_seed{}")]


def one(run):
    m = json.load(open(f"{R}/metrics/{run}_test.json"))
    rows = {(r["method"], r["alpha"]): r for r in csv.DictReader(open(f"{R}/risk/{run}__test_summary.csv"))}
    d, u, c = rows[("default_0.25", "")], rows[("ltt_review_union@fp0.5", "0.1")], rows[("crc", "0.1")]
    return {"map50": 100 * m["map50"], "map50_95": 100 * m["map50_95"], "esc_def": 100 * float(d["inst_escape"]),
            "fa_def": float(d["fp_img"]), "crc_esc": 100 * float(c["inst_escape"]), "crc_viol": 100 * float(c["viol_rate"]),
            "rc_esc": 100 * float(u["inst_escape"]), "rc_viol": 100 * float(u["viol_rate"]),
            "rc_review": 100 * float(u["review"]), "rc_fa": float(u["fp_img"])}


def main():
    lat = json.load(open(f"{R}/latency_d9.json")) if os.path.exists(f"{R}/latency_d9.json") else {}
    out = []
    for ds, pat in SETS:
        for name, key in DETS:
            runs = [pat.format(key, s) for s in range(3) if os.path.exists(f"{R}/risk/{pat.format(key, s)}__test_summary.csv")]
            if not runs:
                continue
            v = [one(r) for r in runs]
            row = {"dataset": ds, "detector": name, "seeds": len(runs)}
            for k in v[0]:
                a = np.array([x[k] for x in v])
                row[k] = float(a.mean())
                row[k + "_sd"] = float(a.std(ddof=1)) if len(a) > 1 else ""
            row["rc_review_min"] = round(min(x["rc_review"] for x in v), 1)
            row["rc_review_max"] = round(max(x["rc_review"] for x in v), 1)
            e = lat.get(pat.format(key, 0))
            if e:
                row.update(params_m=round(e["params"] / 1e6, 2), gflops=round(e["gflops_at_imgsz"], 2),
                           e2e_ms=round(e["end_to_end_4thr"]["total_ms"], 1),
                           model_1thr_ms=round(e["model_only"]["1thr"]["median_ms"], 1),
                           model_4thr_ms=round(e["model_only"]["4thr"]["median_ms"], 1))
            out.append(row)
    os.makedirs(f"{R}/tables", exist_ok=True)
    keys = sorted({k for r in out for k in r}, key=lambda k: list(out[0]).index(k) if k in out[0] else 99)
    with open(f"{R}/tables/d6_comparison.csv", "w", newline="") as f:
        w = csv.DictWriter(f, keys)
        w.writeheader()
        w.writerows(out)
    md = ["| Dataset | Detector | Seeds | mAP@0.5 (%) | mAP@0.5:0.95 (%) | Escape at 0.25 | RC-Inspect: escape / violations "
          "| RC-Inspect: review | Params (M) | GFLOPs | Latency (ms) |", "|---|---|---|---|---|---|---|---|---|---|---|"]
    pm = lambda r, k: f"{r[k]:.1f} ± {r[k + '_sd']:.1f}" if r["seeds"] > 1 else f"{r[k]:.1f}"
    for r in out:
        md.append(f"| {r['dataset']} | {r['detector']} | {r['seeds']} | {pm(r, 'map50')} | {pm(r, 'map50_95')} | "
                  f"{r['esc_def']:.1f} | {r['rc_esc']:.1f} / {r['rc_viol']:.1f} | {r['rc_review']:.1f} | "
                  f"{r.get('params_m', float('nan')):.2f} | {r.get('gflops', float('nan')):.2f} | {r.get('e2e_ms', float('nan')):.1f} |")
    open(f"{R}/tables/d6_comparison.md", "w", encoding="utf-8").write("\n".join(md) + "\n")
    for r in out:
        print(f"{r['dataset'][:7]:8s}{r['detector']:24s}{r['seeds']} mAP {r['map50']:.1f}±{r['map50_sd'] or 0:.1f} "
              f"{r['map50_95']:.1f}±{r['map50_95_sd'] or 0:.1f} esc@.25 {r['esc_def']:.1f} | CRC {r['crc_esc']:.1f}/{r['crc_viol']:.0f}"
              f" | RC {r['rc_esc']:.1f}/{r['rc_viol']:.1f} rev {r['rc_review']:.1f} ({r['rc_review_min']}-{r['rc_review_max']})"
              f" fa {r['rc_fa']:.2f}" + (f" | {r['params_m']}M {r['gflops']}G {r['e2e_ms']}ms" if "e2e_ms" in r else ""))


if __name__ == "__main__":
    main()
