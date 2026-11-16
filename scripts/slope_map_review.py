"""Least-squares slope of review rate (RC-Inspect, alpha 0.10, budget 0.5) and default-threshold escape on test mAP@0.5:0.95
for all NEU-DET models, per model and over architecture means with a 2,000-resample bootstrap interval (seed 0).
usage (from objective1/): python scripts/slope_map_review.py
"""
import csv,glob,json,os,re,numpy as np
NEW=("yolov5nu","yolov5su","yolov8s","yolov10n","yolo11s","rtdetrl","frcnn")
rows=[]
for f in sorted(glob.glob("results/risk/gpu_neu_s0_*_256_seed?__test_summary.csv")):
    run=os.path.basename(f).split("__")[0]; arch=re.match(r"gpu_neu_s0_(.+)_256_seed\d",run).group(1)
    R={(r["method"],r["alpha"]):r for r in csv.DictReader(open(f))}
    m=json.load(open(f"results/metrics/{run}_test.json"))
    rows.append((arch,100*m["map50_95"],100*float(R[("ltt_review_union@fp0.5","0.1")]["review"]),100*float(R[("default_0.25","")]["inst_escape"])))
print(len(rows), sorted({r[0] for r in rows}))
def fit(sel,lab):
    x=np.array([r[1] for r in sel]);y=np.array([r[2] for r in sel]);e=np.array([r[3] for r in sel])
    print(f"{lab}: n={len(sel)} slope_review={np.polyfit(x,y,1)[0]:.2f} r={np.corrcoef(x,y)[0,1]:.2f} | slope_esc={np.polyfit(x,e,1)[0]:.2f} r={np.corrcoef(x,e)[0,1]:.2f}")
    A=sorted({r[0] for r in sel}); xm=np.array([np.mean([r[1] for r in sel if r[0]==a]) for a in A]); ym=np.array([np.mean([r[2] for r in sel if r[0]==a]) for a in A])
    rng=np.random.default_rng(0); bs=[]
    for _ in range(2000):
        i=rng.integers(0,len(A),len(A))
        if np.ptp(xm[i])>0: bs.append(np.polyfit(xm[i],ym[i],1)[0])
    print(f"   arch means n={len(A)} slope={np.polyfit(xm,ym,1)[0]:.2f} CI [{np.percentile(bs,2.5):.2f},{np.percentile(bs,97.5):.2f}] r={np.corrcoef(xm,ym)[0,1]:.2f}")
old=[r for r in rows if r[0] not in NEW]; fit(old,"original")
fit(rows,"all"); fit([r for r in rows if r[0]!="frcnn"],"all but frcnn")
fit([r for r in rows if r[0] in NEW or r[0] in ("yolo11n","yolo12n","yolov8n")],"stock+new")
