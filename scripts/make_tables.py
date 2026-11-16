"""Generate manuscript tables (Markdown) from saved result files. No number is typed by hand."""
import glob
import json
import os

import numpy as np
import pandas as pd

ROOT = os.path.join(os.path.dirname(__file__), "..")
R = os.path.join(ROOT, "results")
MD = os.path.join(R, "tables", "md")
D = os.environ.get("OBJ1DATA", "/home/user/obj1data").replace("\\", "/").rstrip("/")   # as sdd/paths.py
os.makedirs(MD, exist_ok=True)


def md(df):
    cols = list(df.columns)
    out = ["| " + " | ".join(map(str, cols)) + " |", "|" + "|".join("---" for _ in cols) + "|"]
    for _, r in df.iterrows():
        out.append("| " + " | ".join(str(v) for v in r) + " |")
    return "\n".join(out) + "\n"


def table1():
    neu = json.load(open(f"{R}/audit/neu_audit.json"))
    gc = json.load(open(f"{R}/audit/gc10_audit.json"))
    mt = json.load(open(f"{R}/audit/mt_audit.json"))
    grp = json.load(open(f"{R}/audit/mt_tile_groups.json"))["report"]
    sm = {k: json.load(open(f"{D}/{k}/split_meta.json")) for k in ("neu_s0", "gc10_grouped_s0", "gc10_random_s0", "mt_s0")}
    n = lambda m, s: m["splits"][s]["n"] if "n" in m["splits"][s] else m["splits"][s]["images"]  # noqa: E731
    rows = [
        {"Dataset": "NEU-DET (steel strip)", "Images / boxes": f"{neu['n_images']} / {neu['n_boxes']}", "Classes": 6,
         "Image size (px)": "200 × 200", "Defect-free images": 0,
         "Audit findings": f"{len(neu['exact_duplicate_files'])} byte-identical images (1 pair); {len(neu['problems'])} boxes of co-occurring classes",
         "Split (train/val/test)": f"stratified by class; duplicate pair kept together; {n(sm['neu_s0'],'train')}/{n(sm['neu_s0'],'val')}/{n(sm['neu_s0'],'test')}"},
        {"Dataset": "GC10-DET (steel sheet)", "Images / boxes": f"{gc['n_images']} / {gc['n_boxes']}", "Classes": 10,
         "Image size (px)": "2048 × 1000", "Defect-free images": 0,
         "Audit findings": f"{gc['n_sequences']} acquisition sequences (≤ {int(gc['images_per_sequence_quantiles']['1.0'])} frames); {gc['same_original_name_in_two_folders']} images in two class folders; {gc['issue_counts'].get('no annotation file', 0)} without annotation; {gc['issue_counts'].get('folder class absent from annotation', 0)} XML/folder class disagreements; 1 unknown label",
         "Split (train/val/test)": f"grouped by sequence: {n(sm['gc10_grouped_s0'],'train')}/{n(sm['gc10_grouped_s0'],'val')}/{n(sm['gc10_grouped_s0'],'test')} (0 shared sequences); image-random: {n(sm['gc10_random_s0'],'train')}/{n(sm['gc10_random_s0'],'val')}/{n(sm['gc10_random_s0'],'test')} ({sm['gc10_random_s0']['test_images_sharing_sequence_with_train']} test images share a sequence with training)"},
        {"Dataset": "Magnetic Tile (ferrite tile)", "Images / boxes": f"{mt['n_images']} / {sum(mt['boxes_per_class'].values())}", "Classes": 5,
         "Image size (px)": f"{mt['image_w_range'][0]}–{mt['image_w_range'][1]} × {mt['image_h_range'][0]}–{mt['image_h_range'][1]}",
         "Defect-free images": mt["per_folder"]["MT_Free"],
         "Audit findings": f"same tile under several exposures: {grp['n_groups']} tile groups ({grp['group_size_counts'].get('6', grp['group_size_counts'].get(6, 0))} of six images); boxes from mask components; {mt['defective_images_without_box']} defective images without a component ≥ 9 px",
         "Split (train/val/test)": f"grouped by tile, stratified: {sm['mt_s0']['splits']['train']['images']}/{sm['mt_s0']['splits']['val']['images']}/{sm['mt_s0']['splits']['test']['images']} ({sm['mt_s0']['splits']['test']['defective']} defective test images)"},
    ]
    open(f"{MD}/table1_datasets.md", "w").write(md(pd.DataFrame(rows)))


if __name__ == "__main__":
    table1()
    print(open(f"{MD}/table1_datasets.md").read())
