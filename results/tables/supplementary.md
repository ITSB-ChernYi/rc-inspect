# Supplementary material

RC-Inspect: A lightweight risk-controlled deep learning framework for real-time defect inspection in steel and
electronic manufacturing

Code, split lists, cached predictions and the result files behind these tables: https://doi.org/10.5281/zenodo.24269095

## Table S1. YOLOv12n variants on NEU-DET

All variants share the YOLOv12n backbone and the training protocol of Table 3. Attention: channel–spatial attention on
the three backbone outputs. Weighted BiFPN: bidirectional feature pyramid with edge-weighted fusion nodes. Modified
loss: modified box-regression terms with a focal classification loss. Values are means ± standard deviations over
seeds (test partition); escape at the default threshold of 0.25 and review of RC-Inspect ($\alpha=0.10$, $\delta=0.1$,
false-alarm budget 0.5) are means over 500 calibration/evaluation partitions per seed. GFLOPs at the 256-pixel input
used for NEU-DET; Params and GFLOPs of the fused model, as in Table 5 of the paper.

| Variant | Attention | Neck | Loss | Initialisation | Seeds | Params (M) | GFLOPs (256 px) | mAP@0.5 (%) | mAP@0.5:0.95 (%) | Escape at 0.25 (%) | RC-Inspect: review (%) |
|---|---|---|---|---|---|---|---|---|---|---|---|
| YOLOv12n | - | PAN | default | COCO, all layers | 3 | 2.57 | 1.05 | 72.8 ± 1.1 | 40.1 ± 0.6 | 27.6 ± 0.8 | 53.0 ± 2.6 |
| YOLOv12n (backbone-only init.) | - | PAN | default | COCO, backbone only | 3 | 2.56 | 1.04 | 58.9 ± 4.1 | 27.8 ± 3.1 | 42.1 ± 4.0 | 63.7 ± 2.8 |
| Attention only | yes | PAN | default | COCO, backbone only | 1 | 2.61 | 1.06 | 59.2 | 28.5 | 41.5 | 63.7 |
| BiFPN neck only | - | BiFPN (96 ch.) | default | COCO, backbone only | 1 | 2.28 | 1.22 | 64.1 | 32.0 | 37.6 | 59.1 |
| Weighted BiFPN neck only | - | weighted BiFPN (96 ch.) | default | COCO, backbone only | 1 | 2.29 | 1.22 | 64.6 | 32.3 | 38.0 | 57.5 |
| Modified loss only | - | PAN | modified | COCO, backbone only | 1 | 2.56 | 1.04 | 47.6 | 22.1 | 54.5 | 69.5 |
| YOLOv12n-M (default loss) | yes | weighted BiFPN (96 ch.) | default | COCO, backbone only | 3 | 2.35 | 1.25 | 65.5 ± 0.6 | 33.3 ± 0.4 | 35.4 ± 0.4 | 56.9 ± 3.3 |
| YOLOv12n-M | yes | weighted BiFPN (96 ch.) | modified | COCO, all matching layers | 3 | 2.35 | 1.25 | 61.7 ± 1.2 | 31.9 ± 0.4 | 41.0 ± 1.8 | 59.1 ± 6.3 |

## Table S2. Device check

Seed-0 models trained on a 4-core CPU and on the GPU with the same seed, split and settings. Training on a different
device is not bit-identical; all results in the paper use the GPU-trained models. On Magnetic Tile, RC-Inspect sends
every image to review (Section 4.5.3).

| Dataset | Detector (seed 0) | mAP@0.5:0.95, CPU → GPU (%) | Escape at 0.25, CPU → GPU (%) | RC-Inspect review, CPU → GPU (%) |
|---|---|---|---|---|
| Magnetic Tile | YOLO11n | 38.8 → 42.7 | 26.8 → 27.6 | 100.0 → 100.0 |
| NEU-DET | YOLOv12n-M | 31.3 → 31.5 | 37.4 → 43.1 | 69.2 → 62.7 |
| NEU-DET | YOLOv12n-M (default loss) | 32.9 → 33.7 | 36.4 → 34.9 | 55.6 → 60.7 |
| NEU-DET | YOLOv12n (backbone-only init.) | 30.1 → 30.9 | 41.6 → 38.9 | 61.1 → 60.6 |
| NEU-DET | Modified loss only | 21.0 → 22.1 | 59.2 → 54.5 | 73.0 → 69.5 |
| NEU-DET | YOLO11n | 40.5 → 39.7 | 27.5 → 26.9 | 58.1 → 53.8 |
| NEU-DET | YOLOv12n | 40.8 → 39.5 | 28.7 → 28.3 | 53.8 → 51.2 |

## Table S3. Escape among the images decided without review

RC-Inspect ($\alpha=0.10$, $\delta=0.1$, false-alarm budget 0.5), at the operating points of the main analysis (500
partitions per seed; means over seeds). The guarantee of Eq. (4) concerns the escape over all defective images, with
reviewed images counting zero. This table reports the escape among the defective images that are accepted or rejected
without review, which is not guaranteed by the method, and the share of defective images that are accepted.
Partitions without a feasible operating point (full review) are excluded from the escape among non-reviewed images,
its 90th percentile and the share of partitions above 10 %, and the seeds are weighted by their number of feasible
partitions in these three columns (Faster R-CNN on NEU-DET has a feasible point in 79 %, 0.6 % and 100 % of the
partitions of its three seeds).

| Dataset | Detector | Seeds | Escape, all defective images (%) | Review (%) | Escape among non-reviewed images (%) | 90th percentile (%) | Partitions above 10 % (%) | Defective images accepted (%) |
|---|---|---|---|---|---|---|---|---|
| NEU-DET | YOLOv8n | 3 | 3.7 | 52.6 | 7.7 | 11.4 | 19 | 0.00 |
| NEU-DET | YOLO11n | 3 | 4.0 | 56.8 | 9.1 | 13.0 | 35 | 0.08 |
| NEU-DET | YOLOv12n | 3 | 4.1 | 53.0 | 8.6 | 11.8 | 26 | 0.00 |
| NEU-DET | YOLOv5nu | 3 | 4.1 | 57.0 | 9.2 | 12.7 | 35 | 0.01 |
| NEU-DET | YOLOv5su | 3 | 3.8 | 59.2 | 9.2 | 12.6 | 36 | 0.00 |
| NEU-DET | YOLOv8s | 3 | 4.0 | 55.2 | 8.8 | 12.1 | 30 | 0.02 |
| NEU-DET | YOLOv10n | 3 | 4.1 | 55.9 | 9.3 | 12.5 | 37 | 0.01 |
| NEU-DET | YOLO11s | 3 | 3.9 | 50.4 | 7.7 | 10.7 | 16 | 0.00 |
| NEU-DET | RT-DETR-L | 3 | 3.7 | 57.9 | 8.5 | 12.0 | 29 | 0.10 |
| NEU-DET | Faster R-CNN | 3 | 2.2 | 68.9 | 7.1 | 9.7 | 9 | 0.00 |
| NEU-DET | YOLOv12n-M | 3 | 4.0 | 59.1 | 9.8 | 13.9 | 43 | 0.03 |
| NEU-DET | YOLOv12n-M (default loss) | 3 | 4.1 | 56.9 | 9.4 | 13.2 | 39 | 0.04 |
| NEU-DET | YOLOv12n (backbone-only init.) | 3 | 4.1 | 63.7 | 11.1 | 15.1 | 64 | 0.12 |
| NEU-DET | Attention only | 1 | 4.0 | 63.7 | 10.8 | 15.5 | 57 | 0.02 |
| NEU-DET | BiFPN neck only | 1 | 4.1 | 59.1 | 9.8 | 14.1 | 45 | 0.00 |
| NEU-DET | Weighted BiFPN neck only | 1 | 4.1 | 57.5 | 9.6 | 13.2 | 47 | 0.02 |
| NEU-DET | Modified loss only | 1 | 3.9 | 69.5 | 12.8 | 18.9 | 70 | 0.00 |
| GC10-DET (grouped) | YOLO11n | 2 | 4.7 | 60.3 | 11.9 | 16.2 | 68 | 0.01 |
| GC10-DET (random) | YOLO11n | 2 | 4.6 | 47.3 | 8.7 | 12.7 | 33 | 0.13 |
| DeepPCB (grouped) | YOLOv8n | 3 | 3.2 | 13.8 | 3.7 | 4.9 | 0 | 0.00 |
| DeepPCB (grouped) | YOLO11n | 3 | 3.3 | 6.6 | 3.5 | 4.9 | 0 | 0.00 |
| DeepPCB (grouped) | YOLOv12n | 3 | 3.1 | 13.5 | 3.6 | 4.7 | 0 | 0.00 |
| DeepPCB (grouped) | YOLOv5nu | 1 | 3.5 | 4.4 | 3.6 | 4.8 | 0 | 0.00 |
| DeepPCB (grouped) | YOLOv5su | 1 | 3.4 | 0.9 | 3.4 | 4.5 | 0 | 0.00 |
| DeepPCB (grouped) | YOLOv8s | 1 | 3.4 | 3.0 | 3.5 | 4.6 | 0 | 0.00 |
| DeepPCB (grouped) | YOLOv10n | 1 | 3.1 | 6.5 | 3.3 | 4.6 | 0 | 0.00 |
| DeepPCB (grouped) | YOLO11s | 1 | 3.3 | 4.5 | 3.4 | 4.7 | 0 | 0.00 |
| DeepPCB (grouped) | RT-DETR-L | 1 | 3.1 | 17.0 | 3.7 | 4.9 | 0 | 0.00 |
| DeepPCB (grouped) | Faster R-CNN | 1 | 2.8 | 9.1 | 3.1 | 4.2 | 0 | 0.00 |
| DeepPCB (official) | YOLO11n | 2 | 4.7 | 0.0 | 4.7 | 6.0 | 0 | 0.08 |
