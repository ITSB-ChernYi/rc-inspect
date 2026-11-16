# RC-Inspect: risk-controlled decisions for lightweight defect detectors

Code, split lists, cached predictions and result files for the paper

> Lim CY, Lim JMY, Kalavally V, Lim WS. RC-Inspect: A lightweight risk-controlled deep learning framework for
> real-time defect inspection in steel and electronic manufacturing. Submitted to *Computers and Electrical
> Engineering*.

RC-Inspect sits on top of an unchanged object detector. It turns the detector output into accept, reject and
review decisions and calibrates the decision parameters on a small labelled set so that the share of defects that
escape the automatic decisions stays below a target with a stated confidence. The method is implemented in
`sdd/risk.py`; everything else in this repository builds the datasets, trains the detectors and evaluates the
decision rules.

## Contents

| Path | What it holds |
|---|---|
| `sdd/` | Dataset builders (`neu.py`, `gc10.py`, `mt.py`, `deeppcb.py`), the risk layer (`risk.py`), the acquisition-quality gate (`quality.py`), synthetic corruptions (`corrupt.py`), AP from cached predictions (`evalmap.py`) and the re-implemented YOLO-SEC modules (`yolosec.py`) |
| `scripts/` | Training, prediction, analysis and plotting scripts; each starts with a docstring that states its inputs and outputs |
| `configs/` | Model YAML files of the YOLOv12n variants and the job lists used for training |
| `tests/` | Property tests of the risk layer (`pytest tests/`) |
| `data/splits/<split>/` | `train.txt`, `val.txt`, `test.txt` (sorted image stems) and `split_meta.json` (sizes, groups, SHA-256 of each list) |
| `data/preds/<run>/` | Cached predictions of every trained detector on its validation and test partitions (`*.pkl`) |
| `data/preds_quant/<run>/` | Predictions of the quantisation study (FP32 and INT8 ONNX models) |
| `results/` | Result files behind the tables and figures: `metrics/`, `risk/`, `risk_extra/`, `quant/`, `audit/`, `tables/`, `figures/`, latency JSON files |

No images, label files or model weights are included. The images and annotations belong to the dataset authors
and must be obtained from the sources below.

## Datasets

| Dataset | Source used | Split directories |
|---|---|---|
| NEU-DET | https://github.com/siddhartamukherjee/NEU-DET-Steel-Surface-Defect-Detection (commit 351bee2) | `neu_s0` |
| GC10-DET | https://github.com/wyx19980727/GC10-DET-VOC (commit 2459a71) | `gc10_grouped_s0`, `gc10_random_s0` |
| Magnetic Tile | https://github.com/abin24/Magnetic-tile-defect-datasets. (commit cc4eb53) | `mt_s0` |
| DeepPCB | https://github.com/tangsanli5201/DeepPCB (commit 08e98c4) | `deeppcb_grouped_s0`, `deeppcb_official` |

The builders in `sdd/` convert each dataset to YOLO format and write the splits. Each list in `data/splits` is
the sorted list of image stems of one partition, and the SHA-256 of the newline-joined list equals the value in
`split_meta.json`. A rebuilt split can be checked against these files, and `sdd/paths.py: label_digest` together
with `results/audit/pc_reference.json` also checks the label contents.

## Setup

Python 3.12 was used. Install the packages in `requirements.txt` (PyTorch with CUDA for training; CPU is enough
for the analyses that read cached predictions) and point `OBJ1DATA` at the data folder:

```bash
pip install -r requirements.txt
export OBJ1DATA=$PWD/data          # Windows PowerShell: $env:OBJ1DATA = "$PWD/data"
```

Training and prediction (`scripts/train.py`, `scripts/predict.py`, `scripts/frcnn.py`, `scripts/quant_predict.py`)
also need the built datasets under `$OBJ1DATA` and the pretrained weights of Ultralytics and torchvision. The
analysis and plotting scripts only need `data/preds`, `data/preds_quant` and `results/`.

Run the scripts from the repository root, for example

```bash
python scripts/analyze_risk.py --help
python scripts/fig_review_budget.py
```

## Where the numbers in the paper come from

| Paper item | Script | Result files |
|---|---|---|
| Table 4, dataset audit | `sdd/*.py`, `scripts/audit_*.py`, `scripts/group_mt.py` | `data/splits/*/split_meta.json`, `results/audit/` |
| Table 5, detection accuracy and complexity | `scripts/analyze_detection.py`, `scripts/check_models.py`, `scripts/bootstrap_stock_pairs.py` | `results/metrics/`, `results/tables/`, `results/architectures.json` |
| Table 6, detector comparison | `scripts/table_d6.py` | `results/tables/d6_comparison.csv` |
| Tables 7–9, decision rules and their cost | `scripts/analyze_risk.py` | `results/risk/<run>__test*.csv` |
| Table 10, latency | `scripts/bench_latency.py`, `scripts/bench_gate.py` | `results/latency_*.json` |
| Table 11, quantisation | `scripts/quant_predict.py`, `scripts/quant_analyze.py` | `results/quant/` |
| Table 12, acquisition shift | `scripts/analyze_shift.py` | `results/risk/<run>__shift*.csv` |
| Table 13, imperfect review and related measures | `scripts/analyze_review_extra.py` | `results/risk_extra/` |
| Fig. 1 | `scripts/fig_architecture.py` | `results/figures/fig_architecture.*` |
| Figs. 3, 4 and 8 | `scripts/fig_results_redraw.py` | `results/figures/fig3_escape_threshold.*`, `fig4_realised_escape_v2.*`, `fig8_shift.*` |
| Fig. 5 | `scripts/fig_review_budget.py` | `results/figures/fig_review_budget.*` |
| Fig. 6 | `scripts/fig_operating_curves.py` | `results/figures/fig_operating_curves.*` |
| Fig. 7 | `scripts/fig_map_review.py`, `scripts/slope_map_review.py` | `results/figures/fig_map_vs_review.*` |
| Fig. 9 | `scripts/fig_examples.py` (needs the dataset images) | `results/figures/fig_examples.*` |
| Graphical abstract | `scripts/fig_graphical_abstract.py` | `results/figures/graphical_abstract.*` |

Fig. 2 shows dataset images with their annotations and is not generated by a script. `scripts/make_figures.py`
holds the shared plotting style and the figures of an earlier draft.

The per-partition frontier files (`*_frontier.csv`, about 1.3 GB) are not included, except the two read by
`scripts/fig_review_budget.py`. `scripts/analyze_risk.py` regenerates them from the cached predictions.

## Cached predictions

Each `.pkl` file holds a Python dictionary with the keys `pred` and `gt` (both keyed by image stem: `boxes`,
`scores` and `cls` of the detections before any decision threshold, and `boxes` and `cls` of the ground truth
derived from the dataset annotations), `names`, `imgsz` and `seconds`. Load them with `pickle` only from this repository, as pickle files can execute
code.

## Licence

Code: MIT licence (`LICENSE`). Split lists, cached predictions and result files: CC BY 4.0. The datasets remain
under the terms set by their authors.

## Citation

See `CITATION.cff`. Please cite the paper and this archive.
