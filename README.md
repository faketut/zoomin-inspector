# ZoomIn Inspector

An **agentic active-perception defect inspector** built for the OpenCV AI Competition 2026.

Instead of scoring a whole image in one shot, an inspector agent works under a fixed
**crop budget**: it scans the overview for a coarse anomaly heatmap, spends its budget
zooming into the most suspicious regions, re-plans (tighter re-zoom) when uncertain,
and escalates ambiguous cases to a human — logging a full perception → decision → action
trace. The same core workload runs on AWS Graviton via the COOL AMI with a reproducible
latency/throughput/cost benchmark against vanilla baselines.

**Focus paths:** Agentic Vision + Best Use of COOL.

## Quickstart

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt

# Inspect one image (no server needed). Use your own image, or download MVTec:
#   python scripts/download_mvtec.py --categories bottle --out data/mvtec
python -m zoomin.agent data/mvtec/bottle/test/broken_large/000.png --out out/

# With the fitted model behind the reported numbers:
ZOOMIN_MODEL=models/bottle_normality.npz \
  python -m zoomin.agent data/mvtec/bottle/test/broken_large/000.png --out out/

# API + UI
uvicorn app.api:app --port 8000        # POST /inspect
python -m app.ui                       # Gradio UI on :7860

# Reproducibility: benchmark the core workload
python bench/benchmark.py --images data/mvtec/bottle/test --config local --dollars-per-hour 0.0

# Tests
pip install -r requirements-dev.txt
python -m pytest tests/
```

`GET /health` reports the runtime `cv2.__version__` and the active scorer — the
COOL-vs-vanilla evidence the Best Use of COOL award asks for.

Without `ZOOMIN_MODEL`, the service uses `CoarseScorer` (no training data) with
its own calibrated thresholds. With `ZOOMIN_MODEL=<fitted .npz>` it uses
`NormalityScorer`. A set-but-missing path is a startup error rather than a
silent fallback to the weaker scorer.

## How it works

```
upload image
    │  scan_overview  → coarse heatmap (classical OpenCV; ONNX scorer hook via OpenCV 5 DNN)
    ▼
propose top-k regions ──▶ zoom_crop ──▶ inspect_crop ──▶ verdict
   (from the heatmap:        ▲                │
    vision sets the       re-zoom         score ≥ defect → DEFECT
    next tool call)       (re-plan)       unsure         → re-plan: coverage ≥ confirm → DEFECT
                                             else                       → ESCALATE
                                             else        → OK
    │                                              │
    └──────── finalize_report: verdicts + full tool trace + annotated image
```

The agent loop lives in `zoomin/agent.py`. Every tool call is logged with its inputs,
outputs, and rationale — that trace is the Agentic Vision evidence (vision results
*change* the next tool call: heatmap → crop coordinates, inspection coverage → re-zoom
vs escalate). The re-plan reads *coverage* (fraction of the tighter window that is hot),
not peak heat: the re-zoom window is nested inside the first crop, so its peak heat can
only stay level or drop, while coverage reflects how much of the tighter frame the
defect fills.

## Repo map

| Path | What |
|---|---|
| `zoomin/scoring.py` | Coarse heatmap scorer (classical OpenCV baseline) + `ONNXScorer` hook (OpenCV 5 DNN, ENGINE_AUTO) |
| `zoomin/tools.py` | `scan_overview`, `zoom_crop`, `inspect_crop`, `strongest_sub_box` |
| `zoomin/agent.py` | Evidence-driven loop with crop budget, re-planning, escalation, `finalize_report` |
| `zoomin/patchcore_adapter.py` | Optional anomalib PatchCore adapter (guarded import; CPU memory-bank) |
| `zoomin/report.py` | JSON report + annotated overview image |
| `app/api.py` | FastAPI: `POST /inspect`, `GET /health` |
| `app/ui.py` | Gradio UI: annotated image, region table, trace viewer |
| `bench/benchmark.py` | Workload benchmark → ms/image, images/sec, $/10K images, CSV |
| `scripts/download_mvtec.py` | Fetch MVTec AD categories (HF mirror) |
| `docs/REPORT.md` | Technical report skeleton (competition-required sections) |
| `docs/architecture.md` | Architecture diagram + component table |

## Datasets

- **MVTec AD** — 5,354 images / 15 categories: https://www.mvtec.com/company/research/datasets/mvtec-ad
  (HF mirror `Voxel51/mvtec-ad` as backup; `scripts/download_mvtec.py --categories bottle transistor capsule`)
- **VisA** — 10,821 images / 12 categories: https://amazon-visual-anomaly.s3.us-west-2.amazonaws.com/VisA_20220922.tar

Demo on 2–3 categories (bottle / transistor / capsule).

## AWS deployment (competition requirement)

Simple, reproducible, one-command:

1. S3 bucket: dataset + artifacts (`s3://zoomin-inspector/...`).
2. One **c8g.xlarge** from the COOL AMI
   (https://aws.amazon.com/marketplace/pp/prodview-jz6w32bsfv6sg) running
   `docker compose up` → FastAPI on :8000, Gradio on :7860.
3. One short-lived **c7i.xlarge** with vanilla `pip install opencv-python-headless`
   as the benchmark baseline — same container, same script, three configs.

Full step-by-step runbook (account setup → AMI subscribe → 3-config benchmark →
judges' endpoint → cleanup): [docs/AWS_DEPLOY.md](docs/AWS_DEPLOY.md).

```bash
# on each instance:
docker compose up -d
python bench/benchmark.py --images /app/data/mvtec/bottle/test --config cool-graviton \
    --dollars-per-hour 0.1718 --out bench/results.csv
```

## Benchmark protocol (Best Use of COOL evidence)

Same script, same N images, three runs: (A) COOL on Graviton, (B) vanilla pip
opencv on Graviton, (C) vanilla pip opencv on x86. Report: ms/image, images/sec,
$/10K images. Keep the CSVs in `bench/` and paste the table into `docs/REPORT.md`.

## Competition checklist

- [x] OpenCV 5 substantive image analysis (`cv2.__version__` major version asserted in tests)
- [x] Meaningful AWS component (S3 + EC2 serving + benchmark)
- [x] Agentic Vision: trace proves vision output changes later tool calls
- [x] COOL path: Graviton AMI + reproducible benchmark harness
- [x] Pinned dependencies (`requirements.txt`)
- [x] Architecture diagram (`docs/architecture.md`)
- [x] Evaluation incl. failure cases (`docs/REPORT.md` §6)
- [x] Final report numbers (EC2 runs 2026-10-22: benchmarks + 3-category eval in `docs/REPORT.md`, CSVs in `bench/`)
- [x] 5-min demo video (v2, 3:54, EC2 numbers landed; `video/out/zoomin-demo.mp4`)

## Troubleshooting

- **Gradio import fails with `httpx.InvalidURL: Invalid port: ':1]'`** — some sandboxes
  export bracketed IPv6 entries in `NO_PROXY`/`no_proxy` that older httpx versions
  misparse at import time. Workaround: `env -u NO_PROXY -u no_proxy python -m app.ui`.
  Not an issue on the AWS deployment target.

## License

MIT — see [LICENSE](LICENSE).
