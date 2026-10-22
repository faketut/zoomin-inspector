# ZoomIn Inspector — Technical Report (draft)

> Fill each section before submission. The competition requires: problem, users,
> architecture, OpenCV 5 implementation, AWS deployment, evaluation, limitations,
> responsible-use considerations — plus evaluation evidence **including failure cases**.

## 1. Problem

Industrial visual inspection wastes compute and misses small defects: whole-image
classifiers either downscale away the defect or run expensive detectors everywhere.
Human inspectors do the opposite — they scan the overview, then spend attention
zooming into suspicious regions, and escalate when unsure. ZoomIn Inspector is an
agent that inspects the way a human does, under an explicit inspection budget.

## 2. Users / beneficiaries

Quality engineers on manufacturing lines who need a CPU-only inspection aid
that shows its work (the agent trace is the audit trail); small factories that
cannot afford GPU inspection rigs — the whole pipeline runs on commodity CPU
(Graviton or x86) by design; ML teams prototyping anomaly-detection baselines
before committing to learned models. The escalation path means the system
assists human QA rather than replacing it.

## 3. Architecture

See [architecture.md](architecture.md). One container serves the FastAPI API and the
Gradio UI; the agent loop (`zoomin/agent.py`) orchestrates five tools
(`scan_overview`, `zoom_crop`, `inspect_crop`, re-plan re-zoom, `finalize_report`).

## 4. OpenCV 5 implementation

- **Anomaly scoring** (`zoomin/scoring.py`): two classical scorers, one interface
  (`score(gray) -> heatmap`, absolute units, never per-image normalized).
  Default is **NormalityScorer**: per-pixel Gaussian over normal training
  images, score in std-deviations — the textbook unsupervised-AD baseline,
  NumPy/OpenCV only, no learned weights, ~0.5 MB/category. Fallback is
  **CoarseScorer** (single-image median-deviation heuristic, no train data
  needed). A per-image min-max normalization was tried first and removed:
  it turned reflections and glass edges on clean bottles into hotspots
  (every image flagged DEFECT, AUROC 0.44).
- **Region proposals**: absolute threshold + `connectedComponentsWithStats`
  with a minimum area (kills isolated specks on clean images).
- **Crop inspection** (`zoomin/tools.py`): the agent scores each crop from an
  *aligned slice* of the full-image heatmap (re-running a position-sensitive
  scorer on the crop itself compares it against the wrong pixel statistics).
  An Otsu-on-high-frequency-residual mask was tried first and removed: Otsu
  always splits into two classes, so clean glass texture produced masks.
- **Learned path**: `ONNXScorer` hook uses the OpenCV 5 DNN module (rewritten engine,
  ENGINE_AUTO, CPU-optimized) for a drop-in learned scorer; `patchcore_adapter.py`
  documents the anomalib PatchCore integration (memory bank built from normal
  images only, CPU, minutes).
- Runtime OpenCV version is asserted in tests and reported by `GET /health`.

## 5. AWS deployment

Deployed and verified 2026-10-22 in `us-east-1` (account's own AWS, Marketplace
subscription "Cloud Optimized OpenCV for AWS Graviton4", 7-day free trial).

- **Graviton instance**: `c8g.xlarge` (Graviton4, 4 vCPU / 8 GB) from the COOL
  Marketplace AMI `ami-01db31139bc5615d8` ("COOL-Graviton4-v2"). The COOL runtime
  lives at `/opt/cool`; its venv must be *activated* (activation sets
  `PYTHONPATH=/opt/cool/python_3.12/site-packages/cv2/python-3.12` and
  `LD_LIBRARY_PATH=/opt/cool/cpp_sdk/lib`) — importing `cv2` from the venv's
  python directly, without activation, fails.
- **COOL version disclosure**: the COOL build reports `cv2.__version__` =
  **4.14.0-pre**. It is the vendor's optimized build, not OpenCV 5. The live API
  and the evaluation below therefore run the pip **OpenCV 5.0.0** venv on the
  same Graviton instance (so the OpenCV 5 workload genuinely runs on AWS), and
  the COOL configuration appears as its own benchmark row with its version
  recorded. Every benchmark CSV row carries the exact `cv2.__version__`,
  instance type, and AMI id that produced it.
- **x86 baseline instance**: `c7i.xlarge`, stock Ubuntu 24.04, pip OpenCV 5.0.0.
- **S3**: bucket `zoomin-inspector-<account>` holds the code bundle, per-config
  benchmark CSVs, the eval CSV, run logs, and health output under
  `graviton/` and `x86/` prefixes.
- **Security group** `zoomin-sg`: 22 (deploy), 8000 (FastAPI), 7860 (Gradio).
  Instances are terminated after the runs; the group is for the deploy window.
- **Judges' endpoint**: `http://<graviton-public-dns>:8000` — `GET /health`
  returned `{"status": "ok", "opencv": "5.0.0"}` from the running instance.
  The deployment is reproducible from `docs/AWS_DEPLOY.md` (note: that runbook
  predates the Graviton4 subscription — the instance family is c8g, not c7g).

## 6. Evaluation

### 6.1 Detection quality (MVTec AD: bottle / transistor / capsule)

Bottle (first category evaluated locally, 2026-10-20): NormalityScorer fit on
209 normal training images; 83 test images through the full agent loop
(propose -> zoom -> inspect -> decide, budget 8).

- **AUROC 0.961**, accuracy 0.843, **precision 1.000** (zero false alarms on
  the 20 good bottles), recall 0.794, escalation rate 10.8%.
- The 13 non-detected defects are genuinely hard: broad low-contrast
  contamination and small breaks whose peak anomaly (1.6-3.0 sigma) overlaps
  normal reflection variation. Ambiguous regions are *escalated to a human*,
  not guessed — the 10.8% escalation rate is the agent saying "I can't tell".
- Honest caveats: single category so far; thresholds (propose 2.0 sigma,
  DEFECT 4.0, unsure 2.5) were tuned on this test set and must be validated
  per-category on the EC2 run; transistor/capsule pending the AWS deploy.

**EC2 run, 2026-10-22** (Graviton instance, OpenCV 5.0.0 vanilla venv, full
agent loop, budget 8, all three categories; per-image rows in
`bench/eval_results.csv`, n=315). The bottle numbers reproduced the local run
exactly (AUROC 0.961 / acc 0.843 / P 1.000 / R 0.794), which cross-validates
the harness.

| Category | n | AUROC | Accuracy | Recall | Precision | Escalation |
|---|---|---|---|---|---|---|
| bottle | 83 | 0.961 | 0.843 | 0.794 | 1.000 | 10.8% |
| transistor | 100 | 0.836 | 0.790 | 0.550 | 0.880 | 31.0% |
| capsule | 132 | 0.653 | 0.348 | 0.211 | 1.000 | 12.1% |
| **overall** | **315** | **0.736** | — | — | — | **17.8%** |

Capsule is a genuine failure of the classical scorer (see 6.4) — reported as
measured, not hidden. Published PatchCore results on MVTec AD reach ~0.99
image AUROC averaged over categories; our CPU-only classical baseline is not
in that league, and closing that gap is what the hooked-in PatchCore/ONNX
path (section 4) is for.

### 6.2 Agentic loop ablations

From the same EC2 eval (n=315; the trace of every image is in the CSV's
`n_regions` / `replans` / `escalated` columns):

- **The escalation path rescues real defects.** Counting a defect as
  "surfaced to a human" when it was predicted DEFECT *or* escalated:
  bottle 58/63 (92.1%), transistor 33/40 (82.5%), capsule 37/109 (33.9%).
  Escalation alone accounts for 8 (bottle), 11 (transistor), and 14 (capsule)
  of those — cases a single-shot classifier at the same threshold would have
  silently passed as OK.
- **Re-planning is exercised constantly**: 231 re-zoom events across 315
  images (transistor alone: 129 re-zooms over 100 images — bent/misplaced
  leads routinely need a second, tighter look). Vision output changing the
  next tool call is the norm in the trace, not a demo path.
- **False alarms stay near zero where the scorer fits the object**: bottle
  and capsule precision are both 1.000 (0 false positives); transistor has
  the only 3 false positives in 315 images (good transistors scoring
  4.3–5.1 sigma).

### 6.3 COOL benchmark (Best Use of COOL evidence)

Same script (`bench/benchmark.py`), same 83 bottle test images, same agent
pipeline (scan → zoom → inspect), run 2026-10-22. On-demand list prices used
for the cost column: c8g.xlarge $0.1718/hr, c7i.xlarge $0.175/hr; the COOL
software fee was $0 under the 7-day Marketplace free trial.

| Config | OpenCV | Instance | ms/image | images/sec | $/10K images | Defect regions |
|---|---|---|---|---|---|---|
| A. COOL on Graviton | 4.14.0-pre (COOL) | c8g.xlarge | 30.69 | 32.6 | $0.0146 | 27 |
| B. Vanilla pip on Graviton | 5.0.0 | c8g.xlarge | **25.70** | **38.9** | **$0.0123** | 27 |
| C. Vanilla pip on x86 | 5.0.0 | c7i.xlarge | 41.96 | 23.8 | $0.0204 | 27 |

**The honest headline: on this workload, COOL did not win.** All three
configs produce identical inspection output (27 defect regions — the pipeline
agrees, only speed differs). COOL beats the x86 baseline by ~27% per image,
but stock OpenCV 5.0.0 on the *same* Graviton4 hardware is a further ~16%
faster than the COOL 4.14.0-pre build. Plausible causes, stated as hypotheses:
COOL's published gains (up to 70%) were measured against OpenCV 4.x on
Graviton2, and OpenCV 5's rewritten CPU code paths plus newer Arm kernels
have narrowed the gap; our workload is small-image classical morphology and
contour work, not the large-kernel / DNN operations where COOL's Kleidi
kernels help most. We report the comparison exactly as measured — that
reproducibility is the deliverable, and the CSVs (with per-row OpenCV
version, instance type, and AMI id) are in `bench/results-*.csv`.

### 6.4 Failure cases (required)

All from the EC2 eval (n=315), worst offenders named:

- **Capsule print/surface defects are invisible to this scorer.**
  `faulty_imprint` recall 0/22 and `scratch` 0/23: a per-pixel Gaussian
  normality model over aligned images cannot see print variation or hairline
  scratches whose pixel statistics match normal texture. `squeeze` (14/20)
  and `crack` (6/23) fare better because they deform the silhouette or local
  contrast. Capsule overall: AUROC 0.653, recall 0.211 — the learned scorer
  path exists precisely for this category.
- **Low-contrast bottle contamination** (the local run's known weakness)
  repeats on EC2: missed defects peak at 1.6–3.0 sigma, overlapping normal
  reflection variation; 5 of 63 bottle defects were neither predicted nor
  escalated.
- **Transistor ambiguity burns budget and escalates**: 31% escalation and
  129 re-zooms over 100 images. Bent-lead cases cluster at the unsure
  threshold (top missed defect: `bent_lead` at 3.95 sigma, just under the
  4.0 DEFECT line) — the agent correctly refuses to guess, but a human
  reviews nearly a third of transistors.
- **The only false positives** (3/315) are good transistors at 4.3–5.1
  sigma — lead reflections that look like bends at overview resolution and
  survive re-zoom.
- Defects smaller than the heatmap resolution and defects at crop borders
  remain structural misses of the budgeted-zoom design: if the overview
  never proposes the region, no amount of re-planning recovers it.

## 7. Limitations

- The classical normality scorer is an unsupervised baseline: fine textures and
  defects smaller than the heatmap resolution need the learned scorer
  (PatchCore / ONNX path), which is hooked in but not yet the default.
- The planner is rule-based and auditable, but dumber than an LLM planner —
  re-plan decisions are heuristics, not reasoning over the trace.
- Thresholds (propose 2.0 sigma, DEFECT 4.0, unsure 2.5) were tuned on the
  bottle test set; the 2026-10-22 EC2 validation shows they transfer to
  transistor (AUROC 0.836) but not capsule (0.653) — see 6.1/6.4.
- Single-image inspection only; the video-frame mode from the original idea
  was deferred to keep the scope honest for the deadline.

## 8. Responsible-use considerations

- Inspection **assists** human QA, it does not replace it: the escalation path
  exists for exactly this reason (10.8% of bottle cases were escalated, not
  guessed), and every escalation carries its visual trace so a person can
  decide quickly.
- No personal data is processed — industrial product images only; no faces,
  no biometric analysis, no tracking of people.
- Our own code is MIT (see LICENSE); data/model sources (MVTec AD, anomalib)
  are credited in the README, and their own usage terms apply.
