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

TODO: quality engineers on manufacturing lines; small factories that cannot afford
GPU inspection rigs (the whole pipeline is CPU-only by design).

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

TODO: S3 bucket layout; EC2 c7g.xlarge from the COOL AMI running
`docker compose up`; security groups; how judges reach the endpoint.
`GET /health` returns the exact `cv2.__version__` on the instance as COOL evidence.

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

TODO: transistor + capsule on EC2; AUROC/AUPRO vs labels; compare against
published PatchCore numbers where applicable.

### 6.2 Agentic loop ablations

TODO: task success within crop budget; escalation precision/recall;
false-alarm rate with vs without the re-verify re-zoom (the ablation proving the
loop matters, not just the scorer).

### 6.3 COOL benchmark (Best Use of COOL evidence)

TODO: paste the table from `bench/results.csv` — ms/image, images/sec,
$/10K images for (A) COOL on Graviton vs (B) vanilla pip on Graviton vs
(C) vanilla pip on x86. Same script, same images, same N.

### 6.4 Failure cases (required)

TODO — be honest, judges ask for these:
- defects smaller than the heatmap resolution (missed at overview, never proposed);
- texture false positives (e.g. brushed metal) that burn crop budget;
- ambiguous cases correctly escalated (not failures — show the trace);
- re-zoom hitting the crop edge on border defects.

## 7. Limitations

TODO: classical scorer is a baseline — fine textures and tiny defects need the
learned scorer; the rule-based planner is auditable but dumber than an LLM
planner; single-image inspection (video mode is future work).

## 8. Responsible-use considerations

TODO: inspection assists, not replaces, human QA — the escalation path exists for
exactly this reason; no personal data is processed; model/dataset licenses
(MVTec AD, anomalib) are documented in the README.
