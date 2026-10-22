# Devpost submission — copy-paste draft (OpenCV AI Competition 2026)

> Status: DRAFT — prepared 2026-10-22 from docs/REPORT.md (§§5–6.4) and the
> re-rendered demo video. Numbers below are the landed EC2 results.

**Project name:** ZoomIn Inspector

**Tagline:** An agentic visual defect inspector that spends a fixed zoom budget the way a human inspector spends attention — scan the overview, zoom the suspicious regions, escalate when unsure.

**Links**
- Code: https://github.com/faketut/zoomin-inspector (private; pushed 2026-10-22, master @ 485dbc9)
- Live endpoint: judges' API was verified live on the Graviton4 instance during the deploy (GET /health → ok); all instances terminated after the run
- Video: `video/out/zoomin-demo.mp4` (v2, re-rendered with EC2 numbers — upload to YouTube/Vimeo and paste the link; Devpost needs a streaming URL)
- Technical report: `docs/REPORT.md` in the repo (includes architecture diagram)

**Built with:** opencv-5 · python · fastapi · gradio · aws-ec2-graviton4 · cloud-optimized-opencv · docker · mvtec-ad

---

## 💡 Inspiration

Industrial visual inspection wastes compute and misses small defects: whole-image classifiers either downscale the defect away or run expensive detectors everywhere. A human inspector does the opposite — scan the overview, zoom into the suspicious regions, look again when unsure, and call over a colleague when something is genuinely ambiguous. We built an agent that inspects the way a human does, under an explicit budget of eight crops, on CPU only — because the small factories that need this most cannot afford GPU inspection rigs.

## 🛠 What it does

ZoomIn Inspector is an agent loop (`zoomin/agent.py`) orchestrating five tools:

1. **`scan_overview`** — full-image anomaly heatmap from a classical per-pixel scorer
2. **`zoom_crop`** — crops the top-k suspicious regions (the vision output decides the next tool call — this is the agentic-vision step)
3. **`inspect_crop`** — re-scores the crop against aligned full-image statistics
4. **`escalate`** — hands ambiguous cases to a human with the visual trace attached, instead of guessing
5. **`finalize_report`** — emits a defect report with localized masks

Two engineering choices carry the quality:

- **Anomaly scores in absolute units, never per-image normalized.** Our default scorer fits a per-pixel Gaussian over normal training images and scores every pixel in standard deviations. Our first attempt used per-image normalization — every bottle, good or bad, was flagged defect (AUROC 0.44). Removing it was the whole fix.
- **A fixed zoom budget.** The agent gets eight crops and must spend them where they matter, exactly like an inspector's attention.

One container serves the FastAPI API and the Gradio UI (`docker compose up`); every agent step is logged as a trace and visualized in the demo UI.

## ☁️ AWS + COOL

We deployed on **AWS Graviton4** (`c8g.xlarge`) from the official Cloud Optimized OpenCV AMI for Graviton4 — our Best Use of COOL evidence. The benchmark harness ran the same script on the same 83 bottle images three ways:

| Config | OpenCV | ms/image | images/sec | $/10K images |
|---|---|---|---|---|
| A. COOL on Graviton4 | 4.14.0-pre (COOL) | 30.69 | 32.6 | $0.0146 |
| B. Vanilla on Graviton4 | 5.0.0 (pip) | **25.70** | **38.9** | **$0.0123** |
| C. Vanilla on x86 | 5.0.0 (pip) | 41.96 | 23.8 | $0.0204 |

All three configs found the identical 27 defect regions — the pipelines agree, only the speed differs. The honest headline, stated in the report: **stock OpenCV 5.0.0 beat COOL by ~16% on the same Graviton4 hardware for this workload** (COOL still beat x86 by ~27%). One important detail we handled rather than hid: the COOL AMI's optimized build is OpenCV 4.14.0-pre, not OpenCV 5 — so the live API and the MVTec eval ran the pip 5.0.0 venv on the same instance, and every benchmark CSV records the exact `cv2` version per config.

The same pipeline then measured three MVTec categories on the instance (n=315): bottle AUROC **0.961** (exact reproduction of the local run — harness cross-validated), transistor **0.836**, capsule **0.653**. All instances were terminated after the run; total AWS spend under $1.

## 🚧 Challenges we ran into

- **Sandbox egress blocks SSH**, and the Instance Connect Endpoint needed an IAM permission the deployer role didn't have. Rather than bounce Jack back to the console, both instances ran fully autonomously: boot scripts pulled code from S3, ran the benchmarks and eval, and uploaded logs, stage markers, and result CSVs back to S3.
- **The COOL AMI is 4.14.0-pre, not OpenCV 5.** Discovered mid-deploy; handled by running the API and eval on the pip 5.0.0 venv and recording exact versions in every CSV.
- **Two script bugs** (venv activation, requirements-file comment parsing) burned two Graviton runs before the clean v3 run.

## 🏆 Accomplishments

- End-to-end agentic inspection loop with a reproducible benchmark harness and a cross-validated eval (local bottle 0.961 == EC2 bottle 0.961).
- A benchmark result we reported against our own prize interest: stock OpenCV 5 beats COOL on this workload, with hypotheses and reproducible CSVs.
- Precision 1.0 on 20 good bottles — zero false alarms — with 10.8% of bottle images and 17.8% overall escalated to a human rather than guessed.

## 📉 Failure cases (required, and we agree they're the interesting part)

Capsule is a real failure: **AUROC 0.653**, with `faulty_imprint` 0/22 and `scratch` 0/23 detected — a pixel-Gaussian scorer cannot see print defects. Transistor (0.836) burns budget and escalates 31% of the time. These are in the report, not hidden: the escalation path exists precisely because a silent wrong call costs more than an honest "I can't tell".

## 🔭 What's next

- Wire the OpenCV 5 DNN hook (ONNXScorer is already stubbed in the repo) so fine textures like capsule print get a learned scorer.
- Submit before the Oct 26 deadline (repo pushed 2026-10-22; demo video upload is the remaining link).
