# Devpost submission — copy-paste draft (OpenCV AI Competition 2026)

> Status: DRAFT — prepared 2026-10-22 from docs/REPORT.md (§§5–6.4) and the
> re-rendered demo video. Numbers below are the landed EC2 results.

**Project name:** ZoomIn Inspector

**Tagline:** An agentic visual defect inspector that spends a fixed zoom budget the way a human inspector spends attention — scan the overview, zoom the suspicious regions, escalate when unsure.

**Links**
- Code: https://github.com/faketut/zoomin-inspector (private; pushed 2026-10-22, master @ 485dbc9)
- Live endpoint: judges' API was verified live on the Graviton4 instance during the deploy (GET /health → ok); all instances terminated after the run
- Video (THE submission video, upload this one): `video/out/zoomin-combined-demo.mp4` (4:57, under the 5-minute cap) — the narrated slide demo with the live UI recording embedded: slides 1–4, a "LIVE DEMO" title card, the real app inspecting three MVTec bottles live (defects found / escalated / clean pass + full trace, ~0:56 at 1.5×), then slides 5–6. Slides 4–5 re-recorded 2026-10-25 with the post-fix numbers (bottle recall 0.841, escalation 7.2%; transistor 0.840, capsule 0.652, imprint 2/22, scratch 1/23). Slide 6 re-recorded 2026-10-25 with the crop-budget ablation (one crop versus eight adds only 3 detections out of 212). Upload to YouTube/Vimeo and paste the link; Devpost needs a streaming URL.
- The two source videos remain in `video/out/` for reference: `zoomin-demo.mp4` (slides only, 4:00, same re-recorded segments 4–6) and `zoomin-live-demo.mp4` (live only, 1:40, natural speed). The live recording was re-recorded 2026-10-25 with the post-fix pipeline (broken_large/000 now shows 3 DEFECT + 2 ESCALATED, not 5 ESCALATED) using the updated recorder (`video/live/record_live_demo.py`, repo-root from `__file__`, clean-pass marker "0 defect") and runs the fitted bottle NormalityScorer (`ZOOMIN_MODEL=models/bottle_normality.npz`); merge reproducible via `video/live/build_combined.sh`.
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
4. **re-plan / `decide`** — re-zooms the ambiguous region, confirms it as a defect when the tighter window's coverage is high enough, and otherwise hands it to a human with the visual trace attached instead of guessing

Measured honestly: a crop-budget ablation (1 crop vs 8, everything else fixed) adds 3 detections out of 212 defects and no false positives — the overview heatmap decides almost everything, and the loop's real contribution is the audit trail and the human-review channel. 40% of defects (84/212) pass silently because the overview proposes no region at all, which no amount of re-zooming can recover.
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
- Precision 1.0 on 20 good bottles — zero false alarms — with 7.2% of bottle images and 6.3% overall escalated to a human rather than guessed. Overall: AUROC 0.738, precision 0.881, recall 0.557 across 315 test images.

## 📉 Failure cases (required, and we agree they're the interesting part)

Capsule is a real failure: **AUROC 0.652**, with `faulty_imprint` 2/22 and `scratch` 1/23 detected — a pixel-Gaussian scorer cannot see print defects. Transistor (0.840 AUROC) is where the re-plan's precision cost lands: 16 of 60 good transistors are confirmed as DEFECT, dropping transistor precision to 0.667. These are in the report, not hidden: the escalation path exists precisely because a silent wrong call costs more than an honest "I can't tell", and the coverage-confirm threshold is exposed per scorer so an operator can shift that balance back toward human review.

## 🔭 What's next

- Wire the OpenCV 5 DNN hook (ONNXScorer is already stubbed in the repo) so fine textures like capsule print get a learned scorer.
- Submit before the Oct 26 deadline (repo pushed 2026-10-22; demo video upload is the remaining link).
