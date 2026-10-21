"""The inspector agent: an evidence-driven perception → decision → action loop.

Protocol (ReAct-style, rule-based planner — deterministic and auditable):
  1. scan_overview  → heatmap + proposed boxes (vision)
  2. for each box within the crop budget:
       zoom_crop    → crop the suspicious region (action driven by vision output)
       inspect_crop → defect score (vision)
       decide:  score >= defect_threshold  -> DEFECT
                score >= unsure_threshold   -> RE-PLAN: tighter re-zoom, re-inspect;
                                             still unsure -> ESCALATE to human
                else                        -> OK
  3. finalize_report → verdicts + full tool trace + annotated image

Every step is appended to ``trace`` with inputs/outputs/rationale. The trace is
the Agentic Vision deliverable: it shows vision results changing later tool
calls (heatmap → crop coordinates; inspection score → re-zoom vs escalate).
"""

import argparse
import json
import os
import time
from dataclasses import dataclass, field

import cv2
import numpy as np

from . import report as report_mod
from .config import InspectorConfig
from .scoring import CoarseScorer, NormalityScorer
from .tools import inspect_crop, scan_overview, strongest_sub_box, zoom_crop, Box


@dataclass
class TraceStep:
    step: int
    tool: str
    inputs: dict
    outputs: dict
    rationale: str

    def to_dict(self):
        return {"step": self.step, "tool": self.tool, "inputs": self.inputs,
                "outputs": self.outputs, "rationale": self.rationale}


@dataclass
class RegionVerdict:
    region_id: int
    box: Box            # padded box in full-image coordinates
    verdict: str        # OK | DEFECT | ESCALATED
    score: float
    replanned: bool = False


@dataclass
class InspectionReport:
    regions: list[RegionVerdict]
    trace: list[TraceStep]
    elapsed_s: float
    config: InspectorConfig
    annotated: np.ndarray = None

    def summary(self) -> dict:
        counts = {"OK": 0, "DEFECT": 0, "ESCALATED": 0}
        for r in self.regions:
            counts[r.verdict] += 1
        return {
            "n_regions": len(self.regions),
            "verdicts": counts,
            "n_trace_steps": len(self.trace),
            "elapsed_s": round(self.elapsed_s, 3),
            "opencv": cv2.__version__,
        }

    def to_dict(self) -> dict:
        return {
            "summary": self.summary(),
            "regions": [
                {"region_id": r.region_id,
                 "box": {"x": r.box.x, "y": r.box.y, "w": r.box.w, "h": r.box.h},
                 "verdict": r.verdict, "score": round(r.score, 4),
                 "replanned": r.replanned}
                for r in self.regions
            ],
            "trace": [s.to_dict() for s in self.trace],
        }


def run_inspection(image: np.ndarray, config: InspectorConfig = None,
                   scorer=None, patchcore=None) -> InspectionReport:
    config = config or InspectorConfig()
    t0 = time.perf_counter()
    trace: list[TraceStep] = []
    step = 0

    def log(tool, inputs, outputs, rationale):
        nonlocal step
        trace.append(TraceStep(step, tool, inputs, outputs, rationale))
        step += 1

    # 1 — perceive: coarse scan of the whole image
    heat, boxes = scan_overview(image, scorer=scorer, config=config)
    log("scan_overview",
        {"heat_threshold": config.heat_threshold, "min_region_area": config.min_region_area},
        {"n_proposed": len(boxes),
         "top_boxes": [{"x": b.x, "y": b.y, "w": b.w, "h": b.h, "heat": round(b.heat, 3)}
                       for b in boxes[: config.crop_budget]]},
        f"Heatmap proposed {len(boxes)} regions; spending crop budget "
        f"({config.crop_budget}) on the hottest first.")

    # 2 — act on the evidence, region by region
    regions: list[RegionVerdict] = []
    for i, box in enumerate(boxes[: config.crop_budget]):
        crop, padded = zoom_crop(image, box, padding=config.zoom_padding)
        log("zoom_crop",
            {"region_id": i, "box": [box.x, box.y, box.w, box.h]},
            {"padded_box": [padded.x, padded.y, padded.w, padded.h],
             "crop_shape": list(crop.shape)},
            f"Region {i} heat={box.heat:.2f} — zooming in for close inspection.")

        # Aligned heat slice: the full-image heatmap restricted to this crop's
        # footprint (same pixel extents as `crop`). Never re-run a
        # position-sensitive scorer on the crop itself.
        heat_crop = heat[padded.y : padded.y + padded.h,
                         padded.x : padded.x + padded.w]
        result = inspect_crop(crop, config=config, patchcore=patchcore,
                              scorer=scorer, heat=heat_crop)
        log("inspect_crop",
            {"region_id": i},
            {"score": round(result.score, 4), "n_contours": result.n_contours,
             "detail": {k: round(v, 4) if isinstance(v, float) else v
                        for k, v in result.detail.items()}},
            f"Region {i} scored {result.score:.2f}.")

        replanned = False
        if result.score >= config.defect_threshold:
            verdict = "DEFECT"
            rationale = f"Score {result.score:.2f} >= {config.defect_threshold}."
        elif result.score >= config.unsure_threshold:
            # Re-plan: don't guess — look closer at the strongest sub-region.
            replanned = True
            sub = strongest_sub_box(result, crop.shape)
            if sub is not None:
                sw, sh = int(crop.shape[1] * config.rezoom_shrink), int(crop.shape[0] * config.rezoom_shrink)
                cx, cy = sub.x + sub.w // 2, sub.y + sub.h // 2
                tight = Box(max(0, cx - sw // 2), max(0, cy - sh // 2), sw, sh, sub.heat)
                crop2, padded2 = zoom_crop(crop, tight, padding=0.1)
                log("zoom_crop",
                    {"region_id": i, "replan": True,
                     "box": [tight.x, tight.y, tight.w, tight.h]},
                    {"crop_shape": list(crop2.shape)},
                    f"Region {i} ambiguous ({result.score:.2f}) — re-zooming tighter "
                    f"instead of guessing.")
                heat2 = heat_crop[padded2.y : padded2.y + padded2.h,
                                  padded2.x : padded2.x + padded2.w]
                result2 = inspect_crop(crop2, config=config, patchcore=patchcore,
                                       scorer=scorer, heat=heat2)
                log("inspect_crop",
                    {"region_id": i, "replan": True},
                    {"score": round(result2.score, 4), "n_contours": result2.n_contours},
                    f"Re-inspection scored {result2.score:.2f}.")
                if result2.score >= config.defect_threshold:
                    verdict, result = "DEFECT", result2
                    rationale = "Tighter re-zoom confirmed the defect."
                else:
                    verdict = "ESCALATED"
                    rationale = (f"Still ambiguous after re-zoom "
                                 f"({result2.score:.2f}) — escalating to a human.")
            else:
                verdict = "ESCALATED"
                rationale = "Ambiguous but no localizable sub-region — escalating."
        else:
            verdict = "OK"
            rationale = f"Score {result.score:.2f} < {config.unsure_threshold}."

        regions.append(RegionVerdict(i, padded, verdict, result.score, replanned))
        log("decide",
            {"region_id": i},
            {"verdict": verdict, "replanned": replanned},
            rationale)

    # 3 — finalize
    annotated = report_mod.annotate(image, regions)
    elapsed = time.perf_counter() - t0
    rep = InspectionReport(regions, trace, elapsed, config, annotated)
    log("finalize_report", {}, rep.summary(),
        "Assembled verdicts, trace, and annotated image.")
    return rep


def main():
    ap = argparse.ArgumentParser(description="ZoomIn Inspector — run the agent on one image.")
    ap.add_argument("image", help="input image path")
    ap.add_argument("--out", default="out", help="output directory")
    ap.add_argument("--budget", type=int, default=8, help="crop budget")
    ap.add_argument("--model", default=None,
                    help="NormalityScorer .npz model (fit on normal images). "
                         "Without it, falls back to CoarseScorer with its own thresholds.")
    args = ap.parse_args()

    image = cv2.imread(args.image)
    if image is None:
        raise SystemExit(f"cannot read image: {args.image}")
    if args.model:
        config = InspectorConfig(crop_budget=args.budget)  # z-unit defaults
        scorer = NormalityScorer.load(args.model)
    else:
        config = InspectorConfig(crop_budget=args.budget, heat_threshold=0.15,
                                 defect_threshold=0.55, unsure_threshold=0.20)
        scorer = CoarseScorer()
    rep = run_inspection(image, config=config, scorer=scorer)

    os.makedirs(args.out, exist_ok=True)
    with open(os.path.join(args.out, "report.json"), "w") as f:
        json.dump(rep.to_dict(), f, indent=2)
    cv2.imwrite(os.path.join(args.out, "annotated.png"), rep.annotated)
    print(json.dumps(rep.summary(), indent=2))
    print(f"wrote {args.out}/report.json and {args.out}/annotated.png")


if __name__ == "__main__":
    main()
