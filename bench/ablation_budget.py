"""Crop-budget ablation: what does the agentic loop actually add?

The reported detection numbers come from a fixed budget of 8. This script holds
everything else constant and varies only ``crop_budget``, so the difference
between runs is exactly the marginal contribution of spending more crops.

    python bench/ablation_budget.py --data data/mvtec \
        --train data/mvtec_train --budgets 1 2 4 8 --out bench/ablation

It also counts *silent misses*: defect images where the overview proposed no
region at all. Those are the failures no amount of re-zoom or escalation can
recover, so they bound what the loop can ever contribute.

Measured 2026-10-25 (OpenCV 5.0.0, x86, n=315) — see docs/REPORT.md §6.2.
"""

import argparse
import csv
import glob
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import cv2
import numpy as np

from bench.eval_mvtec import auroc, iter_images
from zoomin.agent import run_inspection
from zoomin.config import InspectorConfig
from zoomin.scoring import CoarseScorer, NormalityScorer


def run_budget(items, scorers, budget):
    """Inspect every image at one crop budget; return per-image rows."""
    rows = []
    for cat, defect, label, path in items:
        scorer = scorers[cat]
        config = InspectorConfig.for_scorer(scorer, crop_budget=budget)
        img = cv2.imread(path)
        rep = run_inspection(img, config=config, scorer=scorer)
        verdicts = [r.verdict for r in rep.regions]
        rows.append({
            "category": cat, "defect": defect, "label": label,
            "score": round(max([r.score for r in rep.regions], default=0.0), 4),
            "pred": 1 if "DEFECT" in verdicts else 0,
            "escalated": 1 if ("ESCALATED" in verdicts and "DEFECT" not in verdicts) else 0,
            "n_regions": len(rep.regions),
            "replans": sum(1 for r in rep.regions if r.replanned),
            "ms": round(rep.elapsed_s * 1000, 2),
            "path": path,
        })
    return rows


def summarise(rows):
    y = [r["label"] for r in rows]
    p = [r["pred"] for r in rows]
    tp = sum(1 for a, b in zip(y, p, strict=True) if a == 1 and b == 1)
    fp = sum(1 for a, b in zip(y, p, strict=True) if a == 0 and b == 1)
    fn = sum(1 for a, b in zip(y, p, strict=True) if a == 1 and b == 0)
    tn = sum(1 for a, b in zip(y, p, strict=True) if a == 0 and b == 0)
    silent = [r for r in rows if r["label"] == 1 and r["pred"] == 0 and r["escalated"] == 0]
    no_region = [r for r in rows if r["label"] == 1 and r["n_regions"] == 0]
    return {
        "auroc": auroc([r["score"] for r in rows], y),
        "acc": (tp + tn) / len(rows),
        "precision": tp / (tp + fp) if tp + fp else 0.0,
        "recall": tp / (tp + fn) if tp + fn else 0.0,
        "escalated": sum(r["escalated"] for r in rows) / len(rows),
        "detected": tp,
        "n_defects": tp + fn,
        "silent_miss": len(silent),
        "no_region_at_all": len(no_region),
        "ms_per_image": float(np.mean([r["ms"] for r in rows])),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="data/mvtec")
    ap.add_argument("--train", default=None,
                    help="train-good root; fits a NormalityScorer per category")
    ap.add_argument("--coarse", action="store_true",
                    help="use CoarseScorer everywhere (no training data)")
    ap.add_argument("--budgets", type=int, nargs="+", default=[1, 2, 4, 8])
    ap.add_argument("--out", default="bench/ablation",
                    help="output directory; writes budget-<n>.csv and summary.json")
    args = ap.parse_args()

    items = list(iter_images(args.data))
    if not items:
        raise SystemExit(f"no test images under {args.data}")

    cats = sorted({c for c, *_ in items})
    scorers = {}
    for cat in cats:
        if args.coarse:
            scorers[cat] = CoarseScorer()
            continue
        if not args.train:
            raise SystemExit("need --train <train-good root> or --coarse")
        paths = sorted(glob.glob(os.path.join(args.train, cat, "train", "good", "*.png")))
        if not paths:
            raise SystemExit(f"no train-good images for {cat} under {args.train}")
        print(f"fitting NormalityScorer on {len(paths)} normal images ({cat}) ...")
        scorers[cat] = NormalityScorer.fit(paths)

    os.makedirs(args.out, exist_ok=True)
    summary = {}
    for budget in args.budgets:
        rows = run_budget(items, scorers, budget)
        with open(os.path.join(args.out, f"budget-{budget}.csv"), "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(rows[0]))
            w.writeheader()
            w.writerows(rows)
        summary[budget] = summarise(rows)
        print(f"budget {budget}: {summary[budget]['detected']}/"
              f"{summary[budget]['n_defects']} defects detected")

    with open(os.path.join(args.out, "summary.json"), "w") as f:
        json.dump(summary, f, indent=2)

    print(f"\n{'budget':>7}{'AUROC':>8}{'acc':>7}{'prec':>7}{'rec':>7}{'esc%':>7}"
          f"{'detected':>12}{'silent':>8}{'no-region':>11}{'ms/img':>9}")
    for budget, m in summary.items():
        print(f"{budget:>7}{m['auroc']:>8.3f}{m['acc']:>7.3f}{m['precision']:>7.3f}"
              f"{m['recall']:>7.3f}{m['escalated'] * 100:>6.1f}%"
              f"{m['detected']:>7}/{m['n_defects']:<4}{m['silent_miss']:>8}"
              f"{m['no_region_at_all']:>11}{m['ms_per_image']:>9.1f}")
    print(f"\nwrote {args.out}/budget-*.csv and {args.out}/summary.json")


if __name__ == "__main__":
    main()
