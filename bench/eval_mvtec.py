"""MVTec AD detection eval for ZoomIn Inspector (local, CPU).

Runs the full agent loop on every test image under data/mvtec/<cat>/test/
and reports image-level detection metrics per category and overall.

Image-level rule (honest, fixed before seeing data):
  score  = max region inspect score (0.0 when no region proposed)
  pred   = DEFECT if any region verdict is DEFECT else OK
  (ESCALATED counts as "flagged for human" — reported as escalation rate,
   not as a defect prediction.)

Usage:
  .venv/bin/python bench/eval_mvtec.py --data data/mvtec --out bench/eval_results.csv
"""

import argparse
import csv
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import cv2
import numpy as np

from zoomin.agent import run_inspection
from zoomin.config import InspectorConfig
from zoomin.scoring import CoarseScorer


def auroc(scores, labels):
    """P(score_pos > score_neg) with ties at half; None when single-class."""
    scores = np.asarray(scores, dtype=float)
    labels = np.asarray(labels, dtype=int)
    pos = scores[labels == 1]
    neg = scores[labels == 0]
    if len(pos) == 0 or len(neg) == 0:
        return None
    wins = np.sum(pos[:, None] > neg[None, :]) + 0.5 * np.sum(pos[:, None] == neg[None, :])
    return float(wins / (len(pos) * len(neg)))


def iter_images(root):
    for cat in sorted(os.listdir(root)):
        test_dir = os.path.join(root, cat, "test")
        if not os.path.isdir(test_dir):
            continue
        for defect in sorted(os.listdir(test_dir)):
            ddir = os.path.join(test_dir, defect)
            if not os.path.isdir(ddir):
                continue
            label = 0 if defect == "good" else 1
            for fn in sorted(os.listdir(ddir)):
                if fn.lower().endswith(".png"):
                    yield cat, defect, label, os.path.join(ddir, fn)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="data/mvtec")
    ap.add_argument("--out", default="bench/eval_results.csv")
    ap.add_argument("--budget", type=int, default=8)
    args = ap.parse_args()

    config = InspectorConfig(crop_budget=args.budget)
    scorer = CoarseScorer()
    rows = []
    per_cat = {}

    items = list(iter_images(args.data))
    print(f"{len(items)} test images")
    t_all = time.perf_counter()
    for i, (cat, defect, label, path) in enumerate(items):
        img = cv2.imread(path)
        t0 = time.perf_counter()
        rep = run_inspection(img, config=config, scorer=scorer)
        dt = (time.perf_counter() - t0) * 1000
        verdicts = [r.verdict for r in rep.regions]
        score = max([r.score for r in rep.regions], default=0.0)
        pred = 1 if "DEFECT" in verdicts else 0
        esc = 1 if ("ESCALATED" in verdicts and "DEFECT" not in verdicts) else 0
        replans = sum(1 for r in rep.regions if r.replanned)
        rows.append({"category": cat, "defect": defect, "label": label,
                     "score": round(score, 4), "pred": pred, "escalated": esc,
                     "n_regions": len(rep.regions), "replans": replans,
                     "ms": round(dt, 1), "path": path})
        if (i + 1) % 50 == 0:
            print(f"  {i + 1}/{len(items)} ...")
    print(f"total {time.perf_counter() - t_all:.1f}s")

    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    with open(args.out, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)

    def metrics(rs):
        y = [r["label"] for r in rs]
        s = [r["score"] for r in rs]
        p = [r["pred"] for r in rs]
        tp = sum(1 for a, b in zip(y, p) if a == 1 and b == 1)
        fp = sum(1 for a, b in zip(y, p) if a == 0 and b == 1)
        fn = sum(1 for a, b in zip(y, p) if a == 1 and b == 0)
        return {
            "n": len(rs),
            "auroc": auroc(s, y),
            "acc": (tp + sum(1 for a, b in zip(y, p) if a == 0 and b == 0)) / len(rs),
            "prec": tp / (tp + fp) if tp + fp else 0.0,
            "rec": tp / (tp + fn) if tp + fn else 0.0,
            "esc_rate": sum(r["escalated"] for r in rs) / len(rs),
            "ms/img": float(np.mean([r["ms"] for r in rs])),
        }

    cats = sorted({r["category"] for r in rows})
    print(f"\n{'category':<12}{'n':>5}{'AUROC':>8}{'acc':>7}{'prec':>7}{'rec':>7}{'esc%':>7}{'ms/img':>8}")
    for cat in cats:
        m = metrics([r for r in rows if r["category"] == cat])
        a = f"{m['auroc']:.3f}" if m["auroc"] is not None else "n/a"
        print(f"{cat:<12}{m['n']:>5}{a:>8}{m['acc']:>7.3f}{m['prec']:>7.3f}"
              f"{m['rec']:>7.3f}{m['esc_rate'] * 100:>6.1f}%{m['ms/img']:>8.1f}")
    m = metrics(rows)
    a = f"{m['auroc']:.3f}" if m["auroc"] is not None else "n/a"
    print(f"{'OVERALL':<12}{m['n']:>5}{a:>8}{m['acc']:>7.3f}{m['prec']:>7.3f}"
          f"{m['rec']:>7.3f}{m['esc_rate'] * 100:>6.1f}%{m['ms/img']:>8.1f}")
    print(f"\nwrote {args.out}")


if __name__ == "__main__":
    main()
