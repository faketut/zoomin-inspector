"""Fit a NormalityScorer on a category's NORMAL training images and save it.

Usage:
    python scripts/fit_model.py --train data/mvtec_train/bottle/train/good \
        --out models/bottle_normality.npz

The saved .npz (~0.5 MB) is what app/ui.py and app/api.py load when the
ZOOMIN_MODEL environment variable points at it; without it they fall back
to the single-image CoarseScorer (weaker on real data — see docs/REPORT.md).
"""

import argparse
import glob
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from zoomin.scoring import NormalityScorer


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--train", required=True, help=" dir of normal training images")
    ap.add_argument("--out", required=True, help="output .npz path")
    args = ap.parse_args()
    paths = sorted(glob.glob(os.path.join(args.train, "**", "*.png"), recursive=True))
    assert paths, f"no training images under {args.train}"
    scorer = NormalityScorer.fit(paths)
    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    scorer.save(args.out)
    print(f"fit on {len(paths)} images -> {args.out}")


if __name__ == "__main__":
    main()
