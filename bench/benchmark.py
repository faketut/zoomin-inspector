"""Reproducible benchmark of the core workload (Best Use of COOL evidence).

Measures the exact pipeline the agent runs — scan_overview + zoom_crop +
inspect_crop — over N images and reports latency, throughput, and cost:

    python bench/benchmark.py --images data/mvtec/bottle/test/good \\
        --config cool-graviton --dollars-per-hour 0.145 --out bench/results.csv

Protocol (same script, same images, three runs):
  A. COOL AMI on c7g.xlarge   (--config cool-graviton)
  B. vanilla pip opencv on c7g.xlarge (--config vanilla-graviton)
  C. vanilla pip opencv on c7i.xlarge (--config vanilla-x86)
Keep the CSVs and paste the comparison table into docs/REPORT.md.
"""

import argparse
import csv
import glob
import os
import sys
import time
import urllib.request

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import cv2

from zoomin.agent import run_inspection
from zoomin.config import InspectorConfig
from zoomin.scoring import CoarseScorer, NormalityScorer


def _instance_meta() -> dict:
    """Best-effort EC2 metadata (IMDSv2) for reproducible benchmark provenance."""
    meta = {"instance_type": "unknown", "ami_id": "unknown"}
    try:
        token_req = urllib.request.Request(
            "http://169.254.169.254/latest/api/token",
            headers={"X-aws-ec2-metadata-token-ttl-seconds": "60"},
            method="PUT",
        )
        token = urllib.request.urlopen(token_req, timeout=2).read().decode()
        for key in ("instance-type", "ami-id"):
            req = urllib.request.Request(
                f"http://169.254.169.254/latest/meta-data/{key}",
                headers={"X-aws-ec2-metadata-token": token},
            )
            meta[key.replace("-", "_")] = urllib.request.urlopen(req, timeout=2).read().decode()
    except Exception:
        pass  # not on EC2 (local dev) — provenance stays "unknown"
    return meta


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--images", required=True, help="glob of test images")
    ap.add_argument("--config", required=True, help="run label, e.g. cool-graviton")
    ap.add_argument("--dollars-per-hour", type=float, default=0.0)
    ap.add_argument("--budget", type=int, default=8)
    ap.add_argument("--out", default="bench/results.csv")
    ap.add_argument("--model", default=None,
                    help="NormalityScorer .npz model; without it uses CoarseScorer "
                         "with its own calibration")
    args = ap.parse_args()

    if os.path.isdir(args.images):
        # MVTec-style layout: <category>/test/<good|defect_type>/*.png — recurse.
        paths = sorted(glob.glob(os.path.join(args.images, "**", "*"), recursive=True))
    else:
        paths = sorted(glob.glob(args.images))
    paths = [p for p in paths if p.lower().endswith((".png", ".jpg", ".jpeg", ".bmp"))]
    assert paths, f"no images found under {args.images}"

    # Thresholds always come from the scorer actually in use.
    scorer = NormalityScorer.load(args.model) if args.model else CoarseScorer()
    config = InspectorConfig.for_scorer(scorer, crop_budget=args.budget)
    # Warmup (page caches, one-time inits) — not counted.
    warm = cv2.imread(paths[0])
    run_inspection(warm, config=config, scorer=scorer)

    t0 = time.perf_counter()
    n_defect = 0
    for p in paths:
        img = cv2.imread(p)
        rep = run_inspection(img, config=config, scorer=scorer)
        n_defect += sum(1 for r in rep.regions if r.verdict == "DEFECT")
    elapsed = time.perf_counter() - t0

    n = len(paths)
    ms_per_image = elapsed / n * 1000
    images_per_sec = n / elapsed
    cost_per_10k = args.dollars_per_hour * (elapsed / n * 10_000) / 3600

    meta = _instance_meta()
    row = {
        "config": args.config,
        "opencv": cv2.__version__,
        "instance_type": meta["instance_type"],
        "ami_id": meta["ami_id"],
        "n_images": n,
        "ms_per_image": round(ms_per_image, 2),
        "images_per_sec": round(images_per_sec, 2),
        "defect_regions": n_defect,
        "dollars_per_10k_images": round(cost_per_10k, 4),
    }
    print(row)

    new = not os.path.exists(args.out)
    with open(args.out, "a", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(row))
        if new:
            w.writeheader()
        w.writerow(row)
    print(f"appended to {args.out}")


if __name__ == "__main__":
    main()
