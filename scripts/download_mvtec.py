"""Download MVTec AD categories via the Hugging Face mirror (Voxel51/mvtec-ad).

The mirror stores images as data/data_N/*.png with a samples.json index
(filepath -> category / defect label / split), NOT the original MVTec folder
layout. This script reads the index and materializes the classic layout:

    python scripts/download_mvtec.py --categories bottle transistor capsule --out data/mvtec

produces:
    data/mvtec/<category>/test/<good|defect_type>/img.png

If the official site (https://www.mvtec.com/company/research/datasets/mvtec-ad)
serves you better, download there instead — same layout is expected.
"""

import argparse
import json
import os
import shutil
from collections import Counter
from pathlib import Path

CATEGORIES = ["bottle", "cable", "capsule", "carpet", "grid", "hazelnut",
              "leather", "metal_nut", "pill", "screw", "tile", "toothbrush",
              "transistor", "wood", "zipper"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--categories", nargs="+", default=["bottle", "transistor", "capsule"])
    ap.add_argument("--out", default="data/mvtec")
    ap.add_argument("--split", default="test", choices=["train", "test"])
    args = ap.parse_args()
    for c in args.categories:
        assert c in CATEGORIES, f"unknown category {c}"

    from huggingface_hub import hf_hub_download

    idx_path = hf_hub_download("Voxel51/mvtec-ad", filename="samples.json",
                               repo_type="dataset")
    samples = json.loads(Path(idx_path).read_text())["samples"]
    counts = Counter()
    for s in samples:
        cat = s["category"]["label"]
        if cat not in args.categories or s.get("split") != args.split:
            continue
        defect = s["defect"]["label"] if isinstance(s.get("defect"), dict) else "good"
        src = hf_hub_download("Voxel51/mvtec-ad", filename=s["filepath"],
                              repo_type="dataset")
        dst_dir = os.path.join(args.out, cat, args.split, defect)
        os.makedirs(dst_dir, exist_ok=True)
        dst = os.path.join(dst_dir, os.path.basename(s["filepath"]))
        if not os.path.exists(dst):
            shutil.copyfile(src, dst)
        counts[(cat, defect)] += 1

    total = sum(counts.values())
    print(f"downloaded {total} {args.split} images -> {os.path.abspath(args.out)}")
    for (cat, defect), n in sorted(counts.items()):
        print(f"  {cat}/{defect}: {n}")


if __name__ == "__main__":
    main()
