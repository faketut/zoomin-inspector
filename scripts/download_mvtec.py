"""Download MVTec AD categories via the Hugging Face mirror.

    python scripts/download_mvtec.py --categories bottle transistor capsule --out data/mvtec

If the official site (https://www.mvtec.com/company/research/datasets/mvtec-ad)
serves you better, download there instead — same layout is expected:
  data/mvtec/<category>/test/<good|defect_type>/img.png
"""

import argparse
import os

CATEGORIES = ["bottle", "cable", "capsule", "carpet", "grid", "hazelnut",
              "leather", "metal_nut", "pill", "screw", "tile", "toothbrush",
              "transistor", "wood", "zipper"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--categories", nargs="+", default=["bottle", "transistor", "capsule"])
    ap.add_argument("--out", default="data/mvtec")
    args = ap.parse_args()
    for c in args.categories:
        assert c in CATEGORIES, f"unknown category {c}"

    from huggingface_hub import snapshot_download

    patterns = [f"{c}/**" for c in args.categories]
    snapshot_download(repo_id="Voxel51/mvtec-ad", repo_type="dataset",
                      local_dir=args.out, allow_patterns=patterns)
    print(f"downloaded {args.categories} -> {os.path.abspath(args.out)}")


if __name__ == "__main__":
    main()
