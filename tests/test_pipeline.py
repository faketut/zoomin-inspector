"""Smoke tests on synthetic images: a scratched part must be flagged DEFECT,
a clean part must not be. Also asserts the OpenCV 5 runtime and a non-empty
agent trace (Agentic Vision evidence).

Run:  python tests/test_pipeline.py
"""

import sys

import cv2
import numpy as np

sys.path.insert(0, ".")

from zoomin.agent import run_inspection
from zoomin.config import InspectorConfig
from zoomin.scoring import CoarseScorer


def make_part(defect: bool, size: int = 512, seed: int = 0) -> np.ndarray:
    rng = np.random.default_rng(seed)
    # Light, slightly textured "metal" background. Kept geometrically clean on
    # purpose: legitimate part edges are a documented failure mode of the
    # classical baseline (see docs/REPORT.md) and belong in the eval set,
    # not in the smoke test.
    img = np.full((size, size, 3), 205, np.uint8)
    noise = rng.integers(-6, 7, (size, size, 1)).astype(np.int16)
    img = np.clip(img.astype(np.int16) + noise, 0, 255).astype(np.uint8)
    if defect:
        # Dark scratch across the part.
        cv2.line(img, (size // 4, size // 2 - 40), (3 * size // 4, size // 2 + 40),
                 (40, 40, 40), 7)
    return img


def main():
    assert cv2.__version__.split(".")[0] == "5", f"need OpenCV 5, got {cv2.__version__}"
    print(f"opencv {cv2.__version__} OK")

    config = InspectorConfig(crop_budget=8)
    scorer = CoarseScorer()

    bad = make_part(defect=True)
    rep = run_inspection(bad, config=config, scorer=scorer)
    verdicts = [r.verdict for r in rep.regions]
    print("defective:", rep.summary())
    assert "DEFECT" in verdicts, f"scratch missed! verdicts={verdicts}"

    good = make_part(defect=False, seed=1)
    rep2 = run_inspection(good, config=config, scorer=scorer)
    verdicts2 = [r.verdict for r in rep2.regions]
    print("clean:    ", rep2.summary())
    assert "DEFECT" not in verdicts2, f"false alarm on clean part! {verdicts2}"

    tools = [s.tool for s in rep.trace]
    for required in ("scan_overview", "zoom_crop", "inspect_crop", "finalize_report"):
        assert required in tools, f"trace missing {required}"
    print(f"trace OK ({len(rep.trace)} steps): {tools}")
    print("ALL SMOKE TESTS PASSED")


if __name__ == "__main__":
    main()
