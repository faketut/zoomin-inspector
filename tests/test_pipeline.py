"""Smoke + regression tests for the inspector pipeline.

Run:  python -m pytest tests/          (or: python tests/test_pipeline.py)

Each test below defends a contract that a plausible bug would break. The
regression tests at the bottom pin bugs found in the 2026-10-25 audit.
"""

import os
import sys
from pathlib import Path

import cv2
import numpy as np
import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from zoomin.agent import run_inspection
from zoomin.config import InspectorConfig
from zoomin.scoring import CoarseScorer, NormalityScorer, load_default_scorer
from zoomin.tools import Box, inspect_crop, scan_overview, zoom_crop


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


def test_opencv_5_runtime():
    assert cv2.__version__.split(".")[0] == "5", f"need OpenCV 5, got {cv2.__version__}"


@pytest.mark.parametrize("scorer_cls", [CoarseScorer])
def test_defect_flagged_and_clean_passes(scorer_cls):
    scorer = scorer_cls()
    config = InspectorConfig.for_scorer(scorer)

    rep = run_inspection(make_part(defect=True), config=config, scorer=scorer)
    verdicts = [r.verdict for r in rep.regions]
    assert "DEFECT" in verdicts, f"scratch missed! verdicts={verdicts}"

    rep2 = run_inspection(make_part(defect=False, seed=1), config=config, scorer=scorer)
    verdicts2 = [r.verdict for r in rep2.regions]
    assert "DEFECT" not in verdicts2, f"false alarm on clean part! {verdicts2}"


def test_trace_records_the_agentic_loop():
    scorer = CoarseScorer()
    rep = run_inspection(make_part(defect=True),
                         config=InspectorConfig.for_scorer(scorer), scorer=scorer)
    tools = [s.tool for s in rep.trace]
    for required in ("scan_overview", "zoom_crop", "inspect_crop", "decide",
                     "finalize_report"):
        assert required in tools, f"trace missing {required}: {tools}"
    assert tools[-1] == "finalize_report"


# --- regressions ---------------------------------------------------------

def test_default_deployment_actually_detects():
    """Regression: the deployed default paired CoarseScorer with z-unit
    thresholds (heat_threshold=2.0), so scan_overview proposed nothing at all
    and every image — defective or clean — returned an empty report."""
    scorer = load_default_scorer()  # no ZOOMIN_MODEL in the test env
    assert isinstance(scorer, CoarseScorer)
    config = InspectorConfig.for_scorer(scorer)
    assert config.heat_threshold < 1.0, "z-unit threshold paired with a 0..1 scorer"

    rep = run_inspection(make_part(defect=True), config=config, scorer=scorer)
    assert rep.regions, "default configuration proposed no regions for a scratched part"
    assert "DEFECT" in [r.verdict for r in rep.regions]


def test_config_rejects_uncalibrated_scorer():
    """A scorer with unknown heat units must not get guessed thresholds."""
    class Opaque:
        calibration = None

    with pytest.raises(TypeError, match="calibration"):
        InspectorConfig.for_scorer(Opaque())


def test_config_rejects_unknown_override():
    with pytest.raises(TypeError, match="unknown config field"):
        InspectorConfig.for_scorer(CoarseScorer(), heat_treshold=0.2)  # typo


def test_replan_reads_coverage_not_peak():
    """Regression: re-zoom scored max(heat) over a window nested inside the
    first crop, so the score could never rise and 're-zoom confirmed the
    defect' was unreachable — every re-plan was a guaranteed escalation."""
    scorer = CoarseScorer()
    config = InspectorConfig.for_scorer(scorer)
    img = make_part(defect=True)
    heat, boxes = scan_overview(img, scorer=scorer, config=config)
    assert boxes

    crop, padded = zoom_crop(img, boxes[0], padding=config.zoom_padding)
    heat_crop = heat[padded.y:padded.y + padded.h, padded.x:padded.x + padded.w]
    first = inspect_crop(crop, config=config, heat=heat_crop)

    # A tighter window on the same defect covers strictly more of the frame,
    # so coverage moves even though the peak heat cannot.
    sw = int(crop.shape[1] * config.rezoom_shrink)
    sh = int(crop.shape[0] * config.rezoom_shrink)
    tight = Box(crop.shape[1] // 2 - sw // 2, crop.shape[0] // 2 - sh // 2, sw, sh, 0.0)
    crop2, padded2 = zoom_crop(crop, tight, padding=0.1)
    heat2 = heat_crop[padded2.y:padded2.y + padded2.h, padded2.x:padded2.x + padded2.w]
    second = inspect_crop(crop2, config=config, heat=heat2)

    assert second.score <= first.score + 1e-9, "nested window cannot raise peak heat"
    assert second.coverage > first.coverage, "coverage must reflect the tighter framing"


def test_replan_confirmation_branch_is_reachable():
    """The confirmation branch must be reachable end to end: a re-planned
    region can end as DEFECT, not only as ESCALATED."""
    scorer = CoarseScorer()
    config = InspectorConfig.for_scorer(scorer)
    outcomes = set()
    for width in range(1, 7):
        for dark in (40, 60, 90, 120, 150, 180, 200):
            for seed in range(4):
                # One faint scratch: ambiguous by design, so the region lands in
                # the unsure band and the agent actually re-plans it.
                img = make_part(defect=False, seed=seed)
                cv2.line(img, (128, 216), (384, 296), (dark, dark, dark), width)
                rep = run_inspection(img, config=config, scorer=scorer)
                outcomes |= {(r.verdict, r.replanned) for r in rep.regions}
    assert ("DEFECT", True) in outcomes, f"re-zoom never confirmed a defect: {outcomes}"
    assert ("ESCALATED", True) in outcomes, f"ambiguity never escalated: {outcomes}"


def test_reported_score_matches_trace():
    """Regression: a replanned region reported the FIRST pass's score while its
    trace ended with the re-zoom score — 79/81 escalated regions disagreed."""
    scorer = CoarseScorer()
    config = InspectorConfig.for_scorer(scorer)
    checked = 0
    for width in range(1, 7):
        for dark in (40, 60, 90, 120, 150, 180, 200):
            for seed in range(4):
                img = make_part(defect=False, seed=seed)
                cv2.line(img, (128, 216), (384, 296), (dark, dark, dark), width)
                rep = run_inspection(img, config=config, scorer=scorer)
                for r in rep.regions:
                    if not r.replanned:
                        continue
                    scores = [s.outputs["score"] for s in rep.trace
                              if s.tool == "inspect_crop"
                              and s.inputs.get("region_id") == r.region_id]
                    assert scores, "replanned region has no inspect_crop in trace"
                    # The trace rounds to 4 dp; the report keeps the raw float.
                    assert r.score == pytest.approx(scores[-1], abs=5e-5), \
                        f"verdict score {r.score} != last evidence {scores[-1]}"
                    checked += 1
    assert checked > 0, "no replanned region exercised this test — it proved nothing"


def test_heat_slice_must_be_aligned():
    """A misaligned slice used to be silently dropped, changing the score
    source without any indication in the trace."""
    scorer = CoarseScorer()
    config = InspectorConfig.for_scorer(scorer)
    crop = np.full((50, 50, 3), 200, np.uint8)
    with pytest.raises(ValueError, match="not aligned"):
        inspect_crop(crop, config=config, heat=np.zeros((10, 10), np.float32))


def test_inspect_crop_without_inputs_raises():
    scorer = CoarseScorer()
    with pytest.raises(TypeError, match="aligned"):
        inspect_crop(np.full((50, 50, 3), 200, np.uint8),
                     config=InspectorConfig.for_scorer(scorer))


def test_missing_model_env_fails_loudly(monkeypatch):
    """A set-but-missing ZOOMIN_MODEL must not silently degrade to CoarseScorer."""
    monkeypatch.setenv("ZOOMIN_MODEL", "does/not/exist.npz")
    with pytest.raises(RuntimeError, match="does not exist"):
        load_default_scorer()


def test_model_env_loads_normality_scorer(monkeypatch):
    model = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                         "models", "bottle_normality.npz")
    if not os.path.exists(model):
        pytest.skip("shipped bottle model not present")
    monkeypatch.setenv("ZOOMIN_MODEL", model)
    scorer = load_default_scorer()
    assert isinstance(scorer, NormalityScorer)
    config = InspectorConfig.for_scorer(scorer)
    assert config.defect_threshold == 4.0


def test_onnx_scorer_does_not_normalize_per_image():
    """Regression: ONNXScorer min-max rescaled each heatmap, the exact
    technique the module documents as removed for flagging clean images."""
    from zoomin import scoring
    src = Path(scoring.__file__).read_text(encoding="utf-8")
    body = src.split("class ONNXScorer")[1]
    assert "heat.min()" not in body, "per-image normalization reintroduced"


def test_budget_limits_crop_calls():
    scorer = CoarseScorer()
    config = InspectorConfig.for_scorer(scorer, crop_budget=2)
    img = np.full((512, 512, 3), 205, np.uint8)
    for i in range(6):
        cv2.line(img, (40, 40 + i * 60), (470, 60 + i * 60), (30, 30, 30), 5)
    rep = run_inspection(img, config=config, scorer=scorer)
    assert len(rep.regions) <= 2
