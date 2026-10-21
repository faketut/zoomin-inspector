"""Inspector tools: the agent's perception/decide/act vocabulary.

Each tool is a pure function. The agent in ``agent.py`` calls them in a loop and
logs every call (inputs, outputs, rationale) into the trace — that trace is the
Agentic Vision evidence: vision results *change* subsequent tool calls.
"""

from dataclasses import dataclass, field

import cv2
import numpy as np

from .config import InspectorConfig
from .scoring import CoarseScorer


@dataclass
class Box:
    x: int
    y: int
    w: int
    h: int
    heat: float = 0.0  # mean heat inside the box


@dataclass
class InspectResult:
    score: float                    # 0..1 defect confidence
    mask: np.ndarray                # binary defect mask (crop-sized)
    n_contours: int
    annotated: np.ndarray           # crop with contours drawn
    detail: dict = field(default_factory=dict)


def scan_overview(image: np.ndarray, scorer=None, config: InspectorConfig = None):
    """Tool 1 — coarse scan of the full image.

    Returns a normalized heatmap and the top-k suspicious boxes proposed *from*
    the heatmap. The agent spends its crop budget on these boxes.
    """
    config = config or InspectorConfig()
    scorer = scorer or CoarseScorer()
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if image.ndim == 3 else image
    heat = scorer.score(gray)

    binary = (heat > config.heat_threshold).astype(np.uint8) * 255
    n, labels, stats, centroids = cv2.connectedComponentsWithStats(binary, connectivity=8)
    boxes: list[Box] = []
    for i in range(1, n):  # skip background
        x, y, w, h, area = stats[i]
        if area < config.min_region_area:
            continue
        region_heat = float(heat[y : y + h, x : x + w].mean())
        boxes.append(Box(int(x), int(y), int(w), int(h), region_heat))
    boxes.sort(key=lambda b: b.heat, reverse=True)
    return heat, boxes


def zoom_crop(image: np.ndarray, box: Box, padding: float = 0.15) -> tuple[np.ndarray, Box]:
    """Tool 2 — crop a padded window around a box and return (crop, padded_box)."""
    h, w = image.shape[:2]
    px, py = int(box.w * padding), int(box.h * padding)
    x0, y0 = max(0, box.x - px), max(0, box.y - py)
    x1, y1 = min(w, box.x + box.w + px), min(h, box.y + box.h + py)
    padded = Box(x0, y0, x1 - x0, y1 - y0, box.heat)
    return image[y0:y1, x0:x1].copy(), padded


def _defect_mask(crop_gray: np.ndarray, min_area: int,
                 deviation_floor: float = 0.12, median_ksize: int = 31) -> np.ndarray:
    """Classical defect segmentation: absolute median-deviation + morphology.

    Otsu on a high-frequency residual was tried first and failed on real data:
    Otsu always splits into two classes, so even clean glass texture produced
    a mask (AUROC 0.44 on MVTec bottle, everything flagged). Calibration
    against ground-truth masks (2026-10-20) showed the median-filter deviation
    separates in its tail — defect p99 ~0.29 vs clean p99 ~0.14 — while the
    high-frequency residual does not (Otsu settled at 5-9 gray levels even on
    defect images). Hence an ABSOLUTE floor: a clean crop stays empty, and a
    real defect's connected region survives morphology.
    """
    k = min(median_ksize, (min(crop_gray.shape[:2]) // 2) * 2 + 1)
    k = max(k, 3)
    background = cv2.medianBlur(crop_gray, k)
    dev = cv2.absdiff(crop_gray, background).astype(np.float32) / 255.0
    _, binary = cv2.threshold(dev, deviation_floor, 255, cv2.THRESH_BINARY)
    binary = binary.astype(np.uint8)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    opened = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel)
    n, labels, stats, _ = cv2.connectedComponentsWithStats(opened, connectivity=8)
    mask = np.zeros_like(opened)
    for i in range(1, n):
        if stats[i, cv2.CC_STAT_AREA] >= min_area:
            mask[labels == i] = 255
    return mask


def inspect_crop(crop: np.ndarray, config: InspectorConfig = None,
                 patchcore=None, scorer=None, heat=None) -> InspectResult:
    """Tool 3 — inspect one crop.

    With an aligned ``heat`` slice (the scorer's full-image heatmap restricted
    to this crop's footprint — the normal case): score = peak heat inside the
    crop, in the scorer's absolute units (z / std-deviations for
    NormalityScorer). The heat slice MUST come from the full-image heatmap;
    re-running a position-sensitive scorer (e.g. NormalityScorer) on the crop
    itself would compare it against the wrong pixel statistics.
    With a scorer but no heat slice (translation-invariant scorers only):
    score = peak of scorer.score(crop).
    Without a scorer: falls back to the classical deviation-mask path
    (fraction of mask pixels, boosted by contour count).
    Optional: ``patchcore`` adapter (see patchcore_adapter.py) overrides/augments
    the score with a learned anomaly score — pass ``None`` to skip.
    """
    config = config or InspectorConfig()
    gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY) if crop.ndim == 3 else crop
    if heat is not None and heat.shape[:2] != gray.shape[:2]:
        heat = None  # misaligned slice — fall back to the scorer path
    if heat is not None:
        score = float(np.max(heat))
        mask = (heat > config.unsure_threshold).astype(np.uint8) * 255
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        detail = {"heat_max": score}
    elif scorer is not None:
        heat = scorer.score(gray)
        score = float(heat.max())
        mask = (heat > config.unsure_threshold).astype(np.uint8) * 255
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        detail = {"heat_max": score}
    else:
        mask = _defect_mask(gray, config.min_defect_area,
                            deviation_floor=config.deviation_floor)
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        frac = float(mask.mean() / 255.0)
        # Cap the contour-count term: a speckly mask must not score high on count
        # alone. The count term only applies when a real mask area exists, so a
        # few stray contours on an otherwise empty mask cannot push a clean
        # crop over threshold.
        score = min(1.0, frac * 4.0 + (0.08 * min(len(contours), 5) if frac > 0.002 else 0.0))
        detail = {"classical_score": score, "defect_fraction": frac}
    if patchcore is not None and patchcore.available():
        learned = float(patchcore.score_crop(crop))
        detail["patchcore_score"] = learned
        score = 0.5 * score + 0.5 * learned  # simple, documented fusion

    annotated = crop.copy()
    cv2.drawContours(annotated, contours, -1, (0, 0, 255), 2)
    return InspectResult(score=score, mask=mask, n_contours=len(contours),
                         annotated=annotated, detail=detail)


def strongest_sub_box(result: InspectResult, crop_shape) -> Box | None:
    """Pick the largest defect contour's bounding box (for the re-zoom re-plan)."""
    contours, _ = cv2.findContours(result.mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return None
    c = max(contours, key=cv2.contourArea)
    x, y, w, h = cv2.boundingRect(c)
    return Box(x, y, w, h, heat=result.score)
