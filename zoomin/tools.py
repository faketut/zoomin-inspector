"""Inspector tools: the agent's perception/decide/act vocabulary.

Each tool is a pure function. The agent in ``agent.py`` calls them in a loop and
logs every call (inputs, outputs, rationale) into the trace — that trace is the
Agentic Vision evidence: vision results *change* subsequent tool calls.

Scoring note: a crop is scored from an *aligned slice* of the full-image
heatmap. Re-running a position-sensitive scorer (``NormalityScorer`` compares
each pixel against a per-pixel mean/std fitted in full-image coordinates) on the
crop itself would compare it against the wrong pixel statistics.
"""

from dataclasses import dataclass

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
    score: float                    # peak heat in the crop, in the scorer's units
    mask: np.ndarray                # binary "hot" mask (crop-sized)
    n_contours: int
    annotated: np.ndarray           # crop with contours drawn
    coverage: float = 0.0           # fraction of the crop above unsure_threshold


def scan_overview(image: np.ndarray, scorer=None, config: InspectorConfig = None):
    """Tool 1 — coarse scan of the full image.

    Returns a normalized heatmap and the top-k suspicious boxes proposed *from*
    the heatmap. The agent spends its crop budget on these boxes.
    """
    scorer = scorer or CoarseScorer()
    config = config or InspectorConfig.for_scorer(scorer)
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if image.ndim == 3 else image
    heat = scorer.score(gray)

    binary = (heat > config.heat_threshold).astype(np.uint8) * 255
    n, _labels, stats, _centroids = cv2.connectedComponentsWithStats(binary, connectivity=8)
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


def inspect_crop(crop: np.ndarray, config: InspectorConfig, heat=None,
                 scorer=None) -> InspectResult:
    """Tool 3 — inspect one crop.

    ``heat`` is the scorer's full-image heatmap restricted to this crop's
    footprint (the normal case; the agent passes the aligned slice). Without it,
    a translation-invariant ``scorer`` is run on the crop — valid only for
    scorers whose statistics do not depend on absolute pixel position.

    Returns the peak heat (``score``) *and* ``coverage``: the fraction of the
    crop above ``config.unsure_threshold``. Peak heat alone cannot confirm a
    tighter re-zoom — the re-zoom window is nested inside the previous crop, so
    its peak can only be equal or lower. Coverage is scale-relative (the same
    defect fills more of a tighter window), so it can move either way and is
    what the re-plan decision reads.
    """
    gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY) if crop.ndim == 3 else crop
    if heat is None and scorer is None:
        raise TypeError(
            "inspect_crop needs an aligned `heat` slice or a `scorer`; "
            "without either there is nothing to score the crop with."
        )
    if heat is not None and heat.shape[:2] != gray.shape[:2]:
        raise ValueError(
            f"heat slice {heat.shape[:2]} is not aligned with crop {gray.shape[:2]}"
        )
    if heat is None:
        heat = scorer.score(gray)

    score = float(np.max(heat))
    hot = heat > config.unsure_threshold
    mask = hot.astype(np.uint8) * 255
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    coverage = float(np.count_nonzero(hot) / hot.size)

    annotated = crop.copy()
    cv2.drawContours(annotated, contours, -1, (0, 0, 255), 2)
    return InspectResult(score=score, mask=mask, n_contours=len(contours),
                         annotated=annotated, coverage=coverage)


def strongest_sub_box(result: InspectResult) -> Box | None:
    """Pick the largest hot contour's bounding box (for the re-zoom re-plan)."""
    contours, _ = cv2.findContours(result.mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return None
    c = max(contours, key=cv2.contourArea)
    x, y, w, h = cv2.boundingRect(c)
    return Box(x, y, w, h, heat=result.score)
