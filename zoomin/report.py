"""Report rendering: annotated overview image."""

import cv2
import numpy as np

COLORS = {
    "OK": (0, 200, 0),        # green (BGR)
    "DEFECT": (0, 0, 255),     # red
    "ESCALATED": (0, 165, 255),  # orange
}


def annotate(image: np.ndarray, regions) -> np.ndarray:
    """Draw each region box colored by verdict + label."""
    out = image.copy()
    for r in regions:
        b = r.box
        color = COLORS.get(r.verdict, (255, 255, 255))
        cv2.rectangle(out, (b.x, b.y), (b.x + b.w, b.y + b.h), color, 3)
        label = f"{r.region_id}:{r.verdict} {r.score:.2f}"
        (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.7, 2)
        cv2.rectangle(out, (b.x, b.y - th - 8), (b.x + tw + 6, b.y), color, -1)
        cv2.putText(out, label, (b.x + 3, b.y - 6),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
    return out
