"""Inspector configuration: crop budget, thresholds, geometry."""

from dataclasses import dataclass


@dataclass(frozen=True)
class InspectorConfig:
    # Agentic loop
    crop_budget: int = 8          # max zoom_crop calls per image ("inspection budget")
    heat_threshold: float = 0.45  # heatmap binarization level for region proposals
    min_region_area: int = 120    # px^2, ignore smaller connected components

    # Verdicts
    defect_threshold: float = 0.55  # inspect score >= this -> DEFECT
    unsure_threshold: float = 0.20  # >= this but < defect -> re-zoom, then ESCALATE if still unsure

    # Geometry
    zoom_padding: float = 0.15    # padding around a proposed box, fraction of box size
    rezoom_shrink: float = 0.55   # tighter re-zoom window, fraction of crop size
    min_defect_area: int = 25     # px^2, ignore smaller contours inside a crop


DEFAULT_CONFIG = InspectorConfig()
