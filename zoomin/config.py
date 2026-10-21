"""Inspector configuration: crop budget, thresholds, geometry."""

from dataclasses import dataclass


@dataclass(frozen=True)
class InspectorConfig:
    # Agentic loop
    crop_budget: int = 8          # max zoom_crop calls per image ("inspection budget")
    heat_threshold: float = 2.0   # ABSOLUTE heatmap level for region proposals,
    # in the scorer's units (std-deviations for NormalityScorer; calibrated
    # 2026-10-20 on MVTec bottle: clean images peak ~1-3 (one speck at 3.1),
    # defects average ~6; proposals stay liberal, verdicts stay strict)
    min_region_area: int = 500    # px^2, ignore smaller connected components
    # (kills isolated specks on clean images; real defects span 1k+ px)

    # Verdicts (same units as heat_threshold)
    defect_threshold: float = 4.0   # inspect score >= this -> DEFECT
    unsure_threshold: float = 2.5   # >= this but < defect -> re-zoom, then ESCALATE if still unsure

    # Classical vision
    deviation_floor: float = 0.12  # absolute median-deviation floor for the
    # defect mask (fraction of full gray scale); clean crops stay empty below it

    # Geometry
    zoom_padding: float = 0.15    # padding around a proposed box, fraction of box size
    rezoom_shrink: float = 0.55   # tighter re-zoom window, fraction of crop size
    min_defect_area: int = 25     # px^2, ignore smaller contours inside a crop


DEFAULT_CONFIG = InspectorConfig()
