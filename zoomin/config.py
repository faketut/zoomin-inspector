"""Inspector configuration: crop budget, thresholds, geometry.

Thresholds are expressed in the *scorer's own units* (z / std-deviations for
``NormalityScorer``, gray-level fraction for ``CoarseScorer``), so a config is
only meaningful together with the scorer it was calibrated for. That coupling
used to be implicit: ``InspectorConfig()`` defaulted to z-unit thresholds while
``app/api.py`` paired it with ``CoarseScorer`` (heat in [0, 0.62]), so
``heat > 2.0`` never fired and the deployed service proposed zero regions.
Thresholds now have no defaults — build a config with
:meth:`InspectorConfig.for_scorer`, which reads the calibration published by the
scorer itself. A mismatched pair is no longer constructible by accident.
"""

from dataclasses import dataclass


@dataclass(frozen=True)
class InspectorConfig:
    # Verdicts — same units as the scorer's heatmap (see for_scorer).
    heat_threshold: float      # region-proposal level for scan_overview
    unsure_threshold: float    # >= this but < defect -> re-plan, then ESCALATE
    defect_threshold: float    # inspect score >= this -> DEFECT
    coverage_confirm: float    # crop coverage that confirms a defect on re-zoom

    # Agentic loop
    crop_budget: int = 8       # max zoom_crop calls per image (inspection budget)

    # Region proposals
    min_region_area: int = 500  # px^2, ignore smaller connected components
    # (kills isolated specks on clean images; real defects span 1k+ px)

    # Geometry
    zoom_padding: float = 0.15   # padding around a proposed box, fraction of box size
    rezoom_shrink: float = 0.55  # tighter re-zoom window, fraction of crop size

    @classmethod
    def for_scorer(cls, scorer, **overrides) -> "InspectorConfig":
        """Build a config whose thresholds match ``scorer``'s heat units.

        ``overrides`` may replace any field (typically ``crop_budget``).
        """
        calibration = getattr(scorer, "calibration", None)
        if not calibration:
            raise TypeError(
                f"{type(scorer).__name__} publishes no calibration; its heat units "
                "are unknown, so no threshold can be chosen safely. Pass explicit "
                "heat_threshold/unsure_threshold/defect_threshold calibrated for "
                "that model, or use CoarseScorer/NormalityScorer."
            )
        unknown = set(overrides) - {f for f in cls.__dataclass_fields__}
        if unknown:
            raise TypeError(f"unknown config field(s): {sorted(unknown)}")
        return cls(**{**calibration, **overrides})
