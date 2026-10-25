"""Optional anomalib PatchCore adapter (learned anomaly scores).

PatchCore builds a memory bank of patch features from *normal* images only —
no defective training data needed — and scores a crop by distance to the bank.
On MVTec AD this is a strong, well-published baseline, and the memory-bank
build runs on CPU in minutes.

WHY THIS IS NOT WIRED INTO THE AGENT
------------------------------------
``score_crop`` is deliberately unimplemented. A version that silently returned
a classical score, or a 0.5/0.5 blend of an unimplemented model, would make the
agent's reported score and its trace disagree about where the number came from
— the failure mode this repo treats as a correctness bug. Installing
``anomalib`` and pinning it is a prerequisite for the implementation; until
that is done, callers get an explicit ``unavailable`` adapter that the agent
does not use.

To wire it up:
  1. ``pip install anomalib`` and pin the version in requirements.txt.
  2. Fit a PatchCore model on the category's normal images (anomalib Engine).
  3. Implement ``score_crop`` against that pinned API and return the anomaly
     map's peak, normalized to the model's own calibration — then publish a
     ``calibration`` dict on the adapter like the scorers in ``scoring.py`` do,
     so ``InspectorConfig.for_scorer`` can bind thresholds to it.
"""

import numpy as np


class PatchCoreAdapter:
    """Thin wrapper; ``score_crop`` returns an anomaly score for a crop."""

    def __init__(self, model=None):
        self.model = model

    def available(self) -> bool:
        return self.model is not None

    def score_crop(self, crop: np.ndarray) -> float:
        if self.model is None:
            raise RuntimeError("PatchCore model not loaded — see module docstring.")
        raise NotImplementedError(
            "Wire to your trained anomalib PatchCore model, then publish a "
            "'calibration' dict on this adapter (see module docstring)."
        )
