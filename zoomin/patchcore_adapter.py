"""Optional anomalib PatchCore adapter (learned anomaly scores).

PatchCore builds a memory bank of patch features from *normal* images only —
no defective training data needed — and scores a crop by distance to the bank.
On MVTec AD this is a strong, well-published baseline, and the memory-bank
build runs on CPU in minutes.

This module is a guarded adapter: if ``anomalib`` is not installed,
``available()`` returns False and the classical path in ``tools.inspect_crop``
is used alone. Install with::

    pip install anomalib   # then wire a trained PatchCore model here

Integration sketch (anomalib >= 1.x API — pin and verify against the installed
version before the competition run):

    from anomalib.models import Patchcore
    from anomalib.data import MVTecAD  # or a folder-based datamodule
    model = Patchcore(...)
    engine = Engine(...)  # fit on normal images, then export / predict
"""

import numpy as np


class PatchCoreAdapter:
    """Thin wrapper; ``score_crop`` returns a 0..1 anomaly score for a crop."""

    def __init__(self, model=None):
        self.model = model

    def available(self) -> bool:
        return self.model is not None

    def score_crop(self, crop: np.ndarray) -> float:
        if self.model is None:
            raise RuntimeError("PatchCore model not loaded — see module docstring.")
        # TODO: wire to anomalib Engine.predict on the crop and normalize
        # the anomaly map to [0, 1]. Kept explicit so the integration is
        # verified against the pinned anomalib version, not guessed.
        raise NotImplementedError("Wire to your trained anomalib PatchCore model.")


def try_load(model_path: str | None = None) -> PatchCoreAdapter:
    """Return an adapter; unavailable (classical-only) when anomalib is missing."""
    if model_path is None:
        return PatchCoreAdapter(None)
    try:
        import anomalib  # noqa: F401
    except ImportError:
        return PatchCoreAdapter(None)
    # TODO: load checkpoint -> PatchCoreAdapter(model)
    return PatchCoreAdapter(None)
