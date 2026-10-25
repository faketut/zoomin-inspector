"""Anomaly scoring for the overview scan.

Two scorers share one interface (``score(gray) -> heatmap``) plus a
``calibration`` of the thresholds that belong to its heat units:

- ``CoarseScorer`` — classical OpenCV baseline, no training data required.
  Estimates a smooth background with a median filter, then measures absolute
  deviation from it. Heat is a gray-level fraction: clean images stay below
  ~0.15, defects reach 0.3-0.6.
- ``NormalityScorer`` — classical one-class model: per-pixel Gaussian over
  NORMAL training images, heat in standard deviations (~1-2 normal, >=3
  anomalous).
- ``ONNXScorer`` — hook for a learned scorer via the OpenCV 5 DNN module. Its
  heat scale is model-defined, so it publishes no calibration and
  ``InspectorConfig.for_scorer`` refuses to guess thresholds for it.

Every heatmap is in ABSOLUTE units — never per-image normalized. A clean image
must produce a cold heatmap, otherwise every reflection and glass edge becomes
a hotspot and the agent proposes regions on images with nothing wrong.
(Calibration on MVTec bottle/train, 209 good images, 2026-10-20: median
deviation p99 on clean images is ~0.14; defect regions reach 0.3-0.6. A
per-image min-max normalization was tried first and scored AUROC 0.44 — every
clean image was flagged.)
"""

import os

import cv2
import numpy as np


class CoarseScorer:
    """Classical OpenCV baseline with an ABSOLUTE heat scale.

    Scores each pixel by its absolute deviation from a median-filtered
    background, in units of gray-level fraction, then blurs for spatial
    coherence.

    ``texture_weight`` keeps the old local-variance term as an option, but it
    defaults to 0: on bottle it carried no defect/clean separation (both reach
    ~1.0 on glass texture) and only added false proposals.
    """

    # Thresholds for this scorer's heat units (gray-level fraction in [0, 1]).
    calibration = {
        "heat_threshold": 0.15,    # region proposals: clean images stay below
        "unsure_threshold": 0.20,  # re-plan, then escalate
        "defect_threshold": 0.55,  # a defect fills the crop with deviation
        "coverage_confirm": 0.08,  # thinner than NormalityScorer's 0.20: a
        # median-deviation heat only lights the *edges* of a uniform blob, so
        # the same defect covers less of the window. Calibrated 2026-10-25 on
        # synthetic parts: 36/51 thin scratches confirm at >= 0.08, 0 of 30
        # clean parts reach the re-plan band at all.
    }

    def __init__(self, median_ksize: int = 31, blur_sigma: float = 2.0,
                 texture_weight: float = 0.0):
        self.median_ksize = median_ksize
        self.blur_sigma = blur_sigma
        self.texture_weight = texture_weight

    def score(self, gray: np.ndarray) -> np.ndarray:
        gray = gray.astype(np.float32)
        # Smooth background estimate; defects are local deviations from it.
        background = cv2.medianBlur(gray.astype(np.uint8), self.median_ksize).astype(np.float32)
        heat = cv2.absdiff(gray, background) / 255.0
        if self.texture_weight > 0:
            mean = cv2.boxFilter(gray, -1, (9, 9))
            sqmean = cv2.boxFilter(gray * gray, -1, (9, 9))
            variance = np.maximum(sqmean - mean * mean, 0.0)
            texture = np.clip(np.sqrt(variance) / 32.0, 0.0, 1.0)
            heat = heat + self.texture_weight * texture
        heat = cv2.GaussianBlur(heat, (0, 0), self.blur_sigma)
        return heat.astype(np.float32)


class NormalityScorer:
    """Classical one-class anomaly scorer: per-pixel Gaussian over NORMAL images.

    Fit on defect-free training images (MVTec provides them per category).
    Score = |x - mean| / (std + std_floor), in units of standard deviations —
    an ABSOLUTE scale: ~1-2 is normal variation, >=3 is anomalous, no
    per-image normalization anywhere.

    Still fully classical (NumPy/OpenCV only, no learned weights, CPU/Graviton
    friendly); the "training" is just pixel statistics, ~0.5 MB per category
    as .npz. This is the textbook unsupervised-AD baseline, and on MVTec
    bottle it scores AUROC 0.96 where the single-image heuristic scored 0.44.

    Limitation to state honestly: it assumes roughly aligned images (true for
    bottle, weaker for e.g. screw/hazelnut) — per-category validation on the
    EC2 benchmark run is the real check.
    """

    # Thresholds for this scorer's heat units (std deviations).
    calibration = {
        "heat_threshold": 2.0,     # proposals stay liberal
        "unsure_threshold": 2.5,   # verdicts stay strict
        "defect_threshold": 4.0,
        # Re-plan confirmation, from a sweep on all 315 MVTec test images
        # (bottle/transistor/capsule, 2026-10-25). Raising it trades recall for
        # precision: 0.20 -> P .863 R .566, 0.30 -> P .881 R .557,
        # 0.50 -> P .916 R .514. 0.30 sits at the accuracy plateau with 2x the
        # margin of the old peak-heat path, which never confirmed anything.
        "coverage_confirm": 0.30,
    }

    def __init__(self, mean: np.ndarray, std: np.ndarray,
                 std_floor: float = 8.0, blur_sigma: float = 2.0):
        self.mean = mean.astype(np.float32)
        self.std = std.astype(np.float32)
        self.std_floor = std_floor
        self.blur_sigma = blur_sigma
        self.size = (mean.shape[1], mean.shape[0])  # (w, h)

    @classmethod
    def fit(cls, image_paths: list[str], size: int = 256,
            std_floor: float = 8.0, blur_sigma: float = 2.0) -> "NormalityScorer":
        grays = []
        for p in image_paths:
            img = cv2.imread(p, cv2.IMREAD_GRAYSCALE)
            if img is None:
                raise ValueError(f"cannot read training image: {p}")
            grays.append(cv2.resize(img, (size, size)).astype(np.float32))
        stack = np.stack(grays)
        return cls(stack.mean(axis=0), stack.std(axis=0),
                   std_floor=std_floor, blur_sigma=blur_sigma)

    def save(self, path: str):
        np.savez(path, mean=self.mean, std=self.std,
                 std_floor=np.array(self.std_floor),
                 blur_sigma=np.array(self.blur_sigma))

    @classmethod
    def load(cls, path: str) -> "NormalityScorer":
        z = np.load(path)
        return cls(z["mean"], z["std"], float(z["std_floor"]),
                   float(z["blur_sigma"]))

    def score(self, gray: np.ndarray) -> np.ndarray:
        h, w = gray.shape[:2]
        small = cv2.resize(gray, self.size).astype(np.float32)
        z = np.abs(small - self.mean) / (self.std + self.std_floor)
        z = cv2.GaussianBlur(z, (0, 0), self.blur_sigma)
        heat = cv2.resize(z, (w, h), interpolation=cv2.INTER_LINEAR)
        return heat.astype(np.float32)


class ONNXScorer:
    """Drop-in learned scorer using the OpenCV 5 DNN module.

    Usage:
        scorer = ONNXScorer("defect_scorer.onnx", input_size=(256, 256))
        heat = scorer.score(gray)   # -> float32 heatmap, full resolution

    The heatmap is the model's raw output — deliberately NOT rescaled per
    image, because that makes a clean image as hot as a defective one (the
    failure mode documented in the module docstring). Because the output scale
    is defined by the model, this class publishes no ``calibration``: pass
    thresholds measured for your model to ``InspectorConfig`` explicitly.
    """

    calibration = None  # model-defined units — thresholds must be supplied

    def __init__(self, model_path: str, input_size: tuple[int, int] = (256, 256)):
        self.net = cv2.dnn.readNet(model_path)
        # OpenCV 5: the rewritten DNN engine is selected automatically (ENGINE_AUTO)
        # and is CPU-optimized — ideal for Graviton/COOL deployments.
        self.net.setPreferableBackend(cv2.dnn.DNN_BACKEND_OPENCV)
        self.net.setPreferableTarget(cv2.dnn.DNN_TARGET_CPU)
        self.input_size = input_size

    def score(self, gray: np.ndarray) -> np.ndarray:
        h, w = gray.shape[:2]
        blob = cv2.dnn.blobFromImage(
            gray, 1.0 / 255.0, self.input_size, (0, 0, 0), swapRB=False, crop=False
        )
        self.net.setInput(blob)
        out = self.net.forward().squeeze()
        heat = cv2.resize(out, (w, h), interpolation=cv2.INTER_LINEAR)
        return heat.astype(np.float32)


def load_default_scorer():
    """The scorer selected by the environment, for the serving layer.

    ``ZOOMIN_MODEL=<.npz>`` loads a fitted NormalityScorer (the configuration
    behind the reported MVTec numbers); otherwise CoarseScorer.

    A set-but-unreadable ``ZOOMIN_MODEL`` is a deployment error and raises:
    silently falling back here would serve defects-free "clean" verdicts from
    a model the operator believes is loaded.
    """
    model = os.environ.get("ZOOMIN_MODEL")
    if not model:
        return CoarseScorer()
    if not os.path.exists(model):
        raise RuntimeError(
            f"ZOOMIN_MODEL={model!r} does not exist; unset it to use the "
            "no-training-data CoarseScorer, or point it at a fitted .npz"
        )
    return NormalityScorer.load(model)
