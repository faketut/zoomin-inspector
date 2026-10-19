"""Coarse anomaly scoring for the overview scan.

Two scorers, one interface (``score(gray) -> heatmap in [0, 1]``):

- ``CoarseScorer`` — classical OpenCV baseline. Estimates a smooth background with
  a median filter, then combines absolute deviation with local texture energy.
  Fast, no weights, runs anywhere OpenCV 5 runs (incl. Graviton + COOL).
- ``ONNXScorer`` — hook for a learned scorer via the OpenCV 5 DNN module. The new
  OpenCV 5 DNN engine (rewritten op graph, dynamic shapes, operator fusion) uses
  ENGINE_AUTO by default and is CPU-optimized — drop in any ONNX anomaly model.
"""

import cv2
import numpy as np


class CoarseScorer:
    def __init__(self, median_ksize: int = 31, blur_sigma: float = 2.0):
        self.median_ksize = median_ksize
        self.blur_sigma = blur_sigma

    def score(self, gray: np.ndarray) -> np.ndarray:
        gray = gray.astype(np.float32)
        # Smooth background estimate; defects are local deviations from it.
        background = cv2.medianBlur(gray.astype(np.uint8), self.median_ksize).astype(np.float32)
        deviation = cv2.absdiff(gray, background)
        # Local texture energy: defects also disturb local variance.
        mean = cv2.boxFilter(gray, -1, (9, 9))
        sqmean = cv2.boxFilter(gray * gray, -1, (9, 9))
        variance = np.maximum(sqmean - mean * mean, 0.0)
        local_std = np.sqrt(variance)
        # Fixed reference scale (not self-normalized): uniform sensor noise must
        # stay cold. A local std of ~32 gray levels is a strong texture signal.
        texture = np.clip(local_std / 32.0, 0.0, 1.0)
        heat = deviation / 255.0 + 0.5 * texture
        heat = cv2.GaussianBlur(heat, (0, 0), self.blur_sigma)
        heat = (heat - heat.min()) / (heat.max() - heat.min() + 1e-6)
        return heat.astype(np.float32)


class ONNXScorer:
    """Drop-in learned scorer using the OpenCV 5 DNN module.

    Usage:
        scorer = ONNXScorer("defect_scorer.onnx", input_size=(256, 256))
        heat = scorer.score(gray)   # -> float32 heatmap in [0, 1], full resolution
    """

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
        heat = (heat - heat.min()) / (heat.max() - heat.min() + 1e-6)
        return heat.astype(np.float32)
