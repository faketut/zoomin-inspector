"""FastAPI service: POST /inspect runs the inspector agent on an upload.

Run:  uvicorn app.api:app --port 8000
"""

import base64
import io
import os
import time

import cv2
import numpy as np
from fastapi import FastAPI, File, UploadFile
from fastapi.responses import JSONResponse

from zoomin.agent import run_inspection
from zoomin.config import InspectorConfig
from zoomin.scoring import CoarseScorer, NormalityScorer


def _default_scorer():
    """ZOOMIN_MODEL=<.npz> loads a fitted NormalityScorer (the configuration
    behind the reported MVTec numbers); otherwise CoarseScorer."""
    model = os.environ.get("ZOOMIN_MODEL")
    if model and os.path.exists(model):
        return NormalityScorer.load(model)
    return CoarseScorer()


app = FastAPI(title="ZoomIn Inspector", version="0.1.0")
_scorer = _default_scorer()  # process-wide; swap for ONNXScorer with weights


@app.get("/health")
def health():
    # The COOL award asks for the exact runtime OpenCV build — report it.
    return {"status": "ok", "opencv": cv2.__version__, "time": time.time()}


@app.post("/inspect")
async def inspect(file: UploadFile = File(...), budget: int = 8):
    raw = await file.read()
    image = cv2.imdecode(np.frombuffer(raw, np.uint8), cv2.IMREAD_COLOR)
    if image is None:
        return JSONResponse({"error": "cannot decode image"}, status_code=400)
    config = InspectorConfig(crop_budget=budget)
    rep = run_inspection(image, config=config, scorer=_scorer)
    _, png = cv2.imencode(".png", rep.annotated)
    payload = rep.to_dict()
    payload["annotated_png_base64"] = base64.b64encode(png.tobytes()).decode()
    return payload
