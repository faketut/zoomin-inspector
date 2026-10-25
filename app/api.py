"""FastAPI service: POST /inspect runs the inspector agent on an upload.

Run:  uvicorn app.api:app --port 8000

The endpoint is a plain ``def``, not ``async def``: the agent loop is
CPU-bound, so running it on the event loop would serialize every request onto
one core (measured: 4 concurrent /inspect calls = 1.06x speedup). Starlette
dispatchs plain handlers to a worker threadpool, so uploads are processed in
parallel.
"""

import base64
import dataclasses

import cv2
import numpy as np
from fastapi import FastAPI, File, Query, UploadFile
from fastapi.responses import JSONResponse

from zoomin.agent import run_inspection
from zoomin.config import InspectorConfig
from zoomin.scoring import load_default_scorer


app = FastAPI(title="ZoomIn Inspector", version="0.1.0")
_scorer = load_default_scorer()  # process-wide; swap for ONNXScorer with weights
_config = InspectorConfig.for_scorer(_scorer)  # crop_budget overridden per request


@app.get("/health")
def health():
    # The COOL award asks for the exact runtime OpenCV build — report it.
    return {"status": "ok", "opencv": cv2.__version__, "scorer": type(_scorer).__name__}


@app.post("/inspect")
def inspect(file: UploadFile = File(...),
            budget: int = Query(8, ge=1, le=16, description="max crops inspected")):
    raw = file.file.read()
    image = cv2.imdecode(np.frombuffer(raw, np.uint8), cv2.IMREAD_COLOR)
    if image is None:
        return JSONResponse({"error": "cannot decode image"}, status_code=400)
    config = dataclasses.replace(_config, crop_budget=budget)
    rep = run_inspection(image, config=config, scorer=_scorer)
    _, png = cv2.imencode(".png", rep.annotated)
    payload = rep.to_dict()
    payload["annotated_png_base64"] = base64.b64encode(png.tobytes()).decode()
    return payload
