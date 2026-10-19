"""Gradio UI: upload an image, see the annotated result, region table, and the
agent's full tool trace (the Agentic Vision evidence).

Run:  python -m app.ui   (serves on :7860)
"""

import argparse

import cv2
import gradio as gr
import numpy as np

from zoomin.agent import run_inspection
from zoomin.config import InspectorConfig
from zoomin.scoring import CoarseScorer

_scorer = CoarseScorer()


def inspect_image(img: np.ndarray, budget: int):
    if img is None:
        return None, "upload an image first", None, None
    bgr = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
    rep = run_inspection(bgr, config=InspectorConfig(crop_budget=int(budget)),
                         scorer=_scorer)
    annotated_rgb = cv2.cvtColor(rep.annotated, cv2.COLOR_BGR2RGB)
    rows = [[r.region_id, r.verdict, round(r.score, 3), r.replanned,
             f"({r.box.x},{r.box.y},{r.box.w},{r.box.h})"] for r in rep.regions]
    summary = (f"{rep.summary()['verdicts']} — "
               f"{len(rep.trace)} tool calls in {rep.elapsed_s:.2f}s "
               f"(opencv {cv2.__version__})")
    return annotated_rgb, summary, rows, [s.to_dict() for s in rep.trace]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--server-name", default="127.0.0.1")
    ap.add_argument("--server-port", type=int, default=7860)
    args = ap.parse_args()

    with gr.Blocks(title="ZoomIn Inspector") as demo:
        gr.Markdown("# ZoomIn Inspector — agentic defect inspection")
        gr.Markdown("The agent scans the overview, spends a crop budget on the hottest "
                    "regions, re-zooms when unsure, and escalates the ambiguous ones. "
                    "The trace below shows every tool call.")
        with gr.Row():
            img_in = gr.Image(label="Input image")
            img_out = gr.Image(label="Annotated result")
        budget = gr.Slider(1, 16, value=8, step=1, label="Crop budget")
        btn = gr.Button("Inspect", variant="primary")
        summary = gr.Textbox(label="Summary")
        regions = gr.Dataframe(
            headers=["region", "verdict", "score", "replanned", "box"],
            label="Region verdicts")
        trace = gr.JSON(label="Agent tool trace")
        btn.click(inspect_image, inputs=[img_in, budget],
                  outputs=[img_out, summary, regions, trace])
    demo.launch(server_name=args.server_name, server_port=args.server_port)


if __name__ == "__main__":
    main()
