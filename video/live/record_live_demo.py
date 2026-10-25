#!/usr/bin/env python3
"""Record a live session of the real ZoomIn Gradio UI with Playwright's
stock Chromium (the meta-chromium build blocks loopback via LNA checks).
Three real inspections: DEFECT bottle, ESCALATED bottle, clean bottle.

Needs the UI running (`ZOOMIN_MODEL=models/bottle_normality.npz python -m app.ui`)
and the MVTec bottle test images under data/mvtec (gitignored; fetch with
scripts/download_mvtec.py). Playwright is not a runtime or test dependency:

    pip install playwright && playwright install chromium
"""
import os
import shutil

from playwright.sync_api import sync_playwright

# Repo root, derived from this file: video/live/record_live_demo.py -> repo/
ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
RAW = os.path.join(ROOT, "video/live/raw")
ACTS = [
    ("data/mvtec/bottle/test/broken_large/001-94.png", "DEFECT"),
    ("data/mvtec/bottle/test/broken_large/000-94.png", "ESCALATED"),
    # "0 defect" only appears in the summary when the clean bottle really
    # passed; a bare "tool calls" marker matches every result and asserts nothing.
    ("data/mvtec/bottle/test/good/001-95.png", "0 defect"),
]

shutil.rmtree(RAW, ignore_errors=True)
os.makedirs(RAW, exist_ok=True)

with sync_playwright() as p:
    b = p.chromium.launch(args=["--no-proxy-server",
                                "--disable-features=LocalNetworkAccessChecks"])
    ctx = b.new_context(viewport={"width": 1280, "height": 860},
                        record_video_dir=RAW,
                        record_video_size={"width": 1280, "height": 860})
    pg = ctx.new_page()
    pg.goto("http://127.0.0.1:7860/", timeout=30000)
    pg.wait_for_selector("input[type=file]", state="attached", timeout=30000)
    pg.wait_for_timeout(3000)
    for i, (img, marker) in enumerate(ACTS):
        path = os.path.join(ROOT, img)
        pg.locator("input[type=file]").first.set_input_files(path)
        print(f"act {i+1}: uploaded {img}", flush=True)
        pg.wait_for_timeout(2500)
        pg.get_by_role("button", name="Inspect").first.click()
        try:
            pg.wait_for_function(
                f"() => document.body.innerText.includes('{marker}') && "
                "document.body.innerText.includes('tool calls')",
                timeout=45000)
        except Exception as e:
            print(f"act {i+1}: wait marker failed: {e}", flush=True)
        print(f"act {i+1}: result visible ({marker})", flush=True)
        pg.wait_for_timeout(4000)
        pg.evaluate("window.scrollTo(0, document.body.scrollHeight)")
        pg.wait_for_timeout(4500)
        pg.evaluate("window.scrollTo(0, 0)")
        pg.wait_for_timeout(1200)
        if i < len(ACTS) - 1:
            pg.reload()
            pg.wait_for_selector("input[type=file]", state="attached", timeout=30000)
            pg.wait_for_timeout(2500)
    video = pg.video.path()
    ctx.close()
    b.close()
    print("video:", video, flush=True)
