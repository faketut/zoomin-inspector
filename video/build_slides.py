"""Render the ZoomIn Inspector demo-video slides (1920x1080 PNGs) with PIL.

Usage: python3 build_slides.py  -> video/slides/s1..s6.png
Numbers reflect the MVTec bottle run (2026-10-20) and the EC2 benchmark
results (2026-10-22): COOL 30.69 / vanilla-Graviton 25.70 / vanilla-x86 41.96
ms/image; bottle 0.961, transistor 0.840, capsule 0.652.
"""

from pathlib import Path

from PIL import Image, ImageDraw, ImageFont

OUT = Path(__file__).parent / "slides"
OUT.mkdir(exist_ok=True)
W, H = 1920, 1080

BG = (15, 17, 20)
FG = (235, 238, 242)
MUTED = (140, 148, 158)
ACCENT = (245, 158, 11)      # inspection amber
ACCENT2 = (96, 165, 250)     # blue
GREEN = (110, 190, 105)
WARN = (240, 178, 90)
CARD = (24, 27, 32)
BORDER = (52, 58, 66)

SANS = "/usr/share/fonts/truetype/liberation/LiberationSans-Regular.ttf"
SANSB = "/usr/share/fonts/truetype/liberation/LiberationSans-Bold.ttf"
MONO = "/usr/share/fonts/truetype/liberation/LiberationMono-Regular.ttf"
MONOB = "/usr/share/fonts/truetype/liberation/LiberationMono-Bold.ttf"


def F(size, bold=False, mono=False):
    return ImageFont.truetype(
        MONOB if mono and bold else MONO if mono else SANSB if bold else SANS, size)


def wrap(draw, text, font, max_w):
    words, lines, cur = text.split(), [], ""
    for w in words:
        t = (cur + " " + w).strip()
        if draw.textlength(t, font=font) <= max_w:
            cur = t
        else:
            lines.append(cur)
            cur = w
    if cur:
        lines.append(cur)
    return lines


def canvas():
    img = Image.new("RGB", (W, H), BG)
    return img, ImageDraw.Draw(img)


def footer(d, n):
    d.text((80, H - 60), "ZoomIn Inspector — OpenCV AI Competition 2026",
           font=F(24), fill=MUTED)
    d.text((W - 130, H - 60), f"{n}/6", font=F(24), fill=MUTED)
    d.line([(80, H - 90), (W - 80, H - 90)], fill=BORDER, width=2)


def title(d, text, y=110, size=76, color=FG, x=100):
    d.text((x, y), text, font=F(size, bold=True), fill=color)


def subtitle(d, text, y, x=100, size=36, color=MUTED):
    for line in wrap(d, text, F(size), W - 2 * x):
        d.text((x, y), line, font=F(size), fill=color)
        y += size + 14
    return y


def bullets(d, items, y, size=36, gap=28, x=100, max_w=1650, color=FG):
    for it, c in items:
        d.text((x, y), "›", font=F(size, bold=True), fill=ACCENT)
        ty = y
        for line in wrap(d, it, F(size), max_w - 60):
            d.text((x + 60, ty), line, font=F(size), fill=c or color)
            ty += size + 12
        y = ty + gap
    return y


def card(d, x, y, w, h, headline, sub, hcolor=FG):
    d.rounded_rectangle([x, y, x + w, y + h], radius=18, fill=CARD,
                        outline=BORDER, width=2)
    d.text((x + 40, y + 30), headline, font=F(56, bold=True), fill=hcolor)
    ty = y + 110
    for line in wrap(d, sub, F(30), w - 80):
        d.text((x + 40, ty), line, font=F(30), fill=MUTED)
        ty += 42


def s1():
    img, d = canvas()
    title(d, "ZoomIn Inspector", size=120, y=280)
    subtitle(d, "An agentic visual defect inspector that spends a fixed zoom "
                "budget the way a human inspector spends attention.",
             y=440, size=40)
    y = 700
    for t in ["OpenCV 5  ·  CPU-only", "MVTec AD: bottle AUROC 0.961",
              "Deploy: AWS Graviton + COOL AMI"]:
        d.text((100, y), "›", font=F(36, bold=True), fill=ACCENT)
        d.text((160, y), t, font=F(36), fill=FG)
        y += 64
    d.text((100, y + 40), "Solo entry — Jack (faketut)",
           font=F(32), fill=MUTED)
    footer(d, 1)
    img.save(OUT / "s1.png")


def s2():
    img, d = canvas()
    title(d, "The idea: inspect like a human")
    y = subtitle(d, "Whole-image classifiers downscale the defect away or run "
                    "detectors everywhere. Human inspectors scan, zoom, and "
                    "look again.", y=240, size=34)
    y += 20
    steps = [("SCAN", "overview heatmap — absolute anomaly scores", GREEN),
             ("ZOOM", "spend the crop budget on suspect regions", ACCENT),
             ("RE-VERIFY", "re-zoom ambiguous crops, don't guess", ACCENT2),
             ("ESCALATE", "10.8% said \"I can't tell\" — sent to a human", WARN)]
    x = 100
    for name, sub, col in steps:
        card(d, x, y, 400, 330, name, sub, hcolor=col)
        x += 440
    bullets(d, [("Agent loop: scan_overview → zoom_crop → inspect_crop → "
                 "finalize_report.", None),
                ("Explicit budget: 8 crops per image, logged in the trace.", None)],
            y + 380, size=32)
    footer(d, 2)
    img.save(OUT / "s2.png")


def s3():
    img, d = canvas()
    title(d, "Engineering: absolute scores")
    y = bullets(d, [
        ("NormalityScorer: per-pixel Gaussian fit over normal training images; "
         "score in standard deviations. Never re-normalized per image.", None),
        ("Crop inspection scores an aligned slice of the full-image heatmap — "
         "re-running a position-sensitive scorer on the crop compared it "
         "against the wrong pixel statistics.", None),
    ], y=250, size=33)
    d.text((100, y + 10), "Honest failure, first attempt:", font=F(33, bold=True),
           fill=WARN)
    y += 60
    bullets(d, [
        ("Per-image min-max normalization turned clean reflections and glass "
         "edges into hotspots: every bottle flagged DEFECT, AUROC 0.44.", None),
        ("Removed it. Same scorer, AUROC 0.44 → 0.961. "
         "Also dropped: Otsu-on-residual masks (Otsu always finds two "
         "classes — even in clean glass).", None),
    ], y, size=32)
    footer(d, 3)
    img.save(OUT / "s3.png")


def s4():
    img, d = canvas()
    title(d, "Results: MVTec bottle")
    y = 250
    cards = [("0.961", "AUROC (83 test images)", GREEN),
             ("1.000", "precision — 20/20 good bottles, zero false alarms", GREEN),
             ("0.841", "recall — 10 hard misses, low-contrast / small breaks", ACCENT),
             ("7.2%", "escalated to a human, not guessed", WARN)]
    x = 100
    for h, s, c in cards:
        card(d, x, y, 400, 300, h, s, hcolor=c)
        x += 440
    y += 350
    y = bullets(d, [
        ("Misses are genuinely hard: peak anomaly 1.6–3.0 sigma overlaps "
         "normal reflection variation.", None),
        ("Synthetic controls: AUROC 1.0, 15.6 ms/image. "
         "Scorer: NumPy/OpenCV only, no learned weights, ~0.5 MB/category.", None),
        ("Caveat: thresholds tuned on this set; transistor/capsule were run "
         "on EC2 (0.840 / 0.652) — see the cloud slide.", None),
    ], y, size=30)
    footer(d, 4)
    img.save(OUT / "s4.png")


def s5():
    img, d = canvas()
    title(d, "AWS Graviton4 + COOL: honest numbers")
    y = bullets(d, [
        ("One container: FastAPI API + Gradio UI on c8g.xlarge from the "
         "official Cloud Optimized OpenCV AMI for Graviton4 — our Best Use "
         "of COOL evidence. COOL's optimized build is OpenCV 4.14.0-pre; "
         "the live API runs pip OpenCV 5.0.0.", None),
        ("Same script, same 83 bottle images, three configs — identical "
         "27 defect regions everywhere. All instances terminated; run cost "
         "under $1.", None),
    ], y=228, size=31)
    d.text((100, y + 8), "Benchmark: ms/image · images/sec · $/10K images",
           font=F(33, bold=True), fill=FG)
    y += 60
    x = 100
    for name, sub, hcolor in [
        ("30.69", "A. COOL 4.14.0-pre on Graviton4\n32.6 img/s · $0.0146/10K", FG),
        ("25.70", "B. vanilla 5.0.0 on Graviton4\n38.9 img/s · $0.0123/10K", GREEN),
        ("41.96", "C. vanilla 5.0.0 on x86\n23.8 img/s · $0.0204/10K", FG),
    ]:
        card(d, x, y, 520, 260, name, sub, hcolor)
        x += 560
    bullets(d, [("Honest headline: stock OpenCV 5.0.0 beat COOL by ~16% on "
                 "the same Graviton4 hardware — reported in the paper. MVTec "
                 "on the instance: bottle 0.961, transistor 0.840, capsule "
                 "0.652 — capsule's imprint/scratch misses (2/22, 1/23) and "
                 "transistor's 16 false positives are documented, not hidden.", None)],
            y + 300, size=28)
    footer(d, 5)
    img.save(OUT / "s5.png")


def s6():
    img, d = canvas()
    title(d, "Assists, never replaces, human QA")
    bullets(d, [
        ("Escalation exists because a silent wrong call costs more than an "
         "honest \"I can't tell\" — failure cases are in the report, not "
         "hidden (required, and we agree).", None),
        ("Runs on cheap CPU for small factories that can't afford GPU "
         "inspection rigs. No personal data processed.", None),
        ("Limitation, stated plainly: the classical scorer is a baseline. "
         "Fine textures want the learned scorer — OpenCV 5 DNN hook "
         "(ONNXScorer) is already in the repo.", None),
        ("Ablation: one crop versus eight adds only 3 detections out of "
         "212 — the overview decides nearly everything; the loop's value "
         "is the audit trail and review channel.", None),
    ], y=250, size=33)
    y = 800
    d.text((100, y), "ZoomIn Inspector: inspect like a human, spend like a miser.",
           font=F(38, bold=True), fill=ACCENT)
    footer(d, 6)
    img.save(OUT / "s6.png")


for fn in (s1, s2, s3, s4, s5, s6):
    fn()
    print("rendered", fn.__name__)
