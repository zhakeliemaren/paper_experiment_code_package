"""Render the supplementary C1 barrier figure using the manuscript style."""

from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFont


ROOT = Path(__file__).resolve().parent
OUT = ROOT / "paper_artifacts" / "supp_barrier_c1.png"

# Degree-six certificate retained by the all-degree experiment.
coeffs = np.array([
    0.009639604518186505,
    0.0022466062887889415,
    -0.0046881109927002056,
    0.00017320979020726653,
    0.004557116028820912,
    0.002384915877762419,
    0.00034894106632030414,
])
x = np.linspace(-3.0, 5.0, 1200)
B = sum(value * x**power for power, value in enumerate(coeffs))
rho = 0.02330995481212774

W, H = 1800, 1500
left, top, right, bottom = 170, 270, 1390, 1300
img = Image.new("RGB", (W, H), "white")
draw = ImageDraw.Draw(img)
bold = ImageFont.truetype("C:/Windows/Fonts/arialbd.ttf", 44)
small = ImageFont.truetype("C:/Windows/Fonts/arial.ttf", 25)
tiny = ImageFont.truetype("C:/Windows/Fonts/arial.ttf", 20)
def px(value): return left + (value + 3) / 8 * (right - left)
def py(value): return bottom - (value + 1.0) / 2.0 * (bottom - top)
draw.text((35, 35), "C1", fill="#18272d", font=bold)
draw.text((left + 290, 115), "Projected initial set", fill="#334047", font=small)
draw.text((left + 790, 115), "Projected unsafe set", fill="#334047", font=small)
draw.rectangle((left + 250, 128, left + 280, 158), fill="#16945c")
draw.rectangle((left + 750, 128, left + 780, 158), fill="#d33d49")
draw.text((left + 285, 172), "x1-x2 synchronized slice", fill="#18272d", font=bold)
draw.rectangle((left, top, right, bottom), outline="#263238", width=4, fill="#eef7f8")
vmin, vmax = float(B.min()), float(B.max())
for ix in range(right - left):
    xv = -3 + 8 * ix / (right - left - 1)
    value = float(sum(c * xv**p for p, c in enumerate(coeffs)))
    t = max(0.0, min(1.0, (value - vmin) / max(vmax - vmin, 1e-9)))
    color = (int(238 - 35*t), int(248 - 25*t), int(249 - 130*t))
    draw.line((left + ix, top, left + ix, bottom), fill=color, width=1)
for tick in (-3, -1, 1, 3, 5):
    xx = px(tick)
    draw.line((xx, top, xx, bottom), fill="#d9e2e5", width=2)
    draw.text((xx - 12, bottom + 14), str(tick), fill="#263238", font=small)
for tick in (-1, 0, 1):
    yy = py(tick)
    draw.line((left, yy, right, yy), fill="#d9e2e5", width=2)
    draw.text((left - 58, yy - 14), f"{tick:g}", fill="#263238", font=small)
draw.rectangle((px(0.8), py(1), px(1.2), py(-1)), outline="#176b45", width=5, fill="#b7dfca")
draw.rectangle((px(2.9), py(1), px(3.1), py(-1)), outline="#d33d49", width=5, fill="#efb8bd")
for level in np.linspace(max(vmin, -0.2), min(vmax, 1.0), 12):
    level = float(level)
    crossings = np.where(np.sign(B[:-1] - level) * np.sign(B[1:] - level) <= 0)[0]
    for index in crossings:
        draw.line((px(float(x[index])), top, px(float(x[index])), bottom), fill="#263238", width=2)
draw.line((left, py(rho), right, py(rho)), fill="#1d4f91", width=4)
draw.line((left, py(1.0), right, py(1.0)), fill="#c7353f", width=4)
draw.text((right - 340, py(rho) - 34), f"B=rho={rho:.4f}", fill="#1d4f91", font=tiny)
draw.text((right - 105, py(1.0) - 34), "B=1", fill="#c7353f", font=tiny)
draw.text((right - 475, bottom + 75), "Certified safety lower bound: 0.9767", fill="#263238", font=small)
draw.text((W // 2 - 20, bottom + 24), "x1", fill="#18272d", font=bold)
draw.text((left - 100, top - 42), "x2", fill="#18272d", font=bold)
OUT.parent.mkdir(parents=True, exist_ok=True)
img.save(OUT)
print(OUT)
