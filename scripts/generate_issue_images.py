"""
Generates a small, original, code-drawn illustration for each plant issue in
models/symptom_model/issues_list.json. These are simple flat-style vector-ish
graphics (leaf shapes + symptom overlays) drawn with Pillow -- not photos, not
scraped from anywhere -- so there are no copyright/licensing concerns, and no
internet connection is needed to build them.

Run once from the project root:
    python scripts/generate_issue_images.py

Output: issue_images/<issue_slug>.png (used by pages/1_Plant_Symptom_Diagnosis.py)
"""

import os
import json
import math
import random
from PIL import Image, ImageDraw, ImageFont

random.seed(7)

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT_DIR = os.path.join(ROOT, "issue_images")
os.makedirs(OUT_DIR, exist_ok=True)

W, H = 520, 380
BG = (248, 246, 239)
LEAF_GREEN = (58, 122, 74)
LEAF_GREEN_DARK = (39, 90, 53)
SOIL = (90, 62, 42)
POT = (196, 120, 78)
TEXT_COLOR = (31, 41, 55)


def font(size):
    try:
        return ImageFont.load_default(size=size)
    except TypeError:
        return ImageFont.load_default()


def new_canvas():
    img = Image.new("RGB", (W, H), BG)
    return img, ImageDraw.Draw(img, "RGBA")


def save(img, name, title):
    draw = ImageDraw.Draw(img, "RGBA")
    f = font(20)
    bbox = draw.textbbox((0, 0), title, font=f)
    tw = bbox[2] - bbox[0]
    draw.text(((W - tw) / 2, H - 34), title, fill=TEXT_COLOR, font=f)
    slug = name.lower().replace(" ", "_").replace("(", "").replace(")", "")
    img.save(os.path.join(OUT_DIR, f"{slug}.png"))
    print(f"  saved {slug}.png")


def leaf_shape(draw, cx, cy, w, h, color, rotation=0, droop=0):
    """Draw a simple pointed lens-shaped leaf (two mirrored arcs) with a midrib vein."""
    n = 24
    top = []
    bottom = []
    for i in range(n + 1):
        t = -1 + 2 * i / n  # -1 .. 1 across the width
        x = t * (w / 2)
        frac = (1 - t ** 2) ** 0.5
        top.append((x, -(h / 2) * frac))
        bottom.append((x, (h / 2) * frac))
    pts = top + bottom[::-1]

    rad = math.radians(rotation)
    rot_pts = []
    for x, y in pts:
        if y > 0:  # only droop the lower (outward-hanging) edge
            y += droop * (abs(x) / (w / 2)) ** 1.5
        rx = x * math.cos(rad) - y * math.sin(rad)
        ry = x * math.sin(rad) + y * math.cos(rad)
        rot_pts.append((cx + rx, cy + ry))

    draw.polygon(rot_pts, fill=color)

    rad_mid = math.radians(rotation)
    vein_top = (cx + (-h / 2 + 6) * -math.sin(rad_mid), cy + (-h / 2 + 6) * math.cos(rad_mid))
    vein_bot = (cx + (h / 2 - 6) * -math.sin(rad_mid), cy + (h / 2 - 6) * math.cos(rad_mid))
    draw.line([vein_top, vein_bot], fill=LEAF_GREEN_DARK, width=2)


def sun_icon(draw, cx, cy, r=18, color=(230, 160, 40)):
    """Draw a simple sun (circle + rays) - avoids relying on emoji glyph support in the default font."""
    draw.ellipse([cx - r, cy - r, cx + r, cy + r], fill=color)
    for i in range(8):
        ang = math.radians(i * 45)
        x1 = cx + math.cos(ang) * (r + 6)
        y1 = cy + math.sin(ang) * (r + 6)
        x2 = cx + math.cos(ang) * (r + 14)
        y2 = cy + math.sin(ang) * (r + 14)
        draw.line([(x1, y1), (x2, y2)], fill=color, width=3)


def pot(draw, cx, top_y, width=140, height=70):
    draw.polygon([
        (cx - width / 2, top_y),
        (cx + width / 2, top_y),
        (cx + width / 2 - 14, top_y + height),
        (cx - width / 2 + 14, top_y + height),
    ], fill=POT)
    draw.ellipse([cx - width / 2, top_y - 10, cx + width / 2, top_y + 10], fill=SOIL)


# ---------------------------------------------------------------------------

def img_powdery_mildew():
    img, d = new_canvas()
    pot(d, W / 2, 300)
    leaf_shape(d, W / 2 - 60, 190, 160, 220, LEAF_GREEN, rotation=-12)
    leaf_shape(d, W / 2 + 60, 190, 160, 220, LEAF_GREEN, rotation=12)
    for _ in range(220):
        x = random.uniform(W / 2 - 150, W / 2 + 150)
        y = random.uniform(90, 280)
        r = random.uniform(1, 3)
        d.ellipse([x - r, y - r, x + r, y + r], fill=(255, 255, 255, 180))
    save(img, "Powdery Mildew", "Powdery Mildew")


def img_root_rot():
    img, d = new_canvas()
    pot(d, W / 2, 230, width=150, height=90)
    leaf_shape(d, W / 2, 130, 150, 190, (110, 130, 70), rotation=-8, droop=40)
    d.ellipse([W / 2 - 75, 220, W / 2 + 75, 240], fill=(40, 30, 25, 210))
    for i in range(6):
        x0 = W / 2 - 60 + i * 22
        y0 = 300
        pts = [(x0, y0)]
        for j in range(4):
            pts.append((x0 + random.uniform(-8, 8), y0 + 12 * (j + 1)))
        d.line(pts, fill=(70, 45, 35), width=4, joint="curve")
    save(img, "Root Rot", "Root Rot")


def img_spider_mites():
    img, d = new_canvas()
    pot(d, W / 2, 300)
    leaf_shape(d, W / 2 - 55, 180, 150, 210, (150, 165, 90), rotation=-10)
    leaf_shape(d, W / 2 + 55, 180, 150, 210, (150, 165, 90), rotation=10)
    for _ in range(40):
        x1 = random.uniform(W / 2 - 130, W / 2 + 130)
        y1 = random.uniform(90, 270)
        x2 = x1 + random.uniform(-25, 25)
        y2 = y1 + random.uniform(-25, 25)
        d.line([(x1, y1), (x2, y2)], fill=(255, 255, 255, 140), width=1)
    for _ in range(18):
        x = random.uniform(W / 2 - 130, W / 2 + 130)
        y = random.uniform(90, 270)
        d.ellipse([x - 2, y - 2, x + 2, y + 2], fill=(140, 40, 30))
    save(img, "Spider Mite Infestation", "Spider Mite Infestation")


def img_aphids():
    img, d = new_canvas()
    pot(d, W / 2, 300)
    leaf_shape(d, W / 2, 170, 190, 230, LEAF_GREEN, rotation=0)
    for _ in range(35):
        x = random.uniform(W / 2 - 40, W / 2 + 40)
        y = random.uniform(90, 140)
        r = random.uniform(4, 7)
        d.ellipse([x - r, y - r * 0.7, x + r, y + r * 0.7], fill=(140, 200, 90))
    d.ellipse([W / 2 - 45, 85, W / 2 + 45, 150], outline=(255, 255, 255, 90), width=6)
    save(img, "Aphid Infestation", "Aphid Infestation")


def img_leaf_spot():
    img, d = new_canvas()
    pot(d, W / 2, 300)
    leaf_shape(d, W / 2 - 55, 180, 150, 210, LEAF_GREEN, rotation=-10)
    leaf_shape(d, W / 2 + 55, 180, 150, 210, LEAF_GREEN, rotation=10)
    for _ in range(10):
        x = random.uniform(W / 2 - 130, W / 2 + 130)
        y = random.uniform(100, 260)
        r = random.uniform(6, 12)
        d.ellipse([x - r - 3, y - r - 3, x + r + 3, y + r + 3], fill=(210, 190, 90, 160))
        d.ellipse([x - r, y - r, x + r, y + r], fill=(105, 70, 40))
    save(img, "Fungal Leaf Spot", "Fungal Leaf Spot")


def img_nitrogen_deficiency():
    img, d = new_canvas()
    pot(d, W / 2, 300)
    for i, ang in enumerate([-18, 0, 18]):
        col = (200, 195, 90) if i != 1 else (170, 175, 70)
        leaf_shape(d, W / 2 + ang * 2, 180, 150, 210, col, rotation=ang)
    save(img, "Nitrogen Deficiency", "Nitrogen Deficiency")


def img_overwatering():
    img, d = new_canvas()
    pot(d, W / 2, 260, width=150, height=90)
    d.ellipse([W / 2 - 75, 250, W / 2 + 75, 268], fill=(50, 70, 110, 200))
    leaf_shape(d, W / 2 - 50, 170, 140, 190, (150, 170, 90), rotation=-15, droop=50)
    leaf_shape(d, W / 2 + 50, 170, 140, 190, (150, 170, 90), rotation=15, droop=50)
    save(img, "Overwatering Stress", "Overwatering Stress")


def img_underwatering():
    img, d = new_canvas()
    pot(d, W / 2, 300)
    leaf_shape(d, W / 2 - 55, 180, 145, 200, (120, 140, 80), rotation=-14, droop=25)
    leaf_shape(d, W / 2 + 55, 180, 145, 200, (120, 140, 80), rotation=14, droop=25)
    sun_icon(d, W / 2, 95)
    for i in range(5):
        x0 = W / 2 - 70 + i * 35
        d.line([(x0, 290), (x0 + random.uniform(-6, 6), 305)], fill=(120, 90, 60), width=2)
    save(img, "Underwatering Stress", "Underwatering Stress")


def img_light_deficiency():
    img, d = new_canvas()
    pot(d, W / 2 - 40, 320, width=110, height=50)
    stem_top = (W / 2 + 60, 70)
    d.line([(W / 2 - 40, 320), stem_top], fill=LEAF_GREEN_DARK, width=6)
    for t in [0.3, 0.55, 0.8, 1.0]:
        x = W / 2 - 40 + (stem_top[0] - (W / 2 - 40)) * t
        y = 320 + (stem_top[1] - 320) * t
        leaf_shape(d, x, y, 55, 30, (140, 165, 110), rotation=-20)
    sun_icon(d, W - 70, 55, r=16)
    save(img, "Light Deficiency (Etiolation)", "Light Deficiency (Etiolation)")


def img_sooty_mold():
    img, d = new_canvas()
    pot(d, W / 2, 300)
    leaf_shape(d, W / 2 - 55, 180, 150, 210, LEAF_GREEN, rotation=-10)
    leaf_shape(d, W / 2 + 55, 180, 150, 210, LEAF_GREEN, rotation=10)
    for _ in range(6):
        x = random.uniform(W / 2 - 120, W / 2 + 120)
        y = random.uniform(100, 250)
        r = random.uniform(14, 26)
        d.ellipse([x - r, y - r * 0.6, x + r, y + r * 0.6], fill=(25, 25, 25, 170))
    save(img, "Sooty Mold", "Sooty Mold")


def img_bacterial_blight():
    img, d = new_canvas()
    pot(d, W / 2, 300)
    leaf_shape(d, W / 2, 170, 190, 230, (150, 140, 70), rotation=0)
    for i in range(5):
        y0 = 100 + i * 30
        d.line([(W / 2 - 20, y0), (W / 2 + 60, y0 + 15)], fill=(90, 55, 35), width=6)
    save(img, "Bacterial Leaf Blight", "Bacterial Leaf Blight")


def img_mealybugs():
    img, d = new_canvas()
    pot(d, W / 2, 300)
    leaf_shape(d, W / 2 - 55, 180, 150, 210, LEAF_GREEN, rotation=-10)
    leaf_shape(d, W / 2 + 55, 180, 150, 210, LEAF_GREEN, rotation=10)
    for _ in range(9):
        cx = random.uniform(W / 2 - 30, W / 2 + 30)
        cy = random.uniform(180, 260)
        for _ in range(5):
            x = cx + random.uniform(-10, 10)
            y = cy + random.uniform(-6, 6)
            r = random.uniform(3, 6)
            d.ellipse([x - r, y - r, x + r, y + r], fill=(250, 250, 245, 230))
    save(img, "Mealybug Infestation", "Mealybug Infestation")


ISSUE_RENDERERS = {
    "Powdery Mildew": img_powdery_mildew,
    "Root Rot": img_root_rot,
    "Spider Mite Infestation": img_spider_mites,
    "Aphid Infestation": img_aphids,
    "Fungal Leaf Spot": img_leaf_spot,
    "Nitrogen Deficiency": img_nitrogen_deficiency,
    "Overwatering Stress": img_overwatering,
    "Underwatering Stress": img_underwatering,
    "Light Deficiency (Etiolation)": img_light_deficiency,
    "Sooty Mold": img_sooty_mold,
    "Bacterial Leaf Blight": img_bacterial_blight,
    "Mealybug Infestation": img_mealybugs,
}


if __name__ == "__main__":
    issues_path = os.path.join(ROOT, "models", "symptom_model", "issues_list.json")
    with open(issues_path) as f:
        issues = list(json.load(f).values())

    print(f"Generating {len(issues)} issue illustrations...")
    for issue in issues:
        renderer = ISSUE_RENDERERS.get(issue)
        if renderer is None:
            print(f"  [skip] no renderer defined for '{issue}'")
            continue
        renderer()

    print(f"\nDone. Images saved to {OUT_DIR}/")
