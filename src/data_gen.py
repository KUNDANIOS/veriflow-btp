"""
VeriFlow - Synthetic Test Data Generator
=========================================
Generates a small synthetic image set that recreates BTP's documented
AI-camera failure mode: a real violation/compliance signal whose COLOR is
nearly identical to its background (e.g. an off-white seatbelt against an
off-white shirt, or a dark helmet against dark hair), which trips up
color-distance-based detectors.

Two violation categories are generated:
  - "seatbelt": diagonal seatbelt strap present (compliant) vs absent (violation)
  - "helmet":    helmet present (compliant) vs bare head / hair (violation)

For each image we record ground truth + the exact pixel regions a detector
would inspect, so the baseline detector and verification layer can be kept
simple and fully deterministic.
"""

import os
import random
import csv
import numpy as np
from PIL import Image, ImageDraw

random.seed(42)
np.random.seed(42)

IMG_W, IMG_H = 200, 260
OUT_DIR = os.path.join(os.path.dirname(__file__), "..", "data", "images")
LABELS_CSV = os.path.join(os.path.dirname(__file__), "..", "data", "labels.csv")

# Global "priors" -- the colours a naive colour-distance detector expects
# the BARE state (no seatbelt / no helmet) to look like. These mirror the
# real BTP report: off-white shirts vs off-white seatbelts, and the
# dark-hair prior used to spot helmets.
SEATBELT_PRIOR = np.array([192, 188, 182])   # generic light/neutral shirt tone
HELMET_PRIOR = np.array([68, 54, 46])        # generic dark hair tone


def rand_color_near(base, spread):
    return tuple(int(np.clip(c + random.randint(-spread, spread), 0, 255)) for c in base)


def rand_color_far(base, min_dist):
    while True:
        c = np.array([random.randint(0, 255) for _ in range(3)])
        if np.linalg.norm(c - base) > min_dist:
            return tuple(int(x) for x in c)


def add_noise(img_arr, sigma=4):
    noise = np.random.normal(0, sigma, img_arr.shape)
    return np.clip(img_arr.astype(np.float32) + noise, 0, 255).astype(np.uint8)


def gen_seatbelt_image(idx, violation, hard):
    img = Image.new("RGB", (IMG_W, IMG_H), (110, 112, 116))  # road/background grey
    draw = ImageDraw.Draw(img)

    # shirt colour: sometimes neutral/off-white (near prior), sometimes vivid
    if random.random() < 0.5:
        shirt_color = rand_color_near(SEATBELT_PRIOR, 12)
    else:
        shirt_color = rand_color_far(SEATBELT_PRIOR, 70)

    # torso
    torso_box = (60, 90, 150, 230)
    draw.rectangle(torso_box, fill=shirt_color)

    # head
    draw.ellipse((85, 30, 125, 70), fill=(200, 170, 140))

    feature_region = (75, 110, 135, 160)  # where the strap band would sit

    if not violation:
        if hard:
            strap_color = rand_color_near(SEATBELT_PRIOR, 12)
        else:
            strap_color = rand_color_far(np.array(shirt_color), 60)

        # diagonal strap band
        draw.polygon(
            [(70, 95), (90, 95), (150, 225), (130, 225)],
            fill=strap_color,
        )
        # ribbing texture lines (slightly different shade) -- this is the
        # colour-independent signal VeriFlow's texture check relies on
        for off in (-22, -8, 8, 22):
            shade = tuple(int(np.clip(c + off, 0, 255)) for c in strap_color)
            draw.line([(70 + off, 95), (150 + off, 225)], fill=shade, width=2)
    # else: violation -> torso left as plain shirt_color (no strap)

    arr = add_noise(np.array(img), sigma=3)
    img = Image.fromarray(arr)

    fname = f"seatbelt_{idx:04d}.png"
    img.save(os.path.join(OUT_DIR, fname))
    return {
        "filename": fname,
        "category": "seatbelt",
        "violation": int(violation),
        "hard_case": int(hard),
        "feature_region": feature_region,
        "shirt_color": shirt_color,
    }


def gen_helmet_image(idx, violation, hard):
    img = Image.new("RGB", (IMG_W, IMG_H), (110, 112, 116))
    draw = ImageDraw.Draw(img)

    # body/torso (neutral, not the focus of this category)
    draw.rectangle((60, 140, 150, 250), fill=(90, 100, 130))

    hair_color = rand_color_near(HELMET_PRIOR, 14)

    feature_region = (75, 25, 135, 70)  # head-top region

    # base head (skin) + hair fill
    draw.ellipse((75, 35, 135, 95), fill=(195, 165, 135))  # face/skin
    draw.ellipse(feature_region, fill=hair_color)          # hair on top

    if violation:
        # scribbled hair texture -> many small random strokes -> high edge density
        for _ in range(40):
            x0 = random.randint(75, 130)
            y0 = random.randint(25, 60)
            x1 = x0 + random.randint(-4, 4)
            y1 = y0 + random.randint(3, 7)
            shade = rand_color_near(np.array(hair_color), 25)
            draw.line([(x0, y0), (x1, y1)], fill=shade, width=1)
    else:
        if hard:
            helmet_color = rand_color_near(HELMET_PRIOR, 14)
        else:
            helmet_color = rand_color_far(np.array(hair_color), 60)

        draw.ellipse(feature_region, fill=helmet_color)
        # glossy highlight arc -- smooth, structured edge (low internal noise)
        highlight = tuple(int(np.clip(c + 55, 0, 255)) for c in helmet_color)
        draw.arc(feature_region, start=200, end=300, fill=highlight, width=4)
        # helmet visor line
        draw.line([(78, 65), (132, 65)], fill=(30, 30, 30), width=3)

    arr = add_noise(np.array(img), sigma=3)
    img = Image.fromarray(arr)

    fname = f"helmet_{idx:04d}.png"
    img.save(os.path.join(OUT_DIR, fname))
    return {
        "filename": fname,
        "category": "helmet",
        "violation": int(violation),
        "hard_case": int(hard),
        "feature_region": feature_region,
        "shirt_color": "",
    }


def main(n_per_category=60):
    os.makedirs(OUT_DIR, exist_ok=True)
    rows = []
    idx = 0
    for cat_gen, cat_name in [(gen_seatbelt_image, "seatbelt"), (gen_helmet_image, "helmet")]:
        for i in range(n_per_category):
            violation = i % 2 == 0  # 50/50 split
            # ~35% of the COMPLIANT images are "hard" colour-ambiguous cases
            hard = (not violation) and (random.random() < 0.45)
            row = cat_gen(idx, violation, hard)
            rows.append(row)
            idx += 1

    with open(LABELS_CSV, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["filename", "category", "violation", "hard_case", "feature_region", "shirt_color"])
        writer.writeheader()
        for r in rows:
            writer.writerow(r)

    print(f"Generated {len(rows)} images -> {OUT_DIR}")
    print(f"Labels -> {LABELS_CSV}")


if __name__ == "__main__":
    main()
