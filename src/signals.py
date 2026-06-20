"""
VeriFlow - Detection Signals
=============================
Three independent, deterministic signals computed per image:

1. baseline_color  - mimics today's colour-distance based AI detector.
                      Compares the "feature region" (where a seatbelt strap
                      or helmet would be) to a GLOBAL colour prior for the
                      "bare" state. This is exactly the kind of check that
                      fails on BTP's documented off-white/off-white and
                      dark-helmet/dark-hair cases.

2. texture_edge    - colour-independent. Looks at local gradient structure
                      inside the feature region (seatbelt ribbing creates a
                      strong, regular gradient pattern; bare shirt is flat).
                      For helmets, looks at edge DENSITY (scribbled hair has
                      far more small edges than a smooth helmet shell).

3. contour_shape   - colour-independent. Looks at how many distinct contours
                      / connected regions of similar intensity exist inside
                      the feature region. A helmet (one or two large smooth
                      shapes incl. highlight arc) differs structurally from
                      hair (many tiny strands).

Each signal returns (pred_violation: bool, confidence: float in [0,1]).
"""

import cv2
import numpy as np

SEATBELT_PRIOR = np.array([192, 188, 182])
HELMET_PRIOR = np.array([68, 54, 46])

# Tuned against the synthetic dataset (see calibrate.py)
COLOR_DIST_THRESH = 35.0
TEXTURE_THRESH_SEATBELT = 6.0     # gradient std inside feature region
EDGE_DENSITY_THRESH_HELMET = 0.06  # fraction of pixels that are edges
CONTOUR_THRESH_SEATBELT = 1       # >=2 distinct stripes => strap present
CONTOUR_THRESH_HELMET = 3         # many small contours => hair


def _crop(img, region):
    x0, y0, x1, y1 = region
    return img[y0:y1, x0:x1]


def baseline_color(img, category, feature_region):
    crop = _crop(img, feature_region)
    mean_color = crop.reshape(-1, 3).mean(axis=0)
    prior = SEATBELT_PRIOR if category == "seatbelt" else HELMET_PRIOR
    dist = float(np.linalg.norm(mean_color - prior))

    # If the feature region looks like the "bare" prior -> violation (item absent)
    pred_violation = dist < COLOR_DIST_THRESH
    # confidence = how far from the decision boundary, normalised
    conf = float(np.clip(abs(dist - COLOR_DIST_THRESH) / COLOR_DIST_THRESH, 0, 1))
    return pred_violation, conf, dist


def texture_edge(img, category, feature_region):
    crop = _crop(img, feature_region)
    gray = cv2.cvtColor(crop, cv2.COLOR_RGB2GRAY)

    if category == "seatbelt":
        gx = cv2.Sobel(gray, cv2.CV_32F, 1, 0, ksize=3)
        gy = cv2.Sobel(gray, cv2.CV_32F, 0, 1, ksize=3)
        mag = np.sqrt(gx ** 2 + gy ** 2)
        score = float(mag.std())
        # high gradient std -> ribbing pattern -> strap present -> NOT violation
        pred_violation = score < TEXTURE_THRESH_SEATBELT
        conf = float(np.clip(abs(score - TEXTURE_THRESH_SEATBELT) / max(TEXTURE_THRESH_SEATBELT, 1e-6), 0, 1))
    else:  # helmet
        edges = cv2.Canny(gray, 50, 150)
        density = float((edges > 0).mean())
        # high edge density -> scribbled hair -> violation (no helmet)
        pred_violation = density > EDGE_DENSITY_THRESH_HELMET
        conf = float(np.clip(abs(density - EDGE_DENSITY_THRESH_HELMET) / EDGE_DENSITY_THRESH_HELMET, 0, 1))
        score = density

    return pred_violation, conf, score


def contour_shape(img, category, feature_region):
    crop = _crop(img, feature_region)
    gray = cv2.cvtColor(crop, cv2.COLOR_RGB2GRAY)
    edges = cv2.Canny(gray, 50, 150)
    contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    n = len(contours)

    if category == "seatbelt":
        # ribbing lines produce multiple parallel contours -> strap present -> NOT violation
        pred_violation = n <= CONTOUR_THRESH_SEATBELT
        conf = float(np.clip(abs(n - CONTOUR_THRESH_SEATBELT) / 3.0, 0, 1))
    else:  # helmet
        # scribbled hair produces many small contours -> violation
        pred_violation = n >= CONTOUR_THRESH_HELMET
        conf = float(np.clip(abs(n - CONTOUR_THRESH_HELMET) / 3.0, 0, 1))

    return pred_violation, conf, n


def multiframe_agreement(img, category, feature_region, n_frames=5):
    """Simulate consecutive camera frames via small brightness/contrast/noise
    perturbations, and measure how often baseline_color's prediction stays
    the same as on the original frame."""
    base_pred, _, _ = baseline_color(img, category, feature_region)
    agree = 0
    for i in range(n_frames):
        rng = np.random.RandomState(1000 + i)
        brightness = rng.randint(-15, 16)
        contrast = 1.0 + rng.uniform(-0.08, 0.08)
        perturbed = np.clip(img.astype(np.float32) * contrast + brightness, 0, 255).astype(np.uint8)
        pred, _, _ = baseline_color(perturbed, category, feature_region)
        if pred == base_pred:
            agree += 1
    return agree / n_frames
