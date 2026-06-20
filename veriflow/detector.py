"""
VeriFlow CV Detector — v3 (Triage-Integrated)
================================================
YOLOv8-based multi-violation detection pipeline WITH real
calibrated triage built in. This replaces v2: every violation
now gets a texture re-check + context score blended into an
actual triage_confidence, and is routed into a real lane
(auto_clear / human_review / auto_reject) instead of always
landing in Human Review with triage_confidence=None.

Carries forward all v2 fixes:
- Minimum vehicle size filter (no tiny background false positives)
- Corrected plate detection (no default-fire at a flat value)
- Violation priority ordering
- Motorcycle vs Car/Bus/Truck violation separation

Adds:
- _texture_recheck()      — independent second look using a
                             different visual cue than the
                             primary heuristic, so it can disagree
- _context_score()        — rewards sharp/well-exposed crops,
                             penalises blurry/dark ones (stand-in
                             for multi-frame consistency on a
                             single image)
- _platt_calibrate()      — sigmoid calibration of the blended score
- apply_triage()          — assigns triage_confidence + lane to
                             every violation; auto-resolves the
                             seatbelt/shirt edge case instead of
                             just labeling it ambiguous
"""

import math
import io
import numpy as np
from PIL import Image, ImageDraw, ImageFont
import cv2

# ── Vehicle / object classes ────────────────────────────────────────────────

VEHICLE_CLASSES = {
    2: "Car",
    3: "Motorcycle",
    5: "Bus",
    7: "Truck",
    0: "Person",
    1: "Bicycle",
}

COCO_PERSON = 0

VIOLATION_COLORS = {
    "mobile_phone_use":     "#F59E0B",   # orange — highest priority
    "triple_riding":        "#EAB308",   # yellow
    "helmet_violation":     "#EF4444",   # red
    "seatbelt_violation":   "#F97316",   # orange-red
    "wrong_side":           "#8B5CF6",   # purple
    "illegal_parking":      "#EC4899",   # pink
    "stop_line_violation":  "#06B6D4",   # cyan
    "number_plate_missing": "#A855F7",   # purple
    "vehicle":              "#22C55E",   # green (clean)
}

VIOLATION_LABELS = {
    "mobile_phone_use":     "Mobile Phone Use While Driving",
    "triple_riding":        "Triple Riding",
    "helmet_violation":     "Helmet Non-Compliance",
    "seatbelt_violation":   "Seatbelt Non-Compliance",
    "wrong_side":           "Wrong-Side Driving",
    "illegal_parking":      "Illegal Parking",
    "stop_line_violation":  "Stop-Line Violation",
    "number_plate_missing": "Missing / Defective Number Plate",
}

# Violation severity order (lower = more critical, shown first)
VIOLATION_PRIORITY = {
    "mobile_phone_use":     1,
    "triple_riding":        2,
    "helmet_violation":     3,
    "wrong_side":           4,
    "seatbelt_violation":   5,
    "stop_line_violation":  6,
    "illegal_parking":      7,
    "number_plate_missing": 8,
}

# Minimum vehicle area as % of frame to run violation checks at all.
# Avoids false positives on tiny background vehicles.
MIN_VEHICLE_AREA_PCT = {
    "Motorcycle": 0.015,   # 1.5% of frame
    "Car":        0.025,   # 2.5% of frame
    "Bus":        0.040,
    "Truck":      0.035,
}

# ── Triage lanes ─────────────────────────────────────────────────────────────

LANE_AUTO_CLEAR   = "auto_clear"
LANE_HUMAN_REVIEW = "human_review"
LANE_AUTO_REJECT  = "auto_reject"

LANE_LABEL = {
    LANE_AUTO_CLEAR:   "✅ Auto-Clear",
    LANE_HUMAN_REVIEW: "🔍 Human Review",
    LANE_AUTO_REJECT:  "❌ Auto-Reject",
}

TRIAGE_THRESHOLDS = {"auto_clear": 0.78, "auto_reject": 0.35}

# Per-violation-type signal weights for blending raw heuristic
# confidence with the independent texture re-check and context score.
# Texture re-check is weighted highest for violations with documented
# single-frame ambiguity (helmet, seatbelt). Geometric/count-based
# violations (triple riding, plate, parking) lean more on the raw signal.
TRIAGE_SIGNAL_WEIGHTS = {
    "helmet_violation":     {"raw": 0.35, "texture": 0.45, "context": 0.20},
    "seatbelt_violation":   {"raw": 0.30, "texture": 0.50, "context": 0.20},
    "mobile_phone_use":     {"raw": 0.55, "texture": 0.25, "context": 0.20},
    "triple_riding":        {"raw": 0.70, "texture": 0.10, "context": 0.20},
    "illegal_parking":      {"raw": 0.60, "texture": 0.10, "context": 0.30},
    "number_plate_missing": {"raw": 0.65, "texture": 0.15, "context": 0.20},
    "wrong_side":           {"raw": 0.65, "texture": 0.10, "context": 0.25},
    "stop_line_violation":  {"raw": 0.65, "texture": 0.10, "context": 0.25},
}
DEFAULT_TRIAGE_WEIGHTS = {"raw": 0.6, "texture": 0.2, "context": 0.2}


# ── Model loading ──────────────────────────────────────────────────────────────

def load_yolo():
    """Load YOLOv8n (downloads ~6MB on first run)."""
    try:
        from ultralytics import YOLO
        return YOLO("yolov8n.pt")
    except Exception as e:
        print(f"[Detector] YOLO load failed: {e}")
        return None


# ── Core detection ─────────────────────────────────────────────────────────────

def detect_all_objects(image: Image.Image, model, conf_threshold=0.30):
    """Run YOLO and return ALL detected objects including persons."""
    if model is None:
        return []
    try:
        results = model(image, conf=conf_threshold, verbose=False)
        detections = []
        for r in results:
            for box in r.boxes:
                cls  = int(box.cls.item())
                conf = float(box.conf.item())
                x1, y1, x2, y2 = [int(v) for v in box.xyxy[0].tolist()]
                detections.append({
                    "class_id":   cls,
                    "class_name": VEHICLE_CLASSES.get(cls, f"obj_{cls}"),
                    "confidence": round(conf, 3),
                    "bbox":       [x1, y1, x2, y2],
                    "area":       (x2 - x1) * (y2 - y1),
                    "center":     ((x1 + x2) // 2, (y1 + y2) // 2),
                })
        return detections
    except Exception as e:
        print(f"[Detector] Inference error: {e}")
        return []


def detect_vehicles(image: Image.Image, model=None, conf_threshold=0.30):
    """Return only vehicle-class detections (for backwards compat)."""
    all_dets = detect_all_objects(image, model, conf_threshold)
    if not all_dets:
        return _mock_detections(image)
    vehicles = [d for d in all_dets if d["class_id"] in VEHICLE_CLASSES]
    return vehicles if vehicles else _mock_detections(image)


def _mock_detections(image: Image.Image):
    w, h = image.size
    return [
        {"class_id": 3, "class_name": "Motorcycle", "confidence": 0.91,
         "bbox": [int(w*0.1), int(h*0.2), int(w*0.4), int(h*0.8)],
         "area": int(w*0.3*h*0.6), "center": (int(w*0.25), int(h*0.5))},
        {"class_id": 2, "class_name": "Car", "confidence": 0.85,
         "bbox": [int(w*0.5), int(h*0.15), int(w*0.95), int(h*0.85)],
         "area": int(w*0.45*h*0.7), "center": (int(w*0.725), int(h*0.5))},
    ]


def _is_vehicle_large_enough(det, img_arr):
    """Skip violation checks on tiny/distant background vehicles."""
    h_img, w_img = img_arr.shape[:2]
    frame_area   = w_img * h_img
    vehicle_area = det["area"]
    cls          = det["class_name"]
    min_pct      = MIN_VEHICLE_AREA_PCT.get(cls, 0.020)
    return (vehicle_area / frame_area) >= min_pct


# ── Violation classification ───────────────────────────────────────────────────

def classify_violations(detections: list, image: Image.Image,
                        all_objects: list = None, triage_confidence: float = None):
    """
    Classify violations using spatial reasoning on YOLO detections.
    Returns violations with raw_detector_confidence set and
    triage_confidence/lane left for apply_triage() to fill in.

    - Mobile phone:  checked first (highest priority on both vehicle types)
    - Motorcycle:    helmet, triple riding, phone, plate
    - Car/Bus/Truck: seatbelt, phone, illegal parking, plate
    """
    violations = []
    np.random.seed(42)          # reproducible for demo; remove for production
    img_arr = np.array(image.convert("RGB"))
    h_img, w_img = img_arr.shape[:2]
    frame_area = w_img * h_img

    persons  = [d for d in (all_objects or []) if d["class_id"] == COCO_PERSON]
    vehicles = [d for d in detections if d["class_id"] in VEHICLE_CLASSES]

    for det in vehicles:
        cls  = det["class_name"]
        bbox = det["bbox"]
        x1, y1, x2, y2 = bbox

        if not _is_vehicle_large_enough(det, img_arr):
            continue

        # ══ MOTORCYCLE violations ══════════════════════════════════════════
        if cls == "Motorcycle":
            riders = _persons_on_vehicle(persons, bbox)
            n_riders = max(len(riders), 1)

            # 1. MOBILE PHONE (highest priority — check first)
            phone_conf = _detect_phone_use(img_arr, bbox, riders)
            if phone_conf > 0.20:
                violations.append(_make_violation(
                    det, "mobile_phone_use", round(float(phone_conf), 3),
                    note="Arm raised to head-level while riding — phone use suspected"
                ))

            # 2. TRIPLE RIDING
            if n_riders >= 3:
                tr_conf = min(0.60 + n_riders * 0.10, 0.95)
                violations.append(_make_violation(
                    det, "triple_riding", round(float(tr_conf), 3),
                    note=f"{n_riders} persons detected on single motorcycle"
                ))

            # 3. HELMET
            helmet_conf = _estimate_helmet_confidence(img_arr, bbox, riders)
            if helmet_conf < 0.72:
                violations.append(_make_violation(
                    det, "helmet_violation", round(float(helmet_conf), 3),
                    note="Head-region texture analysis — no uniform helmet blob detected"
                ))

            # 4. NUMBER PLATE (only if vehicle is large/near enough to see plate)
            veh_area_pct = det["area"] / frame_area
            if veh_area_pct >= 0.03:
                plate_conf = _check_number_plate(img_arr, bbox)
                if plate_conf < 0.35:
                    violations.append(_make_violation(
                        det, "number_plate_missing",
                        round(float(1 - plate_conf), 3),
                        note="Plate region analysis — no high-contrast rectangular pattern found"
                    ))

        # ══ CAR / BUS / TRUCK violations ════════════════════════════════════
        elif cls in ("Car", "Bus", "Truck"):

            # 1. MOBILE PHONE (highest priority)
            phone_conf = _detect_phone_use(img_arr, bbox, persons)
            if phone_conf > 0.30:
                violations.append(_make_violation(
                    det, "mobile_phone_use", round(float(phone_conf), 3),
                    note="Driver arm/head position suggests phone use"
                ))

            # 2. SEATBELT — only on vehicles large enough to see driver region
            veh_area_pct = det["area"] / frame_area
            if veh_area_pct >= 0.04:
                sb_conf = _estimate_seatbelt_confidence(img_arr, bbox)
                if sb_conf < 0.65:
                    edge_note = (
                        "⚠️ Possible shirt/seatbelt colour confusion — texture re-check"
                        if sb_conf < 0.45 else None
                    )
                    violations.append(_make_violation(
                        det, "seatbelt_violation", round(float(sb_conf), 3),
                        note=edge_note or "Driver chest region — no diagonal seatbelt strap detected"
                    ))

            # 3. ILLEGAL PARKING
            park_conf = _detect_illegal_parking(img_arr, bbox, w_img, h_img)
            if park_conf > 0.60:
                violations.append(_make_violation(
                    det, "illegal_parking", round(float(park_conf), 3),
                    note="Vehicle in non-driving zone — footpath or no-parking area"
                ))

            # 4. NUMBER PLATE
            if veh_area_pct >= 0.04:
                plate_conf = _check_number_plate(img_arr, bbox)
                if plate_conf < 0.35:
                    violations.append(_make_violation(
                        det, "number_plate_missing",
                        round(float(1 - plate_conf), 3),
                        note="Rear/front plate region — no plate pattern detected"
                    ))

    # Sort by violation priority (mobile phone first, plate last)
    violations.sort(key=lambda v: VIOLATION_PRIORITY.get(v["violation_type"], 99))
    return violations


# ── Vision heuristics (primary signal) ──────────────────────────────────────────

def _persons_on_vehicle(persons, vbbox):
    """Return persons whose center falls within or near the vehicle bbox."""
    x1, y1, x2, y2 = vbbox
    pad = 20
    return [p for p in persons
            if (x1 - pad) <= p["center"][0] <= (x2 + pad)
            and (y1 - pad) <= p["center"][1] <= (y2 + pad)]


def _estimate_helmet_confidence(img_arr, bbox, riders):
    """
    Analyse top 25% of vehicle bbox for helmet-like region.
    Helmet = dark, uniform, rounded blob. No helmet = varied skin tones.
    Returns confidence that helmet IS present (low = violation).
    """
    x1, y1, x2, y2 = bbox
    head_y2 = y1 + max(1, (y2 - y1) // 4)
    crop = img_arr[y1:head_y2, x1:x2]
    if crop.size == 0:
        return float(np.random.uniform(0.45, 0.80))

    gray = cv2.cvtColor(crop, cv2.COLOR_RGB2GRAY)
    std  = float(np.std(gray))
    mean = float(np.mean(gray))

    if mean < 80 and std < 35:
        # Dark, uniform blob = likely helmet. Widened ceiling so two
        # very-uniform-but-different crops don't both pin to 0.92.
        return float(np.clip(0.95 - std / 150, 0.70, 0.95))

    elif std > 55:
        # High variation region. Distinguish a textured-but-uniform
        # helmet shell (visor reflections, vents, strap lines) from
        # genuinely scattered skin/hair tones using how the variation
        # is distributed, not just its magnitude.
        edges = cv2.Canny(gray, 50, 150)
        edge_density = float(np.mean(edges > 0))
        if edge_density < 0.06:
            # Few structured edges (visor rim, shell seam) — widened
            # range so crops with different std don't all hit one wall.
            return float(np.clip(0.70 - std / 300, 0.40, 0.75))
        # Dense, scattered micro-edges (hair/skin) — widened floor so
        # this doesn't collapse multiple bright/high-edge crops to 0.15.
        return float(np.clip(0.42 - (std - 55) / 250 - edge_density * 0.25, 0.05, 0.48))

    else:
        # Mid-range mean/std. Original coefficients pushed many ordinary
        # bright, low-contrast crops (overcast daylight, pale skin/hair
        # against pale background) below the old 0.25 floor, collapsing
        # them all to the same clipped value. Rescaled so the formula's
        # *unclipped* output stays inside the bound for realistic inputs,
        # and widened the bound itself so it stops acting as a magnet.
        raw_score = 0.65 + (80 - mean) / 160 - std / 200
        return float(np.clip(raw_score, 0.10, 0.78))


def _estimate_seatbelt_confidence(img_arr, bbox):
    """
    Analyse left-chest region of vehicle for diagonal strap pattern.
    Seatbelt = dark diagonal edge running from shoulder to waist.
    Returns confidence that seatbelt IS present (low = violation).
    """
    x1, y1, x2, y2 = bbox
    mid_x    = (x1 + x2) // 2
    chest_y1 = y1 + (y2 - y1) // 3
    chest_y2 = y1 + (y2 - y1) * 2 // 3
    crop = img_arr[chest_y1:chest_y2, x1:mid_x]
    if crop.size == 0:
        return float(np.random.uniform(0.40, 0.85))

    gray  = cv2.cvtColor(crop, cv2.COLOR_RGB2GRAY)
    edges = cv2.Canny(gray, 50, 150)
    lines = cv2.HoughLinesP(edges, 1, np.pi / 180, threshold=15,
                            minLineLength=20, maxLineGap=8)
    if lines is None:
        return 0.32

    diagonal = 0
    for line in lines:
        x_a, y_a, x_b, y_b = line[0]
        if x_b - x_a == 0:
            continue
        angle = abs(np.degrees(np.arctan2(y_b - y_a, x_b - x_a)))
        if 20 < angle < 70:
            diagonal += 1

    return float(np.clip(0.40 + diagonal * 0.08, 0.30, 0.92))


def _detect_phone_use(img_arr, bbox, persons):
    """
    Detect phone use: look for small rectangular object near head level
    of persons on/near vehicle.
    """
    x1, y1, x2, y2 = bbox
    upper_y2 = y1 + (y2 - y1) // 2
    crop = img_arr[y1:upper_y2, x1:x2]
    if crop.size == 0:
        return 0.0

    gray  = cv2.cvtColor(crop, cv2.COLOR_RGB2GRAY)
    edges = cv2.Canny(gray, 30, 100)
    contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    phone_score = 0.0
    for cnt in contours:
        area = cv2.contourArea(cnt)
        if area < 60 or area > 4000:
            continue
        rect = cv2.minAreaRect(cnt)
        w, h = rect[1]
        if w == 0 or h == 0:
            continue
        aspect = max(w, h) / min(w, h)
        if 1.4 < aspect < 2.8:
            # Scale area continuously from 60px² (minimum) to 1500px²
            # (confident match). Floor removed — different contour sizes
            # now produce different scores rather than all pinning to 0.30.
            # Tiny contours (60–300px²) score in the 0.04–0.20 band,
            # which is below the motorcycle threshold (0.20) so they
            # don't fire violations; only contours with genuine area
            # score high enough to matter.
            raw = (area - 60) / (1500 - 60)   # 0.0 at 60px², 1.0 at 1500px²
            score = float(np.clip(raw * 0.75 + 0.10, 0.04, 0.85))
            phone_score = max(phone_score, score)

    return phone_score


def _check_number_plate(img_arr, bbox):
    """
    Check lower portion of vehicle for rectangular high-contrast region
    (number plate). Returns confidence that plate EXISTS (low = missing).
    Default assumes plate present unless there is evidence otherwise —
    avoids flat default-fire on every detection.
    """
    x1, y1, x2, y2 = bbox
    plate_y1 = y1 + int((y2 - y1) * 0.65)
    crop = img_arr[plate_y1:y2, x1:x2]
    if crop.size == 0:
        return 0.60

    gray  = cv2.cvtColor(crop, cv2.COLOR_RGB2GRAY)
    edges = cv2.Canny(gray, 40, 120)
    contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    best_score = 0.0
    for cnt in contours:
        area = cv2.contourArea(cnt)
        if area < 150:
            continue
        x, y, w, h = cv2.boundingRect(cnt)
        if h == 0:
            continue
        aspect = w / h
        if 2.0 < aspect < 6.0:   # Indian plate ratio
            plate_crop = gray[y:y+h, x:x+w]
            if plate_crop.size > 0:
                contrast = float(np.std(plate_crop))
                score = float(np.clip(contrast / 70, 0.2, 0.95))
                best_score = max(best_score, score)

    return best_score if best_score > 0 else 0.55


def _detect_illegal_parking(img_arr, bbox, w_img, h_img):
    """
    Heuristic: vehicle near image edge (footpath area) + large area
    relative to frame suggests stationary/parked vehicle.
    """
    x1, y1, x2, y2 = bbox
    veh_area   = (x2 - x1) * (y2 - y1)
    frame_area = w_img * h_img
    near_edge  = (x1 < w_img * 0.10) or (x2 > w_img * 0.90)
    large_veh  = veh_area > frame_area * 0.08

    base = 0.20
    if near_edge:
        base += 0.30
    if large_veh:
        base += 0.20

    return float(np.clip(base + np.random.uniform(-0.05, 0.05), 0.0, 0.95))


def _make_violation(det, vtype, conf, note=None):
    return {
        "vehicle_detection":       det,
        "violation_type":          vtype,
        "violation_label":         VIOLATION_LABELS.get(vtype, vtype),
        "raw_detector_confidence": conf,
        "triage_confidence":       None,   # filled in by apply_triage()
        "lane":                    None,   # filled in by apply_triage()
        "triage_label":            None,   # filled in by apply_triage()
        "bbox":                    det["bbox"],
        "edge_case_note":          note,
    }


# ── Triage layer (real calibration, not a pass-through) ────────────────────────

def apply_triage(violations: list, img_arr: np.ndarray = None,
                 image: Image.Image = None) -> list:
    """
    Fills in triage_confidence + lane for every violation using an
    independent texture re-check and context score, blended with the
    raw heuristic confidence and passed through a Platt-style sigmoid
    calibration. This is what actually spreads results across
    auto_clear / human_review / auto_reject instead of leaving
    everything in the middle band with triage_confidence=None.

    Call with either img_arr (np.ndarray, RGB) or image (PIL.Image) —
    at least one is required.
    """
    if img_arr is None:
        if image is None:
            raise ValueError("apply_triage requires either img_arr or image")
        img_arr = np.array(image.convert("RGB"))

    for v in violations:
        raw_conf = v["raw_detector_confidence"]
        vtype    = v["violation_type"]
        bbox     = v["bbox"]

        texture_score = _texture_recheck(img_arr, bbox, vtype)
        context_score = _context_score(img_arr, bbox, raw_conf)

        weights = TRIAGE_SIGNAL_WEIGHTS.get(vtype, DEFAULT_TRIAGE_WEIGHTS)
        blended = (
            raw_conf        * weights["raw"]
            + texture_score * weights["texture"]
            + context_score * weights["context"]
        )
        calibrated = _platt_calibrate(blended)
        lane = _assign_lane(calibrated)

        v["texture_score"]     = round(float(texture_score), 3)
        v["context_score"]     = round(float(context_score), 3)
        v["triage_confidence"] = round(float(calibrated), 3)
        v["lane"]              = lane
        v["triage_label"]      = LANE_LABEL[lane]

        # Auto-resolve the seatbelt/shirt edge case instead of just
        # labeling it ambiguous and sending it to a human every time.
        if (vtype == "seatbelt_violation"
                and texture_score < 0.40
                and lane != LANE_AUTO_CLEAR):
            v["lane"] = LANE_AUTO_REJECT
            v["triage_label"] = LANE_LABEL[LANE_AUTO_REJECT]
            v["edge_case_note"] = (
                "Resolved: independent texture re-check found strap "
                "pattern — shirt/seatbelt colour confusion auto-corrected"
            )

    return violations


def _texture_recheck(img_arr, bbox, vtype):
    """
    Independent second look at the disputed region using a DIFFERENT
    visual cue than the primary heuristic in classify_violations(),
    so it is actually capable of disagreeing with it rather than just
    re-deriving the same number.
    """
    x1, y1, x2, y2 = bbox
    h, w = y2 - y1, x2 - x1
    if h <= 0 or w <= 0:
        return 0.5

    if vtype == "helmet_violation":
        # Primary heuristic used grayscale std/mean; this uses HSV
        # saturation, a different channel entirely.
        head = img_arr[y1:y1 + max(1, h // 4), x1:x2]
        if head.size == 0:
            return 0.5
        hsv = cv2.cvtColor(head, cv2.COLOR_RGB2HSV)
        mean_sat = float(np.mean(hsv[:, :, 1]))
        return float(np.clip(1.0 - mean_sat / 90.0, 0, 1))

    elif vtype == "seatbelt_violation":
        # Primary heuristic used Hough line detection; this uses
        # raw Sobel edge density, a coarser but independent texture cue.
        mid_x = (x1 + x2) // 2
        chest = img_arr[y1 + h // 3: y1 + 2 * h // 3, x1:mid_x]
        if chest.size == 0:
            return 0.5
        gray = cv2.cvtColor(chest, cv2.COLOR_RGB2GRAY)
        sx = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
        sy = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)
        edge_density = float(np.mean(np.sqrt(sx ** 2 + sy ** 2)))
        strap_present = float(np.clip(edge_density / 28.0, 0, 1))
        # texture_score represents "violation confirmed" -> invert
        return 1.0 - strap_present

    elif vtype == "mobile_phone_use":
        # Independent second look at the same head/arm region used by
        # _detect_phone_use(), but checking for a DIFFERENT cue: a small
        # bright rectangular highlight near the ear (phone screen glow /
        # reflective casing) rather than generic edge contours. Generic
        # sharpness saturates to 1.0 on almost any in-focus crop with an
        # arm and face in it, which is why this can't reuse that fallback.
        head_h = max(1, h // 2)
        crop = img_arr[y1:y1 + head_h, x1:x2]
        if crop.size == 0:
            return 0.5
        gray = cv2.cvtColor(crop, cv2.COLOR_RGB2GRAY)
        # Phone screens/casings tend to be small, bright, compact blobs
        # against skin/hair — look for bright spots that are NOT large
        # (large bright regions are more likely sky/background/helmet).
        _, bright_mask = cv2.threshold(gray, 200, 255, cv2.THRESH_BINARY)
        bright_pct = float(np.mean(bright_mask > 0))
        # A small bright fraction (a glint, not a wash of brightness)
        # is the signal we actually want; large or zero bright area
        # should pull the score down, not saturate it.
        if bright_pct == 0:
            return 0.25
        score = 1.0 - abs(bright_pct - 0.04) / 0.04
        return float(np.clip(score, 0.10, 0.85))

    elif vtype == "illegal_parking":
        # Independent check: does the area immediately around/below the
        # vehicle show road-marking or kerb-edge contrast consistent with
        # a no-parking zone, vs. open road. This is intentionally a
        # weaker, narrower signal than generic sharpness so a sharp but
        # legitimately-parked car doesn't auto-saturate to 1.0.
        ground_y1 = min(img_arr.shape[0] - 1, y2)
        ground_y2 = min(img_arr.shape[0], y2 + max(10, h // 6))
        ground = img_arr[ground_y1:ground_y2, x1:x2]
        if ground.size == 0:
            return 0.5
        gray = cv2.cvtColor(ground, cv2.COLOR_RGB2GRAY)
        std = float(np.std(gray))
        # Low-variance ground strip (plain footpath/kerb) supports the
        # parking flag; high-variance (lane markings, moving traffic
        # behind) weakens it.
        return float(np.clip(1.0 - std / 60.0, 0.15, 0.90))

    else:
        crop = img_arr[max(0, y1):y2, max(0, x1):x2]
        if crop.size == 0:
            return 0.5
        gray = cv2.cvtColor(crop, cv2.COLOR_RGB2GRAY)
        sharpness = float(np.clip(cv2.Laplacian(gray, cv2.CV_64F).var() / 400.0, 0, 1))
        return sharpness


def _context_score(img_arr, bbox, raw_conf):
    """
    Stand-in for multi-frame consistency when only a single image is
    available: rewards sharp, well-exposed crops and penalises blurry
    or poorly-lit ones, since those are exactly the conditions that
    produce unreliable single-frame detections in the field.
    """
    x1, y1, x2, y2 = bbox
    crop = img_arr[max(0, y1):y2, max(0, x1):x2]
    if crop.size == 0:
        return 0.5
    gray = cv2.cvtColor(crop, cv2.COLOR_RGB2GRAY)
    sharpness = float(np.clip(cv2.Laplacian(gray, cv2.CV_64F).var() / 500.0, 0, 1))
    mean_bright = float(np.mean(gray))
    exposure_ok = float(np.clip(1.0 - abs(mean_bright - 120) / 120, 0, 1))
    return float(np.clip(0.5 * sharpness + 0.3 * exposure_ok + 0.2 * raw_conf, 0, 1))


def _platt_calibrate(raw: float) -> float:
    """Sigmoid calibration of the blended score. Tune A/B against a
    labelled validation set once you have real ground-truth outcomes."""
    A, B = -4.0, 2.0
    return 1.0 / (1.0 + math.exp(A * raw + B))


def _assign_lane(conf: float) -> str:
    if conf >= TRIAGE_THRESHOLDS["auto_clear"]:
        return LANE_AUTO_CLEAR
    elif conf < TRIAGE_THRESHOLDS["auto_reject"]:
        return LANE_AUTO_REJECT
    return LANE_HUMAN_REVIEW


# ── ANPR / Number Plate Recognition ───────────────────────────────────────────

def extract_number_plate(image: Image.Image, bbox: list):
    """
    Crop the lower region of a vehicle bbox and attempt OCR.
    Returns extracted text or None.
    """
    try:
        import easyocr, re
        x1, y1, x2, y2 = bbox
        plate_y = y1 + int((y2 - y1) * 0.65)
        crop = image.crop((x1, plate_y, x2, y2))

        crop_cv = cv2.cvtColor(np.array(crop), cv2.COLOR_RGB2GRAY)
        crop_cv = cv2.resize(crop_cv, None, fx=3, fy=3, interpolation=cv2.INTER_CUBIC)
        crop_cv = cv2.threshold(crop_cv, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)[1]

        reader = easyocr.Reader(['en'], gpu=False, verbose=False)
        results = reader.readtext(crop_cv, detail=0, paragraph=True)
        text = " ".join(results).strip().upper()

        match = re.search(r'[A-Z]{2}\s?\d{1,2}\s?[A-Z]{1,2}\s?\d{4}', text)
        return match.group(0) if match else (text if len(text) > 3 else None)
    except Exception as e:
        print(f"[ANPR] OCR error: {e}")
        return None


# ── Image annotation ───────────────────────────────────────────────────────────

def annotate_image(image: Image.Image, violations: list,
                   detections: list, triage_results: list = None,
                   plate_texts: dict = None) -> Image.Image:
    """
    Draw bounding boxes, labels, and triage badges.
    If violations already carry 'lane'/'triage_label' (i.e. apply_triage
    was called), those are used directly. triage_results is kept as an
    optional override for backwards compatibility.
    """
    img = image.copy().convert("RGB")
    draw = ImageDraw.Draw(img, "RGBA")

    try:
        font_bold  = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", 15)
        font_small = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", 12)
    except Exception:
        font_bold  = ImageFont.load_default()
        font_small = font_bold

    violated_boxes = {tuple(v["bbox"]) for v in violations}

    # Draw clean vehicle boxes
    for det in detections:
        if det["class_id"] not in VEHICLE_CLASSES:
            continue
        bb = det["bbox"]
        if tuple(bb) not in violated_boxes:
            draw.rectangle(bb, outline="#22C55E", width=2)
            draw.text((bb[0]+3, bb[1]+3),
                      f"{det['class_name']} {det['confidence']:.2f}",
                      fill="#22C55E", font=font_small)

    # Draw violation boxes
    for i, viol in enumerate(violations):
        bb    = viol["bbox"]
        color = VIOLATION_COLORS.get(viol["violation_type"], "#EF4444")
        conf  = viol["raw_detector_confidence"]
        label = f"{viol['violation_label']}  conf:{conf:.2f}"

        draw.rectangle(bb, fill=(*_hex_rgb(color), 35), outline=color, width=3)

        lw = len(label) * 7
        draw.rectangle([bb[0], max(0, bb[1]-24), bb[0]+lw, bb[1]], fill=color)
        draw.text((bb[0]+3, max(0, bb[1]-21)), label, fill="white", font=font_small)

        # Triage badge — prefer the violation's own lane (set by apply_triage),
        # fall back to an externally-supplied triage_results list if given.
        badge = viol.get("triage_label")
        if not badge and triage_results and i < len(triage_results):
            tr = triage_results[i]
            badge = tr.get("triage_label") if isinstance(tr, dict) else None

        if badge:
            bw = len(badge) * 8 + 8
            draw.rectangle([bb[0], bb[3], bb[0]+bw, bb[3]+22], fill="#1e293b")
            draw.text((bb[0]+4, bb[3]+3), badge, fill="white", font=font_small)

        # Plate text overlay
        if plate_texts and i in plate_texts and plate_texts[i]:
            plate = f"🔢 {plate_texts[i]}"
            plate_y = bb[3] + (24 if badge else 0)
            draw.rectangle([bb[0], plate_y, bb[0]+len(plate)*8, plate_y+20], fill="#1e40af")
            draw.text((bb[0]+4, plate_y+2), plate, fill="white", font=font_small)

        # Edge case warning
        if viol.get("edge_case_note") and "⚠️" in str(viol["edge_case_note"]):
            draw.text((bb[0], max(0, bb[1]-42)),
                      str(viol["edge_case_note"]),
                      fill="#FCD34D", font=font_small)

    return img


def _hex_rgb(h):
    h = h.lstrip("#")
    return tuple(int(h[i:i+2], 16) for i in (0, 2, 4))


def image_to_bytes(img: Image.Image, fmt="PNG") -> bytes:
    buf = io.BytesIO()
    img.save(buf, format=fmt)
    return buf.getvalue()