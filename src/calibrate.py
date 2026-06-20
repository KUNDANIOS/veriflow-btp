import csv
import ast
import numpy as np
from PIL import Image
import sys, os

sys.path.insert(0, os.path.join(os.path.dirname(__file__)))
from signals import baseline_color, texture_edge, contour_shape

DATA_DIR = os.path.join(os.path.dirname(__file__), "..", "data")

rows = list(csv.DictReader(open(os.path.join(DATA_DIR, "labels.csv"))))

groups = {}
for r in rows:
    img = np.array(Image.open(os.path.join(DATA_DIR, "images", r["filename"])).convert("RGB"))
    region = ast.literal_eval(r["feature_region"])
    bp, bc, bd = baseline_color(img, r["category"], region)
    tp, tc, ts = texture_edge(img, r["category"], region)
    cp, cc, cn = contour_shape(img, r["category"], region)

    key = (r["category"], r["violation"], r["hard_case"])
    groups.setdefault(key, []).append((bd, ts, cn, bp, tp, cp, int(r["violation"])))

for key, vals in sorted(groups.items()):
    bds = [v[0] for v in vals]
    tss = [v[1] for v in vals]
    cns = [v[2] for v in vals]
    base_acc = np.mean([v[3] == v[6] for v in vals])
    tex_acc = np.mean([v[4] == v[6] for v in vals])
    con_acc = np.mean([v[5] == v[6] for v in vals])
    print(f"category={key[0]:9s} violation={key[1]} hard={key[2]}  n={len(vals):2d}  "
          f"color_dist[mean={np.mean(bds):6.1f}]  texture[mean={np.mean(tss):6.2f}]  contours[mean={np.mean(cns):4.1f}]  "
          f"baseline_acc={base_acc:.2f} texture_acc={tex_acc:.2f} contour_acc={con_acc:.2f}")
