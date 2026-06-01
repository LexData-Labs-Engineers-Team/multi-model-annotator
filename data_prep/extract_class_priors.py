# ============================================================
# data_prep/extract_class_priors.py
# Reads the CVAT XML used for training and derives per-class
# shape priors that can be enforced at inference time by
# shape_refiner.py — without retraining any model.
#
# Output: SAVE_DIR/class_priors.json
#
# Captured per polygon class:
#   - vertex-count distribution (min, p25, median, p75, p90, max)
#   - area-as-%-of-frame distribution (min, median, p90, max)
# Captured pairwise (between every two polygon classes):
#   - n_pairs   : number of (annotation, annotation) pairs observed
#                 within the same image
#   - mean_iou  : average mask IoU across pairs
#   - max_iou   : largest observed IoU
#   - relation  : "always_zero" | "mixed" | "always_high"
#
# Relation classification is intentionally conservative:
#   always_zero  if max_iou <  ZERO_EPS  (mutually exclusive)
#   always_high  if min_iou >= HIGH_THRESH (one contained in the other)
#   mixed        otherwise
#
# Run: python -m data_prep.extract_class_priors
# ============================================================

import os
import sys
import json
from collections import defaultdict

import numpy as np
import cv2
from tqdm import tqdm

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import config as cfg
from data_prep.split_annotations import parse_cvat_xml


# --- Relation classification thresholds ---
ZERO_EPS    = 0.01    # max_iou below this → always_zero
HIGH_THRESH = 0.50    # min_iou at-or-above this → always_high

# --- IoU rasterization cap (long side) ---
IOU_RASTER_MAX_DIM = 512


def _percentiles(values, ps):
    """Return percentiles dict or empty dict if no values."""
    if not values:
        return {}
    arr = np.array(values, dtype=np.float64)
    return {f"p{p}": float(np.percentile(arr, p)) for p in ps}


def _polygon_mask(pts, h, w, scale=1.0):
    """Rasterize polygon points (list of (x,y)) onto a mask.
    When scale < 1.0, rasterizes onto a smaller canvas for speed."""
    sh, sw = int(h * scale), int(w * scale)
    mask = np.zeros((sh, sw), dtype=np.uint8)
    if len(pts) < 3:
        return mask
    arr = (np.array(pts, dtype=np.float32) * scale).astype(np.int32).reshape(-1, 1, 2)
    cv2.fillPoly(mask, [arr], 1)
    return mask


def _polygon_bbox(pts):
    """Return (x1, y1, x2, y2) bounding box for a polygon."""
    arr = np.array(pts, dtype=np.float32)
    return float(arr[:, 0].min()), float(arr[:, 1].min()), \
           float(arr[:, 0].max()), float(arr[:, 1].max())


def _bboxes_overlap(a, b):
    """Check if two (x1,y1,x2,y2) bounding boxes overlap."""
    return a[0] < b[2] and b[0] < a[2] and a[1] < b[3] and b[1] < a[3]


def _iou(mask_a, mask_b):
    """Binary mask IoU."""
    inter = np.logical_and(mask_a, mask_b).sum()
    if inter == 0:
        return 0.0
    union = np.logical_or(mask_a, mask_b).sum()
    return float(inter) / float(union) if union else 0.0


def extract(xml_path, save_path):
    print(f"Reading CVAT XML : {xml_path}")
    images, _ = parse_cvat_xml(xml_path)
    print(f"Images parsed    : {len(images)}")

    # --- Per-class vertex/area collection (polygons) ---
    vertex_counts = defaultdict(list)
    area_pcts     = defaultdict(list)

    # --- Per-class vertex collection (polylines) ---
    pl_vertex_counts = defaultdict(list)

    # --- Pairwise IoU collection (polygons only) ---
    pair_ious = defaultdict(list)   # frozenset({a, b}) → list of IoU values

    n_images_with_polys = 0
    n_bbox_skipped = 0
    n_iou_computed = 0

    for img in tqdm(images, desc="Extracting class priors", unit="img"):
        # Polylines — collect vertex counts independently.
        for pl in img.get("polylines", []):
            pts = pl["points"]
            if len(pts) < 2:
                continue
            pl_vertex_counts[pl["label"]].append(int(len(pts)))

        polys = img.get("polygons", [])
        if not polys:
            continue
        n_images_with_polys += 1
        h = max(1, int(img.get("height", 0)))
        w = max(1, int(img.get("width",  0)))
        img_area = float(h * w)

        # Per-poly stats (always at full resolution — just contourArea, no raster).
        bboxes = []
        valid  = []
        for p in polys:
            pts = p["points"]
            if len(pts) < 3:
                bboxes.append(None)
                valid.append(False)
                continue
            vertex_counts[p["label"]].append(int(len(pts)))
            poly_arr = np.array(pts, dtype=np.float32)
            area     = float(abs(cv2.contourArea(poly_arr)))
            area_pcts[p["label"]].append(100.0 * area / img_area)
            bboxes.append(_polygon_bbox(pts))
            valid.append(True)

        # Pairwise IoUs — bbox pre-filter + downscaled rasterization.
        if len(polys) < 2:
            continue

        iou_scale = min(1.0, IOU_RASTER_MAX_DIM / max(h, w))

        rasters = [None] * len(polys)
        for i in range(len(polys)):
            for j in range(i + 1, len(polys)):
                if not valid[i] or not valid[j]:
                    continue
                a_lab = polys[i]["label"]
                b_lab = polys[j]["label"]
                if a_lab == b_lab:
                    continue

                if not _bboxes_overlap(bboxes[i], bboxes[j]):
                    key = tuple(sorted((a_lab, b_lab)))
                    pair_ious[key].append(0.0)
                    n_bbox_skipped += 1
                    continue

                if rasters[i] is None:
                    rasters[i] = _polygon_mask(polys[i]["points"], h, w, iou_scale)
                if rasters[j] is None:
                    rasters[j] = _polygon_mask(polys[j]["points"], h, w, iou_scale)

                key = tuple(sorted((a_lab, b_lab)))
                pair_ious[key].append(_iou(rasters[i], rasters[j]))
                n_iou_computed += 1

    print(f"IoU pairs computed: {n_iou_computed}  |  skipped (no bbox overlap): {n_bbox_skipped}")

    # --- Build polygon_classes section ---
    polygon_classes = {}
    for label, counts in vertex_counts.items():
        areas = area_pcts[label]
        polygon_classes[label] = {
            "n_annotations": len(counts),
            "vertex_count" : {
                "min"   : int(np.min(counts)),
                **_percentiles(counts, [25, 50, 75, 90]),
                "max"   : int(np.max(counts)),
                "mean"  : float(np.mean(counts)),
            },
            "area_pct"     : {
                "min"   : float(np.min(areas)),
                **_percentiles(areas, [50, 90]),
                "max"   : float(np.max(areas)),
                "mean"  : float(np.mean(areas)),
            },
        }

    # --- Build polyline_classes section ---
    polyline_classes = {}
    for label, counts in pl_vertex_counts.items():
        polyline_classes[label] = {
            "n_annotations": len(counts),
            "vertex_count" : {
                "min"  : int(np.min(counts)),
                **_percentiles(counts, [25, 50, 75, 90]),
                "max"  : int(np.max(counts)),
                "mean" : float(np.mean(counts)),
            },
        }

    # --- Build overlap_matrix section ---
    overlap_matrix = {}
    for (a, b), ious in pair_ious.items():
        arr     = np.array(ious, dtype=np.float64)
        mean_iou = float(arr.mean())
        min_iou  = float(arr.min())
        max_iou  = float(arr.max())
        if max_iou < ZERO_EPS:
            relation = "always_zero"
        elif min_iou >= HIGH_THRESH:
            relation = "always_high"
        else:
            relation = "mixed"
        overlap_matrix[f"{a}__{b}"] = {
            "n_pairs" : len(ious),
            "mean_iou": mean_iou,
            "min_iou" : min_iou,
            "max_iou" : max_iou,
            "relation": relation,
        }

    # --- Assemble + write ---
    out = {
        "extracted_from"      : xml_path,
        "n_images_total"      : len(images),
        "n_images_with_polys" : n_images_with_polys,
        "polygon_classes"     : polygon_classes,
        "polyline_classes"    : polyline_classes,
        "overlap_matrix"      : overlap_matrix,
        "thresholds"          : {
            "zero_eps"   : ZERO_EPS,
            "high_thresh": HIGH_THRESH,
        },
    }
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    with open(save_path, "w", encoding="utf-8") as f:
        json.dump(out, f, indent=2)
    print(f"Wrote priors     : {save_path}")

    # --- Console summary ---
    print()
    print("=" * 70)
    print("  Per-class polygon stats")
    print("=" * 70)
    print(f"  {'class':<20} {'n':>5} {'verts_min':>10} {'verts_p50':>10} "
          f"{'verts_p90':>10} {'verts_max':>10}")
    for lab in sorted(polygon_classes):
        v = polygon_classes[lab]["vertex_count"]
        print(f"  {lab:<20} {polygon_classes[lab]['n_annotations']:>5} "
              f"{v['min']:>10} {int(v['p50']):>10} "
              f"{int(v['p90']):>10} {v['max']:>10}")

    print()
    print("=" * 70)
    print("  Per-class polyline stats")
    print("=" * 70)
    if polyline_classes:
        print(f"  {'class':<20} {'n':>5} {'verts_min':>10} {'verts_p50':>10} "
              f"{'verts_p90':>10} {'verts_max':>10}")
        for lab in sorted(polyline_classes):
            v = polyline_classes[lab]["vertex_count"]
            print(f"  {lab:<20} {polyline_classes[lab]['n_annotations']:>5} "
                  f"{v['min']:>10} {int(v['p50']):>10} "
                  f"{int(v['p90']):>10} {v['max']:>10}")
    else:
        print("  (no polylines in dataset)")

    print()
    print("=" * 70)
    print("  Pairwise polygon class overlap")
    print("=" * 70)
    print(f"  {'pair':<45} {'n':>5} {'mean':>6} {'max':>6} relation")
    for k in sorted(overlap_matrix):
        m = overlap_matrix[k]
        print(f"  {k:<45} {m['n_pairs']:>5} {m['mean_iou']:>6.3f} "
              f"{m['max_iou']:>6.3f} {m['relation']}")


def main():
    save_path = os.path.join(cfg.SAVE_DIR, "class_priors.json")
    extract(cfg.CVAT_XML, save_path)


if __name__ == "__main__":
    main()
