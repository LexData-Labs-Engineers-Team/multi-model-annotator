# ============================================================
# output_fusion.py — Option 1: Output Fusion
# ============================================================
# Runs after master_test.py has generated predictions.
# Reconciles outputs from all models using spatial relationships:
#
#   1. Cross-model NMS      — suppresses duplicate detections
#                             where bbox and polygon overlap
#   2. Keypoint association — assigns keypoints to their nearest
#                             containing instance
#   3. Polyline association — links polylines that start/end
#                             inside a bbox to that instance
#   4. Tag attachment       — attaches image-level tags as
#                             metadata to all instances
#   5. Confidence re-ranking— re-orders final detections by
#                             fused confidence
#
# Input  : predictions_coco.json from master_test.py
# Output : fused_predictions_coco.json + fused_cvat.xml
#          + fused annotated images + fusion report
#
# Run: python output_fusion.py
# ============================================================

import os
import sys
import cv2
import json
import csv
import glob
import time
import datetime
import numpy as np
from collections import defaultdict
import xml.etree.ElementTree as ET
from xml.dom import minidom

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import config as cfg

# ============================================================
# --- USER SETTINGS ---
# ============================================================

# Folder of original test images (same as used in master_test.py)
IMAGES_DIR          = "/home/lexdata/Documents/LexAnnotate_demo/m_env/multi_pipeline/datasets/flower/images"

# Path to predictions_coco.json from master_test.py
INPUT_PREDICTIONS   = os.path.join(cfg.SAVE_DIR,
                                   "test_output", "predictions_coco.json")

# Output folder for fused results
OUTPUT_DIR          = os.path.join(cfg.SAVE_DIR, "fused_output")

# ---- Fusion thresholds ----

# IoU above this → treat bbox+polygon as duplicate, keep higher confidence
CROSS_NMS_IOU       = 0.5

# A keypoint is "inside" an instance if it falls within this
# expanded ratio of the bbox (1.0 = exact bbox, 1.2 = 20% margin)
KP_BBOX_EXPAND      = 1.15

# A polyline endpoint within this pixel distance of a bbox
# edge is considered associated with that instance
POLY_ENDPOINT_DIST  = 30   # pixels

# Minimum confidence to keep any prediction after fusion
MIN_CONF_AFTER_FUSION = 0.15

# ---- Visualization ----
MASK_ALPHA          = 0.40
FONT_SCALE          = 0.50
BOX_THICKNESS       = 2
POLY_THICKNESS      = 2
KP_RADIUS           = 5

# Supported image extensions
IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".tiff"}

# ============================================================
# --- Color palette ---
# ============================================================

_PALETTE = [
    (  0, 229, 160), (  0, 153, 255), (255, 107,  53),
    (192, 132, 252), (255, 215,   0), (255,  99, 132),
    ( 75, 192, 192), (255, 159,  64), (153, 102, 255),
    ( 54, 162, 235), (255, 206,  86), ( 46, 204, 113),
    (231,  76,  60), ( 52, 152, 219), (155,  89, 182),
]
CLASS_COLOR_MAP = {}

def build_color_map(class_names):
    global CLASS_COLOR_MAP
    CLASS_COLOR_MAP = {
        n: _PALETTE[i % len(_PALETTE)]
        for i, n in enumerate(class_names)
    }

def get_color(label):
    return CLASS_COLOR_MAP.get(label, (200, 200, 200))


# ============================================================
# --- Geometry helpers ---
# ============================================================

def bbox_iou(a, b):
    """
    IoU between two bboxes in [x1, y1, x2, y2] format.
    """
    ax1, ay1, ax2, ay2 = a
    bx1, by1, bx2, by2 = b
    ix1 = max(ax1, bx1)
    iy1 = max(ay1, by1)
    ix2 = min(ax2, bx2)
    iy2 = min(ay2, by2)
    iw  = max(0, ix2 - ix1)
    ih  = max(0, iy2 - iy1)
    inter = iw * ih
    area_a = max(0, ax2 - ax1) * max(0, ay2 - ay1)
    area_b = max(0, bx2 - bx1) * max(0, by2 - by1)
    union  = area_a + area_b - inter
    return inter / union if union > 0 else 0.0


def ann_to_xyxy(ann):
    """
    Convert COCO bbox [x, y, w, h] → [x1, y1, x2, y2].
    Falls back to polygon bounding box if bbox is missing.
    """
    bbox = ann.get("bbox", [])
    if bbox and len(bbox) == 4 and bbox[2] > 0 and bbox[3] > 0:
        x, y, w, h = bbox
        return [x, y, x + w, y + h]

    # Polygon fallback
    seg = ann.get("segmentation", [])
    if seg and isinstance(seg, list) and len(seg) > 0:
        flat = seg[0]
        if len(flat) >= 4:
            xs = flat[0::2]
            ys = flat[1::2]
            return [min(xs), min(ys), max(xs), max(ys)]

    return None


def point_in_expanded_bbox(px, py, bbox_xyxy, expand=1.0):
    """
    Check if point (px, py) lies within an optionally expanded bbox.
    expand=1.2 means bbox is 20% larger in each direction.
    """
    x1, y1, x2, y2 = bbox_xyxy
    cx = (x1 + x2) / 2
    cy = (y1 + y2) / 2
    hw = (x2 - x1) / 2 * expand
    hh = (y2 - y1) / 2 * expand
    return (cx - hw <= px <= cx + hw) and (cy - hh <= py <= cy + hh)


def polyline_endpoints(ann):
    """
    Returns the first and last point of a polyline annotation.
    """
    seg = ann.get("segmentation", [])
    if not seg or not isinstance(seg, list):
        return None, None
    flat = seg[0]
    if len(flat) < 4:
        return None, None
    start = (flat[0],  flat[1])
    end   = (flat[-2], flat[-1])
    return start, end


def point_to_bbox_dist(px, py, bbox_xyxy):
    """
    Minimum distance from point to the nearest edge of a bbox.
    Returns 0 if point is inside the bbox.
    """
    x1, y1, x2, y2 = bbox_xyxy
    dx = max(x1 - px, 0, px - x2)
    dy = max(y1 - py, 0, py - y2)
    return np.sqrt(dx ** 2 + dy ** 2)


# ============================================================
# --- Fusion Steps ---
# ============================================================

def step1_cross_model_nms(annotations):
    """
    Suppresses duplicate detections where a bbox and polygon
    overlap above CROSS_NMS_IOU threshold.
    Keeps the annotation with higher confidence score.

    Returns cleaned annotation list.
    """
    bboxes   = [a for a in annotations if a["shape_type"] == "bbox"]
    polygons = [a for a in annotations if a["shape_type"] == "polygon"]
    others   = [a for a in annotations
                if a["shape_type"] not in ("bbox", "polygon")]

    suppressed = set()   # annotation ids to remove

    for bbox_ann in bboxes:
        bbox_xyxy = ann_to_xyxy(bbox_ann)
        if bbox_xyxy is None:
            continue

        for poly_ann in polygons:
            if poly_ann["id"] in suppressed:
                continue
            poly_xyxy = ann_to_xyxy(poly_ann)
            if poly_xyxy is None:
                continue

            iou = bbox_iou(bbox_xyxy, poly_xyxy)
            if iou >= CROSS_NMS_IOU:
                # Same region detected by both models — suppress lower conf
                bbox_score = bbox_ann.get("score", 0)
                poly_score = poly_ann.get("score", 0)
                if bbox_score >= poly_score:
                    suppressed.add(poly_ann["id"])
                else:
                    suppressed.add(bbox_ann["id"])
                break   # one bbox can only suppress one polygon

    kept = [a for a in annotations if a["id"] not in suppressed]
    return kept, len(suppressed)


def step2_associate_keypoints(annotations):
    """
    Associates each keypoint with its nearest containing instance
    (bbox or polygon). Adds "instance_id" field to keypoint annotations.
    Keypoints with no containing instance are kept but marked unassociated.
    """
    instances  = [a for a in annotations
                  if a["shape_type"] in ("bbox", "polygon")]
    keypoints  = [a for a in annotations if a["shape_type"] == "keypoint"]
    others     = [a for a in annotations
                  if a["shape_type"] not in ("bbox", "polygon", "keypoint")]

    for kp in keypoints:
        kp_data = kp.get("keypoints", [])
        if not kp_data or len(kp_data) < 2:
            kp["instance_id"] = None
            continue

        px, py = kp_data[0], kp_data[1]
        best_inst  = None
        best_area  = float("inf")

        for inst in instances:
            xyxy = ann_to_xyxy(inst)
            if xyxy is None:
                continue
            if point_in_expanded_bbox(px, py, xyxy, KP_BBOX_EXPAND):
                # Among all containing instances, pick the smallest
                # (most specific) one
                x1, y1, x2, y2 = xyxy
                area = (x2 - x1) * (y2 - y1)
                if area < best_area:
                    best_area  = area
                    best_inst  = inst

        kp["instance_id"] = best_inst["id"] if best_inst else None

    return instances + keypoints + others


def step3_associate_polylines(annotations):
    """
    Associates polylines whose endpoints are within POLY_ENDPOINT_DIST
    pixels of a bbox boundary. Adds "instance_id" to polyline annotations.
    """
    instances = [a for a in annotations
                 if a["shape_type"] in ("bbox", "polygon")]
    polylines = [a for a in annotations if a["shape_type"] == "polyline"]
    others    = [a for a in annotations
                 if a["shape_type"] not in
                 ("bbox", "polygon", "polyline")]

    for pl in polylines:
        start, end = polyline_endpoints(pl)
        if start is None:
            pl["instance_id"] = None
            continue

        best_inst = None
        best_dist = float("inf")

        for inst in instances:
            xyxy = ann_to_xyxy(inst)
            if xyxy is None:
                continue
            # Check proximity of either endpoint to the bbox boundary
            d_start = point_to_bbox_dist(start[0], start[1], xyxy)
            d_end   = point_to_bbox_dist(end[0],   end[1],   xyxy)
            min_d   = min(d_start, d_end)
            if min_d < POLY_ENDPOINT_DIST and min_d < best_dist:
                best_dist = min_d
                best_inst = inst

        pl["instance_id"] = best_inst["id"] if best_inst else None

    return instances + polylines + others


def step4_attach_tags(annotations):
    """
    Tags are image-level predictions. Attaches tag labels as
    metadata to all non-tag annotations in the image so downstream
    consumers know what tags co-occur with each detection.
    Keeps tag annotations in the list as-is.
    """
    tags    = [a for a in annotations if a["shape_type"] == "tag"]
    others  = [a for a in annotations if a["shape_type"] != "tag"]
    tag_labels = [t["category_id"] for t in tags]

    for ann in others:
        ann["image_tags"] = tag_labels

    return others + tags


def step5_filter_confidence(annotations):
    """
    Removes any predictions below MIN_CONF_AFTER_FUSION.
    Tags are always kept regardless of score.
    """
    kept = [
        a for a in annotations
        if a["shape_type"] == "tag"
        or a.get("score", 1.0) >= MIN_CONF_AFTER_FUSION
    ]
    removed = len(annotations) - len(kept)
    return kept, removed


# ============================================================
# --- Full fusion pipeline per image ---
# ============================================================

def fuse_image_predictions(annotations):
    """
    Runs all 5 fusion steps on a single image's annotations.

    Returns:
        fused_anns : cleaned and enriched annotation list
        stats      : dict of per-step counts
    """
    stats = {}

    # Step 1 — cross-model NMS
    anns, n_suppressed = step1_cross_model_nms(annotations)
    stats["cross_nms_suppressed"] = n_suppressed

    # Step 2 — keypoint → instance association
    anns = step2_associate_keypoints(anns)
    stats["kp_associated"] = sum(
        1 for a in anns
        if a["shape_type"] == "keypoint"
        and a.get("instance_id") is not None
    )
    stats["kp_unassociated"] = sum(
        1 for a in anns
        if a["shape_type"] == "keypoint"
        and a.get("instance_id") is None
    )

    # Step 3 — polyline → instance association
    anns = step3_associate_polylines(anns)
    stats["poly_associated"] = sum(
        1 for a in anns
        if a["shape_type"] == "polyline"
        and a.get("instance_id") is not None
    )

    # Step 4 — tag attachment
    anns = step4_attach_tags(anns)

    # Step 5 — confidence filter
    anns, n_removed = step5_filter_confidence(anns)
    stats["low_conf_removed"] = n_removed

    stats["final_count"] = len(anns)
    return anns, stats


# ============================================================
# --- Visualization ---
# ============================================================

def draw_fused(image, annotations, cat_map):
    output = image.copy()

    # Masks first
    overlay = output.copy()
    for ann in annotations:
        if ann["shape_type"] != "polygon":
            continue
        label = cat_map.get(ann["category_id"], "unknown")
        color = get_color(label)
        seg   = ann.get("segmentation", [])
        if seg:
            pts = np.array(seg[0], dtype=np.float32).reshape(-1, 2)
            cv2.fillPoly(overlay, [pts.astype(np.int32)], color)
    output = cv2.addWeighted(overlay, MASK_ALPHA,
                             output, 1 - MASK_ALPHA, 0)

    for ann in annotations:
        ptype = ann["shape_type"]
        label = cat_map.get(ann["category_id"], "unknown")
        score = ann.get("score", 1.0)
        color = get_color(label)

        if ptype == "polygon":
            seg = ann.get("segmentation", [])
            if seg:
                pts = np.array(seg[0],
                               dtype=np.float32).reshape(-1, 2)
                cv2.polylines(output, [pts.astype(np.int32)],
                              True, color, 1)
                cx = int(pts[:, 0].mean())
                cy = int(pts[:, 1].mean())
                _draw_label(output, label, score, cx, cy, color)

        elif ptype == "bbox":
            bbox = ann.get("bbox", [])
            if bbox:
                x, y, w, h = [int(v) for v in bbox]
                cv2.rectangle(output, (x, y),
                              (x + w, y + h), color, BOX_THICKNESS)
                _draw_label(output, label, score, x, y, color)

        elif ptype == "polyline":
            seg = ann.get("segmentation", [])
            if seg:
                flat = seg[0]
                pts  = [(int(flat[i]), int(flat[i+1]))
                        for i in range(0, len(flat)-1, 2)]
                for k in range(len(pts) - 1):
                    cv2.line(output, pts[k], pts[k+1],
                             color, POLY_THICKNESS)
                for pt in pts:
                    cv2.circle(output, pt, 3, color, -1)
                if pts:
                    _draw_label(output, label, score,
                                pts[0][0], pts[0][1], color)

        elif ptype == "keypoint":
            kp = ann.get("keypoints", [])
            if kp and len(kp) >= 2:
                cx, cy = int(kp[0]), int(kp[1])
                cv2.circle(output, (cx, cy), KP_RADIUS, color, -1)
                cv2.circle(output, (cx, cy),
                           KP_RADIUS + 1, (0, 0, 0), 1)
                # Draw thin line to associated instance centroid
                inst_id = ann.get("instance_id")
                if inst_id is not None:
                    inst = next(
                        (a for a in annotations if a["id"] == inst_id),
                        None
                    )
                    if inst:
                        ixyxy = ann_to_xyxy(inst)
                        if ixyxy:
                            icx = int((ixyxy[0] + ixyxy[2]) / 2)
                            icy = int((ixyxy[1] + ixyxy[3]) / 2)
                            cv2.line(output, (cx, cy),
                                     (icx, icy), color, 1)

        elif ptype == "tag":
            pass   # drawn separately below

    # Tags as text block
    tag_anns = [a for a in annotations if a["shape_type"] == "tag"]
    if tag_anns:
        y_off = 20
        for ann in tag_anns:
            label = cat_map.get(ann["category_id"], "tag")
            score = ann.get("score", 1.0)
            color = get_color(label)
            text  = f"[TAG] {label} {score*100:.0f}%"
            (tw, th), _ = cv2.getTextSize(
                text, cv2.FONT_HERSHEY_SIMPLEX, FONT_SCALE, 1
            )
            cv2.rectangle(output,
                          (8, y_off - th - 4),
                          (8 + tw + 4, y_off + 2),
                          color, -1)
            cv2.putText(output, text, (10, y_off),
                        cv2.FONT_HERSHEY_SIMPLEX, FONT_SCALE,
                        (0, 0, 0), 1, cv2.LINE_AA)
            y_off += th + 8

    return output


def _draw_label(img, label, score, x, y, color):
    text        = f"{label} {score*100:.0f}%"
    (tw, th), _ = cv2.getTextSize(
        text, cv2.FONT_HERSHEY_SIMPLEX, FONT_SCALE, 1
    )
    y = max(y, th + 8)
    cv2.rectangle(img, (x, y - th - 6),
                  (x + tw + 4, y), color, -1)
    cv2.putText(img, text, (x + 2, y - 3),
                cv2.FONT_HERSHEY_SIMPLEX, FONT_SCALE,
                (0, 0, 0), 1, cv2.LINE_AA)


# ============================================================
# --- CVAT XML builder ---
# ============================================================

def build_cvat_xml(fused_by_image, images_meta, cat_map):
    root = ET.Element("annotations")
    ET.SubElement(root, "version").text = "1.1"
    meta     = ET.SubElement(root, "meta")
    task     = ET.SubElement(meta, "task")
    ET.SubElement(task, "name").text = "LexAnnotate Fused Predictions"
    ET.SubElement(task, "size").text = str(len(fused_by_image))
    ET.SubElement(task, "mode").text = "annotation"
    labels_el= ET.SubElement(task, "labels")
    all_labels = {cat_map.get(cid, f"class_{cid}")
                  for anns in fused_by_image.values()
                  for a in anns
                  for cid in [a["category_id"]]}
    for lname in sorted(all_labels):
        lbl = ET.SubElement(labels_el, "label")
        ET.SubElement(lbl, "name").text  = lname
        ET.SubElement(lbl, "color").text = "#ffffff"
        ET.SubElement(lbl, "type").text  = "any"

    for img_id, (fname, anns) in enumerate(fused_by_image.items()):
        meta_info = images_meta.get(fname, {})
        img_el    = ET.SubElement(root, "image")
        img_el.set("id",     str(img_id))
        img_el.set("name",   fname)
        img_el.set("width",  str(meta_info.get("width",  0)))
        img_el.set("height", str(meta_info.get("height", 0)))

        for ann in anns:
            ptype = ann["shape_type"]
            label = cat_map.get(ann["category_id"], "unknown")
            score = str(round(ann.get("score", 1.0), 4))

            if ptype == "bbox":
                bbox = ann.get("bbox", [])
                if not bbox:
                    continue
                x, y, w, h = bbox
                el = ET.SubElement(img_el, "box")
                el.set("label",    label)
                el.set("xtl",      f"{x:.2f}")
                el.set("ytl",      f"{y:.2f}")
                el.set("xbr",      f"{x+w:.2f}")
                el.set("ybr",      f"{y+h:.2f}")
                el.set("occluded", "0")
                _xml_attr(el, "score", score)
                _xml_attr(el, "fused", "true")

            elif ptype == "polygon":
                seg = ann.get("segmentation", [])
                if not seg:
                    continue
                pts_str = _flat_to_pts_str(seg[0])
                el = ET.SubElement(img_el, "polygon")
                el.set("label",    label)
                el.set("points",   pts_str)
                el.set("occluded", "0")
                _xml_attr(el, "score", score)
                _xml_attr(el, "fused", "true")

            elif ptype == "polyline":
                seg = ann.get("segmentation", [])
                if not seg:
                    continue
                pts_str  = _flat_to_pts_str(seg[0])
                el = ET.SubElement(img_el, "polyline")
                el.set("label",    label)
                el.set("points",   pts_str)
                el.set("occluded", "0")
                _xml_attr(el, "score", score)
                inst_id = ann.get("instance_id")
                _xml_attr(el, "instance_id",
                          str(inst_id) if inst_id else "none")

            elif ptype == "keypoint":
                kp = ann.get("keypoints", [])
                if not kp or len(kp) < 2:
                    continue
                el = ET.SubElement(img_el, "points")
                el.set("label",    label)
                el.set("points",   f"{kp[0]:.2f},{kp[1]:.2f}")
                el.set("occluded", "0")
                _xml_attr(el, "score", score)
                inst_id = ann.get("instance_id")
                _xml_attr(el, "instance_id",
                          str(inst_id) if inst_id else "none")

            elif ptype == "tag":
                el = ET.SubElement(img_el, "tag")
                el.set("label", label)
                _xml_attr(el, "score", score)

    raw      = ET.tostring(root, encoding="unicode")
    reparsed = minidom.parseString(raw)
    return reparsed.toprettyxml(indent="  ", encoding=None)


def _flat_to_pts_str(flat):
    pts = []
    for i in range(0, len(flat) - 1, 2):
        pts.append(f"{flat[i]:.2f},{flat[i+1]:.2f}")
    return ";".join(pts)


def _xml_attr(parent, name, value):
    attr = ET.SubElement(parent, "attribute")
    attr.set("name", name)
    attr.text = value


# ============================================================
# --- Main ---
# ============================================================

def main():
    print("\n" + "═" * 55)
    print("  LexAnnotate — Output Fusion (Option 1)")
    print("═" * 55)

    assert os.path.exists(INPUT_PREDICTIONS), \
        f"predictions_coco.json not found: {INPUT_PREDICTIONS}\n" \
        f"Run master_test.py first."

    os.makedirs(OUTPUT_DIR, exist_ok=True)
    os.makedirs(os.path.join(OUTPUT_DIR, "annotated"), exist_ok=True)

    # Load predictions
    print(f"\n  Loading predictions: {INPUT_PREDICTIONS}")
    with open(INPUT_PREDICTIONS) as f:
        coco = json.load(f)

    # Build lookup maps
    cat_map     = {c["id"]: c["name"] for c in coco["categories"]}
    all_classes = [c["name"] for c in coco["categories"]]
    build_color_map(all_classes)

    images_map  = {img["id"]: img for img in coco["images"]}
    images_meta = {
        img["file_name"]: {
            "width" : img["width"],
            "height": img["height"],
            "id"    : img["id"],
        }
        for img in coco["images"]
    }

    # Group annotations by image
    anns_by_image = defaultdict(list)
    for ann in coco["annotations"]:
        anns_by_image[ann["image_id"]].append(ann)

    print(f"  Images      : {len(images_map)}")
    print(f"  Annotations : {len(coco['annotations'])} (before fusion)")
    print(f"  Categories  : {all_classes}")

    # --------------------------------------------------------
    # Fuse per image
    # --------------------------------------------------------
    print(f"\n--- Running fusion ---")
    fused_by_image   = {}
    images_meta_fname= {}
    all_stats        = []
    total_before     = 0
    total_after      = 0
    start            = time.time()

    for img_info in coco["images"]:
        img_id   = img_info["id"]
        fname    = img_info["file_name"]
        anns     = anns_by_image.get(img_id, [])
        total_before += len(anns)

        fused_anns, stats = fuse_image_predictions(anns)
        total_after      += len(fused_anns)

        fused_by_image[fname]    = fused_anns
        images_meta_fname[fname] = img_info

        stats["image"]          = fname
        stats["before"]         = len(anns)
        stats["after"]          = len(fused_anns)
        all_stats.append(stats)

        print(f"  {fname:<35} "
              f"{len(anns):>3} → {len(fused_anns):>3} "
              f"(−{stats['cross_nms_suppressed']} dup, "
              f"−{stats['low_conf_removed']} low-conf, "
              f"{stats['kp_associated']} kp linked, "
              f"{stats['poly_associated']} pl linked)")

        # Draw and save annotated image
        img_path = os.path.join(IMAGES_DIR, fname)
        if os.path.exists(img_path):
            img = cv2.imread(img_path)
            if img is not None:
                fused_img = draw_fused(img, fused_anns, cat_map)
                cv2.imwrite(
                    os.path.join(OUTPUT_DIR, "annotated", fname),
                    fused_img
                )

    elapsed = time.time() - start

    # --------------------------------------------------------
    # Save fused COCO JSON
    # --------------------------------------------------------
    fused_coco = {
        "info"       : {"description": "LexAnnotate fused predictions"},
        "categories" : coco["categories"],
        "images"     : coco["images"],
        "annotations": [
            ann
            for anns in fused_by_image.values()
            for ann in anns
        ],
    }
    coco_path = os.path.join(OUTPUT_DIR, "fused_predictions_coco.json")
    with open(coco_path, "w") as f:
        json.dump(fused_coco, f, indent=2)

    # --------------------------------------------------------
    # Save fused CVAT XML
    # --------------------------------------------------------
    xml_str  = build_cvat_xml(fused_by_image, images_meta_fname, cat_map)
    xml_path = os.path.join(OUTPUT_DIR, "fused_predictions_cvat.xml")
    with open(xml_path, "w", encoding="utf-8") as f:
        f.write(xml_str)

    # --------------------------------------------------------
    # Save per-image stats CSV
    # --------------------------------------------------------
    stats_path = os.path.join(OUTPUT_DIR, "fusion_stats.csv")
    with open(stats_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=[
            "image", "before", "after",
            "cross_nms_suppressed", "low_conf_removed",
            "kp_associated", "kp_unassociated",
            "poly_associated", "final_count"
        ])
        writer.writeheader()
        writer.writerows(all_stats)

    # --------------------------------------------------------
    # Save fusion report
    # --------------------------------------------------------
    report_path = os.path.join(OUTPUT_DIR, "fusion_report.txt")
    with open(report_path, "w") as f:
        def w(line=""):
            f.write(line + "\n")
        w("═" * 55)
        w("  LexAnnotate — Output Fusion Report")
        w("═" * 55)
        w(f"  Date         : "
          f"{datetime.datetime.now().strftime('%Y-%m-%d %H:%M')}")
        w(f"  Input        : {INPUT_PREDICTIONS}")
        w(f"  Output       : {OUTPUT_DIR}")
        w()
        w("  Fusion settings:")
        w(f"    Cross-NMS IoU threshold  : {CROSS_NMS_IOU}")
        w(f"    Keypoint bbox expand     : {KP_BBOX_EXPAND}")
        w(f"    Polyline endpoint dist   : {POLY_ENDPOINT_DIST}px")
        w(f"    Min confidence threshold : {MIN_CONF_AFTER_FUSION}")
        w()
        w("  Results:")
        w(f"    Images processed         : {len(all_stats)}")
        w(f"    Annotations before fusion: {total_before}")
        w(f"    Annotations after fusion : {total_after}")
        w(f"    Total removed            : {total_before - total_after}")
        w(f"    Reduction                : "
          f"{(total_before-total_after)/max(total_before,1)*100:.1f}%")
        w()
        w("  By step:")
        w(f"    Cross-model NMS removed  : "
          f"{sum(s['cross_nms_suppressed'] for s in all_stats)}")
        w(f"    Low-conf removed         : "
          f"{sum(s['low_conf_removed'] for s in all_stats)}")
        w(f"    Keypoints associated     : "
          f"{sum(s['kp_associated'] for s in all_stats)}")
        w(f"    Keypoints unassociated   : "
          f"{sum(s['kp_unassociated'] for s in all_stats)}")
        w(f"    Polylines associated     : "
          f"{sum(s['poly_associated'] for s in all_stats)}")
        w()
        w(f"  Time         : {elapsed:.1f}s")
        w()
        w("  Output files:")
        w(f"    Annotated images : {OUTPUT_DIR}/annotated/")
        w(f"    Fused CVAT XML   : {xml_path}")
        w(f"    Fused COCO JSON  : {coco_path}")
        w(f"    Per-image stats  : {stats_path}")
        w(f"    This report      : {report_path}")
        w("═" * 55)

    # --------------------------------------------------------
    # Final print
    # --------------------------------------------------------
    print(f"\n{'═'*55}")
    print(f"  Fusion Complete")
    print(f"{'─'*55}")
    print(f"  Before : {total_before} annotations")
    print(f"  After  : {total_after} annotations")
    print(f"  Removed: {total_before - total_after} "
          f"({(total_before-total_after)/max(total_before,1)*100:.1f}%)")
    print(f"{'─'*55}")
    print(f"  Annotated images : {OUTPUT_DIR}/annotated/")
    print(f"  Fused CVAT XML   : {xml_path}")
    print(f"  Fused COCO JSON  : {coco_path}")
    print(f"  Fusion stats     : {stats_path}")
    print(f"  Fusion report    : {report_path}")
    print(f"{'═'*55}\n")


if __name__ == "__main__":
    main()
