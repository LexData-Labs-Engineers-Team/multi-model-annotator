# ============================================================
# master_test.py — Inference & Annotation Pipeline
# ============================================================
# Runs all configured models on a folder of images.
# Outputs:
#   - Annotated images (all annotation types drawn on one image)
#   - CVAT XML 1.1 file (reimportable into CVAT)
#   - COCO JSON of all predictions
#   - Per-image stats CSV
#   - Summary report
#
# Run: python master_test.py
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
from PIL import Image
from collections import defaultdict

import torch
import torch.nn.functional as F
import torchvision.transforms.functional as TF
import xml.etree.ElementTree as ET
from xml.dom import minidom

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import config as cfg

# ============================================================
# --- USER SETTINGS — Edit before running ---
# ============================================================

# Folder of images to run inference on
IMAGES_DIR      = cfg.IMG_DIR

# Output folder
OUTPUT_DIR      = os.path.join(cfg.SAVE_DIR, "test_output")

# ---- Model paths — set to None to skip that model ----
BBOX_MODEL_PATH     = os.path.join(cfg.BBOX_SAVE_DIR,
                                   "train", "weights", "best.pt")
POLYGON_MODEL_PATH  = os.path.join(cfg.POLYGON_SAVE_DIR,
                                   "train", "weights", "best.pt")
KEYPOINT_MODEL_PATH = os.path.join(cfg.KEYPOINT_SAVE_DIR, "best.pt")
KEYPOINT_SEG_PATH   = os.path.join(cfg.KEYPOINT_SAVE_DIR, "best_seg.pt")
POLYLINE_SEG_PATH   = os.path.join(cfg.POLYLINE_SAVE_DIR, "best_seg.pt")
TAG_MODEL_PATH      = os.path.join(cfg.TAG_SAVE_DIR,      "best.pt")
TAG_NAMES_PATH      = os.path.join(cfg.TAG_SAVE_DIR,      "tag_names.json")

# ---- Inference thresholds ----
BBOX_SCORE_THRESH   = cfg.YOLO_SCORE_THRESH
POLY_SCORE_THRESH   = cfg.YOLO_SCORE_THRESH
KP_SCORE_THRESH     = cfg.KP_SCORE_THRESH
KP_SEG_THRESH       = cfg.KP_SEG_THRESH
KP_SEG_NMS_MIN_DIST = cfg.KP_SEG_NMS_MIN_DIST
POLY_SEG_THRESH     = cfg.POLY_SEG_THRESH
TAG_THRESH          = cfg.TAG_SCORE_THRESH

# Polygons larger than this fraction of the frame are flagged as
# suspicious in the per-run diagnostic. Tune by reading the CSV
# output (polygon_diagnostic.csv) after a run.
POLYGON_HUGE_AREA_PCT = 30.0

# ---- Visualization settings ----
DRAW_BOXES          = True
DRAW_MASKS          = True
DRAW_KEYPOINTS      = True
DRAW_POLYLINES      = True
DRAW_TAGS           = True
MASK_ALPHA          = 0.40
BOX_THICKNESS       = 2
KP_RADIUS           = 5
POLY_THICKNESS      = 2
FONT_SCALE          = 0.50
LABEL_THICKNESS     = 1

# Supported image extensions
IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".tiff"}

# ============================================================
# --- Color palette (BGR) — one per class index ---
# ============================================================

_PALETTE = [
    (  0, 229, 160), (  0, 153, 255), (255, 107,  53),
    (192, 132, 252), (255, 215,   0), (255,  99, 132),
    ( 75, 192, 192), (255, 159,  64), (153, 102, 255),
    ( 54, 162, 235), (255, 206,  86), ( 46, 204, 113),
    (231,  76,  60), ( 52, 152, 219), (155,  89, 182),
]

# Built once from class names — same color per class across all images
CLASS_COLOR_MAP = {}

def build_color_map(class_names):
    global CLASS_COLOR_MAP
    CLASS_COLOR_MAP = {
        name: _PALETTE[i % len(_PALETTE)]
        for i, name in enumerate(class_names)
    }

def get_color(class_name):
    return CLASS_COLOR_MAP.get(class_name, (200, 200, 200))


# ============================================================
# --- Model loading helpers ---
# ============================================================

def model_exists(path):
    return path is not None and os.path.exists(path)


def load_yolo(path):
    from ultralytics import YOLO
    model = YOLO(path)
    return model


def load_heatmap_model(path, backbone, device):
    sys.path.insert(0, os.path.join(
        os.path.dirname(os.path.abspath(__file__)), "keypoint_model"
    ))
    from train_keypoint import KeypointHeatmapModel
    ckpt  = torch.load(path, map_location=device)
    model = KeypointHeatmapModel(backbone=backbone, pretrained=False)
    state = ckpt["model_state_dict"] if "model_state_dict" in ckpt else ckpt
    model.load_state_dict(state)
    model.to(device)
    model.eval()
    return model


def load_keypoint_seg(path, device):
    """Load the HRNet keypoint-seg checkpoint and return
    (model, classes, input_size). The checkpoint embeds its own
    backbone / classes / input_size / disk_radius config so we
    don't need cfg values to instantiate the model."""
    from keypoint_seg_model.train_keypoint_seg import HRNetSegModel
    ckpt = torch.load(path, map_location=device, weights_only=False)
    if "config" not in ckpt or "state_dict" not in ckpt:
        raise RuntimeError(
            f"Keypoint-seg checkpoint at {path} is missing 'config' or "
            f"'state_dict' — was it saved by train_keypoint_seg.py?"
        )
    meta       = ckpt["config"]
    backbone   = meta["backbone"]
    classes    = list(meta["classes"])
    input_size = int(meta["input_size"])
    model = HRNetSegModel(
        backbone=backbone, num_classes=len(classes), pretrained=False
    ).to(device)
    model.load_state_dict(ckpt["state_dict"])
    model.eval()
    return model, classes, input_size


def load_polyline_seg(path, device):
    """Load the HRNet polyline-seg checkpoint and return
    (model, classes, input_size). The checkpoint embeds its own
    backbone / classes / input_size config so we don't need cfg values."""
    from polyline_model_working.train_polyline_seg import HRNetSegModel
    ckpt = torch.load(path, map_location=device, weights_only=False)
    if "config" not in ckpt or "state_dict" not in ckpt:
        raise RuntimeError(
            f"Polyline-seg checkpoint at {path} is missing 'config' or "
            f"'state_dict' — was it saved by train_polyline_seg.py?"
        )
    meta       = ckpt["config"]
    backbone   = meta["backbone"]
    classes    = list(meta["classes"])
    input_size = int(meta["input_size"])
    model = HRNetSegModel(
        backbone=backbone, num_classes=len(classes), pretrained=False
    ).to(device)
    model.load_state_dict(ckpt["state_dict"])
    model.eval()
    return model, classes, input_size


def load_tag_model(path, num_tags):
    sys.path.insert(0, os.path.join(
        os.path.dirname(os.path.abspath(__file__)), "tag_model"
    ))
    from train_tag import TagClassifier
    ckpt  = torch.load(path, map_location="cpu")
    model = TagClassifier(num_tags=num_tags, pretrained=False)
    state = ckpt["model_state_dict"] if "model_state_dict" in ckpt else ckpt
    model.load_state_dict(state)
    model.eval()
    return model


# ============================================================
# --- Image preprocessing ---
# ============================================================

def preprocess(img_path, size):
    orig = cv2.imread(img_path)
    if orig is None:
        return None, None, None, None
    orig_h, orig_w = orig.shape[:2]
    pil  = Image.fromarray(cv2.cvtColor(orig, cv2.COLOR_BGR2RGB))
    pil  = pil.resize((size, size), Image.BILINEAR)
    t    = TF.to_tensor(pil)
    t    = TF.normalize(t, cfg.PIXEL_MEAN, cfg.PIXEL_STD)
    t    = t.unsqueeze(0)
    return t, orig, orig_h, orig_w


# ============================================================
# --- Inference per model ---
# ============================================================

def run_bbox(model, img_path, orig_h, orig_w, class_names):
    results = model.predict(
        source  = img_path,
        imgsz   = cfg.INPUT_SIZE,
        conf    = BBOX_SCORE_THRESH,
        iou     = cfg.YOLO_NMS_THRESH,
        device  = cfg.DEVICE,
        verbose = False,
    )
    detections = []
    for r in results:
        boxes = r.boxes
        if boxes is None:
            continue
        for i in range(len(boxes)):
            x1, y1, x2, y2 = boxes.xyxy[i].cpu().numpy().tolist()
            score  = float(boxes.conf[i].cpu())
            cls_id = int(boxes.cls[i].cpu())
            name   = class_names[cls_id] if cls_id < len(class_names) \
                     else f"class_{cls_id}"
            detections.append({
                "type" : "bbox",
                "box"  : [x1, y1, x2, y2],
                "score": score,
                "label": name,
            })
    return detections


def run_polygon(model, img_path, orig_h, orig_w, class_names):
    results = model.predict(
        source  = img_path,
        imgsz   = cfg.INPUT_SIZE,
        conf    = POLY_SCORE_THRESH,
        iou     = cfg.YOLO_NMS_THRESH,
        device  = cfg.DEVICE,
        verbose = False,
    )
    detections = []
    img_area   = float(orig_h * orig_w) or 1.0
    for r in results:
        if r.masks is None:
            continue
        boxes = r.boxes
        for i in range(len(r.masks)):
            # Mask polygon contour in original image coords
            mask_xy = r.masks.xy[i]   # (N, 2) pixel coords
            if len(mask_xy) < 3:
                continue
            # Scale to original image size
            scale_x = orig_w / r.orig_shape[1]
            scale_y = orig_h / r.orig_shape[0]
            pts     = [(float(p[0] * scale_x),
                        float(p[1] * scale_y)) for p in mask_xy]
            score   = float(boxes.conf[i].cpu()) if boxes else 1.0
            cls_id  = int(boxes.cls[i].cpu())    if boxes else 0
            name    = class_names[cls_id] if cls_id < len(class_names) \
                      else f"class_{cls_id}"
            # Diagnostic fields — read by main()'s polygon CSV writer.
            # Underscore prefix marks these as internal/debug so downstream
            # XML/COCO emitters don't need to filter them out.
            poly_arr   = np.array(pts, dtype=np.float32)
            poly_area  = float(cv2.contourArea(poly_arr))
            area_pct   = 100.0 * poly_area / img_area
            detections.append({
                "type"  : "polygon",
                "points": pts,
                "score" : score,
                "label" : name,
                "_diag_area_pct"  : area_pct,
                "_diag_n_vertices": int(len(pts)),
            })
    return detections


def run_keypoints(model, img_path, orig_h, orig_w, device):
    tensor, _, _, _ = preprocess(img_path, cfg.INPUT_SIZE)
    if tensor is None:
        return []
    tensor = tensor.to(device)
    with torch.no_grad():
        heatmap = model(tensor)
        heatmap = F.interpolate(
            heatmap, size=(orig_h, orig_w),
            mode="bilinear", align_corners=False
        )
        heatmap = heatmap.sigmoid().squeeze().cpu().numpy()

    # Extract local maxima as keypoints. NMS window is set from the
    # training heatmap sigma (≈ 2σ+1) so peaks closer than the training
    # blob radius collapse but real separated keypoints survive.
    from scipy.ndimage import maximum_filter
    nms_size      = max(3, 2 * int(cfg.KP_HEATMAP_SIGMA) + 1)
    local_max     = (heatmap == maximum_filter(heatmap, size=nms_size))
    above_thresh  = heatmap > KP_SCORE_THRESH
    peaks         = local_max & above_thresh
    coords        = np.argwhere(peaks)   # (N, 2) in (y, x)

    detections = []
    for (y, x) in coords:
        detections.append({
            "type" : "keypoint",
            "x"    : float(x),
            "y"    : float(y),
            "score": float(heatmap[y, x]),
            "label": "keypoint",
        })
    return detections


def run_keypoints_seg(model, classes, input_size, img_path,
                      orig_h, orig_w, device):
    """HRNet keypoint-seg inference.
    Forward → sigmoid → upsample to original resolution → per class:
    threshold → connected components → centroid + per-component max
    probability as score → greedy distance-based NMS. Emits one
    detection per surviving peak with its true class label."""
    from test_polyline_seg import _preprocess

    bgr = cv2.imread(img_path)
    if bgr is None:
        return []

    tensor = _preprocess(bgr, input_size, device)
    with torch.no_grad():
        logits = model(tensor)
        probs  = torch.sigmoid(logits).float()
        probs  = F.interpolate(probs, size=(orig_h, orig_w),
                               mode="bilinear", align_corners=False)
    probs = probs.squeeze(0).cpu().numpy()   # (C, H, W)

    # NMS distance is configured on the training-grid (input_size); scale
    # it back to the original image so it's correct after upsampling.
    nms_dist = KP_SEG_NMS_MIN_DIST * max(orig_h, orig_w) / float(input_size)

    detections = []
    for c, cname in enumerate(classes):
        prob_map = probs[c]
        mask     = (prob_map > KP_SEG_THRESH).astype(np.uint8)
        if mask.sum() == 0:
            continue
        n_comp, labels, stats, _ = cv2.connectedComponentsWithStats(
            mask, connectivity=8
        )
        # Candidate peaks: one per component, with per-component max prob.
        cands = []
        for comp_id in range(1, n_comp):
            ys, xs = np.where(labels == comp_id)
            if ys.size == 0:
                continue
            comp_probs = prob_map[ys, xs]
            i_max = int(np.argmax(comp_probs))
            cands.append((float(comp_probs[i_max]),
                          float(xs[i_max]), float(ys[i_max])))
        cands.sort(key=lambda t: t[0], reverse=True)
        # Greedy NMS by minimum distance.
        kept = []
        for score, x, y in cands:
            ok = True
            for _, kx, ky in kept:
                if (x - kx) ** 2 + (y - ky) ** 2 < nms_dist ** 2:
                    ok = False
                    break
            if ok:
                kept.append((score, x, y))
        for score, x, y in kept:
            detections.append({
                "type" : "keypoint",
                "x"    : x,
                "y"    : y,
                "score": score,
                "label": cname,
            })
    return detections


def run_polylines_seg(model, classes, input_size, img_path,
                      orig_h, orig_w, device):
    """HRNet polyline-seg inference.
    Forward → sigmoid → per-class probability map → threshold →
    skeletonize → trace 8-connected chains → Douglas-Peucker simplify.
    Reuses the helpers defined in test_polyline_seg.py to keep skeleton
    + trace logic in one place."""
    from test_polyline_seg import (
        _preprocess, _skeletonize, _trace_polylines, _simplify_chain,
        MIN_CHAIN_LEN,
    )

    bgr = cv2.imread(img_path)
    if bgr is None:
        return []

    tensor = _preprocess(bgr, input_size, device)
    with torch.no_grad():
        logits = model(tensor)
        probs  = torch.sigmoid(logits).float()
        probs  = F.interpolate(probs, size=(orig_h, orig_w),
                               mode="bilinear", align_corners=False)
    probs = probs.squeeze(0).cpu().numpy()   # (C, H, W)

    detections = []
    for c, cname in enumerate(classes):
        prob_map = probs[c]
        mask     = (prob_map > POLY_SEG_THRESH).astype(np.uint8) * 255
        if mask.sum() == 0:
            continue
        skel   = _skeletonize(mask)
        chains = _trace_polylines(skel)
        for ch in chains:
            if len(ch) < MIN_CHAIN_LEN:
                continue
            simp = _simplify_chain(ch)
            if len(simp) < 2:
                continue
            # Mean probability along the simplified chain — a reasonable
            # per-polyline confidence proxy.
            xs, ys = zip(*simp)
            xs = np.clip(np.array(xs, dtype=np.int32), 0, orig_w - 1)
            ys = np.clip(np.array(ys, dtype=np.int32), 0, orig_h - 1)
            score = float(prob_map[ys, xs].mean())
            detections.append({
                "type"  : "polyline",
                "points": [(float(x), float(y)) for (x, y) in simp],
                "score" : score,
                "label" : cname,
            })
    return detections


def run_tags(model, img_path, tag_names, device):
    tensor, _, _, _ = preprocess(img_path, cfg.INPUT_SIZE)
    if tensor is None:
        return []
    tensor = tensor.to(device)
    with torch.no_grad():
        logits = model(tensor)
        probs  = logits.sigmoid().squeeze()
        probs  = np.atleast_1d(probs.cpu().numpy())  # ← guarantees 1D

    detections = []
    for i, name in enumerate(tag_names):
        if probs[i] > TAG_THRESH:
            detections.append({
                "type" : "tag",
                "label": name,
                "score": float(probs[i]),
            })
    return detections


# ============================================================
# --- Visualization ---
# ============================================================

def draw_predictions(image, all_preds):
    output = image.copy()

    # --- Masks (drawn first, under everything) ---
    if DRAW_MASKS:
        overlay = output.copy()
        for pred in all_preds:
            if pred["type"] != "polygon":
                continue
            color = get_color(pred["label"])
            pts   = np.array(pred["points"], dtype=np.int32)
            cv2.fillPoly(overlay, [pts], color)
        output = cv2.addWeighted(overlay, MASK_ALPHA,
                                 output, 1 - MASK_ALPHA, 0)
        for pred in all_preds:
            if pred["type"] != "polygon":
                continue
            color = get_color(pred["label"])
            pts   = np.array(pred["points"], dtype=np.int32)
            cv2.polylines(output, [pts], isClosed=True,
                          color=color, thickness=1)

    # --- Bounding boxes ---
    if DRAW_BOXES:
        for pred in all_preds:
            if pred["type"] != "bbox":
                continue
            color        = get_color(pred["label"])
            x1,y1,x2,y2 = [int(v) for v in pred["box"]]
            cv2.rectangle(output, (x1, y1), (x2, y2),
                          color, BOX_THICKNESS)
            _draw_label(output, pred["label"], pred["score"],
                        x1, y1, color)

    # --- Polylines ---
    if DRAW_POLYLINES:
        for pred in all_preds:
            if pred["type"] != "polyline":
                continue
            color = get_color(pred["label"])
            pts   = [(int(p[0]), int(p[1])) for p in pred["points"]]
            for k in range(len(pts) - 1):
                cv2.line(output, pts[k], pts[k + 1],
                         color, POLY_THICKNESS)
            # Draw vertex dots
            for pt in pts:
                cv2.circle(output, pt, 3, color, -1)
            # Label at first point
            if pts:
                _draw_label(output, pred["label"], pred["score"],
                            pts[0][0], pts[0][1], color)

    # --- Keypoints ---
    if DRAW_KEYPOINTS:
        for pred in all_preds:
            if pred["type"] != "keypoint":
                continue
            color = get_color(pred["label"])
            cx    = int(pred["x"])
            cy    = int(pred["y"])
            cv2.circle(output, (cx, cy), KP_RADIUS, color, -1)
            cv2.circle(output, (cx, cy), KP_RADIUS + 1, (0, 0, 0), 1)

    # --- Tags (drawn as text block in top-left corner) ---
    if DRAW_TAGS:
        tag_preds = [p for p in all_preds if p["type"] == "tag"]
        if tag_preds:
            y_offset = 20
            for pred in tag_preds:
                color = get_color(pred["label"])
                text  = f"[TAG] {pred['label']} " \
                        f"{pred['score']*100:.0f}%"
                (tw, th), _ = cv2.getTextSize(
                    text, cv2.FONT_HERSHEY_SIMPLEX, FONT_SCALE, 1
                )
                cv2.rectangle(output,
                              (8, y_offset - th - 4),
                              (8 + tw + 4, y_offset + 2),
                              color, -1)
                cv2.putText(output, text, (10, y_offset),
                            cv2.FONT_HERSHEY_SIMPLEX,
                            FONT_SCALE, (0, 0, 0), LABEL_THICKNESS,
                            cv2.LINE_AA)
                y_offset += th + 8

    # --- Polygon labels (drawn last, on top) ---
    for pred in all_preds:
        if pred["type"] != "polygon":
            continue
        color = get_color(pred["label"])
        pts   = pred["points"]
        if pts:
            cx = int(sum(p[0] for p in pts) / len(pts))
            cy = int(sum(p[1] for p in pts) / len(pts))
            _draw_label(output, pred["label"], pred["score"],
                        cx, cy, color)

    return output


def _draw_label(img, label, score, x, y, color):
    text        = f"{label} {score*100:.0f}%"
    (tw, th), _ = cv2.getTextSize(
        text, cv2.FONT_HERSHEY_SIMPLEX, FONT_SCALE, 1
    )
    y = max(y, th + 8)
    cv2.rectangle(img,
                  (x, y - th - 6),
                  (x + tw + 4, y),
                  color, -1)
    cv2.putText(img, text, (x + 2, y - 3),
                cv2.FONT_HERSHEY_SIMPLEX, FONT_SCALE,
                (0, 0, 0), LABEL_THICKNESS, cv2.LINE_AA)


# ============================================================
# --- CVAT XML 1.1 builder ---
# ============================================================

def build_cvat_xml(all_image_preds, images_meta):
    """
    Builds a CVAT for Images 1.1 XML tree from all predictions.

    all_image_preds : dict — filename → list of prediction dicts
    images_meta     : dict — filename → {width, height}
    """
    root = ET.Element("annotations")
    ET.SubElement(root, "version").text = "1.1"

    # Meta block
    meta  = ET.SubElement(root, "meta")
    task  = ET.SubElement(meta, "task")
    ET.SubElement(task, "name").text = "LexAnnotate Predictions"
    ET.SubElement(task, "size").text = str(len(all_image_preds))
    ET.SubElement(task, "mode").text = "annotation"
    labels_el = ET.SubElement(task, "labels")

    # Collect all unique label names across all predictions
    all_labels = set()
    for preds in all_image_preds.values():
        for p in preds:
            all_labels.add(p["label"])
    for lname in sorted(all_labels):
        lbl = ET.SubElement(labels_el, "label")
        ET.SubElement(lbl, "name").text   = lname
        ET.SubElement(lbl, "color").text  = "#ffffff"
        ET.SubElement(lbl, "type").text   = "any"

    # Image blocks
    for img_id, (fname, preds) in enumerate(all_image_preds.items()):
        meta_info = images_meta.get(fname, {})
        w = meta_info.get("width",  0)
        h = meta_info.get("height", 0)

        img_el = ET.SubElement(root, "image")
        img_el.set("id",     str(img_id))
        img_el.set("name",   fname)
        img_el.set("width",  str(w))
        img_el.set("height", str(h))

        for pred in preds:
            ptype = pred["type"]
            label = pred["label"]
            score = str(round(pred["score"], 4))

            if ptype == "bbox":
                x1, y1, x2, y2 = pred["box"]
                el = ET.SubElement(img_el, "box")
                el.set("label",  label)
                el.set("xtl",    f"{x1:.2f}")
                el.set("ytl",    f"{y1:.2f}")
                el.set("xbr",    f"{x2:.2f}")
                el.set("ybr",    f"{y2:.2f}")
                el.set("occluded", "0")
                _add_attr(el, "score", score)

            elif ptype == "polygon":
                pts_str = _pts_to_str(pred["points"])
                el = ET.SubElement(img_el, "polygon")
                el.set("label",  label)
                el.set("points", pts_str)
                el.set("occluded", "0")
                _add_attr(el, "score", score)

            elif ptype == "polyline":
                pts_str = _pts_to_str(pred["points"])
                el = ET.SubElement(img_el, "polyline")
                el.set("label",  label)
                el.set("points", pts_str)
                el.set("occluded", "0")
                _add_attr(el, "score", score)

            elif ptype == "keypoint":
                el = ET.SubElement(img_el, "points")
                el.set("label",  label)
                el.set("points", f"{pred['x']:.2f},{pred['y']:.2f}")
                el.set("occluded", "0")
                _add_attr(el, "score", score)

            elif ptype == "tag":
                el = ET.SubElement(img_el, "tag")
                el.set("label",  label)
                _add_attr(el, "score", score)

    return root


def _pts_to_str(points):
    return ";".join(f"{p[0]:.2f},{p[1]:.2f}" for p in points)


def _add_attr(parent, name, value):
    attr = ET.SubElement(parent, "attribute")
    attr.set("name", name)
    attr.text = value


def prettify_xml(root):
    raw     = ET.tostring(root, encoding="unicode")
    reparsed= minidom.parseString(raw)
    return reparsed.toprettyxml(indent="  ", encoding=None)


# ============================================================
# --- COCO JSON builder ---
# ============================================================

def build_coco_json(all_image_preds, images_meta, class_names):
    coco = {
        "info"       : {"description": "LexAnnotate predictions"},
        "categories" : [{"id": i + 1, "name": n}
                        for i, n in enumerate(class_names)],
        "images"     : [],
        "annotations": []
    }
    cat_map   = {n: i + 1 for i, n in enumerate(class_names)}
    ann_id    = 1
    for img_id, (fname, preds) in enumerate(
        all_image_preds.items(), start=1
    ):
        meta = images_meta.get(fname, {})
        coco["images"].append({
            "id"       : img_id,
            "file_name": fname,
            "width"    : meta.get("width",  0),
            "height"   : meta.get("height", 0),
        })
        for pred in preds:
            ptype    = pred["type"]
            label    = pred["label"]
            cat_id   = cat_map.get(label, 1)

            ann = {
                "id"          : ann_id,
                "image_id"    : img_id,
                "category_id" : cat_id,
                "score"       : round(pred["score"], 4),
                "shape_type"  : ptype,
                "iscrowd"     : 0,
            }

            if ptype == "bbox":
                x1, y1, x2, y2 = pred["box"]
                ann["bbox"]        = [x1, y1, x2 - x1, y2 - y1]
                ann["area"]        = (x2 - x1) * (y2 - y1)
                ann["segmentation"]= []

            elif ptype == "polygon":
                flat = [v for p in pred["points"] for v in p]
                ann["segmentation"] = [flat]
                x1  = min(p[0] for p in pred["points"])
                y1  = min(p[1] for p in pred["points"])
                x2  = max(p[0] for p in pred["points"])
                y2  = max(p[1] for p in pred["points"])
                ann["bbox"] = [x1, y1, x2 - x1, y2 - y1]
                ann["area"] = (x2 - x1) * (y2 - y1)

            elif ptype == "polyline":
                flat = [v for p in pred["points"] for v in p]
                ann["segmentation"] = [flat]
                ann["bbox"]         = []
                ann["area"]         = 0

            elif ptype == "keypoint":
                ann["keypoints"]    = [pred["x"], pred["y"], 2]
                ann["bbox"]         = []
                ann["segmentation"] = []
                ann["area"]         = 0

            elif ptype == "tag":
                ann["bbox"]         = []
                ann["segmentation"] = []
                ann["area"]         = 0

            coco["annotations"].append(ann)
            ann_id += 1

    return coco


# ============================================================
# --- Main ---
# ============================================================

def main():
    print("\n" + "═" * 55)
    print("  LexAnnotate — Inference Pipeline")
    print("═" * 55)

    os.makedirs(OUTPUT_DIR, exist_ok=True)
    os.makedirs(os.path.join(OUTPUT_DIR, "annotated"), exist_ok=True)

    device = torch.device(
        cfg.DEVICE if torch.cuda.is_available()
        or cfg.DEVICE == "cpu" else "cpu"
    )
    print(f"  Device : {device}")

    # --------------------------------------------------------
    # Collect images
    # --------------------------------------------------------
    image_paths = sorted([
        p for ext in IMAGE_EXTS
        for p in glob.glob(os.path.join(IMAGES_DIR, f"*{ext}"))
    ])
    assert len(image_paths) > 0, \
        f"No images found in: {IMAGES_DIR}"
    print(f"  Images : {len(image_paths)} found")

    # --------------------------------------------------------
    # Load models — only those whose paths exist
    # --------------------------------------------------------
    print("\n--- Loading models ---")
    models_loaded = {}

    if model_exists(BBOX_MODEL_PATH):
        print(f"  ✓ Bbox model      : {BBOX_MODEL_PATH}")
        models_loaded["bbox"] = load_yolo(BBOX_MODEL_PATH)
    else:
        print(f"  ✗ Bbox model      : not found — skipped")

    if model_exists(POLYGON_MODEL_PATH):
        print(f"  ✓ Polygon model   : {POLYGON_MODEL_PATH}")
        models_loaded["polygon"] = load_yolo(POLYGON_MODEL_PATH)
    else:
        print(f"  ✗ Polygon model   : not found — skipped")

    kp_classes    = []
    kp_input_size = cfg.KP_SEG_INPUT_SIZE
    if model_exists(KEYPOINT_SEG_PATH):
        print(f"  ✓ Keypoint model  : {KEYPOINT_SEG_PATH} (seg)")
        (models_loaded["keypoint_seg"],
         kp_classes,
         kp_input_size) = load_keypoint_seg(KEYPOINT_SEG_PATH, device)
    # LEGACY KEYPOINT — kept for revert; see keypoint_seg_model/train_keypoint_seg.py.
    # Uncomment the elif block below to fall back to the heatmap model.
    # elif model_exists(KEYPOINT_MODEL_PATH):
    #     print(f"  ✓ Keypoint model  : {KEYPOINT_MODEL_PATH} (legacy)")
    #     models_loaded["keypoint"] = load_heatmap_model(
    #         KEYPOINT_MODEL_PATH, cfg.KP_BACKBONE, device
    #     )
    else:
        print(f"  ✗ Keypoint model  : not found — skipped")

    polyline_classes    = []
    polyline_input_size = cfg.POLY_SEG_INPUT_SIZE
    if model_exists(POLYLINE_SEG_PATH):
        print(f"  ✓ Polyline model  : {POLYLINE_SEG_PATH}")
        (models_loaded["polyline_seg"],
         polyline_classes,
         polyline_input_size) = load_polyline_seg(POLYLINE_SEG_PATH, device)
        print(f"    classes        : {polyline_classes}")
        print(f"    input size     : {polyline_input_size}")
    else:
        print(f"  ✗ Polyline model  : not found — skipped")

    tag_names = []
    if model_exists(TAG_MODEL_PATH) and model_exists(TAG_NAMES_PATH):
        with open(TAG_NAMES_PATH, encoding="utf-8") as f:
            tag_names = json.load(f)
        print(f"  ✓ Tag model       : {TAG_MODEL_PATH} "
              f"({len(tag_names)} tags)")
        models_loaded["tag"] = load_tag_model(
            TAG_MODEL_PATH, len(tag_names)
        )
    else:
        print(f"  ✗ Tag model       : not found — skipped")

    if not models_loaded:
        print("\n  ERROR: No models found. Check paths in USER SETTINGS.")
        return

    # Collect all class names across YOLO models for color map
    all_class_names = []
    for key in ["bbox", "polygon"]:
        if key in models_loaded:
            names = models_loaded[key].names
            if isinstance(names, dict):
                names = [names[i] for i in sorted(names)]
            for n in names:
                if n not in all_class_names:
                    all_class_names.append(n)
    # Keypoint-seg is per-class — give each label its own color.
    # The hardcoded "keypoint" entry below is only used by the legacy
    # heatmap path, which is class-agnostic; kept for revert.
    for name in kp_classes:
        if name not in all_class_names:
            all_class_names.append(name)
    if "keypoint" not in all_class_names:
        all_class_names.append("keypoint")
    # Polyline-seg is per-class — give each label its own color
    for name in polyline_classes:
        if name not in all_class_names:
            all_class_names.append(name)
    for name in tag_names:
        if name not in all_class_names:
            all_class_names.append(name)

    build_color_map(all_class_names)

    # --------------------------------------------------------
    # Inference loop
    # --------------------------------------------------------
    print(f"\n--- Running inference ---")
    all_image_preds  = {}   # filename → list of preds
    images_meta      = {}   # filename → {width, height}
    per_img_stats    = []
    polygon_diag_rows = []   # one row per polygon detection
    infer_start      = time.time()

    for img_path in image_paths:
        fname    = os.path.basename(img_path)
        orig_img = cv2.imread(img_path)
        if orig_img is None:
            print(f"  Skipping unreadable: {fname}")
            continue

        orig_h, orig_w = orig_img.shape[:2]
        images_meta[fname] = {"width": orig_w, "height": orig_h}

        all_preds = []

        # Bbox
        if "bbox" in models_loaded:
            names = models_loaded["bbox"].names
            if isinstance(names, dict):
                names = [names[i] for i in sorted(names)]
            preds = run_bbox(
                models_loaded["bbox"], img_path,
                orig_h, orig_w, names
            )
            all_preds.extend(preds)

        # Polygon
        if "polygon" in models_loaded:
            names = models_loaded["polygon"].names
            if isinstance(names, dict):
                names = [names[i] for i in sorted(names)]
            preds = run_polygon(
                models_loaded["polygon"], img_path,
                orig_h, orig_w, names
            )
            all_preds.extend(preds)

        # Keypoints (seg)
        if "keypoint_seg" in models_loaded:
            preds = run_keypoints_seg(
                models_loaded["keypoint_seg"], kp_classes, kp_input_size,
                img_path, orig_h, orig_w, device
            )
            all_preds.extend(preds)
        # LEGACY KEYPOINT — kept for revert.
        # if "keypoint" in models_loaded:
        #     preds = run_keypoints(
        #         models_loaded["keypoint"], img_path,
        #         orig_h, orig_w, device
        #     )
        #     all_preds.extend(preds)

        # Polylines (HRNet seg)
        if "polyline_seg" in models_loaded:
            preds = run_polylines_seg(
                models_loaded["polyline_seg"],
                polyline_classes,
                polyline_input_size,
                img_path, orig_h, orig_w, device
            )
            all_preds.extend(preds)

        # Tags
        if "tag" in models_loaded:
            preds = run_tags(
                models_loaded["tag"], img_path,
                tag_names, device
            )
            all_preds.extend(preds)

        all_image_preds[fname] = all_preds

        # Draw and save annotated image
        annotated = draw_predictions(orig_img, all_preds)
        cv2.imwrite(
            os.path.join(OUTPUT_DIR, "annotated", fname), annotated
        )

        # Stats
        type_counts = defaultdict(int)
        for p in all_preds:
            type_counts[p["type"]] += 1

        # Polygon diagnostic: capture per-polygon stats and flag
        # frames whose largest polygon exceeds POLYGON_HUGE_AREA_PCT.
        huge_in_this_img = []
        for p in all_preds:
            if p["type"] != "polygon":
                continue
            area_pct = p.get("_diag_area_pct", 0.0)
            nv       = p.get("_diag_n_vertices", 0)
            polygon_diag_rows.append({
                "image"     : fname,
                "label"     : p["label"],
                "score"     : f"{p['score']:.4f}",
                "n_vertices": nv,
                "area_pct"  : f"{area_pct:.2f}",
                "suspicious": "1" if area_pct >= POLYGON_HUGE_AREA_PCT else "0",
            })
            if area_pct >= POLYGON_HUGE_AREA_PCT:
                huge_in_this_img.append((p["label"], area_pct))
        if huge_in_this_img:
            tag = ", ".join(f"{lab}({pct:.1f}%)"
                            for lab, pct in huge_in_this_img)
            print(f"    ! huge polygon(s) in {fname}: {tag}")

        per_img_stats.append({
            "image"     : fname,
            "total"     : len(all_preds),
            "bbox"      : type_counts["bbox"],
            "polygon"   : type_counts["polygon"],
            "keypoint"  : type_counts["keypoint"],
            "polyline"  : type_counts["polyline"],
            "tag"       : type_counts["tag"],
        })

        print(f"  {fname:<35} "
              f"bbox:{type_counts['bbox']} "
              f"poly:{type_counts['polygon']} "
              f"kp:{type_counts['keypoint']} "
              f"pl:{type_counts['polyline']} "
              f"tag:{type_counts['tag']}")

    # --------------------------------------------------------
    # Save CVAT XML
    # --------------------------------------------------------
    xml_root = build_cvat_xml(all_image_preds, images_meta)
    xml_str  = prettify_xml(xml_root)
    xml_path = os.path.join(OUTPUT_DIR, "predictions_cvat.xml")
    with open(xml_path, "w", encoding="utf-8") as f:
        f.write(xml_str)

    # --------------------------------------------------------
    # Save COCO JSON
    # --------------------------------------------------------
    coco      = build_coco_json(
        all_image_preds, images_meta, all_class_names
    )
    coco_path = os.path.join(OUTPUT_DIR, "predictions_coco.json")
    with open(coco_path, "w", encoding="utf-8") as f:
        json.dump(coco, f, indent=2, ensure_ascii=False)

    # --------------------------------------------------------
    # Save per-image stats CSV
    # --------------------------------------------------------
    stats_path = os.path.join(OUTPUT_DIR, "per_image_stats.csv")
    with open(stats_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=[
            "image", "total", "bbox", "polygon",
            "keypoint", "polyline", "tag"
        ])
        writer.writeheader()
        writer.writerows(per_img_stats)

    # --------------------------------------------------------
    # Save polygon diagnostic CSV (Phase A)
    # --------------------------------------------------------
    diag_path = os.path.join(OUTPUT_DIR, "polygon_diagnostic.csv")
    with open(diag_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=[
            "image", "label", "score", "n_vertices",
            "area_pct", "suspicious",
        ])
        writer.writeheader()
        writer.writerows(polygon_diag_rows)

    # Build per-class polygon summary (count, mean/max area %, # suspicious)
    polygon_class_summary = defaultdict(lambda: {
        "count": 0, "area_sum": 0.0, "area_max": 0.0,
        "n_susp": 0, "n_vertices_sum": 0,
    })
    for row in polygon_diag_rows:
        a   = float(row["area_pct"])
        nv  = int(row["n_vertices"])
        agg = polygon_class_summary[row["label"]]
        agg["count"]          += 1
        agg["area_sum"]       += a
        agg["area_max"]        = max(agg["area_max"], a)
        agg["n_susp"]         += 1 if row["suspicious"] == "1" else 0
        agg["n_vertices_sum"] += nv

    # --------------------------------------------------------
    # Save summary report
    # --------------------------------------------------------
    total_time    = time.time() - infer_start
    total_dets    = sum(s["total"] for s in per_img_stats)
    summary_path  = os.path.join(OUTPUT_DIR, "summary_report.txt")

    with open(summary_path, "w", encoding="utf-8") as f:
        def w(line=""):
            f.write(line + "\n")
        w("═" * 55)
        w("  LexAnnotate — Inference Summary")
        w("═" * 55)
        w(f"  Date         : {datetime.datetime.now().strftime('%Y-%m-%d %H:%M')}")
        w(f"  Images dir   : {IMAGES_DIR}")
        w(f"  Output dir   : {OUTPUT_DIR}")
        w(f"  Device       : {device}")
        w()
        w("  Models used:")
        for key in ["bbox", "polygon", "keypoint_seg", "polyline_seg", "tag"]:
            status = "✓" if key in models_loaded else "✗"
            w(f"    {status}  {key}")
        w()
        w("  Results:")
        w(f"    Images processed : {len(per_img_stats)}")
        w(f"    Total detections : {total_dets}")
        w(f"    Avg per image    : "
          f"{total_dets / max(len(per_img_stats), 1):.1f}")
        w()
        w("  Detections by type:")
        for t in ["bbox", "polygon", "keypoint", "polyline", "tag"]:
            n = sum(s[t] for s in per_img_stats)
            w(f"    {t:<12}: {n}")
        w()
        # Polygon class diagnostic (Phase A) — see polygon_diagnostic.csv
        # for the full per-detection breakdown.
        if polygon_class_summary:
            w("  Polygon diagnostic (per class):")
            w(f"    {'class':<20} {'count':>6} {'mean_area%':>11} "
              f"{'max_area%':>10} {'mean_verts':>11} "
              f"{'n>'+str(int(POLYGON_HUGE_AREA_PCT))+'%':>9}")
            for lab, agg in sorted(polygon_class_summary.items()):
                mean_a = agg["area_sum"] / agg["count"]
                mean_v = agg["n_vertices_sum"] / agg["count"]
                w(f"    {lab:<20} {agg['count']:>6} {mean_a:>11.2f} "
                  f"{agg['area_max']:>10.2f} {mean_v:>11.1f} "
                  f"{agg['n_susp']:>9}")
            w()
        w(f"  Total time       : "
          f"{str(datetime.timedelta(seconds=int(total_time)))}")
        w(f"  Avg per image    : "
          f"{total_time / max(len(image_paths), 1):.2f}s")
        w()
        w("  Output files:")
        w(f"    Annotated images : {OUTPUT_DIR}/annotated/")
        w(f"    CVAT XML         : {xml_path}")
        w(f"    COCO JSON        : {coco_path}")
        w(f"    Per-image stats  : {stats_path}")
        w(f"    Polygon diag     : {diag_path}")
        w(f"    This report      : {summary_path}")
        w("═" * 55)

    # --------------------------------------------------------
    # Final print
    # --------------------------------------------------------
    print(f"\n{'═'*55}")
    print(f"  Inference Complete")
    print(f"{'─'*55}")
    print(f"  Images processed : {len(per_img_stats)}")
    print(f"  Total detections : {total_dets}")
    print(f"  Total time       : "
          f"{str(datetime.timedelta(seconds=int(total_time)))}")
    print(f"{'─'*55}")
    print(f"  Annotated images : {OUTPUT_DIR}/annotated/")
    print(f"  CVAT XML         : {xml_path}")
    print(f"  COCO JSON        : {coco_path}")
    print(f"  Per-image stats  : {stats_path}")
    print(f"  Summary report   : {summary_path}")
    print(f"{'═'*55}\n")


if __name__ == "__main__":
    main()
