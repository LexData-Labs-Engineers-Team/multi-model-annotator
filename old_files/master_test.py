# ============================================================
# unified_model/master_test.py — Unified Model Inference
# ============================================================
# Single forward pass through shared backbone — all heads
# run simultaneously with full contextual awareness.
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
import xml.etree.ElementTree as ET
from xml.dom import minidom

import torch
import torch.nn.functional as F
import torchvision.transforms as T
from scipy.ndimage import maximum_filter

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import config as cfg
from model import UnifiedModel
from model.polyline_head import build_edge_features

# ============================================================
# --- USER SETTINGS ---
# ============================================================

IMAGES_DIR      = "/home/lexdata/Documents/LexAnnotate_demo/datasets/lettuce_img"
MODEL_PATH      = os.path.join(cfg.CHECKPOINT_DIR, "model_final.pt")
OUTPUT_DIR      = cfg.TEST_OUTPUT_DIR

# Override thresholds here if needed
BBOX_THRESH     = cfg.BBOX_SCORE_THRESH
POLY_THRESH     = cfg.POLY_SCORE_THRESH
KP_THRESH       = cfg.KP_SCORE_THRESH
POLY_V_THRESH   = cfg.POLY_S1_THRESH
EDGE_THRESH     = cfg.EDGE_THRESH
TAG_THRESH      = cfg.TAG_THRESH
KP_NMS_WIN      = cfg.KP_NMS_WINDOW
MAX_KP          = cfg.MAX_KP_PER_IMAGE
MAX_POLY        = cfg.MAX_POLY_PER_IMAGE

IMAGE_EXTS      = {".jpg", ".jpeg", ".png", ".bmp", ".tiff"}

# ============================================================
# --- Color map ---
# ============================================================

_PALETTE = [
    (  0,229,160),(  0,153,255),(255,107, 53),(192,132,252),
    (255,215,  0),(255, 99,132),( 75,192,192),(255,159, 64),
    (153,102,255),( 54,162,235),(255,206, 86),( 46,204,113),
    (231, 76, 60),( 52,152,219),(155, 89,182),
]
CLASS_COLOR_MAP = {}

def build_color_map(labels):
    global CLASS_COLOR_MAP
    CLASS_COLOR_MAP = {
        l: _PALETTE[i % len(_PALETTE)]
        for i, l in enumerate(labels)
    }

def get_color(label):
    return CLASS_COLOR_MAP.get(label, (200, 200, 200))

# ============================================================
# --- Load model ---
# ============================================================

def load_model(model_path, device):
    ckpt       = torch.load(model_path, map_location="cpu")
    all_labels = ckpt["all_labels"]
    config     = ckpt.get("config", {})
    num_classes= len(all_labels)

    model = UnifiedModel(
        backbone         = config.get("backbone", cfg.BACKBONE),
        pretrained       = False,
        fpn_channels     = config.get("fpn_channels",
                                      cfg.FPN_OUT_CHANNELS),
        num_bbox_classes = num_classes,
        num_poly_classes = num_classes,
        num_tags         = num_classes,
        anchors          = cfg.BBOX_ANCHORS,
        input_size       = config.get("input_size", cfg.INPUT_SIZE),
        edge_hidden      = cfg.POLY_S2_HIDDEN,
        tag_dropout      = cfg.TAG_DROPOUT,
        w_bbox_obj=1, w_bbox_cls=1, w_bbox_box=1,
        w_poly_mask=1, w_poly_dice=1,
        w_kp_heatmap=1, w_pl_heatmap=1, w_pl_edge=1, w_tag=1,
        active_heads     = set(ckpt.get("active_heads", [
            "bbox","polygon","keypoint","polyline","tag"
        ])),
    )
    model.load_state_dict(ckpt["model_state_dict"])
    model.to(device).eval()
    return model, all_labels


# ============================================================
# --- Preprocessing ---
# ============================================================

def preprocess(img_path, input_size):
    orig = cv2.imread(img_path)
    if orig is None:
        return None, None, None, None
    orig_h, orig_w = orig.shape[:2]
    pil  = Image.fromarray(cv2.cvtColor(orig, cv2.COLOR_BGR2RGB))
    tf   = T.Compose([
        T.Resize((input_size, input_size)),
        T.ToTensor(),
        T.Normalize([0.485,0.456,0.406],[0.229,0.224,0.225]),
    ])
    tensor = tf(pil).unsqueeze(0)
    return tensor, orig, orig_h, orig_w


# ============================================================
# --- Post-processing per head ---
# ============================================================

def decode_bbox(bbox_preds, all_labels, anchors,
                input_size, orig_h, orig_w, thresh):
    """Decode YOLO predictions → list of detection dicts."""
    detections = []
    for scale_idx, pred in enumerate(bbox_preds):
        B, A, H, W, nc5 = pred.shape
        nc       = nc5 - 5
        anch     = torch.tensor(anchors[scale_idx],
                                dtype=torch.float32) / input_size

        pred_xy  = pred[0, :, :, :, :2].sigmoid()
        pred_wh  = pred[0, :, :, :, 2:4]
        pred_obj = pred[0, :, :, :, 4].sigmoid()
        pred_cls = pred[0, :, :, :, 5:].sigmoid()

        for a in range(A):
            for j in range(H):
                for i in range(W):
                    obj = pred_obj[a, j, i].item()
                    if obj < thresh:
                        continue
                    cx = (pred_xy[a,j,i,0].item() + i) / W
                    cy = (pred_xy[a,j,i,1].item() + j) / H
                    aw, ah = anch[a].tolist()
                    bw = torch.exp(pred_wh[a,j,i,0]).item() * aw
                    bh = torch.exp(pred_wh[a,j,i,1]).item() * ah

                    cls_scores = pred_cls[a,j,i]
                    cls_id     = cls_scores.argmax().item()
                    cls_score  = cls_scores[cls_id].item()
                    score      = obj * cls_score

                    if score < thresh:
                        continue

                    x1 = max(0, int((cx - bw/2) * orig_w))
                    y1 = max(0, int((cy - bh/2) * orig_h))
                    x2 = min(orig_w, int((cx + bw/2) * orig_w))
                    y2 = min(orig_h, int((cy + bh/2) * orig_h))

                    label = all_labels[cls_id] \
                            if cls_id < len(all_labels) \
                            else f"class_{cls_id}"
                    detections.append({
                        "type" : "bbox",
                        "box"  : [x1, y1, x2, y2],
                        "score": score,
                        "label": label,
                    })
    return detections


def decode_polygon(seg_logits, all_labels,
                   orig_h, orig_w, thresh):
    """Decode segmentation logits → polygon contour dicts."""
    import cv2 as cv
    probs      = seg_logits[0].softmax(dim=0).cpu().numpy()
    nc         = probs.shape[0] - 1   # exclude background
    detections = []

    for c in range(nc):
        mask = (probs[c] > thresh).astype(np.uint8)
        if mask.sum() == 0:
            continue
        mask_resized = cv.resize(
            mask, (orig_w, orig_h), interpolation=cv.INTER_NEAREST
        )
        contours, _ = cv.findContours(
            mask_resized, cv.RETR_EXTERNAL,
            cv.CHAIN_APPROX_SIMPLE
        )
        label = all_labels[c] if c < len(all_labels) else f"class_{c}"
        for cnt in contours:
            if len(cnt) < 3:
                continue
            pts = [(float(p[0][0]), float(p[0][1]))
                   for p in cnt]
            detections.append({
                "type"  : "polygon",
                "points": pts,
                "score" : float(probs[c].max()),
                "label" : label,
            })
    return detections


def decode_keypoints(heatmap_logits, orig_h, orig_w,
                     thresh, nms_win, max_kp):
    heatmap = heatmap_logits[0, 0].sigmoid().cpu().numpy()
    heatmap_resized = cv2.resize(
        heatmap, (orig_w, orig_h)
    )
    local_max    = (heatmap_resized ==
                    maximum_filter(heatmap_resized, size=nms_win))
    above_thresh = heatmap_resized > thresh
    peaks        = np.argwhere(local_max & above_thresh)

    detections = []
    for (y, x) in peaks:
        detections.append({
            "type" : "keypoint",
            "x"    : float(x),
            "y"    : float(y),
            "score": float(heatmap_resized[y, x]),
            "label": "keypoint",
        })

    detections = sorted(detections,
                        key=lambda d: d["score"], reverse=True)
    return detections[:max_kp]


def decode_polylines(vertex_logits, edge_mlp,
                     orig_h, orig_w, v_thresh,
                     edge_thresh, max_dist, max_poly, device):
    heatmap = vertex_logits[0, 0].sigmoid().cpu().numpy()
    heatmap_resized = cv2.resize(heatmap, (orig_w, orig_h))
    local_max    = (heatmap_resized ==
                    maximum_filter(heatmap_resized, size=35))
    above_thresh = heatmap_resized > v_thresh
    peaks        = np.argwhere(local_max & above_thresh)

    if len(peaks) < 2:
        return []

    verts_norm = [(float(x)/orig_w, float(y)/orig_h)
                  for (y, x) in peaks]

    # Build edge candidates
    edges = []
    for i in range(len(verts_norm)):
        for j in range(len(verts_norm)):
            if i == j:
                continue
            x1, y1 = verts_norm[i]
            x2, y2 = verts_norm[j]
            dist   = np.sqrt((x2-x1)**2 + (y2-y1)**2)
            if dist > max_dist:
                continue
            feat  = build_edge_features(x1, y1, x2, y2)
            feat_t= torch.tensor(feat,
                                 dtype=torch.float32).unsqueeze(0)
            feat_t= feat_t.to(device)
            with torch.no_grad():
                prob = edge_mlp.forward_edge(feat_t).sigmoid().item()
            if prob > edge_thresh:
                edges.append((i, j, prob))

    if not edges:
        return []

    # Reconstruct polylines
    adj = defaultdict(list)
    for (i, j, p) in edges:
        adj[i].append((j, p))
        adj[j].append((i, p))

    visited_e = set()
    visited_n = set()
    polylines  = []
    starts     = [n for n in adj if len(adj[n]) == 1] \
                 or list(adj.keys())

    for start in starts:
        if start in visited_n:
            continue
        path = [start]
        visited_n.add(start)
        cur = start
        while True:
            nbrs = [(nb, p) for (nb, p) in adj[cur]
                    if (cur, nb) not in visited_e
                    and nb not in visited_n]
            if not nbrs:
                break
            nxt, _ = max(nbrs, key=lambda x: x[1])
            visited_e.add((cur, nxt))
            visited_e.add((nxt, cur))
            visited_n.add(nxt)
            path.append(nxt)
            cur = nxt
        if len(path) >= 2:
            pts = [(verts_norm[n][0]*orig_w,
                    verts_norm[n][1]*orig_h)
                   for n in path]
            polylines.append({
                "type"  : "polyline",
                "points": pts,
                "score" : 1.0,
                "label" : "polyline",
            })

    polylines = sorted(polylines,
                       key=lambda x: len(x["points"]),
                       reverse=True)
    return polylines[:max_poly]


def decode_tags(tag_logits, all_labels, thresh):
    probs = tag_logits[0].sigmoid().cpu().numpy()
    probs = np.atleast_1d(probs)
    detections = []
    for i, label in enumerate(all_labels):
        if probs[i] > thresh:
            detections.append({
                "type" : "tag",
                "label": label,
                "score": float(probs[i]),
            })
    return detections


# ============================================================
# --- Visualization ---
# ============================================================

def draw_predictions(image, all_preds):
    output = image.copy()
    overlay = output.copy()

    for pred in all_preds:
        if pred["type"] != "polygon":
            continue
        color = get_color(pred["label"])
        pts   = np.array(pred["points"], dtype=np.int32)
        cv2.fillPoly(overlay, [pts], color)
    output = cv2.addWeighted(overlay, 0.4, output, 0.6, 0)

    for pred in all_preds:
        ptype = pred["type"]
        color = get_color(pred["label"])
        score = pred.get("score", 1.0)

        if ptype == "polygon":
            pts = np.array(pred["points"], dtype=np.int32)
            cv2.polylines(output, [pts], True, color, 1)
            cx = int(pts[:,0].mean())
            cy = int(pts[:,1].mean())
            _draw_label(output, pred["label"], score, cx, cy, color)

        elif ptype == "bbox":
            x1,y1,x2,y2 = [int(v) for v in pred["box"]]
            cv2.rectangle(output, (x1,y1), (x2,y2), color, 2)
            _draw_label(output, pred["label"], score, x1, y1, color)

        elif ptype == "polyline":
            pts = [(int(p[0]),int(p[1])) for p in pred["points"]]
            for k in range(len(pts)-1):
                cv2.line(output, pts[k], pts[k+1], color, 2)
            for pt in pts:
                cv2.circle(output, pt, 3, color, -1)
            if pts:
                _draw_label(output, pred["label"], score,
                            pts[0][0], pts[0][1], color)

        elif ptype == "keypoint":
            cx, cy = int(pred["x"]), int(pred["y"])
            cv2.circle(output, (cx,cy), 5, color, -1)
            cv2.circle(output, (cx,cy), 6, (0,0,0), 1)

    tag_preds = [p for p in all_preds if p["type"] == "tag"]
    if tag_preds:
        y_off = 20
        for pred in tag_preds:
            color = get_color(pred["label"])
            text  = f"[TAG] {pred['label']} {pred['score']*100:.0f}%"
            (tw,th),_ = cv2.getTextSize(
                text, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1
            )
            cv2.rectangle(output, (8,y_off-th-4),
                          (8+tw+4,y_off+2), color, -1)
            cv2.putText(output, text, (10,y_off),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                        (0,0,0), 1, cv2.LINE_AA)
            y_off += th + 8

    return output


def _draw_label(img, label, score, x, y, color):
    text = f"{label} {score*100:.0f}%"
    (tw,th),_ = cv2.getTextSize(
        text, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1
    )
    y = max(y, th+8)
    cv2.rectangle(img, (x,y-th-6), (x+tw+4,y), color, -1)
    cv2.putText(img, text, (x+2,y-3),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                (0,0,0), 1, cv2.LINE_AA)


# ============================================================
# --- CVAT XML + COCO JSON builders (same as pipeline) ---
# ============================================================

def save_cvat_xml(all_image_preds, images_meta, output_path):
    root = ET.Element("annotations")
    ET.SubElement(root, "version").text = "1.1"
    meta  = ET.SubElement(root, "meta")
    task  = ET.SubElement(meta, "task")
    ET.SubElement(task, "name").text = "Unified Model Predictions"

    for img_id, (fname, preds) in enumerate(
        all_image_preds.items()
    ):
        m      = images_meta.get(fname, {})
        img_el = ET.SubElement(root, "image")
        img_el.set("id",     str(img_id))
        img_el.set("name",   fname)
        img_el.set("width",  str(m.get("width",  0)))
        img_el.set("height", str(m.get("height", 0)))

        for pred in preds:
            ptype = pred["type"]
            label = pred["label"]
            score = str(round(pred.get("score", 1.0), 4))

            if ptype == "bbox":
                x1,y1,x2,y2 = pred["box"]
                el = ET.SubElement(img_el, "box")
                el.set("label", label)
                el.set("xtl", f"{x1:.2f}")
                el.set("ytl", f"{y1:.2f}")
                el.set("xbr", f"{x2:.2f}")
                el.set("ybr", f"{y2:.2f}")
                el.set("occluded", "0")
                _attr(el, "score", score)

            elif ptype == "polygon":
                pts_str = ";".join(
                    f"{p[0]:.2f},{p[1]:.2f}"
                    for p in pred["points"]
                )
                el = ET.SubElement(img_el, "polygon")
                el.set("label", label)
                el.set("points", pts_str)
                el.set("occluded", "0")
                _attr(el, "score", score)

            elif ptype == "polyline":
                pts_str = ";".join(
                    f"{p[0]:.2f},{p[1]:.2f}"
                    for p in pred["points"]
                )
                el = ET.SubElement(img_el, "polyline")
                el.set("label", label)
                el.set("points", pts_str)
                el.set("occluded", "0")
                _attr(el, "score", score)

            elif ptype == "keypoint":
                el = ET.SubElement(img_el, "points")
                el.set("label",  label)
                el.set("points",
                       f"{pred['x']:.2f},{pred['y']:.2f}")
                el.set("occluded", "0")
                _attr(el, "score", score)

            elif ptype == "tag":
                el = ET.SubElement(img_el, "tag")
                el.set("label", label)
                _attr(el, "score", score)

    raw      = ET.tostring(root, encoding="unicode")
    reparsed = minidom.parseString(raw)
    xml_str  = reparsed.toprettyxml(indent="  ", encoding=None)
    with open(output_path, "w", encoding="utf-8") as f:
        f.write(xml_str)


def _attr(parent, name, value):
    a = ET.SubElement(parent, "attribute")
    a.set("name", name)
    a.text = value


# ============================================================
# --- Main ---
# ============================================================

def main():
    print("\n" + "═"*55)
    print("  Unified Model — Inference")
    print("═"*55)

    os.makedirs(OUTPUT_DIR, exist_ok=True)
    os.makedirs(os.path.join(OUTPUT_DIR, "annotated"), exist_ok=True)

    device = torch.device(
        cfg.DEVICE if torch.cuda.is_available()
        or cfg.DEVICE == "cpu" else "cpu"
    )
    print(f"  Device : {device}")

    # Load model
    assert os.path.exists(MODEL_PATH), \
        f"Model not found: {MODEL_PATH}"
    print(f"\n--- Loading model ---")
    model, all_labels = load_model(MODEL_PATH, device)
    build_color_map(all_labels)

    total_p = sum(p.numel() for p in model.parameters())
    print(f"  Labels     : {all_labels}")
    print(f"  Parameters : {total_p/1e6:.1f}M")
    print(f"  Active heads: {model.active_heads}")

    # Collect images
    image_paths = sorted([
        p for ext in IMAGE_EXTS
        for p in glob.glob(os.path.join(IMAGES_DIR, f"*{ext}"))
    ])
    assert len(image_paths) > 0, \
        f"No images found in: {IMAGES_DIR}"
    print(f"\n  Images : {len(image_paths)} found")

    # Inference
    all_image_preds = {}
    images_meta     = {}
    per_img_stats   = []
    infer_start     = time.time()

    for img_path in image_paths:
        fname = os.path.basename(img_path)
        tensor, orig_img, orig_h, orig_w = preprocess(
            img_path, cfg.INPUT_SIZE
        )
        if tensor is None:
            print(f"  Skipping: {fname}")
            continue

        images_meta[fname] = {"width": orig_w, "height": orig_h}
        tensor = tensor.to(device)

        with torch.no_grad():
            outputs = model(tensor)

        all_preds = []

        if "bbox" in outputs:
            all_preds.extend(decode_bbox(
                outputs["bbox"], all_labels,
                cfg.BBOX_ANCHORS, cfg.INPUT_SIZE,
                orig_h, orig_w, BBOX_THRESH
            ))

        if "polygon" in outputs:
            all_preds.extend(decode_polygon(
                outputs["polygon"], all_labels,
                orig_h, orig_w, POLY_THRESH
            ))

        if "keypoint" in outputs:
            all_preds.extend(decode_keypoints(
                outputs["keypoint"], orig_h, orig_w,
                KP_THRESH, KP_NMS_WIN, MAX_KP
            ))

        if "polyline_vertex" in outputs:
            all_preds.extend(decode_polylines(
                outputs["polyline_vertex"],
                model.polyline_head,
                orig_h, orig_w,
                POLY_V_THRESH, EDGE_THRESH,
                cfg.POLY_MAX_DIST, MAX_POLY, device
            ))

        if "tag" in outputs:
            all_preds.extend(decode_tags(
                outputs["tag"], all_labels, TAG_THRESH
            ))

        all_image_preds[fname] = all_preds

        # Draw
        annotated = draw_predictions(orig_img, all_preds)
        cv2.imwrite(
            os.path.join(OUTPUT_DIR, "annotated", fname),
            annotated
        )

        type_counts = defaultdict(int)
        for p in all_preds:
            type_counts[p["type"]] += 1
        per_img_stats.append({
            "image"   : fname, "total": len(all_preds),
            "bbox"    : type_counts["bbox"],
            "polygon" : type_counts["polygon"],
            "keypoint": type_counts["keypoint"],
            "polyline": type_counts["polyline"],
            "tag"     : type_counts["tag"],
        })
        print(f"  {fname:<35} "
              f"bbox:{type_counts['bbox']} "
              f"poly:{type_counts['polygon']} "
              f"kp:{type_counts['keypoint']} "
              f"pl:{type_counts['polyline']} "
              f"tag:{type_counts['tag']}")

    # Save outputs
    xml_path = os.path.join(OUTPUT_DIR, "predictions_cvat.xml")
    save_cvat_xml(all_image_preds, images_meta, xml_path)

    stats_path = os.path.join(OUTPUT_DIR, "per_image_stats.csv")
    with open(stats_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=[
            "image","total","bbox","polygon",
            "keypoint","polyline","tag"
        ])
        writer.writeheader()
        writer.writerows(per_img_stats)

    elapsed = time.time() - infer_start
    print(f"\n{'═'*55}")
    print(f"  Inference Complete")
    print(f"  Images    : {len(per_img_stats)}")
    print(f"  Time      : "
          f"{str(datetime.timedelta(seconds=int(elapsed)))}")
    print(f"  CVAT XML  : {xml_path}")
    print(f"  Annotated : {OUTPUT_DIR}/annotated/")
    print(f"  Stats     : {stats_path}")
    print(f"{'═'*55}\n")


if __name__ == "__main__":
    main()
