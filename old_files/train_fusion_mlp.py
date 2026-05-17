# ============================================================
# fusion_mlp/train_fusion_mlp.py — Option 3: Prediction MLP Fusion
# ============================================================
# Trains a small MLP that re-scores predictions from all models
# using context from co-occurring predictions in the same image.
#
# Two operating modes (set FUSION_MODE below):
#   "per_detection" — each prediction is re-scored independently
#                     using its own features + image-level context
#   "per_region"    — predictions that spatially overlap are
#                     re-scored together as a group
#
# Training data: CVAT XML ground truth + master_test.py predictions
# All individual models are FROZEN — only the MLP trains.
#
# Run training : python fusion_mlp/train_fusion_mlp.py
# Inference    : called automatically from master_test.py
# ============================================================

import os
import sys
import json
import csv
import time
import datetime
import random
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import xml.etree.ElementTree as ET

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import config as cfg

# ============================================================
# --- USER SETTINGS ---
# ============================================================

# CVAT XML ground truth — used to build training labels
GT_XML_PATH     = cfg.CVAT_XML

# predictions_coco.json from master_test.py — model predictions
# on the same images as GT_XML_PATH
PRED_JSON_PATH  = os.path.join(cfg.SAVE_DIR,
                               "test_output", "predictions_coco.json")

# Where to save the trained MLP
MLP_SAVE_DIR    = os.path.join(cfg.SAVE_DIR, "fusion_mlp")

# ---- Mode switch ----
# "per_detection" : each prediction re-scored independently
# "per_region"    : overlapping predictions re-scored as a group
FUSION_MODE     = "per_detection"   # or "per_region"

# ---- Training hyperparameters ----
MLP_HIDDEN_DIM  = 512
MLP_EPOCHS      = 1000
MLP_BATCH_SIZE  = 64
MLP_LR          = 1e-4
MLP_WEIGHT_DECAY= 1e-4
MLP_DROPOUT     = 0.25
MLP_VAL_RATIO   = 0.3
MLP_SEED        = 42

# ---- Per-region mode settings ----
# Two predictions are "in the same region" if their bbox IoU
# exceeds this threshold
REGION_IOU_THRESH = 0.1

# ---- IoU threshold to match a prediction to a GT annotation ----
# A prediction with IoU >= this against any GT box is a true positive
TP_IOU_THRESH   = 0.35

# ============================================================
# --- Feature dimensions ---
# ============================================================
#
# Per-detection features (17 values):
#   score           (1)  — model confidence
#   shape_type_ohe  (5)  — one-hot: bbox/polygon/polyline/keypoint/tag
#   norm_area       (1)  — bbox area / image area
#   norm_cx, norm_cy(2)  — bbox center normalized
#   norm_w, norm_h  (2)  — bbox size normalized
#   aspect_ratio    (1)  — w/h
#   n_same_type     (1)  — count of same-type predictions in image
#   n_other_type    (1)  — count of other-type predictions in image
#   max_iou_same    (1)  — max IoU with same-type predictions
#   max_iou_other   (1)  — max IoU with different-type predictions
#   tag_present     (1)  — 1 if any tag prediction exists in image
#
# Per-region features (additional group features appended):
#   n_in_region     (1)  — number of predictions overlapping this one
#   type_diversity  (1)  — number of unique types in region
#   mean_score_region(1) — mean score of all predictions in region
#   max_score_region(1)  — max score in region

PER_DET_DIM    = 17
PER_REGION_DIM = PER_DET_DIM + 4

SHAPE_TYPES    = ["bbox", "polygon", "polyline", "keypoint", "tag"]


# ============================================================
# --- Geometry helpers ---
# ============================================================

def ann_to_xyxy(ann, img_w, img_h):
    bbox = ann.get("bbox", [])
    if bbox and len(bbox) == 4 and bbox[2] > 0 and bbox[3] > 0:
        x, y, w, h = bbox
        return [x, y, x + w, y + h]
    seg = ann.get("segmentation", [])
    if seg and isinstance(seg, list) and len(seg) > 0:
        flat = seg[0]
        if len(flat) >= 4:
            xs = flat[0::2]
            ys = flat[1::2]
            return [min(xs), min(ys), max(xs), max(ys)]
    kp = ann.get("keypoints", [])
    if kp and len(kp) >= 2:
        r = 10
        return [kp[0]-r, kp[1]-r, kp[0]+r, kp[1]+r]
    return [0, 0, img_w, img_h]


def iou(a, b):
    ax1, ay1, ax2, ay2 = a
    bx1, by1, bx2, by2 = b
    ix1  = max(ax1, bx1)
    iy1  = max(ay1, by1)
    ix2  = min(ax2, bx2)
    iy2  = min(ay2, by2)
    iw   = max(0, ix2 - ix1)
    ih   = max(0, iy2 - iy1)
    inter= iw * ih
    area_a = max(0, ax2-ax1) * max(0, ay2-ay1)
    area_b = max(0, bx2-bx1) * max(0, by2-by1)
    union  = area_a + area_b - inter
    return inter / union if union > 0 else 0.0


# ============================================================
# --- Feature extraction ---
# ============================================================

def extract_per_detection_features(ann, all_anns_in_image,
                                   img_w, img_h):
    """
    Extracts a 17-dimensional feature vector for one prediction.
    """
    score      = float(ann.get("score", 1.0))
    shape_type = ann.get("shape_type", "bbox")
    xyxy       = ann_to_xyxy(ann, img_w, img_h)

    x1, y1, x2, y2 = xyxy
    bw   = max(0, x2 - x1)
    bh   = max(0, y2 - y1)
    area = bw * bh
    img_area = max(img_w * img_h, 1)

    norm_area = area / img_area
    norm_cx   = ((x1 + x2) / 2) / max(img_w, 1)
    norm_cy   = ((y1 + y2) / 2) / max(img_h, 1)
    norm_w    = bw / max(img_w, 1)
    norm_h    = bh / max(img_h, 1)
    aspect    = bw / max(bh, 1e-6)

    # One-hot shape type
    ohe = [1.0 if shape_type == t else 0.0 for t in SHAPE_TYPES]

    # Context: counts of same/other type predictions in image
    same_type  = [a for a in all_anns_in_image
                  if a["id"] != ann["id"]
                  and a.get("shape_type") == shape_type]
    other_type = [a for a in all_anns_in_image
                  if a["id"] != ann["id"]
                  and a.get("shape_type") != shape_type]

    n_same  = len(same_type)
    n_other = len(other_type)

    # Max IoU with same-type predictions
    max_iou_same = 0.0
    for other in same_type:
        other_xyxy = ann_to_xyxy(other, img_w, img_h)
        v = iou(xyxy, other_xyxy)
        if v > max_iou_same:
            max_iou_same = v

    # Max IoU with different-type predictions
    max_iou_other = 0.0
    for other in other_type:
        other_xyxy = ann_to_xyxy(other, img_w, img_h)
        v = iou(xyxy, other_xyxy)
        if v > max_iou_other:
            max_iou_other = v

    # Tag presence in image
    tag_present = 1.0 if any(
        a.get("shape_type") == "tag"
        for a in all_anns_in_image
    ) else 0.0

    feat = [score] + ohe + [
        norm_area, norm_cx, norm_cy,
        norm_w, norm_h, aspect,
        float(n_same), float(n_other),
        max_iou_same, max_iou_other,
        tag_present
    ]
    return np.array(feat, dtype=np.float32)   # (17,)


def extract_per_region_features(ann, all_anns_in_image,
                                 img_w, img_h):
    """
    Extends per-detection features with 4 region-level features.
    Returns a 21-dimensional feature vector.
    """
    base_feat  = extract_per_detection_features(
        ann, all_anns_in_image, img_w, img_h
    )
    xyxy       = ann_to_xyxy(ann, img_w, img_h)

    # Find all predictions overlapping this one
    overlapping = []
    for other in all_anns_in_image:
        if other["id"] == ann["id"]:
            continue
        other_xyxy = ann_to_xyxy(other, img_w, img_h)
        if iou(xyxy, other_xyxy) >= REGION_IOU_THRESH:
            overlapping.append(other)

    n_in_region      = float(len(overlapping))
    type_diversity   = float(len({
        a.get("shape_type") for a in overlapping
    }))
    scores_in_region = [float(a.get("score", 1.0))
                        for a in overlapping]
    mean_score_region= float(np.mean(scores_in_region)) \
                       if scores_in_region else 0.0
    max_score_region = float(max(scores_in_region)) \
                       if scores_in_region else 0.0

    region_feat = np.array([
        n_in_region, type_diversity,
        mean_score_region, max_score_region
    ], dtype=np.float32)

    return np.concatenate([base_feat, region_feat])   # (21,)


def extract_features(ann, all_anns, img_w, img_h, mode):
    if mode == "per_region":
        return extract_per_region_features(ann, all_anns, img_w, img_h)
    return extract_per_detection_features(ann, all_anns, img_w, img_h)


# ============================================================
# --- Ground truth matching ---
# ============================================================

def parse_gt_xml(xml_path):
    """
    Parses CVAT XML ground truth.
    Returns dict: image_filename → list of GT annotation dicts
    Each GT dict has: shape_type, label, xyxy
    """
    tree     = ET.parse(xml_path)
    root     = tree.getroot()
    gt_by_img= {}

    for img_el in root.findall("image"):
        fname = img_el.get("name", "")
        anns  = []

        for el in img_el.findall("box"):
            xtl = float(el.get("xtl", 0))
            ytl = float(el.get("ytl", 0))
            xbr = float(el.get("xbr", 0))
            ybr = float(el.get("ybr", 0))
            anns.append({
                "shape_type": "bbox",
                "label"     : el.get("label", ""),
                "xyxy"      : [xtl, ytl, xbr, ybr],
            })

        for el in img_el.findall("polygon"):
            pts  = _parse_pts(el.get("points", ""))
            if pts:
                xs = [p[0] for p in pts]
                ys = [p[1] for p in pts]
                anns.append({
                    "shape_type": "polygon",
                    "label"     : el.get("label", ""),
                    "xyxy"      : [min(xs), min(ys),
                                   max(xs), max(ys)],
                })

        for el in img_el.findall("polyline"):
            pts  = _parse_pts(el.get("points", ""))
            if pts:
                xs = [p[0] for p in pts]
                ys = [p[1] for p in pts]
                anns.append({
                    "shape_type": "polyline",
                    "label"     : el.get("label", ""),
                    "xyxy"      : [min(xs), min(ys),
                                   max(xs), max(ys)],
                })

        for el in img_el.findall("points"):
            pts = _parse_pts(el.get("points", ""))
            for pt in pts:
                r = 10
                anns.append({
                    "shape_type": "keypoint",
                    "label"     : el.get("label", ""),
                    "xyxy"      : [pt[0]-r, pt[1]-r,
                                   pt[0]+r, pt[1]+r],
                })

        for el in img_el.findall("tag"):
            anns.append({
                "shape_type": "tag",
                "label"     : el.get("label", ""),
                "xyxy"      : None,
            })

        gt_by_img[fname] = anns

    return gt_by_img


def _parse_pts(points_str):
    pts = []
    for pair in points_str.strip().split(";"):
        pair = pair.strip()
        if not pair:
            continue
        parts = pair.split(",")
        if len(parts) == 2:
            try:
                pts.append((float(parts[0]), float(parts[1])))
            except ValueError:
                continue
    return pts


def is_true_positive(pred_ann, gt_anns, img_w, img_h):
    """
    Returns 1.0 if prediction matches any GT annotation
    of the same shape_type with IoU >= TP_IOU_THRESH.
    Tags are matched by label presence alone (no geometry).
    """
    pred_type = pred_ann.get("shape_type", "bbox")

    if pred_type == "tag":
        pred_label = pred_ann.get("category_id_name", "")
        return 1.0 if any(
            gt["shape_type"] == "tag"
            and gt["label"] == pred_label
            for gt in gt_anns
        ) else 0.0

    pred_xyxy = ann_to_xyxy(pred_ann, img_w, img_h)
    same_type_gt = [
        gt for gt in gt_anns
        if gt["shape_type"] == pred_type
        and gt["xyxy"] is not None
    ]

    for gt in same_type_gt:
        if iou(pred_xyxy, gt["xyxy"]) >= TP_IOU_THRESH:
            return 1.0

    return 0.0


# ============================================================
# --- Dataset ---
# ============================================================

class FusionMLPDataset(Dataset):
    """
    Builds (feature_vector, label) pairs from predictions + GT.
    label = 1.0 if prediction matches GT, 0.0 otherwise.
    """

    def __init__(self, pred_json_path, gt_xml_path, mode, seed=42):
        random.seed(seed)
        self.mode    = mode
        self.samples = []   # (feat, label)

        # Load predictions
        with open(pred_json_path) as f:
            coco = json.load(f)

        cat_map   = {c["id"]: c["name"] for c in coco["categories"]}
        images_map= {img["id"]: img for img in coco["images"]}

        from collections import defaultdict
        anns_by_img = defaultdict(list)
        for ann in coco["annotations"]:
            # Attach label name for tag matching
            ann["category_id_name"] = cat_map.get(
                ann["category_id"], ""
            )
            anns_by_img[ann["image_id"]].append(ann)

        # Load ground truth
        gt_by_img = parse_gt_xml(gt_xml_path)

        # Build samples
        n_pos = 0
        n_neg = 0
        for img_info in coco["images"]:
            img_id = img_info["id"]
            fname  = img_info["file_name"]
            img_w  = img_info.get("width",  640)
            img_h  = img_info.get("height", 640)

            preds  = anns_by_img.get(img_id, [])
            gt_anns= gt_by_img.get(fname, [])

            if not preds:
                continue

            for ann in preds:
                feat  = extract_features(
                    ann, preds, img_w, img_h, mode
                )
                label = is_true_positive(ann, gt_anns, img_w, img_h)
                self.samples.append((feat, label))
                if label > 0.5:
                    n_pos += 1
                else:
                    n_neg += 1

        random.shuffle(self.samples)
        print(f"    Fusion MLP dataset: "
              f"{len(self.samples)} samples | "
              f"pos={n_pos} | neg={n_neg} | "
              f"mode={mode}")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        feat, label = self.samples[idx]
        return (
            torch.tensor(feat,  dtype=torch.float32),
            torch.tensor(label, dtype=torch.float32),
        )


# ============================================================
# --- MLP Model ---
# ============================================================

class FusionMLP(nn.Module):
    """
    Small MLP that takes a feature vector and outputs a
    refined confidence score for the prediction.
    Input dim adapts to mode: 17 (per_detection) or 21 (per_region).
    """

    def __init__(self, input_dim, hidden_dim=256, dropout=0.3):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.LayerNorm(hidden_dim // 2),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, hidden_dim // 4),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim // 4, 1),
        )
        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                nn.init.zeros_(m.bias)

    def forward(self, x):
        return self.net(x).squeeze(-1)   # (B,) logits

    def predict_score(self, x):
        """Returns refined confidence in [0, 1]."""
        with torch.no_grad():
            return self.forward(x).sigmoid()


# ============================================================
# --- Training ---
# ============================================================

def train():
    print("\n" + "═" * 55)
    print(f"  Fusion MLP Training — mode: {FUSION_MODE}")
    print("═" * 55)

    os.makedirs(MLP_SAVE_DIR, exist_ok=True)
    device = torch.device(cfg.DEVICE if torch.cuda.is_available()
                          else "cpu")
    print(f"  Device : {device}")

    assert os.path.exists(PRED_JSON_PATH), \
        f"Predictions not found: {PRED_JSON_PATH}\n" \
        f"Run master_test.py first."
    assert os.path.exists(GT_XML_PATH), \
        f"GT XML not found: {GT_XML_PATH}"

    # Dataset
    full_ds = FusionMLPDataset(
        pred_json_path = PRED_JSON_PATH,
        gt_xml_path    = GT_XML_PATH,
        mode           = FUSION_MODE,
        seed           = MLP_SEED,
    )
    assert len(full_ds) > 0, \
        "No training samples built. Check that PRED_JSON_PATH " \
        "and GT_XML_PATH cover the same images."

    n_val   = max(1, int(len(full_ds) * MLP_VAL_RATIO))
    n_train = len(full_ds) - n_val
    train_ds, val_ds = torch.utils.data.random_split(
        full_ds, [n_train, n_val],
        generator=torch.Generator().manual_seed(MLP_SEED)
    )

    # Compute pos_weight for class imbalance
    labels     = [full_ds[i][1].item() for i in range(len(full_ds))]
    n_pos      = sum(1 for l in labels if l > 0.5)
    n_neg      = len(labels) - n_pos
    pos_weight = torch.tensor(
        [n_neg / max(n_pos, 1)], dtype=torch.float32
    ).to(device)
    print(f"  pos_weight : {pos_weight.item():.2f} "
          f"(neg/pos ratio)")

    train_loader = DataLoader(
        train_ds, batch_size=MLP_BATCH_SIZE,
        shuffle=True, num_workers=0
    )
    val_loader   = DataLoader(
        val_ds, batch_size=MLP_BATCH_SIZE,
        shuffle=False, num_workers=0
    )

    print(f"  Train samples : {n_train}")
    print(f"  Val samples   : {n_val}")

    # Model
    input_dim = PER_REGION_DIM if FUSION_MODE == "per_region" \
                else PER_DET_DIM
    model = FusionMLP(
        input_dim  = input_dim,
        hidden_dim = MLP_HIDDEN_DIM,
        dropout    = MLP_DROPOUT,
    ).to(device)
    print(f"  Input dim     : {input_dim}")
    print(f"  Parameters    : "
          f"{sum(p.numel() for p in model.parameters()):,}")

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=MLP_LR, weight_decay=MLP_WEIGHT_DECAY
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=MLP_EPOCHS
    )

    log_path  = os.path.join(MLP_SAVE_DIR, "train_log.csv")
    best_val  = float("inf")
    best_path = os.path.join(MLP_SAVE_DIR, "fusion_mlp_best.pt")
    start     = time.time()

    with open(log_path, "w", newline="") as f:
        csv.writer(f).writerow([
            "epoch", "train_loss", "val_loss",
            "val_acc", "val_f1", "epoch_time"
        ])

    for epoch in range(1, MLP_EPOCHS + 1):
        t0 = time.time()

        # Train
        model.train()
        train_loss = 0.0
        for feats, labels in train_loader:
            feats  = feats.to(device)
            labels = labels.to(device)
            logits = model(feats)
            loss   = F.binary_cross_entropy_with_logits(
                logits, labels, pos_weight=pos_weight
            )
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            train_loss += loss.item()
        train_loss /= len(train_loader)

        # Validate
        model.eval()
        val_loss   = 0.0
        all_preds  = []
        all_labels = []
        with torch.no_grad():
            for feats, labels in val_loader:
                feats  = feats.to(device)
                labels = labels.to(device)
                logits = model(feats)
                val_loss += F.binary_cross_entropy_with_logits(
                    logits, labels, pos_weight=pos_weight
                ).item()
                preds = (logits.sigmoid() > 0.5).float()
                all_preds.append(preds.cpu())
                all_labels.append(labels.cpu())
        val_loss /= len(val_loader)

        all_preds  = torch.cat(all_preds)
        all_labels = torch.cat(all_labels)
        acc  = (all_preds == all_labels).float().mean().item()
        tp   = (all_preds * all_labels).sum().item()
        fp   = (all_preds * (1 - all_labels)).sum().item()
        fn   = ((1 - all_preds) * all_labels).sum().item()
        prec = tp / max(tp + fp, 1)
        rec  = tp / max(tp + fn, 1)
        f1   = 2 * prec * rec / max(prec + rec, 1e-6)

        scheduler.step()
        ep_time = time.time() - t0

        if val_loss < best_val:
            best_val = val_loss
            torch.save({
                "epoch"           : epoch,
                "model_state_dict": model.state_dict(),
                "input_dim"       : input_dim,
                "hidden_dim"      : MLP_HIDDEN_DIM,
                "fusion_mode"     : FUSION_MODE,
                "val_loss"        : best_val,
            }, best_path)

        with open(log_path, "a", newline="") as f:
            csv.writer(f).writerow([
                epoch, f"{train_loss:.6f}", f"{val_loss:.6f}",
                f"{acc:.4f}", f"{f1:.4f}", f"{ep_time:.2f}"
            ])

        if epoch % 10 == 0 or epoch == 1:
            print(f"  Epoch {epoch:>4}/{MLP_EPOCHS} | "
                  f"train {train_loss:.4f} | "
                  f"val {val_loss:.4f} | "
                  f"acc {acc*100:.1f}% | "
                  f"F1 {f1*100:.1f}% | "
                  f"{ep_time:.2f}s")

    elapsed = str(datetime.timedelta(
        seconds=int(time.time() - start)
    ))
    print(f"\n  Training complete | Time: {elapsed}")
    print(f"  Best model → {best_path}")
    print(f"  Log        → {log_path}")
    return best_path


# ============================================================
# --- Inference helper (called from master_test.py) ---
# ============================================================

def load_fusion_mlp(model_path):
    """
    Loads a saved FusionMLP checkpoint.
    Returns (model, fusion_mode, input_dim).
    Call this once at startup in master_test.py.
    """
    ckpt       = torch.load(model_path, map_location="cpu")
    input_dim  = ckpt["input_dim"]
    hidden_dim = ckpt.get("hidden_dim", MLP_HIDDEN_DIM)
    mode       = ckpt.get("fusion_mode", "per_detection")
    model      = FusionMLP(input_dim=input_dim,
                           hidden_dim=hidden_dim)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()
    return model, mode, input_dim


def apply_fusion_mlp(model, all_preds, img_w, img_h,
                     mode, device, score_thresh=0.3):
    """
    Re-scores a list of predictions for one image using
    the trained FusionMLP. Returns filtered + re-scored list.

    Args:
        model       : loaded FusionMLP
        all_preds   : list of prediction dicts for one image
                      (same format as master_test.py produces)
        img_w/img_h : image dimensions
        mode        : "per_detection" or "per_region"
        device      : torch device
        score_thresh: predictions below this after re-scoring
                      are dropped

    Returns:
        re-scored and filtered prediction list
    """
    if not all_preds:
        return all_preds

    # Build feature matrix
    feats = np.stack([
        extract_features(ann, all_preds, img_w, img_h, mode)
        for ann in all_preds
    ], axis=0)   # (N, input_dim)

    feat_t = torch.tensor(feats, dtype=torch.float32).to(device)
    with torch.no_grad():
        new_scores = model.predict_score(feat_t).cpu().numpy()

    # Update scores and filter
    refined = []
    for ann, new_score in zip(all_preds, new_scores):
        if float(new_score) >= score_thresh:
            ann = dict(ann)   # don't mutate original
            ann["score_original"] = ann.get("score", 1.0)
            ann["score"]          = float(new_score)
            refined.append(ann)

    return refined


# ============================================================
# --- Entry point ---
# ============================================================

if __name__ == "__main__":
    train()
