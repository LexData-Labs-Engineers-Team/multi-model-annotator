# ============================================================
# attention_fusion.py — Late Fusion with Cross-Modal Attention
# ============================================================
# Extracts intermediate features from each frozen specialist
# model via forward hooks, projects them to a common spatial
# resolution, fuses via 2-layer cross-attention, and produces
# refined predictions for all annotation types.
#
# All specialist models are FROZEN.
# Only the fusion module trains.
#
# Structure:
#   AttentionFusion          — main module
#   FeatureExtractor         — wraps each model with hooks
#   ModalityProjector        — projects each model's features
#                              to common dim + resolution
#   CrossModalAttention      — 2-layer transformer attention
#   RefinedHeads             — lightweight prediction heads
#                              on fused features
#
# Training : python attention_fusion.py --mode train
# Inference: python attention_fusion.py --mode test
# ============================================================

import os
import sys
import cv2
import csv
import json
import glob
import time
import math
import random
import datetime
import argparse
import numpy as np
from collections import defaultdict
from PIL import Image

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as T
import torchvision.transforms.functional as TF
import xml.etree.ElementTree as ET
from xml.dom import minidom
from scipy.ndimage import maximum_filter

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import config as cfg

# ============================================================
# --- USER SETTINGS ---
# ============================================================

# Ground truth XML for training supervision
GT_XML_PATH         = cfg.CVAT_XML
IMG_DIR             = cfg.IMG_DIR

# Trained specialist model paths
BBOX_MODEL_PATH     = os.path.join(cfg.BBOX_SAVE_DIR,
                                   "train", "weights", "best.pt")
POLYGON_MODEL_PATH  = os.path.join(cfg.POLYGON_SAVE_DIR,
                                   "train", "weights", "best.pt")
KEYPOINT_MODEL_PATH = os.path.join(cfg.KEYPOINT_SAVE_DIR, "best.pt")
POLYLINE_S1_PATH    = os.path.join(cfg.POLYLINE_SAVE_DIR, "s1_best.pt")
POLYLINE_S2_PATH    = os.path.join(cfg.POLYLINE_SAVE_DIR, "s2_best.pt")
TAG_MODEL_PATH      = os.path.join(cfg.TAG_SAVE_DIR,      "best.pt")
TAG_NAMES_PATH      = os.path.join(cfg.TAG_SAVE_DIR,      "tag_names.json")

# Fusion module save directory
FUSION_SAVE_DIR     = os.path.join(cfg.SAVE_DIR, "attention_fusion")

# Test images (for inference mode)
TEST_IMAGES_DIR     = "/home/lexdata/Documents/LexAnnotate_demo/m_env/multi_pipeline/datasets/flower/images"
TEST_OUTPUT_DIR     = os.path.join(cfg.SAVE_DIR, "fusion_test_output")

# ---- Fusion architecture ----
FUSION_DIM          = 64    # common projection dimension
FUSION_SPATIAL      = 5     # common spatial resolution (20×20)
NUM_HEADS           = 4      # attention heads
NUM_LAYERS          = 2      # cross-attention layers
DROPOUT             = 0.25

# ---- Training ----
EPOCHS              = 60
BATCH_SIZE          = 4
LR                  = 1e-4
WEIGHT_DECAY        = 1e-4
VAL_RATIO           = 0.3
RANDOM_SEED         = 42
NUM_WORKERS         = 2      # set to cfg.NUM_WORKERS after testing

# ---- Inference thresholds ----
BBOX_THRESH         = cfg.YOLO_SCORE_THRESH
POLY_THRESH         = cfg.YOLO_SCORE_THRESH
KP_THRESH           = cfg.KP_SCORE_THRESH
POLY_V_THRESH       = cfg.POLY_S1_SCORE_THRESH  
EDGE_THRESH         = cfg.EDGE_THRESH
TAG_THRESH          = cfg.TAG_SCORE_THRESH

IMAGE_EXTS          = {".jpg", ".jpeg", ".png", ".bmp", ".tiff"}

# ============================================================
# --- Feature Extractor (forward hooks) ---
# ============================================================

class FeatureExtractor:
    """
    Wraps a model and attaches forward hooks to extract
    intermediate feature maps without modifying the model.

    For YOLO models: hooks the last backbone/neck layer.
    For ResNet heatmap models: hooks the decoder bottleneck.
    For EfficientNet tag model: hooks the backbone output.
    """

    def __init__(self):
        self.features = {}
        self._hooks   = []

    def register(self, model, layer, name):
        def hook_fn(module, input, output):
            # Clone immediately to escape inference_mode context
            if isinstance(output, torch.Tensor):
                self.features[name] = output.detach().clone()
            else:
                # Some layers return tuples
                self.features[name] = output[0].detach().clone()
        h = layer.register_forward_hook(hook_fn)
        self._hooks.append(h)

    def clear(self):
        self.features = {}

    def remove(self):
        for h in self._hooks:
            h.remove()
        self._hooks = []


def register_yolo_hook(yolo_model, extractor, name):
    """
    Register hook on YOLO model's last C2f layer in backbone.
    Extracts rich spatial features before the detection head.
    """
    try:
        # Ultralytics YOLO — model.model is the nn.Sequential
        # Layer -4 is typically the last C2f before detection
        target_layer = yolo_model.model.model[-4]
        extractor.register(yolo_model, target_layer, name)
        return True
    except Exception:
        # Fallback — hook last module
        layers = list(yolo_model.model.model.children())
        if layers:
            extractor.register(yolo_model, layers[-2], name)
            return True
    return False


def register_resnet_hook(resnet_model, extractor, name):
    """
    Register hook on ResNet heatmap model's dec0 layer
    (decoder bottleneck — rich low-level + high-level features).
    """
    extractor.register(resnet_model, resnet_model.dec0, name)


def register_tag_hook(tag_model, extractor, name):
    """
    Register hook on EfficientNet backbone's last features.
    """
    extractor.register(tag_model, tag_model.backbone, name)


# ============================================================
# --- Modality Projector ---
# ============================================================

class ModalityProjector(nn.Module):
    """
    Projects each model's feature map to a common spatial
    resolution (FUSION_SPATIAL × FUSION_SPATIAL) and a
    common channel dimension (FUSION_DIM).

    Handles both spatial (B, C, H, W) and global (B, C)
    feature tensors.
    """

    def __init__(self, in_channels, fusion_dim, fusion_spatial):
        super().__init__()
        self.fusion_spatial = fusion_spatial
        self.fusion_dim     = fusion_dim

        # Spatial projection — conv to target channels
        self.spatial_proj = nn.Sequential(
            nn.Conv2d(in_channels, fusion_dim, 1, bias=False),
            nn.BatchNorm2d(fusion_dim),
            nn.ReLU(inplace=True),
        )

        # Global projection for 1D features (e.g. tag backbone)
        self.global_proj = nn.Sequential(
            nn.Linear(in_channels, fusion_dim * fusion_spatial
                      * fusion_spatial),
            nn.ReLU(inplace=True),
        )

        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.xavier_uniform_(m.weight)
            elif isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                nn.init.zeros_(m.bias)

    def forward(self, x):
        """
        x: (B, C, H, W) or (B, C, 1, 1) or (B, C)
        Returns: (B, fusion_dim, fusion_spatial, fusion_spatial)
        """
        if x.dim() == 2:
            # Global feature (B, C) — from tag backbone GAP
            B = x.shape[0]
            out = self.global_proj(x)
            return out.reshape(
                B, self.fusion_dim,
                self.fusion_spatial, self.fusion_spatial
            )

        if x.dim() == 4:
            # Spatial feature (B, C, H, W)
            proj = self.spatial_proj(x)   # (B, D, H, W)
            return F.interpolate(
                proj,
                size=(self.fusion_spatial, self.fusion_spatial),
                mode="bilinear", align_corners=False
            )

        # 3D — unlikely but handle gracefully
        return x


# ============================================================
# --- Cross-Modal Attention ---
# ============================================================

class CrossModalAttentionLayer(nn.Module):
    """
    Single cross-modal attention layer.
    Each modality attends to all other modalities.
    Uses standard multi-head attention with residual + norm.
    """

    def __init__(self, dim, num_heads, dropout=0.1):
        super().__init__()
        self.attn   = nn.MultiheadAttention(
            dim, num_heads, dropout=dropout, batch_first=True
        )
        self.norm1  = nn.LayerNorm(dim)
        self.norm2  = nn.LayerNorm(dim)
        self.ffn    = nn.Sequential(
            nn.Linear(dim, dim * 4),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(dim * 4, dim),
            nn.Dropout(dropout),
        )

    def forward(self, x):
        """
        x: (B, N*S, D)
           N = number of modalities
           S = spatial_tokens per modality
           D = fusion_dim
        """
        # Self-attention across all modality tokens
        attn_out, _ = self.attn(x, x, x)
        x = self.norm1(x + attn_out)
        x = self.norm2(x + self.ffn(x))
        return x


class CrossModalAttention(nn.Module):
    """
    2-layer cross-modal attention over projected modality features.
    Input: list of (B, D, S, S) feature maps, one per modality.
    Output: list of (B, D, S, S) fused feature maps.
    """

    def __init__(self, fusion_dim, fusion_spatial,
                 num_modalities, num_heads, num_layers, dropout):
        super().__init__()
        self.fusion_dim     = fusion_dim
        self.fusion_spatial = fusion_spatial
        self.num_modalities = num_modalities
        self.S              = fusion_spatial * fusion_spatial

        # Learnable positional encoding per modality
        self.pos_embed = nn.Parameter(
            torch.randn(1, num_modalities * self.S, fusion_dim)
            * 0.02
        )

        # Modality type embedding — helps attention distinguish sources
        self.modal_embed = nn.Embedding(num_modalities, fusion_dim)

        # Attention layers
        self.layers = nn.ModuleList([
            CrossModalAttentionLayer(fusion_dim, num_heads, dropout)
            for _ in range(num_layers)
        ])

        self.out_norm = nn.LayerNorm(fusion_dim)

    def forward(self, modal_features):
        """
        modal_features: list of N tensors, each (B, D, S, S)
        Returns: list of N tensors, each (B, D, S, S)
        """
        B  = modal_features[0].shape[0]
        S  = self.S
        D  = self.fusion_dim
        N  = len(modal_features)

        # Flatten spatial dims and concatenate modalities
        # Each (B, D, S, S) → (B, S, D)
        tokens = []
        for i, feat in enumerate(modal_features):
            # (B, D, S, S) → (B, S*S, D)
            t = feat.flatten(2).permute(0, 2, 1)   # (B, S², D)
            # Add modality embedding
            me = self.modal_embed(
                torch.tensor(i, device=feat.device)
            )
            t  = t + me.unsqueeze(0).unsqueeze(0)
            tokens.append(t)

        # Concatenate all modality tokens
        x = torch.cat(tokens, dim=1)   # (B, N*S², D)

        # Add positional encoding (trim/pad if needed)
        pos = self.pos_embed[:, :x.shape[1], :]
        x   = x + pos

        # Apply attention layers
        for layer in self.layers:
            x = layer(x)

        x = self.out_norm(x)

        # Split back into per-modality feature maps
        fused = []
        for i in range(N):
            t = x[:, i*S:(i+1)*S, :]          # (B, S², D)
            t = t.permute(0, 2, 1)             # (B, D, S²)
            t = t.reshape(B, D,
                          self.fusion_spatial,
                          self.fusion_spatial) # (B, D, Sf, Sf)
            fused.append(t)

        return fused


# ============================================================
# --- Refined Prediction Heads ---
# ============================================================

class RefinedBboxHead(nn.Module):
    """
    Refines bbox predictions using fused features.
    Takes fused bbox features + original bbox predictions
    and outputs a correction factor.
    """

    def __init__(self, fusion_dim, num_classes):
        super().__init__()
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.fc   = nn.Sequential(
            nn.Flatten(),
            nn.Linear(fusion_dim, 256),
            nn.ReLU(inplace=True),
            nn.Dropout(0.2),
            nn.Linear(256, num_classes + 1),  # class scores
        )

    def forward(self, fused_bbox_feat):
        return self.fc(self.pool(fused_bbox_feat))


class RefinedPolygonHead(nn.Module):
    """
    Upsamples fused polygon features to output segmentation mask.
    """

    def __init__(self, fusion_dim, num_classes, output_size):
        super().__init__()
        self.output_size = output_size
        self.decoder = nn.Sequential(
            nn.Conv2d(fusion_dim, 128, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 64, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, num_classes + 1, 1),
        )

    def forward(self, fused_poly_feat):
        up = F.interpolate(
            fused_poly_feat,
            size=(self.output_size, self.output_size),
            mode="bilinear", align_corners=False
        )
        return self.decoder(up)


class RefinedHeatmapHead(nn.Module):
    """
    Produces a refined heatmap from fused features.
    Used for both keypoint and polyline vertex heads.
    """

    def __init__(self, fusion_dim, output_size):
        super().__init__()
        self.output_size = output_size
        self.decoder = nn.Sequential(
            nn.Conv2d(fusion_dim, 64, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 32, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 1, 1),
        )
        nn.init.constant_(self.decoder[-1].bias, -2.0)

    def forward(self, fused_feat):
        up = F.interpolate(
            fused_feat,
            size=(self.output_size, self.output_size),
            mode="bilinear", align_corners=False
        )
        return self.decoder(up)


class RefinedTagHead(nn.Module):
    """
    Refines tag predictions from fused features.
    """

    def __init__(self, fusion_dim, num_tags):
        super().__init__()
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.fc   = nn.Sequential(
            nn.Flatten(),
            nn.Linear(fusion_dim, 128),
            nn.ReLU(inplace=True),
            nn.Dropout(0.2),
            nn.Linear(128, num_tags),
        )

    def forward(self, fused_tag_feat):
        return self.fc(self.pool(fused_tag_feat))


# ============================================================
# --- Main Fusion Module ---
# ============================================================

class AttentionFusion(nn.Module):
    """
    Full attention fusion module.
    Accepts extracted features from all specialist models,
    fuses them via cross-modal attention, and produces
    refined predictions for all annotation types.
    """

    MODALITIES = ["bbox", "polygon", "keypoint",
                  "polyline", "tag"]

    def __init__(
        self,
        # Feature channel dims from each specialist model
        bbox_feat_dim    : int,
        polygon_feat_dim : int,
        kp_feat_dim      : int,
        poly_feat_dim    : int,
        tag_feat_dim     : int,
        # Class counts
        num_bbox_classes : int,
        num_poly_classes : int,
        num_tags         : int,
        # Architecture
        fusion_dim       : int = FUSION_DIM,
        fusion_spatial   : int = FUSION_SPATIAL,
        num_heads        : int = NUM_HEADS,
        num_layers       : int = NUM_LAYERS,
        dropout          : float = DROPOUT,
        input_size       : int = 320,
    ):
        super().__init__()
        self.fusion_spatial   = fusion_spatial
        self.input_size       = input_size
        self.num_bbox_classes = num_bbox_classes
        self.num_poly_classes = num_poly_classes
        self.num_tags         = num_tags

        # ---- Projectors — one per modality ----
        self.proj_bbox    = ModalityProjector(
            bbox_feat_dim,    fusion_dim, fusion_spatial
        )
        self.proj_polygon = ModalityProjector(
            polygon_feat_dim, fusion_dim, fusion_spatial
        )
        self.proj_kp      = ModalityProjector(
            kp_feat_dim,      fusion_dim, fusion_spatial
        )
        self.proj_poly    = ModalityProjector(
            poly_feat_dim,    fusion_dim, fusion_spatial
        )
        self.proj_tag     = ModalityProjector(
            tag_feat_dim,     fusion_dim, fusion_spatial
        )

        # ---- Cross-modal attention ----
        self.attention = CrossModalAttention(
            fusion_dim     = fusion_dim,
            fusion_spatial = fusion_spatial,
            num_modalities = 5,
            num_heads      = num_heads,
            num_layers     = num_layers,
            dropout        = dropout,
        )

        # ---- Refined heads ----
        self.refined_bbox    = RefinedBboxHead(
            fusion_dim, num_bbox_classes
        )
        self.refined_polygon = RefinedPolygonHead(
            fusion_dim, num_poly_classes,
            output_size = input_size // 4
        )
        self.refined_kp      = RefinedHeatmapHead(
            fusion_dim, input_size
        )
        self.refined_poly    = RefinedHeatmapHead(
            fusion_dim, input_size
        )
        self.refined_tag     = RefinedTagHead(
            fusion_dim, num_tags
        )

    def forward(self, feat_dict):
        """
        feat_dict: dict with keys matching MODALITIES,
                   each value is a feature tensor from that model.

        Returns:
            refined: dict of refined predictions per modality
            fused_feats: list of fused feature maps (for inspection)
        """
        # Project all modalities to common dim + resolution
        p_bbox = self.proj_bbox(feat_dict["bbox"])
        p_poly = self.proj_polygon(feat_dict["polygon"])
        p_kp   = self.proj_kp(feat_dict["keypoint"])
        p_pl   = self.proj_poly(feat_dict["polyline"])
        p_tag  = self.proj_tag(feat_dict["tag"])

        # Cross-modal attention
        fused = self.attention([p_bbox, p_poly, p_kp, p_pl, p_tag])
        f_bbox, f_poly, f_kp, f_pl, f_tag = fused

        # Refined predictions
        refined = {
            "bbox"    : self.refined_bbox(f_bbox),
            "polygon" : self.refined_polygon(f_poly),
            "keypoint": self.refined_kp(f_kp),
            "polyline": self.refined_poly(f_pl),
            "tag"     : self.refined_tag(f_tag),
        }
        return refined, fused


# ============================================================
# --- Feature channel detection helpers ---
# ============================================================

def probe_feature_dims(models_dict, device, input_size):
    """
    Runs a dummy forward pass to detect output channel dims
    from each model's hooked layer.
    """
    dummy  = torch.zeros(1, 3, input_size, input_size).to(device)
    dims   = {}
    extractor = FeatureExtractor()

    # Register hooks
    if "bbox" in models_dict:
        register_yolo_hook(models_dict["bbox"], extractor, "bbox")
    if "polygon" in models_dict:
        register_yolo_hook(models_dict["polygon"],
                           extractor, "polygon")
    if "keypoint" in models_dict:
        register_resnet_hook(models_dict["keypoint"],
                             extractor, "keypoint")
    if "polyline" in models_dict:
        register_resnet_hook(models_dict["polyline"],
                             extractor, "polyline")
    if "tag" in models_dict:
        register_tag_hook(models_dict["tag"], extractor, "tag")

    # Dummy forward
    with torch.no_grad():
        for name, model in models_dict.items():
            if name in ("bbox", "polygon"):
                model.predict(dummy, verbose=False)
            else:
                model(dummy)

    for name, feat in extractor.features.items():
        if feat.dim() == 4:
            dims[name] = feat.shape[1]   # channel dim
        elif feat.dim() == 2:
            dims[name] = feat.shape[1]
        else:
            dims[name] = feat.reshape(1, -1).shape[1]

    extractor.remove()
    return dims


# ============================================================
# --- Dataset ---
# ============================================================

def parse_gt_xml(xml_path):
    """Parse CVAT XML — same parser as split_annotations.py."""
    tree      = ET.parse(xml_path)
    root      = tree.getroot()
    all_labels= set()
    images    = []

    for img_el in root.findall("image"):
        img = {
            "name"     : img_el.get("name",   ""),
            "width"    : int(img_el.get("width",  0)),
            "height"   : int(img_el.get("height", 0)),
            "bboxes"   : [],
            "polygons" : [],
            "polylines": [],
            "keypoints": [],
            "tags"     : [],
        }
        for el in img_el.findall("box"):
            label = el.get("label", "")
            all_labels.add(label)
            img["bboxes"].append({
                "label": label,
                "xtl": float(el.get("xtl", 0)),
                "ytl": float(el.get("ytl", 0)),
                "xbr": float(el.get("xbr", 0)),
                "ybr": float(el.get("ybr", 0)),
            })
        for el in img_el.findall("polygon"):
            label = el.get("label", "")
            all_labels.add(label)
            pts = _parse_pts(el.get("points", ""))
            if len(pts) >= 3:
                img["polygons"].append(
                    {"label": label, "points": pts}
                )
        for el in img_el.findall("polyline"):
            label = el.get("label", "")
            all_labels.add(label)
            pts = _parse_pts(el.get("points", ""))
            if len(pts) >= 2:
                img["polylines"].append(
                    {"label": label, "points": pts}
                )
        for el in img_el.findall("points"):
            label = el.get("label", "")
            all_labels.add(label)
            pts = _parse_pts(el.get("points", ""))
            for pt in pts:
                img["keypoints"].append(
                    {"label": label, "x": pt[0], "y": pt[1]}
                )
        for el in img_el.findall("tag"):
            label = el.get("label", "")
            all_labels.add(label)
            img["tags"].append({"label": label})

        images.append(img)

    return images, sorted(all_labels)


def _parse_pts(pts_str):
    pts = []
    for pair in pts_str.strip().split(";"):
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


class FusionDataset(Dataset):
    """
    Returns image path + all GT annotations per image.
    The feature extraction and model forward passes happen
    in the training loop — not here — to allow gradient
    checkpointing and hook management.
    """

    def __init__(self, images, img_dir, input_size,
                 all_labels, tag_names, augment=True):
        self.img_dir    = img_dir
        self.input_size = input_size
        self.all_labels = all_labels
        self.tag_names  = tag_names
        self.label_to_id= {l: i for i, l in enumerate(all_labels)}
        self.tag_to_id  = {t: i for i, t in enumerate(tag_names)}
        self.augment    = augment
        self.samples    = [
            img for img in images
            if any([img["bboxes"], img["polygons"],
                    img["polylines"], img["keypoints"],
                    img["tags"]])
        ]
        self.tf = T.Compose([
            T.Resize((input_size, input_size)),
            T.ToTensor(),
            T.Normalize([0.485, 0.456, 0.406],
                        [0.229, 0.224, 0.225]),
        ])

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        img_info = self.samples[idx]
        img_path = os.path.join(self.img_dir, img_info["name"])
        image    = Image.open(img_path).convert("RGB")
        orig_w, orig_h = image.size
        tensor   = self.tf(image)

        # Build GT targets
        target = {
            "img_path" : img_path,
            "orig_w"   : orig_w,
            "orig_h"   : orig_h,
            # bbox: (N, 5) — [cls, cx, cy, w, h] normalized
            "gt_bboxes"    : self._build_bboxes(
                img_info, orig_w, orig_h
            ),
            # segmentation heatmap (num_classes, H, W)
            "gt_seg"       : self._build_seg(
                img_info, orig_w, orig_h
            ),
            # keypoint heatmap (1, H, W)
            "gt_kp_heatmap": self._build_kp_heatmap(
                img_info, orig_w, orig_h
            ),
            # polyline vertex heatmap (1, H, W)
            "gt_pl_heatmap": self._build_pl_heatmap(
                img_info, orig_w, orig_h
            ),
            # tag binary vector (num_tags,)
            "gt_tags"      : self._build_tags(img_info),
        }
        return tensor, target, img_path

    def _build_bboxes(self, img_info, orig_w, orig_h):
        boxes = []
        for ann in img_info["bboxes"]:
            cls_id = self.label_to_id.get(ann["label"])
            if cls_id is None:
                continue
            xtl, ytl = ann["xtl"], ann["ytl"]
            xbr, ybr = ann["xbr"], ann["ybr"]
            bw = xbr - xtl
            bh = ybr - ytl
            if bw <= 0 or bh <= 0:
                continue
            cx = (xtl + bw / 2) / orig_w
            cy = (ytl + bh / 2) / orig_h
            boxes.append([
                float(cls_id),
                max(0., min(1., cx)),
                max(0., min(1., cy)),
                max(0., min(1., bw / orig_w)),
                max(0., min(1., bh / orig_h)),
            ])
        if boxes:
            return torch.tensor(boxes, dtype=torch.float32)
        return torch.zeros((0, 5), dtype=torch.float32)

    def _build_seg(self, img_info, orig_w, orig_h):
        import cv2
        nc   = len(self.all_labels)
        size = self.input_size // 4
        seg  = torch.zeros(nc, size, size, dtype=torch.float32)
        for ann in img_info["polygons"]:
            cls_id = self.label_to_id.get(ann["label"])
            if cls_id is None:
                continue
            pts = np.array(ann["points"], dtype=np.float32)
            pts[:, 0] = pts[:, 0] * size / orig_w
            pts[:, 1] = pts[:, 1] * size / orig_h
            mask = np.zeros((size, size), dtype=np.uint8)
            cv2.fillPoly(mask, [pts.astype(np.int32)], 1)
            seg[cls_id] = torch.maximum(
                seg[cls_id],
                torch.from_numpy(mask).float()
            )
        return seg

    def _build_kp_heatmap(self, img_info, orig_w, orig_h):
        hm   = np.zeros(
            (self.input_size, self.input_size), dtype=np.float32
        )
        sigma= cfg.KP_HEATMAP_SIGMA
        for ann in img_info["keypoints"]:
            sx = ann["x"] * self.input_size / orig_w
            sy = ann["y"] * self.input_size / orig_h
            _paint_gaussian(hm, sx, sy, sigma, self.input_size)
        return torch.tensor(hm, dtype=torch.float32).unsqueeze(0)

    def _build_pl_heatmap(self, img_info, orig_w, orig_h):
        hm   = np.zeros(
            (self.input_size, self.input_size), dtype=np.float32
        )
        sigma= cfg.POLY_S1_HEATMAP_SIGMA
        for ann in img_info["polylines"]:
            for (px, py) in ann["points"]:
                sx = px * self.input_size / orig_w
                sy = py * self.input_size / orig_h
                _paint_gaussian(hm, sx, sy, sigma, self.input_size)
        return torch.tensor(hm, dtype=torch.float32).unsqueeze(0)

    def _build_tags(self, img_info):
        vec = torch.zeros(len(self.tag_names), dtype=torch.float32)
        for ann in img_info["tags"]:
            idx = self.tag_to_id.get(ann["label"])
            if idx is not None:
                vec[idx] = 1.0
        return vec


def _paint_gaussian(heatmap, cx, cy, sigma, size):
    x0 = max(0, int(cx) - 3 * sigma)
    x1 = min(size, int(cx) + 3 * sigma + 1)
    y0 = max(0, int(cy) - 3 * sigma)
    y1 = min(size, int(cy) + 3 * sigma + 1)
    for y in range(y0, y1):
        for x in range(x0, x1):
            d2  = (x - cx) ** 2 + (y - cy) ** 2
            val = math.exp(-d2 / (2 * sigma ** 2))
            if val > heatmap[y, x]:
                heatmap[y, x] = val


def collate_fn(batch):
    images    = torch.stack([b[0] for b in batch])
    targets   = [b[1] for b in batch]
    img_paths = [b[2] for b in batch]
    return images, targets, img_paths


# ============================================================
# --- Loss ---
# ============================================================

class FusionLoss(nn.Module):

    def __init__(self, num_poly_classes):
        super().__init__()
        self.num_poly_classes = num_poly_classes
        self.ce  = nn.CrossEntropyLoss()
        self.bce = nn.BCEWithLogitsLoss()
        self.mse = nn.MSELoss()

    def forward(self, refined, targets):
        device     = refined["tag"].device
        B          = refined["tag"].shape[0]
        loss_dict  = {}
        total_loss = torch.tensor(0.0, device=device)

        # ---- Bbox loss — CE on class scores ----
        bbox_logits = refined["bbox"]   # (B, nc+1)
        gt_cls      = torch.zeros(B, dtype=torch.long, device=device)
        for b, t in enumerate(targets):
            gt_b = t["gt_bboxes"]
            if len(gt_b) > 0:
                gt_cls[b] = int(gt_b[0, 0].item())
        l_bbox = self.ce(bbox_logits, gt_cls)
        loss_dict["bbox"]   = l_bbox.item()
        total_loss          = total_loss + l_bbox

        # ---- Polygon segmentation loss ----
        seg_logits = refined["polygon"]   # (B, nc+1, H, W)
        nc         = self.num_poly_classes
        H, W       = seg_logits.shape[2:]
        seg_gt     = torch.full((B, H, W), nc,
                                dtype=torch.long, device=device)
        for b, t in enumerate(targets):
            gt_seg = t["gt_seg"].to(device)   # (nc, H, W)
            gt_seg_r = F.interpolate(
                gt_seg.unsqueeze(0), size=(H, W), mode="nearest"
            ).squeeze(0)
            for c in range(nc):
                seg_gt[b][gt_seg_r[c] > 0.5] = c
        l_seg = self.ce(seg_logits, seg_gt)
        loss_dict["polygon"]= l_seg.item()
        total_loss          = total_loss + l_seg

        # ---- Keypoint heatmap loss ----
        kp_pred = refined["keypoint"].sigmoid()   # (B, 1, H, W)
        kp_gt   = torch.stack(
            [t["gt_kp_heatmap"].to(device) for t in targets]
        )
        kp_gt_r = F.interpolate(
            kp_gt, size=kp_pred.shape[2:],
            mode="bilinear", align_corners=False
        )
        l_kp = self.mse(kp_pred, kp_gt_r)
        loss_dict["keypoint"]= l_kp.item()
        total_loss           = total_loss + l_kp

        # ---- Polyline vertex heatmap loss ----
        pl_pred = refined["polyline"].sigmoid()   # (B, 1, H, W)
        pl_gt   = torch.stack(
            [t["gt_pl_heatmap"].to(device) for t in targets]
        )
        pl_gt_r = F.interpolate(
            pl_gt, size=pl_pred.shape[2:],
            mode="bilinear", align_corners=False
        )
        l_pl = self.mse(pl_pred, pl_gt_r)
        loss_dict["polyline"]= l_pl.item()
        total_loss           = total_loss + l_pl

        # ---- Tag loss ----
        tag_pred = refined["tag"]   # (B, num_tags)
        tag_gt   = torch.stack(
            [t["gt_tags"].to(device) for t in targets]
        )
        l_tag = self.bce(tag_pred, tag_gt)
        loss_dict["tag"]   = l_tag.item()
        total_loss         = total_loss + l_tag

        loss_dict["total"] = total_loss.item()
        return total_loss, loss_dict


# ============================================================
# --- Model loading helpers ---
# ============================================================

def load_all_models(device):
    """Load all specialist models. Returns dict of loaded models."""
    from ultralytics import YOLO

    models = {}

    def _exists(p):
        return p is not None and os.path.exists(p)

    if _exists(BBOX_MODEL_PATH):
        print(f"  ✓ Bbox      : {BBOX_MODEL_PATH}")
        m = YOLO(BBOX_MODEL_PATH)
        m.model.to(device).eval()
        for p in m.model.parameters():
            p.requires_grad = False
        models["bbox"] = m
    else:
        print(f"  ✗ Bbox      : not found")

    if _exists(POLYGON_MODEL_PATH):
        print(f"  ✓ Polygon   : {POLYGON_MODEL_PATH}")
        m = YOLO(POLYGON_MODEL_PATH)
        m.model.to(device).eval()
        for p in m.model.parameters():
            p.requires_grad = False
        models["polygon"] = m
    else:
        print(f"  ✗ Polygon   : not found")

    if _exists(KEYPOINT_MODEL_PATH):
        print(f"  ✓ Keypoint  : {KEYPOINT_MODEL_PATH}")
        sys.path.insert(0, os.path.join(
            os.path.dirname(os.path.abspath(__file__)),
            "keypoint_model"
        ))
        from train_keypoint import KeypointHeatmapModel
        ckpt = torch.load(KEYPOINT_MODEL_PATH, map_location="cpu")
        m    = KeypointHeatmapModel(
            backbone=cfg.KP_BACKBONE, pretrained=False
        )
        state = ckpt.get("model_state_dict", ckpt)
        m.load_state_dict(state)
        m.to(device).eval()
        for p in m.parameters():
            p.requires_grad = False
        models["keypoint"] = m
    else:
        print(f"  ✗ Keypoint  : not found")

    if _exists(POLYLINE_S1_PATH):
        print(f"  ✓ Polyline  : {POLYLINE_S1_PATH}")
        sys.path.insert(0, os.path.join(
            os.path.dirname(os.path.abspath(__file__)),
            "keypoint_model"
        ))
        from train_keypoint import KeypointHeatmapModel
        ckpt = torch.load(POLYLINE_S1_PATH, map_location="cpu")
        m    = KeypointHeatmapModel(
            backbone=cfg.POLY_S1_BACKBONE, pretrained=False
        )
        state = ckpt.get("model_state_dict", ckpt)
        m.load_state_dict(state)
        m.to(device).eval()
        for p in m.parameters():
            p.requires_grad = False
        models["polyline"] = m
    else:
        print(f"  ✗ Polyline  : not found")

    if _exists(TAG_MODEL_PATH):
        print(f"  ✓ Tag       : {TAG_MODEL_PATH}")
        sys.path.insert(0, os.path.join(
            os.path.dirname(os.path.abspath(__file__)),
            "tag_model"
        ))
        from train_tag import TagClassifier
        ckpt      = torch.load(TAG_MODEL_PATH, map_location="cpu")
        tag_names = ckpt.get("tag_names", [])
        m         = TagClassifier(
            num_tags=len(tag_names), pretrained=False
        )
        state = ckpt.get("model_state_dict", ckpt)
        m.load_state_dict(state)
        m.to(device).eval()
        for p in m.parameters():
            p.requires_grad = False
        models["tag"]      = m
        models["tag_names"]= tag_names
    else:
        print(f"  ✗ Tag       : not found")

    return models


def extract_features_batch(images, models_dict,
                            extractor, device,
                            img_paths=None):
    extractor.clear()
    feat_dict = {}

    with torch.no_grad():
        if "bbox" in models_dict:
            # Pass file paths to YOLO — it handles its own preprocessing
            src = img_paths if img_paths else images
            models_dict["bbox"].predict(src, verbose=False)
            if "bbox" in extractor.features:
                feat_dict["bbox"] = extractor.features["bbox"].detach().clone()

        if "polygon" in models_dict:
            src = img_paths if img_paths else images
            models_dict["polygon"].predict(src, verbose=False)
            if "polygon" in extractor.features:
                feat_dict["polygon"] = extractor.features["polygon"].detach().clone()

        if "keypoint" in models_dict:
            models_dict["keypoint"](images)
            if "keypoint" in extractor.features:
                feat_dict["keypoint"] = extractor.features["keypoint"].detach().clone()

        if "polyline" in models_dict:
            models_dict["polyline"](images)
            if "polyline" in extractor.features:
                feat_dict["polyline"] = extractor.features["polyline"].detach().clone()

        if "tag" in models_dict:
            models_dict["tag"](images)
            if "tag" in extractor.features:
                f = extractor.features["tag"]
                if f.dim() == 4:
                    f = f.flatten(2).mean(dim=2)
                feat_dict["tag"] = f.detach().clone()

    return feat_dict


# ============================================================
# --- Training ---
# ============================================================

def train():
    print("\n" + "═"*55)
    print("  Attention Fusion — Training")
    print("═"*55)

    random.seed(RANDOM_SEED)
    torch.manual_seed(RANDOM_SEED)
    os.makedirs(FUSION_SAVE_DIR, exist_ok=True)

    device = torch.device(
        cfg.DEVICE if torch.cuda.is_available()
        or cfg.DEVICE == "cpu" else "cpu"
    )
    print(f"  Device : {device}")

    # Load specialist models
    print("\n--- Loading specialist models ---")
    models_dict = load_all_models(device)

    assert len(models_dict) >= 2, \
        "Need at least 2 specialist models to train fusion."

    tag_names = models_dict.pop("tag_names", [])

    # Register hooks
    extractor = FeatureExtractor()
    if "bbox"     in models_dict:
        register_yolo_hook(models_dict["bbox"],
                           extractor, "bbox")
    if "polygon"  in models_dict:
        register_yolo_hook(models_dict["polygon"],
                           extractor, "polygon")
    if "keypoint" in models_dict:
        register_resnet_hook(models_dict["keypoint"],
                             extractor, "keypoint")
    if "polyline" in models_dict:
        register_resnet_hook(models_dict["polyline"],
                             extractor, "polyline")
    if "tag"      in models_dict:
        register_tag_hook(models_dict["tag"], extractor, "tag")

    # Probe feature dimensions
    # ---- Probe ALL model feature dims including YOLO ----
    print("\n--- Probing feature dimensions ---")
    feat_dims = {}
    extractor.clear()

    # For YOLO models use a dummy image file on disk
    # (YOLO predict needs a file path or numpy array, not a normalized tensor)
    import tempfile, cv2 as _cv2
    dummy_np  = np.zeros((cfg.INPUT_SIZE, cfg.INPUT_SIZE, 3), dtype=np.uint8)
    tmp_file  = os.path.join(tempfile.gettempdir(), "dummy_probe.jpg")
    _cv2.imwrite(tmp_file, dummy_np)

    with torch.no_grad():
        if "bbox" in models_dict:
            extractor.clear()
            models_dict["bbox"].predict(tmp_file, verbose=False)
            if "bbox" in extractor.features:
                f = extractor.features["bbox"]
                feat_dims["bbox"] = f.shape[1]
                print(f"    bbox     : {f.shape}")

        if "polygon" in models_dict:
            extractor.clear()
            models_dict["polygon"].predict(tmp_file, verbose=False)
            if "polygon" in extractor.features:
                f = extractor.features["polygon"]
                feat_dims["polygon"] = f.shape[1]
                print(f"    polygon  : {f.shape}")

        dummy_t = torch.zeros(
            1, 3, cfg.INPUT_SIZE, cfg.INPUT_SIZE
        ).to(device)

        if "keypoint" in models_dict:
            extractor.clear()
            models_dict["keypoint"](dummy_t)
            if "keypoint" in extractor.features:
                f = extractor.features["keypoint"]
                feat_dims["keypoint"] = f.shape[1]
                print(f"    keypoint : {f.shape}")

        if "polyline" in models_dict:
            extractor.clear()
            models_dict["polyline"](dummy_t)
            if "polyline" in extractor.features:
                f = extractor.features["polyline"]
                feat_dims["polyline"] = f.shape[1]
                print(f"    polyline : {f.shape}")

        if "tag" in models_dict:
            extractor.clear()
            models_dict["tag"](dummy_t)
            if "tag" in extractor.features:
                f = extractor.features["tag"]
                if f.dim() == 4:
                    f = f.flatten(2).mean(dim=2)
                feat_dims["tag"] = f.shape[1]
                print(f"    tag      : {f.shape}")

    extractor.clear()
    os.remove(tmp_file)

    # Safe fallbacks for any missing modality
    feat_dims.setdefault("bbox",     256)
    feat_dims.setdefault("polygon",  256)
    feat_dims.setdefault("keypoint", 16)
    feat_dims.setdefault("polyline", 16)
    feat_dims.setdefault("tag",      1280)

    print(f"  Feature dims: {feat_dims}")

    # Parse ground truth
    print("\n--- Parsing ground truth XML ---")
    images, all_labels = parse_gt_xml(GT_XML_PATH)
    print(f"  Images : {len(images)}")
    print(f"  Labels : {all_labels}")
    if not tag_names:
        tag_names = sorted({
            ann["label"]
            for img in images
            for ann in img["tags"]
        })
    print(f"  Tags   : {tag_names}")

    # Dataset
    full_ds = FusionDataset(
        images     = images,
        img_dir    = IMG_DIR,
        input_size = cfg.INPUT_SIZE,
        all_labels = all_labels,
        tag_names  = tag_names,
        augment    = True,
    )
    n_val   = max(1, int(len(full_ds) * VAL_RATIO))
    n_train = len(full_ds) - n_val
    train_ds, val_ds = torch.utils.data.random_split(
        full_ds, [n_train, n_val],
        generator=torch.Generator().manual_seed(RANDOM_SEED)
    )
    val_ds.dataset.augment = False

    train_loader = DataLoader(
        train_ds, batch_size=BATCH_SIZE, shuffle=True,
        num_workers=NUM_WORKERS, collate_fn=collate_fn
    )
    val_loader = DataLoader(
        val_ds, batch_size=BATCH_SIZE, shuffle=False,
        num_workers=NUM_WORKERS, collate_fn=collate_fn
    )
    print(f"  Train : {n_train} | Val : {n_val}")

    # Build fusion model
    fusion_model = AttentionFusion(
        bbox_feat_dim    = feat_dims.get("bbox",    256),
        polygon_feat_dim = feat_dims.get("polygon", 256),
        kp_feat_dim      = feat_dims.get("keypoint",256),
        poly_feat_dim    = feat_dims.get("polyline",256),
        tag_feat_dim     = feat_dims.get("tag",     1280),
        num_bbox_classes = len(all_labels),
        num_poly_classes = len(all_labels),
        num_tags         = max(len(tag_names), 1),
        fusion_dim       = FUSION_DIM,
        fusion_spatial   = FUSION_SPATIAL,
        num_heads        = NUM_HEADS,
        num_layers       = NUM_LAYERS,
        dropout          = DROPOUT,
        input_size       = cfg.INPUT_SIZE,
    ).to(device)

    total_p = sum(p.numel() for p in fusion_model.parameters())
    print(f"\n  Fusion params : {total_p/1e6:.2f}M")

    criterion = FusionLoss(num_poly_classes=len(all_labels))
    optimizer = torch.optim.AdamW(
        fusion_model.parameters(),
        lr=LR, weight_decay=WEIGHT_DECAY
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=EPOCHS
    )

    log_path  = os.path.join(FUSION_SAVE_DIR, "train_log.csv")
    best_val  = float("inf")
    best_path = os.path.join(FUSION_SAVE_DIR, "fusion_best.pt")
    start     = time.time()

    with open(log_path, "w", newline="") as f:
        csv.writer(f).writerow([
            "epoch", "train_loss", "val_loss", "epoch_time"
        ])

    for epoch in range(1, EPOCHS + 1):
        t0 = time.time()

        # ---- Train ----
        fusion_model.train()
        train_loss = 0.0
        for images_batch, targets, img_paths in train_loader:
            images_batch = images_batch.to(device)
            feat_dict    = extract_features_batch(
                images_batch, models_dict, extractor, device,
                img_paths=img_paths
            
            )
            if len(feat_dict) < 2:
                continue

            # Fill missing modalities with zeros
            for mod in AttentionFusion.MODALITIES:
                if mod not in feat_dict:
                    ref = next(iter(feat_dict.values()))
                    feat_dict[mod] = torch.zeros(
                        ref.shape[0], 1,
                        *ref.shape[2:]
                        if ref.dim() == 4
                        else (1,),
                        device=device
                    )

            refined, _ = fusion_model(feat_dict)
            loss, _    = criterion(refined, targets)
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(
                fusion_model.parameters(), 1.0
            )
            optimizer.step()
            train_loss += loss.item()
        train_loss /= max(len(train_loader), 1)

        # ---- Validate ----
        fusion_model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for images_batch, targets, img_paths in val_loader:
                images_batch = images_batch.to(device)
                feat_dict    = extract_features_batch(
                    images_batch, models_dict, extractor, device,
                    img_paths=img_paths
                )
                if len(feat_dict) < 2:
                    continue
                for mod in AttentionFusion.MODALITIES:
                    if mod not in feat_dict:
                        ref = next(iter(feat_dict.values()))
                        feat_dict[mod] = torch.zeros_like(
                            ref[:, :1]
                        )
                refined, _ = fusion_model(feat_dict)
                loss, _    = criterion(refined, targets)
                val_loss  += loss.item()
        val_loss /= max(len(val_loader), 1)

        scheduler.step()
        ep_time = time.time() - t0

        if val_loss < best_val:
            best_val = val_loss
            torch.save({
                "epoch"            : epoch,
                "model_state_dict" : fusion_model.state_dict(),
                "all_labels"       : all_labels,
                "tag_names"        : tag_names,
                "feat_dims"        : feat_dims,
                "val_loss"         : best_val,
            }, best_path)

        with open(log_path, "a", newline="") as f:
            csv.writer(f).writerow([
                epoch, f"{train_loss:.6f}",
                f"{val_loss:.6f}", f"{ep_time:.1f}"
            ])

        if epoch % 10 == 0 or epoch == 1:
            print(f"  Epoch {epoch:>4}/{EPOCHS} | "
                  f"train {train_loss:.4f} | "
                  f"val {val_loss:.4f} | "
                  f"{ep_time:.1f}s")

    extractor.remove()
    elapsed = str(datetime.timedelta(
        seconds=int(time.time() - start)
    ))
    print(f"\n  Training complete | Time: {elapsed}")
    print(f"  Best model → {best_path}")


# ============================================================
# --- Inference ---
# ============================================================

def test():
    print("\n" + "═"*55)
    print("  Attention Fusion — Inference")
    print("═"*55)

    os.makedirs(TEST_OUTPUT_DIR, exist_ok=True)
    os.makedirs(os.path.join(TEST_OUTPUT_DIR, "annotated"),
                exist_ok=True)

    device = torch.device(
        cfg.DEVICE if torch.cuda.is_available()
        or cfg.DEVICE == "cpu" else "cpu"
    )

    best_path = os.path.join(FUSION_SAVE_DIR, "fusion_best.pt")
    assert os.path.exists(best_path), \
        f"Fusion model not found: {best_path}\nRun training first."

    ckpt       = torch.load(best_path, map_location="cpu")
    all_labels = ckpt["all_labels"]
    tag_names  = ckpt["tag_names"]
    feat_dims  = ckpt["feat_dims"]

    # Load specialist models
    print("\n--- Loading specialist models ---")
    models_dict = load_all_models(device)
    models_dict.pop("tag_names", None)

    # Register hooks
    extractor = FeatureExtractor()
    if "bbox"     in models_dict:
        register_yolo_hook(models_dict["bbox"],
                           extractor, "bbox")
    if "polygon"  in models_dict:
        register_yolo_hook(models_dict["polygon"],
                           extractor, "polygon")
    if "keypoint" in models_dict:
        register_resnet_hook(models_dict["keypoint"],
                             extractor, "keypoint")
    if "polyline" in models_dict:
        register_resnet_hook(models_dict["polyline"],
                             extractor, "polyline")
    if "tag"      in models_dict:
        register_tag_hook(models_dict["tag"], extractor, "tag")

    # Load fusion model
    fusion_model = AttentionFusion(
        bbox_feat_dim    = feat_dims.get("bbox",    256),
        polygon_feat_dim = feat_dims.get("polygon", 256),
        kp_feat_dim      = feat_dims.get("keypoint",256),
        poly_feat_dim    = feat_dims.get("polyline",256),
        tag_feat_dim     = feat_dims.get("tag",     1280),
        num_bbox_classes = len(all_labels),
        num_poly_classes = len(all_labels),
        num_tags         = max(len(tag_names), 1),
        fusion_dim       = FUSION_DIM,
        fusion_spatial   = FUSION_SPATIAL,
        num_heads        = NUM_HEADS,
        num_layers       = NUM_LAYERS,
        dropout          = DROPOUT,
        input_size       = cfg.INPUT_SIZE,
    ).to(device)
    fusion_model.load_state_dict(ckpt["model_state_dict"])
    fusion_model.eval()

    # Color map
    color_map = {
        l: [(0,229,160),(0,153,255),(255,107,53),
            (192,132,252),(255,215,0),(255,99,132),
            (75,192,192),(255,159,64),(153,102,255),
            (54,162,235)][i % 10]
        for i, l in enumerate(all_labels + tag_names)
    }

    tf = T.Compose([
        T.Resize((cfg.INPUT_SIZE, cfg.INPUT_SIZE)),
        T.ToTensor(),
        T.Normalize([0.485,0.456,0.406],[0.229,0.224,0.225]),
    ])

    image_paths = sorted([
        p for ext in IMAGE_EXTS
        for p in glob.glob(
            os.path.join(TEST_IMAGES_DIR, f"*{ext}")
        )
    ])
    assert len(image_paths) > 0, \
        f"No images in: {TEST_IMAGES_DIR}"

    print(f"  Images : {len(image_paths)}")
    xml_root = ET.Element("annotations")
    ET.SubElement(xml_root, "version").text = "1.1"

    for img_path in image_paths:
        fname  = os.path.basename(img_path)
        orig   = cv2.imread(img_path)
        if orig is None:
            continue
        orig_h, orig_w = orig.shape[:2]
        pil    = Image.fromarray(
            cv2.cvtColor(orig, cv2.COLOR_BGR2RGB)
        )
        tensor = tf(pil).unsqueeze(0).to(device)

        extractor.clear()
        feat_dict = extract_features_batch(
            tensor, models_dict, extractor, device,
            img_paths=[img_path]
        )
        for mod in AttentionFusion.MODALITIES:
            if mod not in feat_dict:
                ref = next(iter(feat_dict.values()))
                feat_dict[mod] = torch.zeros_like(ref[:, :1])

        with torch.no_grad():
            refined, _ = fusion_model(feat_dict)

        output    = orig.copy()
        img_el    = ET.SubElement(xml_root, "image")
        img_el.set("name",   fname)
        img_el.set("width",  str(orig_w))
        img_el.set("height", str(orig_h))

        # ---- Draw bbox ----
        bbox_probs = refined["bbox"][0].softmax(0).cpu().numpy()
        cls_id     = bbox_probs[:-1].argmax()
        if bbox_probs[cls_id] > BBOX_THRESH:
            label = all_labels[cls_id] \
                    if cls_id < len(all_labels) else "unknown"
            color = color_map.get(label, (200,200,200))
            el = ET.SubElement(img_el, "tag")
            el.set("label", f"pred_bbox:{label}")

        # ---- Draw polygon ----
        seg_logits = refined["polygon"]   # (1, nc+1, H, W)
        seg_probs  = seg_logits[0].softmax(0).cpu().numpy()
        seg_up     = cv2.resize(
            seg_probs.transpose(1,2,0),
            (orig_w, orig_h)
        ).transpose(2,0,1)
        overlay = output.copy()
        for c in range(len(all_labels)):
            mask = (seg_up[c] > POLY_THRESH).astype(np.uint8)
            if mask.sum() == 0:
                continue
            label = all_labels[c]
            color = color_map.get(label, (200,200,200))
            overlay[mask == 1] = color
            cnts, _ = cv2.findContours(
                mask, cv2.RETR_EXTERNAL,
                cv2.CHAIN_APPROX_SIMPLE
            )
            pts_list = []
            for cnt in cnts:
                if len(cnt) >= 3:
                    pts_list.append(cnt)
                    el = ET.SubElement(img_el, "polygon")
                    el.set("label", label)
                    pts_str = ";".join(
                        f"{p[0][0]:.1f},{p[0][1]:.1f}"
                        for p in cnt
                    )
                    el.set("points", pts_str)
            cv2.drawContours(output, pts_list, -1, color, 1)
        output = cv2.addWeighted(overlay, 0.35, output, 0.65, 0)

        # ---- Draw keypoints ----
        kp_hm = refined["keypoint"][0,0].sigmoid().cpu().numpy()
        kp_hm = cv2.resize(kp_hm, (orig_w, orig_h))
        lmax  = (kp_hm == maximum_filter(kp_hm, size=35))
        for (y, x) in np.argwhere(lmax & (kp_hm > KP_THRESH)):
            cv2.circle(output, (int(x), int(y)), 5,
                       (0,229,160), -1)
            el = ET.SubElement(img_el, "points")
            el.set("label", "keypoint")
            el.set("points", f"{x:.1f},{y:.1f}")

        # ---- Draw polylines ----
        pl_hm = refined["polyline"][0,0].sigmoid().cpu().numpy()
        pl_hm = cv2.resize(pl_hm, (orig_w, orig_h))
        lmax  = (pl_hm == maximum_filter(pl_hm, size=35))
        verts = np.argwhere(lmax & (pl_hm > POLY_V_THRESH))
        if len(verts) >= 2:
            # Simple nearest-neighbour chain
            used  = set()
            chain = [0]
            used.add(0)
            while len(chain) < len(verts):
                cur = chain[-1]
                cy, cx = verts[cur]
                best_d = float("inf")
                best_j = -1
                for j in range(len(verts)):
                    if j in used:
                        continue
                    ny, nx = verts[j]
                    d = math.sqrt(
                        (nx-cx)**2 + (ny-cy)**2
                    )
                    if d < best_d:
                        best_d = d
                        best_j = j
                if best_j == -1 or best_d > cfg.POLY_MAX_DIST \
                        * orig_w:
                    break
                chain.append(best_j)
                used.add(best_j)
            pts = [(int(verts[i][1]), int(verts[i][0]))
                   for i in chain]
            for k in range(len(pts)-1):
                cv2.line(output, pts[k], pts[k+1],
                         (255,107,53), 2)
            if len(pts) >= 2:
                el = ET.SubElement(img_el, "polyline")
                el.set("label", "polyline")
                el.set("points", ";".join(
                    f"{p[0]:.1f},{p[1]:.1f}" for p in pts
                ))

        # ---- Draw tags ----
        tag_probs = refined["tag"][0].sigmoid().cpu().numpy()
        tag_probs = np.atleast_1d(tag_probs)
        y_off = 20
        for i, tname in enumerate(tag_names):
            if i < len(tag_probs) and \
                    tag_probs[i] > TAG_THRESH:
                color = color_map.get(tname, (255,215,0))
                text  = f"[TAG] {tname} " \
                        f"{tag_probs[i]*100:.0f}%"
                (tw,th),_ = cv2.getTextSize(
                    text, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1
                )
                cv2.rectangle(output, (8,y_off-th-4),
                              (8+tw+4,y_off+2), color, -1)
                cv2.putText(output, text, (10,y_off),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                            (0,0,0), 1, cv2.LINE_AA)
                y_off += th + 8
                el = ET.SubElement(img_el, "tag")
                el.set("label", tname)

        cv2.imwrite(
            os.path.join(TEST_OUTPUT_DIR, "annotated", fname),
            output
        )
        print(f"  {fname}")

    # Save CVAT XML
    raw      = ET.tostring(xml_root, encoding="unicode")
    reparsed = minidom.parseString(raw)
    xml_str  = reparsed.toprettyxml(indent="  ", encoding=None)
    xml_path = os.path.join(TEST_OUTPUT_DIR,
                            "fusion_predictions.xml")
    with open(xml_path, "w", encoding="utf-8") as f:
        f.write(xml_str)

    extractor.remove()
    print(f"\n  CVAT XML  → {xml_path}")
    print(f"  Annotated → {TEST_OUTPUT_DIR}/annotated/")


# ============================================================
# --- Entry point ---
# ============================================================

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--mode", choices=["train", "test"],
        default="train",
        help="train: train fusion module | test: run inference"
    )
    args = parser.parse_args()

    if args.mode == "train":
        train()
    else:
        test()
