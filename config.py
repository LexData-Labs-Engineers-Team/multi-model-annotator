# # ============================================================
# # config.py — Pipeline Configuration
# # Edit the PATHS section before running master_train.py
# # ============================================================

# import os

# # ============================================================
# # --- PATHS --- Edit these before running
# # ============================================================

# # Combined COCO JSON exported from CVAT (via converter.py)
# COCO_JSON       = "/home/lexdata/Documents/LexAnnotate_demo/m_env/multi_pipeline/datasets/flower/annotations/annotations.json"

# # Folder containing all images
# IMG_DIR         = "/home/lexdata/Documents/LexAnnotate_demo/m_env/multi_pipeline/datasets/bonsai_v2/images"

# # Root output folder — all models, splits, and logs saved here
# SAVE_DIR        = "/home/lexdata/Documents/LexAnnotate_demo/m_env/multi_pipeline/saved/bonsai_v2"

# # ============================================================
# # --- DERIVED PATHS — auto-set, do not edit ---
# # ============================================================

# # Data preparation outputs
# DATA_DIR            = os.path.join(SAVE_DIR, "data")
# BBOX_JSON           = os.path.join(DATA_DIR, "bbox_annotations.json")
# POLYGON_JSON        = os.path.join(DATA_DIR, "polygon_annotations.json")
# KEYPOINT_JSON       = os.path.join(DATA_DIR, "keypoint_annotations.json")
# POLYLINE_JSON       = os.path.join(DATA_DIR, "polyline_annotations.json")
# TAG_JSON            = os.path.join(DATA_DIR, "tag_annotations.json")

# # YOLO format data directories
# YOLO_BBOX_DIR       = os.path.join(DATA_DIR, "yolo_bbox")
# YOLO_POLYGON_DIR    = os.path.join(DATA_DIR, "yolo_polygon")

# # Model save directories
# BBOX_SAVE_DIR       = os.path.join(SAVE_DIR, "bbox_model")
# POLYGON_SAVE_DIR    = os.path.join(SAVE_DIR, "polygon_model")
# KEYPOINT_SAVE_DIR   = os.path.join(SAVE_DIR, "keypoint_model")
# POLYLINE_SAVE_DIR   = os.path.join(SAVE_DIR, "polyline_model")
# TAG_SAVE_DIR        = os.path.join(SAVE_DIR, "tag_model")

# # Master log
# MASTER_LOG          = os.path.join(SAVE_DIR, "master_train_log.txt")

# # ============================================================
# # --- GENERAL TRAINING ---
# # ============================================================

# DEVICE          = "cpu"        # "cuda" or "cpu"
# INPUT_SIZE      = 320           # input image size for all models
# VAL_RATIO       = 0.3          # fraction of data used for validation
# RANDOM_SEED     = 42
# NUM_WORKERS     = 2

# # ImageNet normalization — shared across all models
# PIXEL_MEAN      = [0.485, 0.456, 0.406]
# PIXEL_STD       = [0.229, 0.224, 0.225]

# # ============================================================
# # --- YOLO (BBOX + POLYGON MODELS) ---
# # ============================================================

# YOLO_MODEL_SIZE     = "yolov8n-seg.pt"   # n=nano — change to s/m/l/x as needed
# YOLO_EPOCHS         = 100
# YOLO_BATCH_SIZE     = 4
# YOLO_LR             = 0.01
# YOLO_PATIENCE       = 20                 # early stopping patience
# YOLO_SCORE_THRESH   = 0.25
# YOLO_NMS_THRESH     = 0.45
# YOLO_AUGMENT        = True

# # ============================================================
# # --- KEYPOINT MODEL ---
# # ============================================================

# KP_BACKBONE         = "resnet50"         # resnet18 / resnet50
# KP_PRETRAINED       = True
# KP_EPOCHS           = 80
# KP_BATCH_SIZE       = 8
# KP_LR               = 1e-4
# KP_WEIGHT_DECAY     = 1e-4
# KP_HEATMAP_SIGMA    = 8                  # gaussian sigma for heatmap targets
# KP_SCORE_THRESH     = 0.3               # minimum heatmap peak value to keep
# KP_CHECKPOINT_EVERY = 10

# # ============================================================
# # --- POLYLINE MODEL ---
# # ============================================================

# # Stage 1 — vertex heatmap (same architecture as keypoint model)
# POLY_S1_BACKBONE        = "resnet50"
# POLY_S1_PRETRAINED      = True
# POLY_S1_EPOCHS          = 80
# POLY_S1_BATCH_SIZE      = 8
# POLY_S1_LR              = 1e-4
# POLY_S1_WEIGHT_DECAY    = 1e-4
# POLY_S1_HEATMAP_SIGMA   = 6
# POLY_S1_SCORE_THRESH    = 0.3
# POLY_S1_CHECKPOINT_EVERY= 10

# # Stage 2 — edge connectivity MLP
# POLY_S2_EPOCHS          = 50
# POLY_S2_BATCH_SIZE      = 32
# POLY_S2_LR              = 1e-4
# POLY_S2_WEIGHT_DECAY    = 1e-4
# POLY_S2_HIDDEN_DIM      = 128
# POLY_S2_MAX_DIST        = 0.15          # max normalized distance to consider edge
# POLY_S2_CHECKPOINT_EVERY= 10

# # ============================================================
# # --- TAG MODEL ---
# # ============================================================

# TAG_BACKBONE        = "efficientnet_b0"
# TAG_PRETRAINED      = True
# TAG_EPOCHS          = 60
# TAG_BATCH_SIZE      = 16
# TAG_LR              = 1e-4
# TAG_WEIGHT_DECAY    = 1e-4
# TAG_SCORE_THRESH    = 0.5               # sigmoid threshold for tag presence
# TAG_CHECKPOINT_EVERY= 10


# ============================================================
# config.py — Pipeline Configuration
# Edit the PATHS section before running master_train.py
# ============================================================

import os

# ============================================================
# --- PATHS --- Edit these before running
# ============================================================

# CVAT for Images 1.1 XML — exported directly from CVAT
# This is the single source of truth for all annotation types
CVAT_XML        = "/home/lexdata/Documents/LexAnnotate_demo/m_env/multi_pipeline/datasets/lettuce_v2/annotations/annotations.xml"

# Folder containing all images
IMG_DIR         = "/home/lexdata/Documents/LexAnnotate_demo/m_env/multi_pipeline/datasets/lettuce_v2/images"

# Root output folder — all models, splits, and logs saved here
SAVE_DIR        = "/home/lexdata/Documents/LexAnnotate_demo/m_env/multi_pipeline/saved/lettuce_v2"

# ============================================================
# --- DERIVED PATHS — auto-set, do not edit ---
# ============================================================

# YOLO format data directories (converted from XML at runtime)
DATA_DIR            = os.path.join(SAVE_DIR, "data")
YOLO_BBOX_DIR       = os.path.join(DATA_DIR, "yolo_bbox")
YOLO_POLYGON_DIR    = os.path.join(DATA_DIR, "yolo_polygon")

# Model save directories
BBOX_SAVE_DIR       = os.path.join(SAVE_DIR, "bbox_model")
POLYGON_SAVE_DIR    = os.path.join(SAVE_DIR, "polygon_model")
KEYPOINT_SAVE_DIR   = os.path.join(SAVE_DIR, "keypoint_model")
POLYLINE_SAVE_DIR   = os.path.join(SAVE_DIR, "polyline_model")
TAG_SAVE_DIR        = os.path.join(SAVE_DIR, "tag_model")

# Master log
MASTER_LOG          = os.path.join(SAVE_DIR, "master_train_log.txt")

# ============================================================
# --- GENERAL TRAINING ---
# ============================================================

DEVICE          = "cpu"        # "cuda" or "cpu"
INPUT_SIZE      = 640           # input image size for all models
VAL_RATIO       = 0.4          # fraction of data used for validation
RANDOM_SEED     = 42
NUM_WORKERS     = 4

# ImageNet normalization — shared across all models
PIXEL_MEAN      = [0.485, 0.456, 0.406]
PIXEL_STD       = [0.229, 0.224, 0.225]

# ============================================================
# --- YOLO (BBOX + POLYGON MODELS) ---
# ============================================================

YOLO_BBOX_MODEL_SIZE     = "yolov8n.pt"
YOLO_POLYGON_MODEL_SIZE     = "yolov8n-seg.pt"
YOLO_EPOCHS         = 200
YOLO_BATCH_SIZE     = 8
YOLO_LR             = 0.01
YOLO_PATIENCE       = 100
YOLO_SCORE_THRESH   = 0.25
YOLO_NMS_THRESH     = 0.45
YOLO_AUGMENT        = True

# ============================================================
# --- KEYPOINT MODEL ---
# ============================================================

KP_BACKBONE         = "resnet50"
KP_PRETRAINED       = True
KP_EPOCHS           = 200
KP_BATCH_SIZE       = 8
KP_LR               = 1e-4
KP_WEIGHT_DECAY     = 1e-4
KP_HEATMAP_SIGMA    = 8
KP_SCORE_THRESH     = 0.3
KP_CHECKPOINT_EVERY = 10

# ============================================================
# --- POLYLINE MODEL ---
# ============================================================

POLY_S1_BACKBONE        = "resnet50"
POLY_S1_PRETRAINED      = True
POLY_S1_EPOCHS          = 200
POLY_S1_BATCH_SIZE      = 8
POLY_S1_LR              = 1e-4
POLY_S1_WEIGHT_DECAY    = 1e-4
POLY_S1_HEATMAP_SIGMA   = 6
POLY_S1_SCORE_THRESH    = 0.3
POLY_S1_CHECKPOINT_EVERY= 10

POLY_S2_EPOCHS          = 200
POLY_S2_BATCH_SIZE      = 16
POLY_S2_LR              = 1e-4
POLY_S2_WEIGHT_DECAY    = 1e-4
POLY_S2_HIDDEN_DIM      = 128
POLY_S2_MAX_DIST        = 0.15
POLY_S2_CHECKPOINT_EVERY= 10

# ============================================================
# --- TAG MODEL ---
# ============================================================

TAG_BACKBONE        = "efficientnet_b0"
TAG_PRETRAINED      = True
TAG_EPOCHS          = 100
TAG_BATCH_SIZE      = 32
TAG_LR              = 1e-4
TAG_WEIGHT_DECAY    = 1e-4
TAG_SCORE_THRESH    = 0.5
TAG_CHECKPOINT_EVERY= 10


EDGE_THRESH = 0.5
