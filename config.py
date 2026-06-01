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
CVAT_XML        = r"D:\muhtasim\model-trn\multi_pipeline\datasets\cricket\cvat_dfghjkl.xml"

# Folder containing all images
IMG_DIR         = r"D:\muhtasim\model-trn\multi_pipeline\datasets\cricket\images"

# Root output folder — all models, splits, and logs saved here
SAVE_DIR        = r"D:\muhtasim\model-trn\multi_pipeline\saved\cricket"

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
# --- MODEL ENABLE/DISABLE (True = train, False = skip) ---
# ============================================================

TRAIN_BBOX      = True
TRAIN_POLYGON   = True
TRAIN_KEYPOINT  = True
TRAIN_POLYLINE  = True
TRAIN_TAG       = True

# ============================================================
# --- GENERAL TRAINING ---
# ============================================================

DEVICE          = "cuda"        # "cuda" or "cpu"
INPUT_SIZE      = 640           # input image size for all models
VAL_RATIO       = 0.2          # fraction of data used for validation
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
YOLO_PATIENCE       = 20
YOLO_SCORE_THRESH   = 0.25
YOLO_NMS_THRESH     = 0.45
YOLO_AUGMENT        = True
YOLO_OPTIMIZER      = 'AdamW'
YOLO_MOSAIC         = 1
YOLO_RETINA_MASKS   = True

# ============================================================
# --- POLYGON-SEG (HRNet per-class semantic segmentation) ---
# When POLYGON_USE_SEG is True, master_train.py routes the
# polygon step to polygon_seg_model/train_polygon_seg.py
# (HRNet-W18 per-class segmentation + BCE+Dice). Set False to
# fall back to YOLO-seg (polygon_model/train_polygon.py).
# ============================================================

POLYGON_USE_SEG              = False
POLYGON_SEG_BACKBONE         = "hrnet_w18"
POLYGON_SEG_PRETRAINED       = True
POLYGON_SEG_INPUT_SIZE       = 640
POLYGON_SEG_EPOCHS           = 100
POLYGON_SEG_BATCH_SIZE       = 4
POLYGON_SEG_LR               = 1e-4
POLYGON_SEG_WEIGHT_DECAY     = 1e-4
POLYGON_SEG_WARMUP_ITERS     = 500
POLYGON_SEG_BCE_WEIGHT       = 1.0
POLYGON_SEG_DICE_WEIGHT      = 1.0
POLYGON_SEG_THRESH           = 0.5
POLYGON_SEG_CHECKPOINT_EVERY = 20
POLYGON_SEG_CLASSES          = []     # auto-populated at runtime by master_train.py
POLYGON_SEG_MIN_AREA         = 100    # min contour area (px) to filter noise at inference
POLYGON_SEG_EPSILON_RATIO    = 0.002  # Douglas-Peucker epsilon as fraction of contour perimeter

# ============================================================
# --- TAG MODEL ---
# ============================================================

TAG_BACKBONE        = "efficientnet_b0"
TAG_PRETRAINED      = True
TAG_EPOCHS          = 40
TAG_BATCH_SIZE      = 64
TAG_LR              = 1e-4
TAG_WEIGHT_DECAY    = 1e-4
TAG_SCORE_THRESH    = 0.5
TAG_CHECKPOINT_EVERY= 10

# ============================================================
# --- SHAPE REFINER (data-driven polygon postprocessing) ---
# Enable to apply shape_refiner.py at master_test.py inference:
#   Pass 1: simplifies each polygon to its class's median vertex
#           count (derived from class_priors.json).
#   Pass 2: enforces mutual exclusion for always_zero class pairs.
# Disable to compare against raw model output.
# ============================================================

SHAPE_REFINER_ENABLED = True
CLASS_PRIORS_PATH     = os.path.join(SAVE_DIR, "class_priors.json")

# ============================================================
# --- POLYLINE-SEG (HRNet-W18 per-class segmentation) ---
# master_train.py routes the polyline step to
# polyline_model_working/train_polyline_seg.py
# (HRNet-W18 per-class segmentation + BCE+Dice).
# ============================================================

POLY_SEG_BACKBONE         = "hrnet_w18"
POLY_SEG_PRETRAINED       = True
POLY_SEG_INPUT_SIZE       = 640
POLY_SEG_EPOCHS           = 100
POLY_SEG_BATCH_SIZE       = 4
POLY_SEG_LR               = 1e-4
POLY_SEG_WEIGHT_DECAY     = 1e-4
POLY_SEG_WARMUP_ITERS     = 500
POLY_SEG_MASK_THICKNESS   = 5
POLY_SEG_BCE_WEIGHT       = 1.0
POLY_SEG_DICE_WEIGHT      = 1.0
POLY_SEG_THRESH           = 0.5
POLY_SEG_CHECKPOINT_EVERY = 20
POLY_SEG_CLASSES          = []   # auto-populated at runtime by master_train.py

# ============================================================
# --- KEYPOINT-SEG (HRNet per-class segmentation) ---
# master_train.py routes the keypoint step to
# keypoint_seg_model/train_keypoint_seg.py
# (HRNet-W18 per-class small-disk segmentation + BCE+Dice).
# ============================================================

KP_SEG_BACKBONE          = "hrnet_w18"
KP_SEG_PRETRAINED        = True
KP_SEG_INPUT_SIZE        = 640
KP_SEG_EPOCHS            = 100
KP_SEG_BATCH_SIZE        = 4
KP_SEG_LR                = 1e-4
KP_SEG_WEIGHT_DECAY      = 1e-4
KP_SEG_WARMUP_ITERS      = 500
KP_SEG_DISK_RADIUS       = 6      # px on input_size grid; ~6/640 ≈ 0.9% of image
KP_SEG_BCE_WEIGHT        = 1.0
KP_SEG_DICE_WEIGHT       = 1.0
KP_SEG_THRESH            = 0.5
KP_SEG_CHECKPOINT_EVERY  = 20
KP_SEG_CLASSES           = []     # auto-populated at runtime by master_train.py
KP_SEG_NMS_MIN_DIST      = 8      # px on input_size grid; min separation between peaks
