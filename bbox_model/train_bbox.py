# # ============================================================
# # bbox_model/train_bbox.py — YOLOv8n-seg Bbox Training
# # ============================================================

# import os
# import sys
# import json
# import time
# import datetime

# sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
# import config as cfg
# from data_prep.coco_to_yolo import coco_to_yolo


# def train(log_fn=print):
#     log_fn("\n" + "─" * 50)
#     log_fn("  BBOX MODEL — YOLOv8n-seg")
#     log_fn("─" * 50)

#     try:
#         from ultralytics import YOLO
#     except ImportError:
#         raise ImportError(
#             "ultralytics not installed. Run: pip install ultralytics"
#         )

#     os.makedirs(cfg.BBOX_SAVE_DIR, exist_ok=True)

#     # Convert COCO JSON → YOLO format
#     log_fn("\n  Preparing YOLO bbox dataset...")
#     yaml_path = coco_to_yolo(
#         coco_json_path = cfg.BBOX_JSON,
#         img_dir        = cfg.IMG_DIR,
#         output_dir     = cfg.YOLO_BBOX_DIR,
#         val_ratio      = cfg.VAL_RATIO,
#         random_seed    = cfg.RANDOM_SEED,
#         mode           = "bbox",
#     )

#     # Load model
#     log_fn(f"\n  Loading {cfg.YOLO_MODEL_SIZE}...")
#     model = YOLO(cfg.YOLO_MODEL_SIZE)

#     # Train
#     log_fn(f"  Starting training — {cfg.YOLO_EPOCHS} epochs...")
#     start = time.time()

#     results = model.train(
#         data        = yaml_path,
#         epochs      = cfg.YOLO_EPOCHS,
#         imgsz       = cfg.INPUT_SIZE,
#         batch       = cfg.YOLO_BATCH_SIZE,
#         lr0         = cfg.YOLO_LR,
#         patience    = cfg.YOLO_PATIENCE,
#         device      = cfg.DEVICE,
#         project     = cfg.BBOX_SAVE_DIR,
#         name        = "train",
#         exist_ok    = True,
#         augment     = cfg.YOLO_AUGMENT,
#         val         = True,
#         save        = True,
#         save_period = 10,
#         verbose     = True,
#         conf        = cfg.YOLO_SCORE_THRESH,
#         iou         = cfg.YOLO_NMS_THRESH,
#         # Bbox-focused — disable mask loss
#         overlap_mask= False,
#     )

#     elapsed = str(datetime.timedelta(seconds=int(time.time() - start)))
#     best_model = os.path.join(
#         cfg.BBOX_SAVE_DIR, "train", "weights", "best.pt"
#     )

#     log_fn(f"\n  Bbox model training complete")
#     log_fn(f"  Time    : {elapsed}")
#     log_fn(f"  Best model → {best_model}")

#     return best_model


# ============================================================
# bbox_model/train_bbox.py — YOLOv8n-seg Bbox Training
# Reads directly from CVAT XML — no intermediate COCO JSON
# ============================================================

import os
import sys
import time
import datetime

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import config as cfg
from data_prep.split_annotations import (
    parse_cvat_xml, filter_images_by_type,
    get_labels_for_type, BBOX_TYPE
)
from data_prep.coco_to_yolo import xml_to_yolo


def train(images, log_fn=print):
    """
    Args:
        images : parsed image list from parse_cvat_xml()
        log_fn : logging function from master_train.py
    """
    log_fn("\n" + "─" * 50)
    log_fn(f"  BBOX MODEL — {cfg.YOLO_BBOX_MODEL_SIZE}")
    log_fn("─" * 50)

    try:
        from ultralytics import YOLO
    except ImportError:
        raise ImportError(
            "ultralytics not installed. Run: pip install ultralytics"
        )

    os.makedirs(cfg.BBOX_SAVE_DIR, exist_ok=True)
    os.makedirs(cfg.YOLO_BBOX_DIR, exist_ok=True)

    # Get label names for bbox annotations
    label_names = get_labels_for_type(images, BBOX_TYPE)
    log_fn(f"  Classes ({len(label_names)}): {label_names}")

    # Convert XML → YOLO format
    log_fn("\n  Preparing YOLO bbox dataset...")
    yaml_path = xml_to_yolo(
        images      = images,
        ann_type    = "bbox",
        label_names = label_names,
        img_dir     = cfg.IMG_DIR,
        output_dir  = cfg.YOLO_BBOX_DIR,
        val_ratio   = cfg.VAL_RATIO,
        random_seed = cfg.RANDOM_SEED,
    )

    # Load and train
    log_fn(f"\n  Loading {cfg.YOLO_BBOX_MODEL_SIZE}...")
    model = YOLO(cfg.YOLO_BBOX_MODEL_SIZE)

    log_fn(f"  Starting training — {cfg.YOLO_EPOCHS} epochs...")
    start = time.time()

    model.train(
        data        = yaml_path,
        epochs      = cfg.YOLO_EPOCHS,
        imgsz       = cfg.INPUT_SIZE,
        batch       = cfg.YOLO_BATCH_SIZE,
        lr0         = cfg.YOLO_LR,
        patience    = cfg.YOLO_PATIENCE,
        device      = cfg.DEVICE,
        project     = cfg.BBOX_SAVE_DIR,
        name        = "train",
        exist_ok    = True,
        augment     = cfg.YOLO_AUGMENT,
        val         = True,
        save        = True,
        save_period = 10,
        verbose     = True,
        conf        = cfg.YOLO_SCORE_THRESH,
        iou         = cfg.YOLO_NMS_THRESH,
        overlap_mask= False,
    )

    elapsed    = str(datetime.timedelta(seconds=int(time.time() - start)))
    best_model = os.path.join(
        cfg.BBOX_SAVE_DIR, "train", "weights", "best.pt"
    )
    log_fn(f"\n  Bbox training complete | Time: {elapsed}")
    log_fn(f"  Best model → {best_model}")
    return best_model