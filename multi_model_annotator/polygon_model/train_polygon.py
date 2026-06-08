# # ============================================================
# # polygon_model/train_polygon.py — YOLOv8n-seg Polygon Training
# # ============================================================

# import os
# import sys
# import time
# import datetime

# sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
# import config as cfg
# from data_prep.coco_to_yolo import coco_to_yolo


# def train(log_fn=print):
#     log_fn("\n" + "─" * 50)
#     log_fn("  POLYGON MODEL — YOLOv8n-seg")
#     log_fn("─" * 50)

#     try:
#         from ultralytics import YOLO
#     except ImportError:
#         raise ImportError(
#             "ultralytics not installed. Run: pip install ultralytics"
#         )

#     os.makedirs(cfg.POLYGON_SAVE_DIR, exist_ok=True)

#     # Convert COCO JSON → YOLO format with polygon segmentation points
#     log_fn("\n  Preparing YOLO polygon dataset...")
#     yaml_path = coco_to_yolo(
#         coco_json_path = cfg.POLYGON_JSON,
#         img_dir        = cfg.IMG_DIR,
#         output_dir     = cfg.YOLO_POLYGON_DIR,
#         val_ratio      = cfg.VAL_RATIO,
#         random_seed    = cfg.RANDOM_SEED,
#         mode           = "polygon",
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
#         project     = cfg.POLYGON_SAVE_DIR,
#         name        = "train",
#         exist_ok    = True,
#         augment     = cfg.YOLO_AUGMENT,
#         val         = True,
#         save        = True,
#         save_period = 10,
#         verbose     = True,
#         conf        = cfg.YOLO_SCORE_THRESH,
#         iou         = cfg.YOLO_NMS_THRESH,
#         overlap_mask= True,     # enable mask overlap for polygons
#         mask_ratio  = 4,        # mask downsample ratio
#     )

#     elapsed    = str(datetime.timedelta(seconds=int(time.time() - start)))
#     best_model = os.path.join(
#         cfg.POLYGON_SAVE_DIR, "train", "weights", "best.pt"
#     )

#     log_fn(f"\n  Polygon model training complete")
#     log_fn(f"  Time      : {elapsed}")
#     log_fn(f"  Best model → {best_model}")

#     return best_model


# ============================================================
# polygon_model/train_polygon.py — YOLOv8n-seg Polygon Training
# Reads directly from CVAT XML — no intermediate COCO JSON
# ============================================================

import os
import sys
import time
import datetime

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from multi_model_annotator import config as cfg
from multi_model_annotator.data_prep.split_annotations import (
    parse_cvat_xml, filter_images_by_type,
    get_labels_for_type, POLYGON_TYPE
)
from multi_model_annotator.data_prep.coco_to_yolo import xml_to_yolo


def train(images, log_fn=print):
    """
    Args:
        images : parsed image list from parse_cvat_xml()
        log_fn : logging function from master_train.py
    """
    log_fn("\n" + "─" * 50)
    log_fn(f"  POLYGON MODEL — {cfg.YOLO_POLYGON_MODEL_SIZE}")
    log_fn("─" * 50)

    try:
        from ultralytics import YOLO
    except ImportError:
        raise ImportError(
            "ultralytics not installed. Run: pip install ultralytics"
        )

    os.makedirs(cfg.POLYGON_SAVE_DIR, exist_ok=True)
    os.makedirs(cfg.YOLO_POLYGON_DIR, exist_ok=True)

    label_names = get_labels_for_type(images, POLYGON_TYPE)
    log_fn(f"  Classes ({len(label_names)}): {label_names}")

    log_fn("\n  Preparing YOLO polygon dataset...")
    yaml_path = xml_to_yolo(
        images      = images,
        ann_type    = "polygon",
        label_names = label_names,
        img_dir     = cfg.IMG_DIR,
        output_dir  = cfg.YOLO_POLYGON_DIR,
        val_ratio   = cfg.VAL_RATIO,
        random_seed = cfg.RANDOM_SEED,
    )

    log_fn(f"\n  Loading {cfg.YOLO_POLYGON_MODEL_SIZE}...")
    model = YOLO(cfg.YOLO_POLYGON_MODEL_SIZE)

    log_fn(f"  Starting training — {cfg.YOLO_EPOCHS} epochs...")
    start = time.time()

    model.train(
        data        = yaml_path,
        epochs      = cfg.YOLO_EPOCHS,
        imgsz       = cfg.INPUT_SIZE,
        batch       = cfg.YOLO_BATCH_SIZE,
        workers     = cfg.NUM_WORKERS,   # fewer dataloader workers → less pinned host memory (8 GB OOM fix)
        cache       = False,             # don't hold the dataset in RAM
        lr0         = cfg.YOLO_LR,
        patience    = cfg.YOLO_PATIENCE,
        device      = cfg.DEVICE,
        project     = cfg.POLYGON_SAVE_DIR,
        name        = "train",
        exist_ok    = True,
        augment     = cfg.YOLO_AUGMENT,
        val         = True,
        save        = True,
        save_period = 10,
        verbose     = True,
        conf        = cfg.YOLO_SCORE_THRESH,
        iou         = cfg.YOLO_NMS_THRESH,
        overlap_mask= True,
        mask_ratio  = 2,
        optimizer   = cfg.YOLO_OPTIMIZER,
        mosaic      = cfg.YOLO_MOSAIC,
        retina_masks= cfg.YOLO_RETINA_MASKS,
    )

    elapsed    = str(datetime.timedelta(seconds=int(time.time() - start)))
    best_model = os.path.join(
        cfg.POLYGON_SAVE_DIR, "train", "weights", "best.pt"
    )
    log_fn(f"\n  Polygon training complete | Time: {elapsed}")
    log_fn(f"  Best model → {best_model}")
    return best_model