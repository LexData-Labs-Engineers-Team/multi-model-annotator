# ============================================================
# master_train.py — Sequential Training Pipeline
# ============================================================
# Parses CVAT XML once, detects which annotation types are
# present, and trains only the relevant models in order.
# All train scripts receive the parsed image list directly —
# no intermediate COCO JSON is written or read.
#
# Run: python master_train.py
# ============================================================

import os
import sys
import json
import time
import datetime

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from multi_model_annotator import config as cfg
from multi_model_annotator.data_prep.split_annotations import (
    parse_cvat_xml,
    detect_annotation_types,
    BBOX_TYPE, POLYGON_TYPE, KEYPOINT_TYPE, POLYLINE_TYPE, TAG_TYPE
)

# ============================================================
# --- Logging ---
# ============================================================

os.makedirs(cfg.SAVE_DIR, exist_ok=True)
_log_file = open(cfg.MASTER_LOG, "w", buffering=1, encoding="utf-8")


def log(msg=""):
    print(msg)
    _log_file.write(msg + "\n")


# ============================================================
# --- Path validation ---
# ============================================================

def validate_paths():
    errors = []
    if not os.path.exists(cfg.CVAT_XML):
        errors.append(f"CVAT XML not found       : {cfg.CVAT_XML}")
    if not os.path.exists(cfg.IMG_DIR):
        errors.append(f"Images folder not found  : {cfg.IMG_DIR}")
    if errors:
        for e in errors:
            log(f"  ERROR: {e}")
        raise FileNotFoundError(
            "Fix the above path errors in config.py before running."
        )


# ============================================================
# --- Main ---
# ============================================================

def main():
    total_start = time.time()
    all_types   = [BBOX_TYPE, POLYGON_TYPE, KEYPOINT_TYPE,
                   POLYLINE_TYPE, TAG_TYPE]

    log("\n" + "═" * 55)
    log("  LexAnnotate — Sequential Training Pipeline")
    log("═" * 55)
    log(f"  CVAT XML   : {cfg.CVAT_XML}")
    log(f"  Images     : {cfg.IMG_DIR}")
    log(f"  Output     : {cfg.SAVE_DIR}")

    # --------------------------------------------------------
    # Step 1 — Validate paths
    # --------------------------------------------------------
    log("\n--- Step 1: Validating paths ---")
    validate_paths()
    log("  All paths OK.")

    # --------------------------------------------------------
    # Step 2 — Parse XML once
    # --------------------------------------------------------
    log("\n--- Step 2: Parsing CVAT XML ---")
    images, all_labels = parse_cvat_xml(cfg.CVAT_XML)
    log(f"  Images parsed  : {len(images)}")
    log(f"  Labels found   : {all_labels}")

    # --------------------------------------------------------
    # Step 3 — Detect annotation types
    # --------------------------------------------------------
    log("\n--- Step 3: Detecting annotation types ---")
    present_types, counts = detect_annotation_types(cfg.CVAT_XML)

    log(f"\n  Annotation types:")
    for t in all_types:
        if t in present_types:
            log(f"    ✓  {t:<12} — {counts.get(t, 0)} annotations")
        else:
            log(f"    ✗  {t:<12} — not present, model will be skipped")

    if not present_types:
        log("\n  ERROR: No annotations found in XML.")
        return

    train_flags = {
        BBOX_TYPE: cfg.TRAIN_BBOX, POLYGON_TYPE: cfg.TRAIN_POLYGON,
        KEYPOINT_TYPE: cfg.TRAIN_KEYPOINT, POLYLINE_TYPE: cfg.TRAIN_POLYLINE,
        TAG_TYPE: cfg.TRAIN_TAG,
    }
    disabled = sorted(t for t, on in train_flags.items() if not on)
    will_train = sorted(t for t in present_types if train_flags.get(t, True))
    will_skip  = sorted(set(all_types) - present_types | set(disabled))

    if disabled:
        log(f"\n  Disabled    : {disabled}")
    log(f"  Will train  : {will_train}")
    log(f"  Will skip   : {will_skip}")

    os.makedirs(cfg.DATA_DIR, exist_ok=True)

    # --------------------------------------------------------
    # Step 4 — Sequential training
    # --------------------------------------------------------
    log("\n--- Step 4: Training models ---")
    log("  Order: bbox → polygon → keypoint → polyline (s1→s2) → tag\n")

    results  = {}
    failures = {}

    # --- Bbox ---
    if not cfg.TRAIN_BBOX:
        log("\n  [1/5] BBOX — disabled in config")
    elif BBOX_TYPE not in present_types:
        log("\n  [1/5] BBOX — skipped (no annotations)")
    else:
        log("\n" + "═" * 55)
        log("  [1/5] BBOX MODEL")
        log("═" * 55)
        try:
            from multi_model_annotator.bbox_model.train_bbox import train as train_bbox
            best = train_bbox(images=images, log_fn=log)
            results[BBOX_TYPE] = best
        except Exception as e:
            log(f"  FAILED: {e}")
            failures[BBOX_TYPE] = str(e)

    # --- Polygon ---
    if not cfg.TRAIN_POLYGON:
        log("\n  [2/5] POLYGON — disabled in config")
    elif POLYGON_TYPE not in present_types:
        log("\n  [2/5] POLYGON — skipped (no annotations)")
    elif getattr(cfg, "POLYGON_USE_SEG", False):
        log("\n" + "═" * 55)
        log("  [2/5] POLYGON MODEL — Segmentation (HRNet)")
        log("═" * 55)
        try:
            from multi_model_annotator.data_prep.split_annotations import get_labels_for_type
            classes = get_labels_for_type(images, POLYGON_TYPE)
            log(f"  Polygon classes (auto-derived): {classes}")
            from multi_model_annotator.polygon_seg_model.train_polygon_seg import (
                train as train_polygon_seg
            )
            best = train_polygon_seg(images=images, log_fn=log, classes=classes)
            results[POLYGON_TYPE] = best
        except Exception as e:
            log(f"  FAILED: {e}")
            failures[POLYGON_TYPE] = str(e)
    else:
        log("\n" + "═" * 55)
        log("  [2/5] POLYGON MODEL — YOLO-seg")
        log("═" * 55)
        try:
            from multi_model_annotator.polygon_model.train_polygon import train as train_polygon
            best = train_polygon(images=images, log_fn=log)
            results[POLYGON_TYPE] = best
        except Exception as e:
            log(f"  FAILED: {e}")
            failures[POLYGON_TYPE] = str(e)

    # --- Keypoint ---
    if not cfg.TRAIN_KEYPOINT:
        log("\n  [3/5] KEYPOINT — disabled in config")
    elif KEYPOINT_TYPE not in present_types:
        log("\n  [3/5] KEYPOINT — skipped (no annotations)")
    else:
        log("\n" + "═" * 55)
        log("  [3/5] KEYPOINT MODEL — Segmentation (HRNet)")
        log("═" * 55)
        try:
            from multi_model_annotator.data_prep.split_annotations import get_labels_for_type
            classes = get_labels_for_type(images, KEYPOINT_TYPE)
            log(f"  Keypoint classes (auto-derived): {classes}")
            from multi_model_annotator.keypoint_seg_model.train_keypoint_seg import (
                train as train_kp_seg
            )
            best = train_kp_seg(images=images, log_fn=log, classes=classes)
            results[KEYPOINT_TYPE] = best
        except Exception as e:
            log(f"  FAILED: {e}")
            failures[KEYPOINT_TYPE] = str(e)

    # --- Polyline ---
    if not cfg.TRAIN_POLYLINE:
        log("\n  [4/5] POLYLINE — disabled in config")
    elif POLYLINE_TYPE not in present_types:
        log("\n  [4/5] POLYLINE — skipped (no annotations)")
    else:
        log("\n" + "═" * 55)
        log("  [4/5] POLYLINE MODEL — Segmentation (HRNet)")
        log("═" * 55)
        try:
            from multi_model_annotator.data_prep.split_annotations import get_labels_for_type
            classes = get_labels_for_type(images, POLYLINE_TYPE)
            log(f"  Polyline classes (auto-derived): {classes}")
            from multi_model_annotator.polyline_model_working.train_polyline_seg import (
                train as train_seg
            )
            best = train_seg(images=images, log_fn=log, classes=classes)
            results[POLYLINE_TYPE] = best
        except Exception as e:
            log(f"  FAILED: {e}")
            failures[POLYLINE_TYPE] = str(e)

    # --- Tag ---
    if not cfg.TRAIN_TAG:
        log("\n  [5/5] TAG — disabled in config")
    elif TAG_TYPE not in present_types:
        log("\n  [5/5] TAG — skipped (no annotations)")
    else:
        log("\n" + "═" * 55)
        log("  [5/5] TAG MODEL")
        log("═" * 55)
        try:
            from multi_model_annotator.tag_model.train_tag import train as train_tag
            best = train_tag(images=images, log_fn=log)
            results[TAG_TYPE] = best
        except Exception as e:
            log(f"  FAILED: {e}")
            failures[TAG_TYPE] = str(e)

    # --------------------------------------------------------
    # Step 5 — Summary
    # --------------------------------------------------------
    total_elapsed = str(
        datetime.timedelta(seconds=int(time.time() - total_start))
    )

    log("\n" + "═" * 55)
    log("  Pipeline Complete — Summary")
    log("─" * 55)
    log(f"  Total time : {total_elapsed}")
    log(f"\n  Trained successfully:")
    if results:
        for name, path in results.items():
            log(f"    ✓  {name:<22} → {path}")
    else:
        log("    None")

    if failures:
        log(f"\n  Failed:")
        for name, err in failures.items():
            log(f"    ✗  {name:<22} : {err}")

    log(f"\n  Full log → {cfg.MASTER_LOG}")
    log("═" * 55 + "\n")

    # Save summary JSON
    summary_path = os.path.join(cfg.SAVE_DIR, "training_summary.json")
    with open(summary_path, "w") as f:
        json.dump({
            "cvat_xml"     : cfg.CVAT_XML,
            "present_types": list(present_types),
            "skipped_types": list(set(all_types) - present_types),
            "results"      : results,
            "failures"     : failures,
            "total_time"   : total_elapsed,
        }, f, indent=2)

    log(f"  Summary JSON → {summary_path}")
    _log_file.close()


if __name__ == "__main__":
    main()