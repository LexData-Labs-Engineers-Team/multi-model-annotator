# # ============================================================
# # master_train.py — Sequential Training Pipeline
# # ============================================================
# # Scans the COCO JSON, detects which annotation types are present,
# # and trains only the relevant models in order.
# #
# # Run: python master_train.py
# # ============================================================

# import os
# import sys
# import json
# import time
# import datetime

# sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
# import config as cfg
# from data_prep.split_annotations import (
#     detect_annotation_types,
#     split_by_type,
#     BBOX_TYPE, POLYGON_TYPE, KEYPOINT_TYPE, POLYLINE_TYPE, TAG_TYPE
# )

# # ============================================================
# # --- Logging ---
# # ============================================================

# os.makedirs(cfg.SAVE_DIR, exist_ok=True)
# _log_file = open(cfg.MASTER_LOG, "w", buffering=1, encoding="utf-8")


# def log(msg=""):
#     print(msg)
#     _log_file.write(msg + "\n")


# # ============================================================
# # --- Dataset validation ---
# # ============================================================

# def validate_paths():
#     errors = []
#     if not os.path.exists(cfg.COCO_JSON):
#         errors.append(f"COCO JSON not found      : {cfg.COCO_JSON}")
#     if not os.path.exists(cfg.IMG_DIR):
#         errors.append(f"Images directory not found: {cfg.IMG_DIR}")
#     if errors:
#         for e in errors:
#             log(f"  ERROR: {e}")
#         raise FileNotFoundError(
#             "Fix the above path errors in config.py before running."
#         )


# # ============================================================
# # --- Main ---
# # ============================================================

# def main():
#     total_start = time.time()

#     log("\n" + "═" * 55)
#     log("  LexAnnotate — Sequential Training Pipeline")
#     log("═" * 55)
#     log(f"  COCO JSON  : {cfg.COCO_JSON}")
#     log(f"  Images     : {cfg.IMG_DIR}")
#     log(f"  Output     : {cfg.SAVE_DIR}")
#     log(f"  Log        : {cfg.MASTER_LOG}")

#     # --------------------------------------------------------
#     # Step 1 — Validate paths
#     # --------------------------------------------------------
#     log("\n--- Step 1: Validating paths ---")
#     validate_paths()
#     log("  All paths OK.")

#     # --------------------------------------------------------
#     # Step 2 — Detect annotation types
#     # --------------------------------------------------------
#     log("\n--- Step 2: Scanning dataset for annotation types ---")
#     present_types, counts = detect_annotation_types(cfg.COCO_JSON)

#     log(f"\n  Annotation types detected:")
#     all_types = [BBOX_TYPE, POLYGON_TYPE, KEYPOINT_TYPE,
#                  POLYLINE_TYPE, TAG_TYPE]
#     for t in all_types:
#         if t in present_types:
#             log(f"    ✓  {t:<12} — {counts.get(t, 0)} annotations")
#         else:
#             log(f"    ✗  {t:<12} — not present, model will be skipped")

#     if not present_types:
#         log("\n  ERROR: No annotations found in the COCO JSON.")
#         log("  Check that converter.py has been run and the JSON is valid.")
#         return

#     # Confirm training plan
#     log(f"\n  Models to be trained: "
#         f"{sorted(present_types)}")
#     log(f"  Models to be skipped: "
#         f"{sorted(set(all_types) - present_types)}")

#     # Polyline requires both stages
#     if POLYLINE_TYPE in present_types:
#         log(f"  Note: Polyline model trains in 2 stages "
#             f"(vertex heatmap → edge connectivity)")

#     # --------------------------------------------------------
#     # Step 3 — Split annotations by type
#     # --------------------------------------------------------
#     log("\n--- Step 3: Splitting annotations by type ---")
#     os.makedirs(cfg.DATA_DIR, exist_ok=True)
#     split_paths = split_by_type(cfg.COCO_JSON, cfg.DATA_DIR, present_types)

#     # --------------------------------------------------------
#     # Step 4 — Sequential training
#     # --------------------------------------------------------
#     log("\n--- Step 4: Training models ---")
#     log(f"  Training order: bbox → polygon → keypoint → "
#         f"polyline (s1 → s2) → tag\n")

#     results  = {}
#     failures = {}

#     # --- Bbox ---
#     if BBOX_TYPE in present_types:
#         log("\n" + "═" * 55)
#         log("  [1/5] BBOX MODEL")
#         log("═" * 55)
#         try:
#             from bbox_model.train_bbox import train as train_bbox
#             best = train_bbox(log_fn=log)
#             results[BBOX_TYPE] = best
#         except Exception as e:
#             log(f"  FAILED: {e}")
#             failures[BBOX_TYPE] = str(e)
#     else:
#         log("\n  [1/5] BBOX MODEL — skipped (no bbox annotations)")

#     # --- Polygon ---
#     if POLYGON_TYPE in present_types:
#         log("\n" + "═" * 55)
#         log("  [2/5] POLYGON MODEL")
#         log("═" * 55)
#         try:
#             from polygon_model.train_polygon import train as train_polygon
#             best = train_polygon(log_fn=log)
#             results[POLYGON_TYPE] = best
#         except Exception as e:
#             log(f"  FAILED: {e}")
#             failures[POLYGON_TYPE] = str(e)
#     else:
#         log("\n  [2/5] POLYGON MODEL — skipped (no polygon annotations)")

#     # --- Keypoint ---
#     if KEYPOINT_TYPE in present_types:
#         log("\n" + "═" * 55)
#         log("  [3/5] KEYPOINT MODEL")
#         log("═" * 55)
#         try:
#             from keypoint_model.train_keypoint import train as train_keypoint
#             best = train_keypoint(log_fn=log)
#             results[KEYPOINT_TYPE] = best
#         except Exception as e:
#             log(f"  FAILED: {e}")
#             failures[KEYPOINT_TYPE] = str(e)
#     else:
#         log("\n  [3/5] KEYPOINT MODEL — skipped (no keypoint annotations)")

#     # --- Polyline (2 stages) ---
#     if POLYLINE_TYPE in present_types:
#         log("\n" + "═" * 55)
#         log("  [4/5] POLYLINE MODEL — Stage 1")
#         log("═" * 55)
#         s1_ok = False
#         try:
#             from polyline_model.train_polyline_s1 import train as train_s1
#             best_s1 = train_s1(log_fn=log)
#             results[f"{POLYLINE_TYPE}_s1"] = best_s1
#             s1_ok = True
#         except Exception as e:
#             log(f"  FAILED (Stage 1): {e}")
#             failures[f"{POLYLINE_TYPE}_s1"] = str(e)

#         if s1_ok:
#             log("\n" + "─" * 55)
#             log("  [4/5] POLYLINE MODEL — Stage 2")
#             log("─" * 55)
#             try:
#                 from polyline_model.train_polyline_s2 import train as train_s2
#                 best_s2 = train_s2(log_fn=log)
#                 results[f"{POLYLINE_TYPE}_s2"] = best_s2
#             except Exception as e:
#                 log(f"  FAILED (Stage 2): {e}")
#                 failures[f"{POLYLINE_TYPE}_s2"] = str(e)
#         else:
#             log("\n  Polyline Stage 2 skipped — Stage 1 failed.")
#     else:
#         log("\n  [4/5] POLYLINE MODEL — skipped (no polyline annotations)")

#     # --- Tag ---
#     if TAG_TYPE in present_types:
#         log("\n" + "═" * 55)
#         log("  [5/5] TAG MODEL")
#         log("═" * 55)
#         try:
#             from tag_model.train_tag import train as train_tag
#             best = train_tag(log_fn=log)
#             results[TAG_TYPE] = best
#         except Exception as e:
#             log(f"  FAILED: {e}")
#             failures[TAG_TYPE] = str(e)
#     else:
#         log("\n  [5/5] TAG MODEL — skipped (no tag annotations)")

#     # --------------------------------------------------------
#     # Step 5 — Summary
#     # --------------------------------------------------------
#     total_elapsed = str(
#         datetime.timedelta(seconds=int(time.time() - total_start))
#     )

#     log("\n" + "═" * 55)
#     log("  Pipeline Complete — Summary")
#     log("─" * 55)
#     log(f"  Total time : {total_elapsed}")
#     log(f"\n  Trained models:")
#     if results:
#         for model_name, path in results.items():
#             log(f"    ✓  {model_name:<20} → {path}")
#     else:
#         log("    None")

#     if failures:
#         log(f"\n  Failed models:")
#         for model_name, err in failures.items():
#             log(f"    ✗  {model_name:<20} : {err}")

#     log(f"\n  Full log saved → {cfg.MASTER_LOG}")
#     log("═" * 55 + "\n")

#     # Save results summary JSON
#     summary_path = os.path.join(cfg.SAVE_DIR, "training_summary.json")
#     with open(summary_path, "w") as f:
#         json.dump({
#             "present_types" : list(present_types),
#             "skipped_types" : list(set(all_types) - present_types),
#             "results"       : results,
#             "failures"      : failures,
#             "total_time"    : total_elapsed,
#         }, f, indent=2)

#     log(f"  Results JSON → {summary_path}")

#     _log_file.close()


# if __name__ == "__main__":
#     main()


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
import config as cfg
from data_prep.split_annotations import (
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

    log(f"\n  Will train  : {sorted(present_types)}")
    log(f"  Will skip   : {sorted(set(all_types) - present_types)}")

    os.makedirs(cfg.DATA_DIR, exist_ok=True)

    # --------------------------------------------------------
    # Step 4 — Sequential training
    # --------------------------------------------------------
    log("\n--- Step 4: Training models ---")
    log("  Order: bbox → polygon → keypoint → polyline (s1→s2) → tag\n")

    results  = {}
    failures = {}

    # --- Bbox ---
    if BBOX_TYPE in present_types:
        log("\n" + "═" * 55)
        log("  [1/5] BBOX MODEL")
        log("═" * 55)
        try:
            from bbox_model.train_bbox import train as train_bbox
            best = train_bbox(images=images, log_fn=log)
            results[BBOX_TYPE] = best
        except Exception as e:
            log(f"  FAILED: {e}")
            failures[BBOX_TYPE] = str(e)
    else:
        log("\n  [1/5] BBOX — skipped")

    # --- Polygon ---
    if POLYGON_TYPE in present_types:
        log("\n" + "═" * 55)
        log("  [2/5] POLYGON MODEL")
        log("═" * 55)
        try:
            from polygon_model.train_polygon import train as train_polygon
            best = train_polygon(images=images, log_fn=log)
            results[POLYGON_TYPE] = best
        except Exception as e:
            log(f"  FAILED: {e}")
            failures[POLYGON_TYPE] = str(e)
    else:
        log("\n  [2/5] POLYGON — skipped")

    # --- Keypoint ---
    if KEYPOINT_TYPE in present_types:
        if getattr(cfg, "KP_USE_SEG", True):
            # New path: HRNet per-class disk segmentation
            log("\n" + "═" * 55)
            log("  [3/5] KEYPOINT MODEL — Segmentation (HRNet)")
            log("═" * 55)
            try:
                from data_prep.split_annotations import get_labels_for_type
                classes = get_labels_for_type(images, KEYPOINT_TYPE)
                log(f"  Keypoint classes (auto-derived): {classes}")
                from keypoint_seg_model.train_keypoint_seg import (
                    train as train_kp_seg
                )
                best = train_kp_seg(images=images, log_fn=log, classes=classes)
                results[KEYPOINT_TYPE] = best
            except Exception as e:
                log(f"  FAILED: {e}")
                failures[KEYPOINT_TYPE] = str(e)
        else:
            # LEGACY KEYPOINT — kept for revert; see train_keypoint_seg.py.
            # Uncomment the block below and set cfg.KP_USE_SEG=False to use it.
            # ----------------------------------------------------------
            # log("\n" + "═" * 55)
            # log("  [3/5] KEYPOINT MODEL (legacy heatmap)")
            # log("═" * 55)
            # try:
            #     from keypoint_model.train_keypoint import train as train_kp
            #     best = train_kp(images=images, log_fn=log)
            #     results[KEYPOINT_TYPE] = best
            # except Exception as e:
            #     log(f"  FAILED: {e}")
            #     failures[KEYPOINT_TYPE] = str(e)
            log("\n  [3/5] KEYPOINT — legacy path is commented out; "
                "set KP_USE_SEG=True or restore the legacy block above.")
    else:
        log("\n  [3/5] KEYPOINT — skipped")

    # --- Polyline ---
    if POLYLINE_TYPE in present_types:
        if getattr(cfg, "POLY_USE_SEG", True):
            # New path: HRNet per-class segmentation
            log("\n" + "═" * 55)
            log("  [4/5] POLYLINE MODEL — Segmentation (HRNet)")
            log("═" * 55)
            try:
                from data_prep.split_annotations import get_labels_for_type
                classes = get_labels_for_type(images, POLYLINE_TYPE)
                log(f"  Polyline classes (auto-derived): {classes}")
                from polyline_model_working.train_polyline_seg import (
                    train as train_seg
                )
                best = train_seg(images=images, log_fn=log, classes=classes)
                results[POLYLINE_TYPE] = best
            except Exception as e:
                log(f"  FAILED: {e}")
                failures[POLYLINE_TYPE] = str(e)
        else:
            # Legacy 2-stage path (kept for A/B comparison)
            log("\n" + "═" * 55)
            log("  [4/5] POLYLINE MODEL — Stage 1 (legacy)")
            log("═" * 55)
            s1_ok = False
            try:
                from polyline_model.train_polyline_s1 import train as train_s1
                best_s1 = train_s1(images=images, log_fn=log)
                results[f"{POLYLINE_TYPE}_s1"] = best_s1
                s1_ok = True
            except Exception as e:
                log(f"  FAILED (Stage 1): {e}")
                failures[f"{POLYLINE_TYPE}_s1"] = str(e)

            if s1_ok:
                log("\n" + "─" * 55)
                log("  [4/5] POLYLINE MODEL — Stage 2 (legacy)")
                log("─" * 55)
                try:
                    from polyline_model.train_polyline_s2 import train as train_s2
                    best_s2 = train_s2(images=images, log_fn=log)
                    results[f"{POLYLINE_TYPE}_s2"] = best_s2
                except Exception as e:
                    log(f"  FAILED (Stage 2): {e}")
                    failures[f"{POLYLINE_TYPE}_s2"] = str(e)
            else:
                log("  Stage 2 skipped — Stage 1 failed.")
    else:
        log("\n  [4/5] POLYLINE — skipped")

    # --- Tag ---
    if TAG_TYPE in present_types:
        log("\n" + "═" * 55)
        log("  [5/5] TAG MODEL")
        log("═" * 55)
        try:
            from tag_model.train_tag import train as train_tag
            best = train_tag(images=images, log_fn=log)
            results[TAG_TYPE] = best
        except Exception as e:
            log(f"  FAILED: {e}")
            failures[TAG_TYPE] = str(e)
    else:
        log("\n  [5/5] TAG — skipped")

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