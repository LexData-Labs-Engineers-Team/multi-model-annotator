# # ============================================================
# # data_prep/split_annotations.py
# # Splits a combined COCO JSON into per-annotation-type JSONs.
# # Also detects which annotation types are present.
# # Called by master_train.py — not meant to be run standalone.
# # ============================================================

# import os
# import json
# from collections import defaultdict


# # Annotation type identifiers
# # shape_type is set by converter.py when converting CVAT XML → COCO JSON
# BBOX_TYPE     = "bbox"
# POLYGON_TYPE  = "polygon"
# KEYPOINT_TYPE = "keypoint"
# POLYLINE_TYPE = "polyline"
# TAG_TYPE      = "tag"

# ALL_TYPES = [BBOX_TYPE, POLYGON_TYPE, KEYPOINT_TYPE, POLYLINE_TYPE, TAG_TYPE]


# def detect_annotation_types(coco_json_path):
#     """
#     Scans a COCO JSON and returns which annotation types are present.

#     Returns:
#         present : set of annotation type strings
#         counts  : dict of type → count
#     """
#     with open(coco_json_path) as f:
#         data = json.load(f)

#     counts  = defaultdict(int)
#     present = set()

#     for ann in data.get("annotations", []):
#         shape = ann.get("shape_type", None)

#         if shape == POLYLINE_TYPE:
#             counts[POLYLINE_TYPE] += 1
#             present.add(POLYLINE_TYPE)

#         elif shape == KEYPOINT_TYPE or (
#             "keypoints" in ann and ann.get("keypoints")
#         ):
#             counts[KEYPOINT_TYPE] += 1
#             present.add(KEYPOINT_TYPE)

#         elif shape == TAG_TYPE:
#             counts[TAG_TYPE] += 1
#             present.add(TAG_TYPE)

#         elif shape == POLYGON_TYPE or (
#             ann.get("segmentation") and
#             isinstance(ann["segmentation"], list) and
#             len(ann["segmentation"]) > 0 and
#             ann.get("bbox") and
#             _is_polygon(ann)
#         ):
#             counts[POLYGON_TYPE] += 1
#             present.add(POLYGON_TYPE)

#         elif ann.get("bbox") and len(ann.get("bbox", [])) == 4:
#             counts[BBOX_TYPE] += 1
#             present.add(BBOX_TYPE)

#     # Check for image-level tags stored separately
#     for img in data.get("images", []):
#         if img.get("tags"):
#             counts[TAG_TYPE] += len(img["tags"])
#             present.add(TAG_TYPE)

#     return present, dict(counts)


# def _is_polygon(ann):
#     """
#     Heuristic to distinguish polygon from bbox-only annotation.
#     A polygon has segmentation points that form a non-rectangular shape
#     or has more than 4 unique points.
#     """
#     seg = ann.get("segmentation", [])
#     if not seg or not isinstance(seg, list):
#         return False
#     flat = seg[0] if isinstance(seg[0], list) else seg
#     # More than 4 points (8 values) = likely a polygon not just a box
#     return len(flat) > 8


# def split_by_type(coco_json_path, output_dir, present_types):
#     """
#     Splits the combined COCO JSON into one JSON per annotation type.
#     Only creates files for types in present_types.

#     Returns:
#         output_paths: dict of type → output json path
#     """
#     os.makedirs(output_dir, exist_ok=True)

#     with open(coco_json_path) as f:
#         data = json.load(f)

#     images_map = {img["id"]: img for img in data["images"]}

#     # Group annotations by type
#     type_anns = defaultdict(list)

#     for ann in data.get("annotations", []):
#         shape = ann.get("shape_type", None)

#         if shape == POLYLINE_TYPE and POLYLINE_TYPE in present_types:
#             type_anns[POLYLINE_TYPE].append(ann)

#         elif (shape == KEYPOINT_TYPE or "keypoints" in ann) \
#                 and KEYPOINT_TYPE in present_types:
#             type_anns[KEYPOINT_TYPE].append(ann)

#         elif shape == TAG_TYPE and TAG_TYPE in present_types:
#             type_anns[TAG_TYPE].append(ann)

#         elif (shape == POLYGON_TYPE or _is_polygon(ann)) \
#                 and POLYGON_TYPE in present_types:
#             type_anns[POLYGON_TYPE].append(ann)

#         elif ann.get("bbox") and BBOX_TYPE in present_types:
#             # Only pure bbox (non-polygon) annotations
#             if not _is_polygon(ann):
#                 type_anns[BBOX_TYPE].append(ann)

#     output_paths = {}

#     type_to_path = {
#         BBOX_TYPE    : os.path.join(output_dir, "bbox_annotations.json"),
#         POLYGON_TYPE : os.path.join(output_dir, "polygon_annotations.json"),
#         KEYPOINT_TYPE: os.path.join(output_dir, "keypoint_annotations.json"),
#         POLYLINE_TYPE: os.path.join(output_dir, "polyline_annotations.json"),
#         TAG_TYPE     : os.path.join(output_dir, "tag_annotations.json"),
#     }

#     for ann_type, anns in type_anns.items():
#         if not anns:
#             continue

#         # Collect only images that have this annotation type
#         used_image_ids = {ann["image_id"] for ann in anns}
#         used_images    = [
#             images_map[iid]
#             for iid in used_image_ids
#             if iid in images_map
#         ]

#         out = {
#             "info"       : data.get("info", {}),
#             "licenses"   : data.get("licenses", []),
#             "categories" : data.get("categories", []),
#             "images"     : used_images,
#             "annotations": anns,
#         }

#         out_path = type_to_path[ann_type]
#         with open(out_path, "w") as f:
#             json.dump(out, f, indent=2)

#         output_paths[ann_type] = out_path
#         print(f"    [{ann_type}] {len(anns)} annotations "
#               f"across {len(used_images)} images → {out_path}")

#     return output_paths


# ============================================================
# data_prep/split_annotations.py
# Parses CVAT for Images 1.1 XML directly.
# Detects which annotation types are present and extracts
# per-type annotation data — no intermediate COCO JSON needed.
# Called by master_train.py — not meant to be run standalone.
# ============================================================

import xml.etree.ElementTree as ET
from collections import defaultdict

# Annotation type constants
BBOX_TYPE     = "bbox"
POLYGON_TYPE  = "polygon"
KEYPOINT_TYPE = "keypoint"
POLYLINE_TYPE = "polyline"
TAG_TYPE      = "tag"
ALL_TYPES     = [BBOX_TYPE, POLYGON_TYPE, KEYPOINT_TYPE,
                 POLYLINE_TYPE, TAG_TYPE]


# ============================================================
# --- Core XML parsing ---
# ============================================================

def _fix_polygon_winding(points):
    import numpy as np, math
    pts = np.array(points, dtype=np.float32)
    cx  = float(pts[:, 0].mean())
    cy  = float(pts[:, 1].mean())
    angles = [math.atan2(p[1]-cy, p[0]-cx) for p in pts]
    return [p for _, p in sorted(zip(angles, points))]

def parse_cvat_xml(xml_path):
    """
    Parses a CVAT for Images 1.1 XML file.

    Returns:
        images : list of dicts, each with:
                   id, name, width, height,
                   bboxes, polygons, polylines, keypoints, tags
        labels : sorted list of all unique label names
    """
    tree = ET.parse(xml_path)
    root = tree.getroot()

    all_labels = set()
    images     = []

    for img_el in root.findall("image"):
        img = {
            "id"       : img_el.get("id",     "0"),
            "name"     : img_el.get("name",   ""),
            "width"    : int(img_el.get("width",  0)),
            "height"   : int(img_el.get("height", 0)),
            "bboxes"   : [],
            "polygons" : [],
            "polylines": [],
            "keypoints": [],
            "tags"     : [],
        }

        # --- Bounding boxes ---
        for el in img_el.findall("box"):
            label = el.get("label", "")
            all_labels.add(label)
            img["bboxes"].append({
                "label": label,
                "xtl"  : float(el.get("xtl", 0)),
                "ytl"  : float(el.get("ytl", 0)),
                "xbr"  : float(el.get("xbr", 0)),
                "ybr"  : float(el.get("ybr", 0)),
            })

        # --- Polygons ---
        for el in img_el.findall("polygon"):
            label = el.get("label", "")
            all_labels.add(label)
            pts = _parse_points(el.get("points", ""))
            if len(pts) >= 3:
                img["polygons"].append({"label": label, "points": pts})

        # --- Polylines ---
        for el in img_el.findall("polyline"):
            label = el.get("label", "")
            all_labels.add(label)
            pts   = _parse_points(el.get("points", ""))
            if len(pts) >= 2:
                img["polylines"].append({
                    "label" : label,
                    "points": pts,
                })

        # --- Keypoints (stored as <points> with single point) ---
        for el in img_el.findall("points"):
            label = el.get("label", "")
            all_labels.add(label)
            pts   = _parse_points(el.get("points", ""))
            for pt in pts:
                img["keypoints"].append({
                    "label": label,
                    "x"    : pt[0],
                    "y"    : pt[1],
                })

        # --- Tags ---
        for el in img_el.findall("tag"):
            label = el.get("label", "")
            all_labels.add(label)
            img["tags"].append({"label": label})

        images.append(img)

    return images, sorted(all_labels)


def _parse_points(points_str):
    """
    Parses CVAT points string: "x1,y1;x2,y2;..." → [(x1,y1), ...]
    """
    pts = []
    if not points_str.strip():
        return pts
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


# ============================================================
# --- Detect annotation types ---
# ============================================================

def detect_annotation_types(xml_path):
    """
    Scans CVAT XML and returns which annotation types are present.

    Returns:
        present : set of type strings
        counts  : dict of type → count
    """
    images, _ = parse_cvat_xml(xml_path)
    counts    = defaultdict(int)

    for img in images:
        counts[BBOX_TYPE]     += len(img["bboxes"])
        counts[POLYGON_TYPE]  += len(img["polygons"])
        counts[POLYLINE_TYPE] += len(img["polylines"])
        counts[KEYPOINT_TYPE] += len(img["keypoints"])
        counts[TAG_TYPE]      += len(img["tags"])

    present = {t for t, c in counts.items() if c > 0}
    return present, dict(counts)


# ============================================================
# --- Get label names per type ---
# ============================================================

def get_labels_for_type(images, ann_type):
    """
    Returns sorted list of unique label names for a given type.
    Used to build class lists for YOLO yaml and dataset classes.
    """
    labels = set()
    key    = {
        BBOX_TYPE    : "bboxes",
        POLYGON_TYPE : "polygons",
        POLYLINE_TYPE: "polylines",
        KEYPOINT_TYPE: "keypoints",
        TAG_TYPE     : "tags",
    }[ann_type]

    for img in images:
        for ann in img[key]:
            labels.add(ann["label"])

    return sorted(labels)


# ============================================================
# --- Filter images to those containing a given type ---
# ============================================================

def filter_images_by_type(images, ann_type):
    """
    Returns only images that have at least one annotation
    of the given type.
    """
    key = {
        BBOX_TYPE    : "bboxes",
        POLYGON_TYPE : "polygons",
        POLYLINE_TYPE: "polylines",
        KEYPOINT_TYPE: "keypoints",
        TAG_TYPE     : "tags",
    }[ann_type]

    return [img for img in images if img[key]]