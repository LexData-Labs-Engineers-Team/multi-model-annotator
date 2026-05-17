# # ============================================================
# # data_prep/coco_to_yolo.py
# # Converts a COCO JSON to YOLO segmentation .txt format.
# # Handles both bbox-only and polygon annotations.
# # Called by train_bbox.py and train_polygon.py.
# # ============================================================

# import os
# import json
# import shutil
# import random
# from collections import defaultdict


# def coco_to_yolo(
#     coco_json_path : str,
#     img_dir        : str,
#     output_dir     : str,
#     val_ratio      : float = 0.15,
#     random_seed    : int   = 42,
#     mode           : str   = "polygon",   # "polygon" or "bbox"
# ):
#     """
#     Converts COCO JSON → YOLO format directory structure:

#         output_dir/
#             dataset.yaml
#             train/
#                 images/   ← symlinks or copies of training images
#                 labels/   ← .txt label files
#             val/
#                 images/
#                 labels/

#     Each .txt label file has one line per annotation:
#         For polygon: <class_id> x1 y1 x2 y2 ... xn yn  (normalized)
#         For bbox:    <class_id> cx cy w h                (normalized)

#     Args:
#         coco_json_path : path to COCO JSON
#         img_dir        : folder containing images
#         output_dir     : where to write YOLO dataset
#         val_ratio      : fraction of images for validation
#         random_seed    : reproducibility
#         mode           : "polygon" uses segmentation points,
#                          "bbox" uses bbox only
#     """
#     random.seed(random_seed)
#     os.makedirs(output_dir, exist_ok=True)

#     with open(coco_json_path) as f:
#         data = json.load(f)

#     # Build category map: coco_id → yolo_id (0-based)
#     categories   = data.get("categories", [])
#     cat_id_to_yolo = {
#         cat["id"]: idx
#         for idx, cat in enumerate(categories)
#     }
#     class_names  = [cat["name"] for cat in categories]

#     # Build image map
#     images_map   = {img["id"]: img for img in data["images"]}

#     # Group annotations by image id
#     anns_by_image = defaultdict(list)
#     for ann in data.get("annotations", []):
#         anns_by_image[ann["image_id"]].append(ann)

#     # Split into train/val
#     image_ids    = list(images_map.keys())
#     random.shuffle(image_ids)
#     n_val        = max(1, int(len(image_ids) * val_ratio))
#     val_ids      = set(image_ids[:n_val])
#     train_ids    = set(image_ids[n_val:])

#     splits = {"train": train_ids, "val": val_ids}

#     for split_name, split_ids in splits.items():
#         img_out_dir = os.path.join(output_dir, split_name, "images")
#         lbl_out_dir = os.path.join(output_dir, split_name, "labels")
#         os.makedirs(img_out_dir, exist_ok=True)
#         os.makedirs(lbl_out_dir, exist_ok=True)

#         for image_id in split_ids:
#             img_info = images_map[image_id]
#             fname    = img_info["file_name"]
#             w        = img_info.get("width",  1)
#             h        = img_info.get("height", 1)

#             # Copy image
#             src = os.path.join(img_dir, fname)
#             dst = os.path.join(img_out_dir, fname)
#             if os.path.exists(src) and not os.path.exists(dst):
#                 shutil.copy2(src, dst)

#             # Write label file
#             anns      = anns_by_image.get(image_id, [])
#             lbl_name  = os.path.splitext(fname)[0] + ".txt"
#             lbl_path  = os.path.join(lbl_out_dir, lbl_name)

#             lines = []
#             for ann in anns:
#                 cat_id   = ann.get("category_id")
#                 yolo_id  = cat_id_to_yolo.get(cat_id)
#                 if yolo_id is None:
#                     continue

#                 bbox = ann.get("bbox", [])
#                 if not bbox or len(bbox) != 4:
#                     continue

#                 x, y, bw, bh = bbox

#                 if mode == "polygon":
#                     seg = ann.get("segmentation", [])
#                     if seg and isinstance(seg, list) and len(seg) > 0:
#                         flat = seg[0]
#                         if len(flat) >= 6:
#                             # Normalize polygon points
#                             pts = []
#                             for i in range(0, len(flat), 2):
#                                 px = max(0.0, min(1.0, flat[i]     / w))
#                                 py = max(0.0, min(1.0, flat[i + 1] / h))
#                                 pts.extend([f"{px:.6f}", f"{py:.6f}"])
#                             lines.append(f"{yolo_id} " + " ".join(pts))
#                             continue

#                 # Fallback to bbox (also used when mode == "bbox")
#                 cx  = max(0.0, min(1.0, (x + bw / 2) / w))
#                 cy  = max(0.0, min(1.0, (y + bh / 2) / h))
#                 nw  = max(0.0, min(1.0, bw / w))
#                 nh  = max(0.0, min(1.0, bh / h))
#                 lines.append(
#                     f"{yolo_id} {cx:.6f} {cy:.6f} {nw:.6f} {nh:.6f}"
#                 )

#             with open(lbl_path, "w") as f:
#                 f.write("\n".join(lines))

#     # Write dataset.yaml
#     yaml_path = os.path.join(output_dir, "dataset.yaml")
#     with open(yaml_path, "w") as f:
#         f.write(f"path : {output_dir}\n")
#         f.write(f"train: train/images\n")
#         f.write(f"val  : val/images\n\n")
#         f.write(f"nc   : {len(class_names)}\n")
#         f.write(f"names: {class_names}\n")

#     print(f"    YOLO dataset written → {output_dir}")
#     print(f"    Train images : {len(train_ids)}")
#     print(f"    Val images   : {len(val_ids)}")
#     print(f"    Classes      : {class_names}")
#     print(f"    dataset.yaml : {yaml_path}")

#     return yaml_path


# ============================================================
# data_prep/coco_to_yolo.py
# Converts parsed CVAT XML image data → YOLO .txt format.
# Reads directly from the parsed image list — no COCO JSON.
# Called by train_bbox.py and train_polygon.py.
# ============================================================

import os
import shutil
import random


def xml_to_yolo(
    images       : list,
    ann_type     : str,
    label_names  : list,
    img_dir      : str,
    output_dir   : str,
    val_ratio    : float = 0.15,
    random_seed  : int   = 42,
):
    """
    Converts parsed XML image data → YOLO segmentation .txt format.

    Directory structure created:
        output_dir/
            dataset.yaml
            train/
                images/
                labels/
            val/
                images/
                labels/

    Label format:
        Polygon : <class_id> x1 y1 x2 y2 ... xn yn  (normalized)
        Bbox    : <class_id> cx cy w h                (normalized)

    Args:
        images      : list of image dicts from parse_cvat_xml()
        ann_type    : "bbox" or "polygon"
        label_names : sorted list of class names (defines class ids)
        img_dir     : folder containing source images
        output_dir  : where to write YOLO dataset
        val_ratio   : fraction of images for validation
        random_seed : for reproducible splits
    """
    random.seed(random_seed)

    label_to_id = {name: i for i, name in enumerate(label_names)}
    ann_key     = "bboxes" if ann_type == "bbox" else "polygons"

    # Filter to images that have this annotation type
    valid_images = [img for img in images if img[ann_key]]
    random.shuffle(valid_images)

    n_val    = max(1, int(len(valid_images) * val_ratio))
    val_imgs = valid_images[:n_val]
    trn_imgs = valid_images[n_val:]

    splits = {"train": trn_imgs, "val": val_imgs}

    for split_name, split_imgs in splits.items():
        img_out = os.path.join(output_dir, split_name, "images")
        lbl_out = os.path.join(output_dir, split_name, "labels")
        os.makedirs(img_out, exist_ok=True)
        os.makedirs(lbl_out, exist_ok=True)

        for img in split_imgs:
            fname = img["name"]
            w     = img["width"]
            h     = img["height"]

            if w == 0 or h == 0:
                continue

            # Copy image
            src = os.path.join(img_dir, fname)
            dst = os.path.join(img_out, fname)
            if os.path.exists(src) and not os.path.exists(dst):
                shutil.copy2(src, dst)

            # Write label file
            base     = os.path.splitext(fname)[0]
            lbl_path = os.path.join(lbl_out, base + ".txt")
            lines    = []

            if ann_type == "bbox":
                for ann in img["bboxes"]:
                    cls_id = label_to_id.get(ann["label"])
                    if cls_id is None:
                        continue
                    xtl, ytl = ann["xtl"], ann["ytl"]
                    xbr, ybr = ann["xbr"], ann["ybr"]
                    bw   = xbr - xtl
                    bh   = ybr - ytl
                    cx   = max(0.0, min(1.0, (xtl + bw / 2) / w))
                    cy   = max(0.0, min(1.0, (ytl + bh / 2) / h))
                    nw   = max(0.0, min(1.0, bw / w))
                    nh   = max(0.0, min(1.0, bh / h))
                    lines.append(
                        f"{cls_id} {cx:.6f} {cy:.6f} {nw:.6f} {nh:.6f}"
                    )

            elif ann_type == "polygon":
                for ann in img["polygons"]:
                    cls_id = label_to_id.get(ann["label"])
                    if cls_id is None:
                        continue
                    pts = ann["points"]
                    if len(pts) < 3:
                        continue
                    norm_pts = []
                    for (px, py) in pts:
                        nx = max(0.0, min(1.0, px / w))
                        ny = max(0.0, min(1.0, py / h))
                        norm_pts.extend([f"{nx:.6f}", f"{ny:.6f}"])
                    lines.append(f"{cls_id} " + " ".join(norm_pts))

            with open(lbl_path, "w") as f:
                f.write("\n".join(lines))

    # Write dataset.yaml
    yaml_path = os.path.join(output_dir, "dataset.yaml")
    with open(yaml_path, "w") as f:
        f.write(f"path : {output_dir}\n")
        f.write(f"train: train/images\n")
        f.write(f"val  : val/images\n\n")
        f.write(f"nc   : {len(label_names)}\n")
        f.write(f"names: {label_names}\n")

    print(f"    YOLO dataset written → {output_dir}")
    print(f"    Train images : {len(trn_imgs)}")
    print(f"    Val images   : {len(val_imgs)}")
    print(f"    Classes ({len(label_names)}): {label_names}")
    print(f"    dataset.yaml : {yaml_path}")

    return yaml_path