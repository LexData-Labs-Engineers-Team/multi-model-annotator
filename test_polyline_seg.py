# ============================================================
# test_polyline_seg.py — Standalone inference for the
# HRNet-W18 polyline segmentation model.
#
# Loads a `best_seg.pt` checkpoint trained by
# polyline_model_working/train_polyline_seg.py, runs it on every
# image in IMAGES_DIR, skeletonises the per-class probability
# masks, traces polylines, and writes annotated JPEGs to OUTPUT_DIR.
#
# Run: python test_polyline_seg.py
# ============================================================

import os
import sys
import glob
import time
import datetime
from collections import defaultdict

import cv2
import numpy as np
import torch
import torch.nn.functional as F

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import config as cfg
from polyline_model_working.train_polyline_seg import HRNetSegModel

# ============================================================
# --- USER SETTINGS — edit before running ---
# ============================================================

MODEL_PATH    = os.path.join(cfg.POLYLINE_SAVE_DIR, "best_seg.pt")
IMAGES_DIR    = cfg.IMG_DIR
OUTPUT_DIR    = os.path.join(cfg.SAVE_DIR, "polyline_seg_test")

SCORE_THRESH  = cfg.POLY_SEG_THRESH   # per-class probability cutoff
DEVICE        = cfg.DEVICE            # "cuda" or "cpu"

MIN_CHAIN_LEN = 5                     # drop traced chains shorter than this (pixels)
DP_EPSILON    = 1.5                   # Douglas-Peucker simplification tolerance

LINE_THICKNESS = 2
VERTEX_RADIUS  = 3
FONT_SCALE     = 0.5
LEGEND_BG      = (30, 30, 30)
LEGEND_FG      = (240, 240, 240)

IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".tiff"}

# Per-class BGR colors — same palette as master_test.py for consistency.
_PALETTE = [
    (  0, 229, 160), (  0, 153, 255), (255, 107,  53),
    (192, 132, 252), (255, 215,   0), (255,  99, 132),
    ( 75, 192, 192), (255, 159,  64), (153, 102, 255),
    ( 54, 162, 235), (255, 206,  86), ( 46, 204, 113),
]


# ============================================================
# --- Skeletonization backend (ximgproc preferred, skimage fallback)
# ============================================================

_SKELETONIZE_BACKEND = None


def _skeletonize(mask_uint8):
    """Return a uint8 (0/255) skeleton of the binary input mask.
    Prefers cv2.ximgproc.thinning (Zhang-Suen) when available; falls
    back to skimage.morphology.skeletonize. Backend choice is reported
    on first call only."""
    global _SKELETONIZE_BACKEND
    if _SKELETONIZE_BACKEND is None:
        try:
            _ = cv2.ximgproc.THINNING_ZHANGSUEN
            _SKELETONIZE_BACKEND = "ximgproc"
            print(f"  Skeletonize backend : cv2.ximgproc.thinning (Zhang-Suen)")
        except AttributeError:
            try:
                from skimage.morphology import skeletonize as _sk_skel  # noqa: F401
                _SKELETONIZE_BACKEND = "skimage"
                print(f"  Skeletonize backend : skimage.morphology.skeletonize")
            except ImportError:
                raise ImportError(
                    "Neither cv2.ximgproc nor scikit-image is available.\n"
                    "Install one of:\n"
                    "  pip install opencv-contrib-python\n"
                    "  pip install scikit-image"
                )

    if _SKELETONIZE_BACKEND == "ximgproc":
        return cv2.ximgproc.thinning(mask_uint8, cv2.ximgproc.THINNING_ZHANGSUEN)
    else:
        from skimage.morphology import skeletonize as _sk_skel
        bool_mask = mask_uint8 > 0
        skel = _sk_skel(bool_mask)
        return (skel.astype(np.uint8) * 255)


# ============================================================
# --- Trace polylines from a skeleton image
# ============================================================

_NEIGHBOR_OFFSETS = [(-1, -1), (-1, 0), (-1, 1),
                    ( 0, -1),          ( 0, 1),
                    ( 1, -1), ( 1, 0), ( 1, 1)]


def _trace_polylines(skel_uint8):
    """Walk an 8-connected skeleton, returning a list of pixel chains.
    Each chain is a list of (x, y) tuples (image-space pixel coords).
    Endpoints are degree-1 skeleton pixels; junctions (degree >= 3)
    terminate a chain. Closed loops (no endpoints) are also captured."""
    skel = (skel_uint8 > 0).astype(np.uint8)
    H, W = skel.shape

    # Precompute degree for every skeleton pixel (single pass over ys, xs).
    ys, xs = np.nonzero(skel)
    if ys.size == 0:
        return []

    degree = np.zeros_like(skel, dtype=np.int8)
    for y, x in zip(ys, xs):
        d = 0
        for dy, dx in _NEIGHBOR_OFFSETS:
            ny, nx = y + dy, x + dx
            if 0 <= ny < H and 0 <= nx < W and skel[ny, nx]:
                d += 1
        degree[y, x] = d

    visited = np.zeros_like(skel, dtype=bool)
    chains  = []

    def _neighbors(y, x):
        for dy, dx in _NEIGHBOR_OFFSETS:
            ny, nx = y + dy, x + dx
            if 0 <= ny < H and 0 <= nx < W and skel[ny, nx]:
                yield ny, nx

    def _walk(start_y, start_x):
        # Build a chain from a starting pixel, stopping at the next
        # junction (degree >= 3), already-visited pixel, or dead end.
        chain   = [(start_x, start_y)]
        visited[start_y, start_x] = True
        cy, cx  = start_y, start_x
        while True:
            next_pix = None
            for ny, nx in _neighbors(cy, cx):
                if visited[ny, nx]:
                    continue
                # Junction is included as endpoint of this chain, but
                # we don't continue past it.
                next_pix = (ny, nx)
                if degree[ny, nx] >= 3:
                    visited[ny, nx] = True
                    chain.append((nx, ny))
                    return chain
                break
            if next_pix is None:
                return chain
            ny, nx = next_pix
            visited[ny, nx] = True
            chain.append((nx, ny))
            cy, cx = ny, nx

    # Pass 1 — start from endpoints (degree == 1).
    endpoints = list(zip(*np.where(degree == 1)))
    for (y, x) in endpoints:
        if visited[y, x]:
            continue
        chain = _walk(y, x)
        if len(chain) >= MIN_CHAIN_LEN:
            chains.append(chain)

    # Pass 2 — remaining skeleton pixels (closed loops / dangling segments).
    remaining = list(zip(*np.where(skel & ~visited)))
    for (y, x) in remaining:
        if visited[y, x]:
            continue
        chain = _walk(y, x)
        if len(chain) >= MIN_CHAIN_LEN:
            chains.append(chain)

    return chains


def _simplify_chain(chain):
    """Douglas-Peucker simplification of a pixel chain."""
    if len(chain) < 3:
        return chain
    pts  = np.array(chain, dtype=np.int32).reshape(-1, 1, 2)
    simp = cv2.approxPolyDP(pts, DP_EPSILON, closed=False)
    return [(int(p[0][0]), int(p[0][1])) for p in simp]


# ============================================================
# --- Model loader
# ============================================================

def load_seg_model(model_path, device):
    """Load the HRNet seg checkpoint. Reads the embedded config dict
    so backbone / input_size / classes / mask_thickness are picked up
    from training rather than re-specified here."""
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model not found: {model_path}")

    ckpt = torch.load(model_path, map_location=device, weights_only=False)
    if "config" not in ckpt or "state_dict" not in ckpt:
        raise RuntimeError(
            f"Checkpoint at {model_path} is missing 'config' or 'state_dict' "
            f"keys — was it saved by train_polyline_seg.py?"
        )

    meta        = ckpt["config"]
    backbone    = meta["backbone"]
    classes     = list(meta["classes"])
    input_size  = int(meta["input_size"])
    mask_thick  = int(meta.get("mask_thickness", cfg.POLY_SEG_MASK_THICKNESS))

    model = HRNetSegModel(
        backbone=backbone,
        num_classes=len(classes),
        pretrained=False,
    ).to(device)
    model.load_state_dict(ckpt["state_dict"])
    model.eval()

    return model, classes, input_size, mask_thick


# ============================================================
# --- Preprocess
# ============================================================

_IMAGENET_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
_IMAGENET_STD  = np.array([0.229, 0.224, 0.225], dtype=np.float32)


def _preprocess(bgr, input_size, device):
    rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
    rgb = cv2.resize(rgb, (input_size, input_size), interpolation=cv2.INTER_AREA)
    arr = rgb.astype(np.float32) / 255.0
    arr = (arr - _IMAGENET_MEAN) / _IMAGENET_STD
    tensor = torch.from_numpy(arr.transpose(2, 0, 1)).contiguous().unsqueeze(0)
    return tensor.to(device)


# ============================================================
# --- Inference + tracing for one image
# ============================================================

def _infer_image(model, bgr, input_size, classes, device):
    """Returns: dict {class_name: [chain, chain, ...]} where each
    chain is a list of (x, y) pixel coords in original image space."""
    orig_h, orig_w = bgr.shape[:2]
    tensor = _preprocess(bgr, input_size, device)

    with torch.no_grad():
        logits = model(tensor)                              # (1, C, S, S)
        probs  = torch.sigmoid(logits).float()
        probs  = F.interpolate(probs, size=(orig_h, orig_w),
                               mode="bilinear", align_corners=False)
    probs = probs.squeeze(0).cpu().numpy()                  # (C, H, W)

    per_class_chains = {}
    for c, cname in enumerate(classes):
        mask = (probs[c] > SCORE_THRESH).astype(np.uint8) * 255
        if mask.sum() == 0:
            per_class_chains[cname] = []
            continue
        skel   = _skeletonize(mask)
        chains = _trace_polylines(skel)
        chains = [_simplify_chain(ch) for ch in chains if len(ch) >= MIN_CHAIN_LEN]
        per_class_chains[cname] = chains

    return per_class_chains


# ============================================================
# --- Drawing
# ============================================================

def _draw_annotations(bgr, per_class_chains, classes):
    out = bgr.copy()
    color_map = {c: _PALETTE[i % len(_PALETTE)] for i, c in enumerate(classes)}

    for cname, chains in per_class_chains.items():
        color = color_map[cname]
        for chain in chains:
            if len(chain) < 2:
                continue
            pts = np.array(chain, dtype=np.int32).reshape(-1, 1, 2)
            cv2.polylines(out, [pts], isClosed=False, color=color,
                          thickness=LINE_THICKNESS, lineType=cv2.LINE_AA)
            for (x, y) in chain:
                cv2.circle(out, (int(x), int(y)), VERTEX_RADIUS, color, -1)

    # Legend bar at the top.
    legend_h = int(20 + 24 * len(classes))
    if legend_h > 0 and out.shape[0] > legend_h:
        cv2.rectangle(out, (0, 0), (240, legend_h), LEGEND_BG, -1)
        y = 22
        for cname in classes:
            color = color_map[cname]
            cv2.rectangle(out, (10, y - 12), (30, y + 2), color, -1)
            cv2.putText(out, cname, (40, y),
                        cv2.FONT_HERSHEY_SIMPLEX, FONT_SCALE,
                        LEGEND_FG, 1, cv2.LINE_AA)
            y += 24

    return out


# ============================================================
# --- Main
# ============================================================

def main():
    print("\n" + "═" * 55)
    print("  Polyline-Seg Inference — Standalone Test")
    print("═" * 55)

    device = torch.device(
        "cuda" if (DEVICE == "cuda" and torch.cuda.is_available()) else "cpu"
    )
    print(f"  Device       : {device}")
    print(f"  Model        : {MODEL_PATH}")
    print(f"  Images dir   : {IMAGES_DIR}")
    print(f"  Output dir   : {OUTPUT_DIR}")
    print(f"  Score thresh : {SCORE_THRESH}")

    os.makedirs(OUTPUT_DIR, exist_ok=True)

    # --- Load model ---
    model, classes, input_size, mask_thick = load_seg_model(MODEL_PATH, device)
    print(f"\n  Loaded model:")
    print(f"    Backbone     : {cfg.POLY_SEG_BACKBONE}")
    print(f"    Classes ({len(classes)}): {classes}")
    print(f"    Input size   : {input_size}")
    print(f"    Mask thick   : {mask_thick}")

    # --- Collect images ---
    image_paths = sorted([
        p for ext in IMAGE_EXTS
        for p in glob.glob(os.path.join(IMAGES_DIR, f"*{ext}"))
    ])
    if not image_paths:
        raise RuntimeError(f"No images found in {IMAGES_DIR}")
    print(f"\n  Images found : {len(image_paths)}")

    # --- Inference loop ---
    print(f"\n--- Running inference ---")
    total_chains   = 0
    per_class_tot  = defaultdict(int)
    t0             = time.time()

    for i, img_path in enumerate(image_paths, 1):
        fname = os.path.basename(img_path)
        bgr   = cv2.imread(img_path)
        if bgr is None:
            print(f"  [{i:3d}/{len(image_paths)}] {fname:<40} unreadable, skipped")
            continue

        per_class_chains = _infer_image(
            model, bgr, input_size, classes, device
        )
        n_img = sum(len(v) for v in per_class_chains.values())
        for c, chains in per_class_chains.items():
            per_class_tot[c] += len(chains)
        total_chains += n_img

        annotated = _draw_annotations(bgr, per_class_chains, classes)
        out_path  = os.path.join(OUTPUT_DIR, fname)
        cv2.imwrite(out_path, annotated)

        detail = " ".join(f"{c}={len(per_class_chains[c])}" for c in classes)
        print(f"  [{i:3d}/{len(image_paths)}] {fname:<40} chains={n_img:3d} | {detail}")

    elapsed = time.time() - t0

    # --- Summary ---
    print(f"\n{'═' * 55}")
    print(f"  Done")
    print(f"{'─' * 55}")
    print(f"  Images processed : {len(image_paths)}")
    print(f"  Total polylines  : {total_chains}")
    print(f"  Per-class counts :")
    for c in classes:
        print(f"    {c:<20} : {per_class_tot[c]}")
    print(f"  Total time       : "
          f"{str(datetime.timedelta(seconds=int(elapsed)))}")
    print(f"  Avg / image      : "
          f"{elapsed / max(len(image_paths), 1):.2f}s")
    print(f"  Output dir       : {OUTPUT_DIR}")
    print(f"{'═' * 55}\n")


if __name__ == "__main__":
    main()
