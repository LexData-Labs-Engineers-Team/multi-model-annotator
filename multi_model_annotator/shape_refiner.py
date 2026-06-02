# ============================================================
# shape_refiner.py
# Data-driven postprocessing for polygon predictions.
#
# Loads priors derived by data_prep/extract_class_priors.py and:
#   Pass 1 — simplifies each polygon to its class's median vertex
#            count (iterative Douglas-Peucker).
#   Pass 2 — enforces mutual exclusion for class pairs flagged
#            "always_zero" in the priors (with n_pairs >= MIN_PAIR_OBS).
#            For each exclusion pair (A, B), the lower-confidence
#            polygon is clipped by the higher-confidence one.
#
# Designed to be called once from master_test.py after all base
# predictions have been collected for an image, before drawing
# or exporting. No model retraining required.
# ============================================================

import json

import cv2
import numpy as np

# Shapely is the right tool for polygon Boolean ops, but it's not a
# hard dependency — we fall back to a cv2 raster path if missing.
try:
    from shapely.geometry import Polygon as _ShPoly
    from shapely.validation import make_valid as _sh_make_valid
    _HAS_SHAPELY = True
except ImportError:
    _HAS_SHAPELY = False


MIN_PAIR_OBS         = 20    # ignore overlap-matrix entries below this
MIN_POLYGON_AREA_PX  = 200   # drop polygons smaller than this after clipping
MAX_DP_ITERS         = 50    # safety cap on iterative Douglas-Peucker

# Classes whose predictions should be image-aware-clipped to vegetation
# (ExG > 0) before geometric simplification. Add labels here if more
# classes need leaf-aware refinement; the rule is identical for each.
#
# DISABLED 2026-05-18: ExG-based clipping was tried on tree_canopy and
# hurt overall performance (likely fragmented the canopy too aggressively
# and/or the ExG threshold was wrong for this lighting/foliage). The
# Pass 0 block is gated on this set being non-empty, so an empty set
# short-circuits the entire pass. Re-enable by adding label(s) here.
VEGETATION_CLIP_CLASSES = set()


# ============================================================
# --- Priors loading
# ============================================================

def load_priors(path):
    """Load class_priors.json. Returns the parsed dict, or None if
    the file is missing (caller should skip refinement in that case)."""
    try:
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    except FileNotFoundError:
        return None


def _exclusion_pairs(priors):
    """Return a set of frozenset({class_a, class_b}) for every pair
    in the overlap matrix flagged 'always_zero' with enough observations."""
    pairs = set()
    for key, info in priors.get("overlap_matrix", {}).items():
        if info.get("relation") != "always_zero":
            continue
        if int(info.get("n_pairs", 0)) < MIN_PAIR_OBS:
            continue
        # Keys were written as "a__b" with (a, b) sorted; recover.
        a, b = key.split("__", 1)
        pairs.add(frozenset((a, b)))
    return pairs


def _target_vertex_counts(priors):
    """Map class_name → median vertex count (rounded to int) for polygons.
    Falls back to 4 for any class missing from priors."""
    out = {}
    for label, info in priors.get("polygon_classes", {}).items():
        v = info.get("vertex_count", {})
        # Prefer p50; fall back to mean.
        med = v.get("p50", v.get("mean", 4.0))
        out[label] = max(3, int(round(float(med))))
    return out


def _target_polyline_vertex_counts(priors):
    """Map class_name → median vertex count for polylines. Floor of 2
    (start + end). Empty dict if priors has no polyline_classes."""
    out = {}
    for label, info in priors.get("polyline_classes", {}).items():
        v = info.get("vertex_count", {})
        med = v.get("p50", v.get("mean", 2.0))
        out[label] = max(2, int(round(float(med))))
    return out


# ============================================================
# --- Pass 1: per-class vertex simplification
# ============================================================

def _simplify_to_target(pts, target_vertices, closed):
    """Iterative Douglas-Peucker. Doubles epsilon until the simplified
    shape has <= target_vertices points (or MAX_DP_ITERS reached).
    Returns simplified pts as list of (x, y) tuples. If the input is
    already at-or-below target, returns it as-is.

    closed=True  → polygon path (output must have >= 3 vertices)
    closed=False → polyline path (output must have >= 2 vertices)
    """
    floor = 3 if closed else 2
    if len(pts) <= target_vertices:
        return pts

    arr = np.array(pts, dtype=np.float32).reshape(-1, 1, 2)
    # Starting epsilon: 1 px in image coordinates. Polygons/polylines
    # in this project live at original image resolution (thousands of
    # px), so 1 px is a sensible floor that resolves quickly.
    eps = 1.0
    best = arr
    for _ in range(MAX_DP_ITERS):
        simp = cv2.approxPolyDP(arr, eps, closed=closed)
        best = simp
        if len(simp) <= target_vertices:
            break
        eps *= 2.0
    # cv2.approxPolyDP can return a degenerate result if eps gets huge
    # (below floor). Reject and keep the unsimplified input.
    if len(best) < floor:
        return pts
    return [(float(p[0][0]), float(p[0][1])) for p in best]


# ============================================================
# --- Pass 2: mutual exclusion via polygon difference
# ============================================================

def _to_shapely(pts):
    """Build a shapely polygon, repairing self-intersections if needed."""
    poly = _ShPoly(pts)
    if not poly.is_valid:
        poly = _sh_make_valid(poly)
    return poly


def _shapely_difference(a_pts, b_pts):
    """Return list of polygon point-lists representing (A - B), or [] if empty."""
    a = _to_shapely(a_pts)
    b = _to_shapely(b_pts)
    diff = a.difference(b)
    if diff.is_empty:
        return []
    out = []
    # difference can yield Polygon, MultiPolygon, or GeometryCollection.
    geoms = getattr(diff, "geoms", [diff])
    for g in geoms:
        if g.geom_type != "Polygon":
            continue
        ext = list(g.exterior.coords)
        if len(ext) >= 3:
            out.append([(float(x), float(y)) for x, y in ext])
    return out


def _raster_difference(a_pts, b_pts, canvas_h, canvas_w):
    """cv2-only fallback for polygon difference. Rasterizes onto a
    shared canvas, computes A AND-NOT B, recovers contours."""
    mask_a = np.zeros((canvas_h, canvas_w), dtype=np.uint8)
    mask_b = np.zeros((canvas_h, canvas_w), dtype=np.uint8)
    a_arr  = np.array(a_pts, dtype=np.int32).reshape(-1, 1, 2)
    b_arr  = np.array(b_pts, dtype=np.int32).reshape(-1, 1, 2)
    cv2.fillPoly(mask_a, [a_arr], 1)
    cv2.fillPoly(mask_b, [b_arr], 1)
    kept = mask_a & (1 - mask_b)
    contours, _ = cv2.findContours(kept, cv2.RETR_EXTERNAL,
                                   cv2.CHAIN_APPROX_NONE)
    out = []
    for c in contours:
        if len(c) < 3:
            continue
        out.append([(float(p[0][0]), float(p[0][1])) for p in c])
    return out


def _polygon_difference(a_pts, b_pts, canvas_h, canvas_w):
    """Dispatch to shapely if available, otherwise raster fallback."""
    if _HAS_SHAPELY:
        try:
            return _shapely_difference(a_pts, b_pts)
        except Exception:
            # Some pathological inputs trip shapely; fall through.
            pass
    return _raster_difference(a_pts, b_pts, canvas_h, canvas_w)


# ============================================================
# --- Pass 0: image-aware vegetation clip (ExG-based)
# ============================================================

def _vegetation_mask(bgr):
    """Returns uint8 mask, 255 where ExG > 0 (vegetation pixels).
    Cleaned with morphological open (3) then close (5) — removes
    pinhole speckles, fills small leaf-gaps.

    ExG = 2G - R - B (Woebbecke 1995). Positive for vegetation, near-
    zero/negative for sky, soil, structures. Single-threshold, no
    parameter tuning per-image."""
    img = bgr.astype(np.int16)
    # cv2 loads BGR, so channels are: 0=B, 1=G, 2=R
    exg  = 2 * img[:, :, 1] - img[:, :, 2] - img[:, :, 0]
    mask = (exg > 0).astype(np.uint8) * 255
    k_open  = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    k_close = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN,  k_open)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, k_close)
    return mask


def _clip_polygon_by_mask(pts, mask, h, w):
    """Returns list of polygon point-lists representing the input
    polygon AND'd with `mask`. The result may split the polygon
    into multiple disconnected pieces (one per surviving contour)."""
    canopy = np.zeros((h, w), dtype=np.uint8)
    arr = np.array(pts, dtype=np.int32).reshape(-1, 1, 2)
    cv2.fillPoly(canopy, [arr], 255)
    kept = cv2.bitwise_and(canopy, mask)
    contours, _ = cv2.findContours(kept, cv2.RETR_EXTERNAL,
                                   cv2.CHAIN_APPROX_NONE)
    out = []
    for c in contours:
        if len(c) < 3:
            continue
        out.append([(float(p[0][0]), float(p[0][1])) for p in c])
    return out


# ============================================================
# --- Diagnostic helpers (refresh _diag_* fields after refinement)
# ============================================================

def _refresh_diag(pred, img_h, img_w):
    """Recompute _diag_area_pct and _diag_n_vertices in place so the
    polygon_diagnostic.csv reflects post-refinement geometry."""
    pts = pred["points"]
    if len(pts) < 3:
        pred["_diag_area_pct"]   = 0.0
        pred["_diag_n_vertices"] = int(len(pts))
        return
    arr  = np.array(pts, dtype=np.float32)
    area = float(abs(cv2.contourArea(arr)))
    img_area = float(img_h * img_w) or 1.0
    pred["_diag_area_pct"]   = 100.0 * area / img_area
    pred["_diag_n_vertices"] = int(len(pts))


def _polygon_area_px(pts):
    if len(pts) < 3:
        return 0.0
    arr = np.array(pts, dtype=np.float32)
    return float(abs(cv2.contourArea(arr)))


# ============================================================
# --- Entry point
# ============================================================

def refine_predictions(all_preds, priors, img_h, img_w, image=None):
    """Apply image-aware and geometric refinement to the polygon /
    polyline entries in `all_preds`. Non-polygon/-polyline predictions
    are passed through untouched. Returns a new list — the input is
    not mutated.

    Pipeline:
      Pass 0  image-aware vegetation clip (only if `image` is provided
              and a polygon's class is in VEGETATION_CLIP_CLASSES)
      Pass 1a polygon simplification (iterative Douglas-Peucker)
      Pass 1b polyline simplification
      Pass 2  polygon mutual exclusion (always_zero pairs)

    img_h, img_w : original image dimensions.
    image        : optional BGR image (cv2-loaded). Enables Pass 0.
    """
    if priors is None:
        return list(all_preds)

    targets       = _target_vertex_counts(priors)
    pl_targets    = _target_polyline_vertex_counts(priors)
    excl_pairs    = _exclusion_pairs(priors)

    # Split predictions: polygons (full refinement), polylines (simplify
    # only), and everything else (passed through).
    polys     = [dict(p) for p in all_preds if p.get("type") == "polygon"]
    polylines = [dict(p) for p in all_preds if p.get("type") == "polyline"]
    other     = [p for p in all_preds
                 if p.get("type") not in ("polygon", "polyline")]

    # --- Pass 0: image-aware vegetation clip ---
    # Computed lazily — we only build the (expensive) vegetation mask
    # the first time we hit a polygon whose class wants it.
    if image is not None and VEGETATION_CLIP_CLASSES:
        veg_mask = None
        new_polys = []
        for p in polys:
            if p["label"] not in VEGETATION_CLIP_CLASSES:
                new_polys.append(p)
                continue
            if veg_mask is None:
                veg_mask = _vegetation_mask(image)
            parts = _clip_polygon_by_mask(p["points"], veg_mask,
                                          img_h, img_w)
            for part in parts:
                if _polygon_area_px(part) < MIN_POLYGON_AREA_PX:
                    continue
                new_p = dict(p)
                new_p["points"] = part
                new_polys.append(new_p)
        polys = new_polys

    # --- Pass 1a: polygon simplification ---
    for p in polys:
        target = targets.get(p["label"])
        if target is None:
            continue
        new_pts = _simplify_to_target(p["points"], target, closed=True)
        if new_pts is not p["points"]:
            p["points"] = new_pts

    # --- Pass 1b: polyline simplification ---
    # No mutual-exclusion pass for polylines — they're 1D curves and
    # don't have meaningful area-IoU. Endpoints are preserved by DP,
    # so target=2 collapses a noisy straight-line trace to its endpoints.
    for pl in polylines:
        target = pl_targets.get(pl["label"])
        if target is None:
            continue
        new_pts = _simplify_to_target(pl["points"], target, closed=False)
        if new_pts is not pl["points"]:
            pl["points"] = new_pts

    # --- Pass 2: mutual exclusion ---
    # For each polygon P, clip by every higher-confidence exclusion-
    # paired polygon Q. Order-independent in the higher-conf direction;
    # ties are broken by iteration order (deterministic for a given run).
    refined = []
    for i, p in enumerate(polys):
        a_label = p["label"]
        a_score = float(p.get("score", 0.0))
        # Collect higher-confidence Qs that exclude A.
        clippers = []
        for j, q in enumerate(polys):
            if i == j:
                continue
            b_label = q["label"]
            if frozenset((a_label, b_label)) not in excl_pairs:
                continue
            b_score = float(q.get("score", 0.0))
            if b_score > a_score:
                clippers.append(q)
        if not clippers:
            _refresh_diag(p, img_h, img_w)
            refined.append(p)
            continue

        # Apply each clipper in turn. The geometry may split into
        # multiple parts after the first cut; subsequent cuts apply
        # to every surviving part.
        current_parts = [p["points"]]
        for q in clippers:
            next_parts = []
            for part in current_parts:
                diff_parts = _polygon_difference(part, q["points"],
                                                 img_h, img_w)
                next_parts.extend(diff_parts)
            current_parts = next_parts
            if not current_parts:
                break

        # Emit surviving parts as separate detections, each inheriting
        # P's score and label. Drop parts below MIN_POLYGON_AREA_PX.
        for part in current_parts:
            if _polygon_area_px(part) < MIN_POLYGON_AREA_PX:
                continue
            new_pred = dict(p)
            new_pred["points"] = part
            _refresh_diag(new_pred, img_h, img_w)
            refined.append(new_pred)

    return refined + polylines + other
