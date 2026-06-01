# Legacy / Removed Code Archive

This document archives the commented-out and legacy code removed from the
codebase during the **2026-06-01 cleanup pass** (removal of the legacy
heatmap-keypoint and 2-stage-polyline paths, the dead `attention_fusion.py`
experiment, the `old_files/` folder, and assorted stale commented blocks).

**Anchor commit:** all line numbers below refer to the file state as of commit
`c1b7d24` — the last commit *before* this cleanup. After removal, line numbers in
the live files shift, so read these numbers together with the commit reference,
not against the current files.

**Recovering anything in full from git:**
```bash
git show c1b7d24:<path>                 # print a file exactly as it was
git show c1b7d24:<path> > recovered.py  # restore it to a new file
```

Sections are named by source file. Large stale blocks and whole deleted files are
recorded as `[POINTER]` entries (recoverable via the command above); focused
legacy code is reproduced verbatim.

---

## config.py

### [POINTER] Lines 1–124 — stale commented-out duplicate of the entire old config
A full commented-out mirror of an earlier config: old Linux paths, the retired
`COCO_JSON` workflow, `flower` / `bonsai_v2` datasets, and pre-segmentation
`KEYPOINT` / `POLYLINE` / `TAG` parameter blocks. Superseded by the live config
that followed it. Recover: `git show c1b7d24:config.py` (lines 1–124).

### Lines 236–244 — legacy KEYPOINT (heatmap) hyperparameters
Used only by the removed `keypoint_model/train_keypoint.py` (single-channel
Gaussian heatmap trainer). Superseded by the `KP_SEG_*` block.
```python
# ============================================================
# --- KEYPOINT MODEL ---
# ============================================================

KP_BACKBONE         = "resnet50"
KP_PRETRAINED       = True
KP_EPOCHS           = 50
KP_BATCH_SIZE       = 8
KP_LR               = 1e-4
KP_WEIGHT_DECAY     = 1e-4
KP_HEATMAP_SIGMA    = 8
KP_SCORE_THRESH     = 0.3
KP_CHECKPOINT_EVERY = 10
```

### Lines 246–266 — legacy POLYLINE 2-stage hyperparameters (S1 heatmap + S2 edge MLP)
Used only by the removed `polyline_model/train_polyline_s1.py` /
`train_polyline_s2.py`. Superseded by the `POLY_SEG_*` block.
```python
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
```

### Line 282 — legacy edge-connectivity threshold (S2 edge MLP)
```python
EDGE_THRESH = 0.5
```

### Lines 296–304 — POLY_USE_SEG toggle (removed; seg path is now the only path)
The surrounding `POLY_SEG_*` config block is **kept**; only the toggle line and
its legacy-fallback wording were removed.
```python
# ============================================================
# --- POLYLINE-SEG (new, working) ---
# When POLY_USE_SEG is True (default), master_train.py routes the
# polyline step to polyline_model_working/train_polyline_seg.py
# (HRNet-W18 per-class segmentation + BCE+Dice). Set False to
# fall back to the legacy 2-stage (S1 heatmap + S2 edge MLP) path.
# ============================================================

POLY_USE_SEG              = True
```

### Lines 320–329 — KP_USE_SEG toggle (removed; seg path is now the only path)
The surrounding `KP_SEG_*` config block is **kept**; only the toggle line and its
legacy-fallback wording were removed.
```python
# ============================================================
# --- KEYPOINT-SEG (new, HRNet per-class segmentation) ---
# When KP_USE_SEG is True (default), master_train.py routes the
# keypoint step to keypoint_seg_model/train_keypoint_seg.py
# (HRNet-W18 per-class small-disk segmentation + BCE+Dice).
# Set False to fall back to the legacy single-channel Gaussian
# heatmap trainer at keypoint_model/train_keypoint.py.
# ============================================================

KP_USE_SEG               = True
```

---

## master_train.py

### [POINTER] Lines 1–261 — stale commented-out duplicate of the old training flow
A full commented-out mirror of an earlier `master_train.py`: `log()`,
`validate_paths()`, and `main()` built around the retired `COCO_JSON` /
`split_by_type` workflow and the 2-stage polyline plan. Superseded by the live
code that followed it (live module starts at line 262). Recover:
`git show c1b7d24:master_train.py` (lines 1–261).

### Lines 462–477 — legacy keypoint `else` branch (heatmap trainer)
Removed when the `KP_USE_SEG` toggle was dropped; the keypoint step is now
unconditionally the HRNet seg trainer.
```python
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
```

### Lines 500–527 — legacy polyline `else` branch (2-stage S1 → S2 A/B path)
Live code (ran when `POLY_USE_SEG=False`). Removed when the toggle was dropped;
the polyline step is now unconditionally the HRNet seg trainer.
```python
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
```

---

## master_test.py

### Line 53 — legacy keypoint checkpoint path (orphaned after heatmap path removed)
```python
KEYPOINT_MODEL_PATH = os.path.join(cfg.KEYPOINT_SAVE_DIR, "best.pt")
```

### Line 62 — legacy keypoint score-threshold constant
```python
KP_SCORE_THRESH     = cfg.KP_SCORE_THRESH
```

### Lines 129–140 — `load_heatmap_model()` (legacy heatmap loader, unused)
```python
def load_heatmap_model(path, backbone, device):
    sys.path.insert(0, os.path.join(
        os.path.dirname(os.path.abspath(__file__)), "keypoint_model"
    ))
    from train_keypoint import KeypointHeatmapModel
    ckpt  = torch.load(path, map_location=device)
    model = KeypointHeatmapModel(backbone=backbone, pretrained=False)
    state = ckpt["model_state_dict"] if "model_state_dict" in ckpt else ckpt
    model.load_state_dict(state)
    model.to(device)
    model.eval()
    return model
```

### Lines 384–416 — `run_keypoints()` (legacy heatmap inference, unused)
Its only call site was already commented out (see lines 1069–1075 below).
```python
def run_keypoints(model, img_path, orig_h, orig_w, device):
    tensor, _, _, _ = preprocess(img_path, cfg.INPUT_SIZE)
    if tensor is None:
        return []
    tensor = tensor.to(device)
    with torch.no_grad():
        heatmap = model(tensor)
        heatmap = F.interpolate(
            heatmap, size=(orig_h, orig_w),
            mode="bilinear", align_corners=False
        )
        heatmap = heatmap.sigmoid().squeeze().cpu().numpy()

    # Extract local maxima as keypoints. NMS window is set from the
    # training heatmap sigma (≈ 2σ+1) so peaks closer than the training
    # blob radius collapse but real separated keypoints survive.
    from scipy.ndimage import maximum_filter
    nms_size      = max(3, 2 * int(cfg.KP_HEATMAP_SIGMA) + 1)
    local_max     = (heatmap == maximum_filter(heatmap, size=nms_size))
    above_thresh  = heatmap > KP_SCORE_THRESH
    peaks         = local_max & above_thresh
    coords        = np.argwhere(peaks)   # (N, 2) in (y, x)

    detections = []
    for (y, x) in coords:
        detections.append({
            "type" : "keypoint",
            "x"    : float(x),
            "y"    : float(y),
            "score": float(heatmap[y, x]),
            "label": "keypoint",
        })
    return detections
```

### Lines 926–932 — commented legacy keypoint loader `elif` (in model-load section)
```python
    # LEGACY KEYPOINT — kept for revert; see keypoint_seg_model/train_keypoint_seg.py.
    # Uncomment the elif block below to fall back to the heatmap model.
    # elif model_exists(KEYPOINT_MODEL_PATH):
    #     print(f"  ✓ Keypoint model  : {KEYPOINT_MODEL_PATH} (legacy)")
    #     models_loaded["keypoint"] = load_heatmap_model(
    #         KEYPOINT_MODEL_PATH, cfg.KP_BACKBONE, device
    #     )
```

### Lines 979–985 — legacy hardcoded `"keypoint"` color-map entry
The two-line legacy note (979–980) and the hardcoded append (984–985) were
removed. The live per-class `kp_classes` loop (981–983) was kept.
```python
    # The hardcoded "keypoint" entry below is only used by the legacy
    # heatmap path, which is class-agnostic; kept for revert.
    ...
    if "keypoint" not in all_class_names:
        all_class_names.append("keypoint")
```

### Lines 1069–1075 — commented legacy keypoint inference call site
```python
        # LEGACY KEYPOINT — kept for revert.
        # if "keypoint" in models_loaded:
        #     preds = run_keypoints(
        #         models_loaded["keypoint"], img_path,
        #         orig_h, orig_w, device
        #     )
        #     all_preds.extend(preds)
```

---

## Deleted whole files (recoverable via `git show c1b7d24:<path>`)

### [POINTER] keypoint_model/train_keypoint.py
Legacy single-channel Gaussian-heatmap keypoint trainer (`KeypointHeatmapModel`,
`_build_heatmap`, `train`). Superseded by `keypoint_seg_model/train_keypoint_seg.py`
(HRNet per-class disk segmentation). The `keypoint_model/__init__.py` package
marker was removed with it.

### [POINTER] polyline_model/train_polyline_s1.py
Legacy polyline Stage 1 — ResNet50 vertex-heatmap model (reused
`KeypointHeatmapModel`). Superseded by `polyline_model_working/train_polyline_seg.py`.

### [POINTER] polyline_model/train_polyline_s2.py
Legacy polyline Stage 2 — `EdgeMLP` edge-connectivity classifier + edge feature
builder. Superseded by the single-stage HRNet seg polyline model.

### [POINTER] attention_fusion.py
Standalone experimental "Late Fusion with Cross-Modal Attention" script
(`python attention_fusion.py --mode train|test`). Never imported by the active
pipeline; last consumer of the legacy `train_keypoint` / `POLY_S1_*` code. Removed
so the legacy keypoint/polyline code and config could be fully retired.

### [POINTER] old_files/ (entire folder)
Deprecated experiment scripts, superseded by the current `master_*` + per-model
trainers:
- `old_files/master_test.py`
- `old_files/master_test_dep.py`
- `old_files/master_test_mlp.py`
- `old_files/output_fusion.py`       (Option 1: output fusion)
- `old_files/train_fusion_mlp.py`    (Option 3: prediction-MLP fusion)
