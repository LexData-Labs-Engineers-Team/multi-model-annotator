# Plan — Improve generalization, redesign keypoint + tag, integrate polyline-seg

## Context

The pipeline is an **auto-labeller**: a single CVAT XML is parsed once, used to train five per-task models (bbox / polygon / keypoint / polyline / tag), and the resulting models then label unseen images by emitting a new CVAT XML + COCO JSON via `master_test.py`. Today the bbox + polygon models (YOLO) work; the **keypoint, tag, and old 2-stage polyline models do not generalize**. A separate, far stronger polyline implementation lives at `polyline_model_working/train_polyline_seg.py` (HRNet-W18 + per-class segmentation + BCE+Dice + AMP + warmup-cosine) but isn't wired into `config.py` / `master_train.py` / `master_test.py`. Inference confirms: keypoints miss real points and emit spurious peaks, tags collapse/fire incorrectly, polyline_working is essentially right but needs fine-tuning, and all of the heatmap/MLP-based models train acceptably yet fail on unseen images — a classic small-dataset + weak-recipe failure.

This plan **adopts the working-polyline recipe as the standard** and applies it to keypoint and tag, integrates polyline-seg end-to-end (training + config + inference), and tightens generalization with stronger augmentation, AMP, warmup+cosine, EMA, and class-aware losses. Per user: redesign keypoint + tag (not just tune); keypoints become **per-class heatmap channels**; **strong augmentation**; **re-derive** the seg inference code (skeletonize + trace) in the plan rather than waiting on the existing script.

## Goals

1. Integrate `polyline_model_working/train_polyline_seg.py` into the pipeline (config + master_train + master_test) and fine-tune it.
2. Replace `keypoint_model/train_keypoint.py` with an HRNet + **per-class** small-blob segmentation model trained with BCE+Dice, AMP, warmup+cosine, strong aug.
3. Replace `tag_model/train_tag.py` recipe: keep multi-label framing but swap to a stronger pretrained backbone (timm ConvNeXt-Tiny or EfficientNetV2-S), add class-balanced loss / pos_weight, strong aug, AMP, warmup+cosine, and threshold calibration on val.
4. Improve generalization across all custom models with a shared training-utility module: stronger augmentation, AMP autocast/GradScaler, AdamW + warmup+cosine, EMA, deterministic val split, early-stop on val mIoU / F1, val-time threshold sweep.

## Approach

### A. Polyline-seg integration (highest leverage; least risk)

1. **Config additions** in `code/config.py` (new block; values mirror the working file):
   - `POLY_SEG_BACKBONE = "hrnet_w18"`, `POLY_SEG_PRETRAINED = True`
   - `POLY_SEG_INPUT_SIZE = 640`, `POLY_SEG_EPOCHS = 200`, `POLY_SEG_BATCH_SIZE = 4`
   - `POLY_SEG_LR = 1e-4`, `POLY_SEG_WEIGHT_DECAY = 1e-4`, `POLY_SEG_WARMUP_ITERS = 500`
   - `POLY_SEG_MASK_THICKNESS = 5`, `POLY_SEG_BCE_WEIGHT = 1.0`, `POLY_SEG_DICE_WEIGHT = 1.0`
   - `POLY_SEG_THRESH = 0.5`, `POLY_SEG_CHECKPOINT_EVERY = 20`
   - `POLY_SEG_CLASSES` — auto-populated by `master_train.py` via `get_labels_for_type(images, POLYLINE_TYPE)` so it tracks the XML (avoid hard-coding); fall back to a manual override in config if needed.

2. **Move + rename** `polyline_model_working/train_polyline_seg.py` → `polyline_model/train_polyline_seg.py` (keep the old `train_polyline_s1.py` / `train_polyline_s2.py` until verified, then delete). Drop the working folder once moved.

3. **Wire into `master_train.py`** at the polyline step: replace the S1+S2 path with a single `from polyline_model.train_polyline_seg import train as train_polyline_seg` call. Pass `cfg.POLY_SEG_CLASSES` (or auto-populate just before the call).

4. **Re-derive inference** for the seg model in `master_test.py`:
   - Load checkpoint payload (`state_dict` + saved `config` block with classes / input_size / mask_thickness).
   - For each image: BGR→RGB resize to `input_size`, ImageNet normalize, forward → `sigmoid` → per-class probability map at original resolution (bilinear up).
   - Threshold at `POLY_SEG_THRESH` → binary mask per class.
   - **Skeletonize**: prefer `cv2.ximgproc.thinning(mask, THINNING_ZHANGSUEN)` if `opencv-contrib-python` is available; otherwise fall back to `skimage.morphology.skeletonize`. Add a one-line dependency note in the plan's verification section.
   - **Trace polylines from skeleton**: build an 8-connected pixel graph on the skeleton; identify endpoints (degree 1) and junctions (degree ≥ 3); walk from each endpoint to the next endpoint/junction marking visited pixels, recording the pixel chain; for any remaining unvisited skeleton pixels (closed loops), pick an arbitrary start. Douglas-Peucker simplify the chain (`cv2.approxPolyDP`, epsilon ≈ 1–2 px) to get final polyline vertices. Return `{type:"polyline", points:[(x,y)], score:mean_prob, label:class_name}`.
   - Replace the existing `run_polylines` + `_reconstruct_polylines` block (`master_test.py:280–405`) with `run_polylines_seg`. Update model-loader dispatch (`master_test.py:791–798`) to load the new `best_seg.pt` instead of `s1_best.pt` / `s2_best.pt`.

5. **Fine-tune knob**: expose `POLY_SEG_RESUME_FROM` in config; if set, load weights and continue training with a lower LR (e.g. `1e-5`) for `POLY_SEG_FT_EPOCHS`. Add 4 lines in the trainer to support resume.

### B. Keypoint redesign — per-class HRNet small-blob segmentation

Rationale: the current single-channel Gaussian-heatmap MSE recipe is exactly the recipe that's already failing for polylines; the working-polyline approach (per-class dense BCE+Dice mask on a strong pretrained backbone) is the proven fix.

1. **New file** `keypoint_model/train_keypoint_seg.py`:
   - Reuse `HRNetSegModel` from `train_polyline_seg.py` (after the move, import it).
   - Dataset rasterizes **per-class small disks** (radius from new `KP_DISK_RADIUS` config, ~5–8 px at 640) at each keypoint location on a `(C, S, S)` mask. C = number of keypoint labels from `get_labels_for_type(images, KEYPOINT_TYPE)`.
   - Same BCE+Dice loss, AMP, AdamW, warmup+cosine, strong aug (color jitter, ±10° rotation with mask warp, hflip, scale 0.8–1.2 with both-image-and-mask transform, light Gaussian blur, random erasing on image only).
   - Save best by per-class mIoU; also save `kp_classes.json` alongside `best.pt` for inference.

2. **Inference** in `master_test.py`:
   - Replace `run_keypoints` (`master_test.py:247–277`).
   - Threshold per-class probability map → connected-components on each channel → centroid of each component as a predicted keypoint; score = max probability inside the component. NMS by minimum-distance (>= disk radius) to drop near-duplicates.
   - Emit `{type:"keypoint", x, y, score, label:class_name}` (currently the label is hardcoded `"keypoint"` — this preserves the class).

3. **Config additions**: `KP_SEG_BACKBONE`, `KP_SEG_INPUT_SIZE`, `KP_DISK_RADIUS`, `KP_SEG_EPOCHS`, `KP_SEG_BATCH_SIZE`, `KP_SEG_LR`, `KP_SEG_WD`, `KP_SEG_WARMUP_ITERS`, `KP_SEG_BCE_WEIGHT`, `KP_SEG_DICE_WEIGHT`, `KP_SEG_THRESH`, `KP_SEG_NMS_RADIUS`.

4. Delete (later) the heatmap-MSE `train_keypoint.py` after verification — but keep it until polyline_s1 has been removed too, since `train_polyline_s1.py` imports `KeypointHeatmapModel` and `_build_heatmap` from it. With S1/S2 gone (step A.2), it's safe to drop.

### C. Tag model redesign — stronger backbone, class-balanced, calibrated threshold

1. **New file** `tag_model/train_tag_v2.py`:
   - **Backbone**: timm `convnext_tiny` (ImageNet pretrained) with global-avg-pool head — better generalization than EfficientNet-B0 on small data.
   - **Loss**: `BCEWithLogitsLoss(pos_weight=...)` where `pos_weight[c] = (N - n_pos[c]) / max(n_pos[c], 1)` to fix class imbalance on small data. Optionally add **asymmetric loss** (`AsymmetricLoss`, gamma_neg=4, gamma_pos=1) — well-known to help multi-label classification with imbalance.
   - **Recipe**: AMP, AdamW, warmup+cosine, EMA of weights, strong aug (color jitter, random resized crop 0.7–1.0, hflip, random erasing, mild rotation ±10°).
   - **Threshold calibration**: at end of training, sweep per-class thresholds on val (0.1..0.9 step 0.05) to maximize per-class F1; save `tag_thresholds.json` alongside `tag_names.json`. Inference uses these per-class thresholds instead of a single `TAG_THRESH`.
   - Save best by val macro-F1, not val loss (loss can decrease while macro-F1 stays bad under imbalance).

2. **Inference** in `master_test.py`:
   - Update `run_tags` (`master_test.py:408–426`) to load `tag_thresholds.json`; if missing, fall back to scalar `cfg.TAG_SCORE_THRESH`.
   - Load model via timm by reading the saved `backbone` name from the checkpoint payload (so future backbone changes don't break inference).

3. **Config additions**: `TAG_BACKBONE = "convnext_tiny"`, `TAG_SEG_INPUT_SIZE = 320`, `TAG_USE_ASYM_LOSS = True`, `TAG_GAMMA_NEG = 4`, `TAG_GAMMA_POS = 1`, `TAG_WARMUP_ITERS`, plus existing `TAG_LR/TAG_WD/TAG_EPOCHS/TAG_BATCH_SIZE` reused.

### D. Shared generalization improvements (apply across B + C, optionally for future use)

Add a small shared utility `code/training_utils.py` to avoid duplication:
- `make_warmup_cosine_lambda(warmup_iters, total_iters)` — exactly the closure used in `train_polyline_seg.py:269–274`.
- `ModelEMA` — standard exponential moving average wrapper (decay 0.9998), used for val + saved as `best_ema.pt`.
- `StrongAugment` — a single `albumentations`-free numpy/cv2 pipeline shared by KP-seg and tag (geometric ops with mask awareness flag).
- `deterministic_split(images, val_ratio, seed)` — deterministic, reproducible val split that the working polyline already does inline.
- `bce_dice_loss` + `compute_iou` (lift these from `train_polyline_seg.py` so they're reused by KP-seg).

This consolidates the recipe into one place so the polyline, keypoint, and (eventually) any future segmentation model use identical training infrastructure.

### E. Files

**Modify:**
- `code/config.py` — append `POLY_SEG_*`, `KP_SEG_*`, `TAG_*` blocks (preserve existing keys; old keys can be deleted once inference is migrated).
- `code/master_train.py:425–454` — replace the two-stage polyline block with a single `train_polyline_seg(images, log_fn=log)` call.
- `code/master_train.py:410–423` — replace `train_kp` call with `train_keypoint_seg`.
- `code/master_train.py:456–469` — point tag to `train_tag_v2`.
- `code/master_test.py:121–157` — replace model-loader helpers (`load_heatmap_model`, `load_edge_model`, `load_tag_model`) with seg/timm-aware loaders that read `backbone`+`classes` from the checkpoint payload.
- `code/master_test.py:247–277` (keypoint), `280–405` (polyline), `408–426` (tag) — rewrite each with the new inference functions described above.
- `code/master_test.py:46–55` — update model-path constants to point at `best_seg.pt` / per-class artifacts.

**Create:**
- `code/training_utils.py` (shared utilities).
- `code/polyline_model/train_polyline_seg.py` (move from `polyline_model_working/`).
- `code/keypoint_model/train_keypoint_seg.py` (new, mirrors polyline-seg).
- `code/tag_model/train_tag_v2.py` (new).

**Delete after verification:**
- `code/polyline_model_working/` (entire folder, once moved).
- `code/polyline_model/train_polyline_s1.py`, `train_polyline_s2.py`.
- `code/keypoint_model/train_keypoint.py` (only after S1/S2 are removed, since they import from it).
- `code/tag_model/train_tag.py` (after `train_tag_v2.py` is verified end-to-end).

### F. Dependencies

- `timm` — already used by the working polyline file. Confirm installed (`pip show timm`).
- `opencv-contrib-python` — needed for `cv2.ximgproc.thinning` (optional; `skimage.morphology.skeletonize` fallback is acceptable).
- `scikit-image` — for skeletonization fallback and connected components.

### G. Reusable existing utilities

- `data_prep.split_annotations.parse_cvat_xml` — single source of truth; do not re-parse.
- `data_prep.split_annotations.get_labels_for_type` — for auto-populating per-class lists for polyline-seg + KP-seg.
- `data_prep.split_annotations.filter_images_by_type` — drop irrelevant images before training each model.
- `HRNetSegModel` + `bce_dice_loss` + `compute_iou` from `train_polyline_seg.py` — lift into `training_utils.py` and reuse.

## Verification

1. **Smoke test**: run `python master_train.py` on the current `lettuce_v2` dataset (or smallest available) with `*_EPOCHS = 2` overrides — confirm all three model trainers start, complete one epoch, and write checkpoints + per-class CSV logs.
2. **Polyline-seg correctness**: confirm `master_test.py` loads `best_seg.pt`, produces non-empty `polylines` in the output CVAT XML on a couple of train images, and that visual `annotated/*.jpg` shows lines tracing the right structures (catches skeletonize+trace regressions).
3. **Keypoint per-class**: inference output CVAT XML should now have `points` elements with the **correct `label`** (not hardcoded `"keypoint"`).
4. **Tag thresholds**: confirm `tag_thresholds.json` written; confirm inference uses per-class thresholds (print loaded thresholds at startup).
5. **Generalization probe**: hold out a small set of unseen images (not in training XML), run `master_test.py` on them, eyeball the annotated outputs and `summary_report.txt`. Compare detection counts vs. previous run to confirm the fix.
6. **Resume / fine-tune**: set `POLY_SEG_RESUME_FROM=best_seg.pt` and `POLY_SEG_FT_EPOCHS=20` with `POLY_SEG_LR=1e-5`; confirm training resumes and val mIoU stays at or improves above the original.
