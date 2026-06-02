# ============================================================
# keypoint_seg_model/train_keypoint_seg.py
# HRNet-W18 per-class keypoint segmentation trainer.
# Each keypoint is rasterized as a small filled disk on a
# per-class mask channel; the network learns these masks with
# BCE + Dice loss. Inference (in master_test.py) thresholds the
# per-class probability map, takes connected-component
# centroids, and applies distance-based NMS.
#
# Reuses HRNetSegModel + bce_dice_loss + compute_iou from
# polyline_model_working/train_polyline_seg.py — the planned
# shared training_utils.py extraction will collapse the two
# consumers into one import path with no behavior change.
# ============================================================

import os
import sys
import csv
import time
import json
import math
import datetime

import cv2
import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from multi_model_annotator import config as cfg
from multi_model_annotator.polyline_model_working.train_polyline_seg import (
    HRNetSegModel, bce_dice_loss, compute_iou,
)


# ============================================================
# --- Dataset: rasterize keypoints as per-class disks ---
# ============================================================

class KeypointSegDataset(Dataset):
    """Each sample yields:
       img_t  (3, S, S)  — float32, ImageNet-normalized
       mask_t (C, S, S)  — float32, 0/1; channel c is the union of all
                           filled disks centered at keypoints of class c.
    """

    def __init__(self, images, img_dir, input_size, class_names,
                 disk_radius, augment):
        self.img_dir     = img_dir
        self.input_size  = input_size
        self.class_names = list(class_names)
        self.class_idx   = {n: i for i, n in enumerate(class_names)}
        self.disk_radius = disk_radius
        self.augment     = augment
        # Drop images that have no keypoints in our class list — they
        # contribute only background and bias the per-class IoU low.
        self.samples = [
            img for img in images
            if any(k["label"] in self.class_idx
                   for k in img.get("keypoints", []))
        ]

    def __len__(self):
        return len(self.samples)

    def _rasterize_masks(self, keypoints, orig_w, orig_h, S):
        """Return (C, S, S) uint8 mask, 1 inside each per-class disk."""
        C = len(self.class_names)
        masks = np.zeros((C, S, S), dtype=np.uint8)
        for kp in keypoints:
            c = self.class_idx.get(kp["label"])
            if c is None:
                continue
            x = int(round(kp["x"] * S / orig_w))
            y = int(round(kp["y"] * S / orig_h))
            if 0 <= x < S and 0 <= y < S:
                cv2.circle(masks[c], (x, y), self.disk_radius,
                           color=1, thickness=-1, lineType=cv2.LINE_8)
        return masks

    def __getitem__(self, idx):
        info = self.samples[idx]
        path = os.path.join(self.img_dir, info["name"])
        bgr  = cv2.imread(path)
        if bgr is None:
            raise FileNotFoundError(f"unreadable image: {path}")
        orig_h, orig_w = bgr.shape[:2]
        S = self.input_size

        img  = cv2.resize(bgr, (S, S), interpolation=cv2.INTER_AREA)
        img  = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        mask = self._rasterize_masks(info["keypoints"], orig_w, orig_h, S)

        if self.augment:
            # Color jitter (multiplicative) — image only.
            if np.random.rand() < 0.5:
                gain = np.random.uniform(0.7, 1.3, size=3).astype(np.float32)
                img  = np.clip(img.astype(np.float32) * gain, 0, 255).astype(np.uint8)
            # Small rotation ±5°; image and every mask channel rotated
            # identically so disk centers stay aligned with image content.
            if np.random.rand() < 0.5:
                angle = float(np.random.uniform(-5.0, 5.0))
                M = cv2.getRotationMatrix2D((S / 2, S / 2), angle, 1.0)
                img  = cv2.warpAffine(img, M, (S, S),
                                      flags=cv2.INTER_LINEAR,
                                      borderMode=cv2.BORDER_REFLECT_101)
                rot = []
                for c in range(mask.shape[0]):
                    rot.append(cv2.warpAffine(
                        mask[c], M, (S, S),
                        flags=cv2.INTER_NEAREST,
                        borderMode=cv2.BORDER_CONSTANT, borderValue=0))
                mask = np.stack(rot, axis=0)

        # ImageNet normalize — matches timm HRNet pretraining.
        img  = img.astype(np.float32) / 255.0
        mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
        std  = np.array([0.229, 0.224, 0.225], dtype=np.float32)
        img  = (img - mean) / std
        img_t  = torch.from_numpy(img.transpose(2, 0, 1)).contiguous()
        mask_t = torch.from_numpy(mask.astype(np.float32))
        return img_t, mask_t


# ============================================================
# --- Train entry point ---
# ============================================================

def train(images, log_fn=print, classes=None):
    """Train HRNet-W18 keypoint segmentation model.
    `classes` (if given) takes precedence over cfg.KP_SEG_CLASSES so
    callers can auto-derive the class list from parsed CVAT XML.
    Returns path to best checkpoint.
    """
    log_fn("\n" + "-" * 50)
    log_fn("  KEYPOINT MODEL — HRNet-W18 per-class disk segmentation")
    log_fn("-" * 50)

    classes_cfg = list(cfg.KP_SEG_CLASSES) if cfg.KP_SEG_CLASSES else []
    classes     = list(classes) if classes else classes_cfg
    if not classes:
        raise RuntimeError(
            "No keypoint classes — pass classes=... or set cfg.KP_SEG_CLASSES."
        )
    num_classes  = len(classes)
    input_size   = int(cfg.KP_SEG_INPUT_SIZE)
    epochs       = int(cfg.KP_SEG_EPOCHS)
    batch_size   = int(cfg.KP_SEG_BATCH_SIZE)
    lr           = float(cfg.KP_SEG_LR)
    weight_decay = float(cfg.KP_SEG_WEIGHT_DECAY)
    warmup_iters = int(cfg.KP_SEG_WARMUP_ITERS)
    disk_radius  = int(cfg.KP_SEG_DISK_RADIUS)
    bce_w        = float(cfg.KP_SEG_BCE_WEIGHT)
    dice_w       = float(cfg.KP_SEG_DICE_WEIGHT)
    save_every   = int(cfg.KP_SEG_CHECKPOINT_EVERY)
    val_thresh   = float(cfg.KP_SEG_THRESH)

    save_dir = cfg.KEYPOINT_SAVE_DIR
    os.makedirs(save_dir, exist_ok=True)
    log_fn(f"  Classes      : {classes}")
    log_fn(f"  Input size   : {input_size}")
    log_fn(f"  Batch        : {batch_size}")
    log_fn(f"  Epochs       : {epochs}")
    log_fn(f"  LR / WD      : {lr} / {weight_decay}")
    log_fn(f"  Disk radius  : {disk_radius} px")
    log_fn(f"  Save dir     : {save_dir}")

    # --- Train / val split (deterministic, eligibility-filtered) ---
    val_ratio = float(getattr(cfg, "VAL_RATIO", 0.2))
    rng = np.random.RandomState(getattr(cfg, "RANDOM_SEED", 42))
    eligible = [
        img for img in images
        if any(k["label"] in classes for k in img.get("keypoints", []))
    ]
    if len(eligible) < 2:
        raise RuntimeError(
            f"Not enough images with target keypoint classes "
            f"({len(eligible)}). Need >= 2."
        )
    idx = np.arange(len(eligible))
    rng.shuffle(idx)
    n_val = max(1, int(len(idx) * val_ratio))
    val_idx, tr_idx = set(idx[:n_val].tolist()), set(idx[n_val:].tolist())
    tr_imgs = [eligible[i] for i in sorted(tr_idx)]
    va_imgs = [eligible[i] for i in sorted(val_idx)]
    log_fn(f"  Train images : {len(tr_imgs)}")
    log_fn(f"  Val images   : {len(va_imgs)}")

    tr_ds = KeypointSegDataset(tr_imgs, cfg.IMG_DIR, input_size, classes,
                               disk_radius, augment=True)
    va_ds = KeypointSegDataset(va_imgs, cfg.IMG_DIR, input_size, classes,
                               disk_radius, augment=False)
    tr_dl = DataLoader(tr_ds, batch_size=batch_size, shuffle=True,
                       num_workers=cfg.NUM_WORKERS, pin_memory=True,
                       drop_last=True, persistent_workers=cfg.NUM_WORKERS > 0)
    va_dl = DataLoader(va_ds, batch_size=batch_size, shuffle=False,
                       num_workers=cfg.NUM_WORKERS, pin_memory=True,
                       persistent_workers=cfg.NUM_WORKERS > 0)

    # --- Model / optim / scheduler ---
    device = torch.device(cfg.DEVICE if torch.cuda.is_available()
                          or cfg.DEVICE == "cpu" else "cpu")
    log_fn(f"  Device       : {device}")
    model = HRNetSegModel(
        backbone=cfg.KP_SEG_BACKBONE,
        num_classes=num_classes,
        pretrained=cfg.KP_SEG_PRETRAINED,
    ).to(device)

    n_params = sum(p.numel() for p in model.parameters())
    log_fn(f"  Params       : {n_params / 1e6:.1f}M")

    optim = torch.optim.AdamW(model.parameters(), lr=lr,
                              weight_decay=weight_decay)
    iters_per_epoch = max(1, len(tr_dl))
    total_iters     = epochs * iters_per_epoch

    def lr_lambda(step):
        if step < warmup_iters:
            return float(step) / float(max(1, warmup_iters))
        progress = (step - warmup_iters) / float(
            max(1, total_iters - warmup_iters))
        return 0.5 * (1.0 + math.cos(math.pi * progress))

    sched = torch.optim.lr_scheduler.LambdaLR(optim, lr_lambda)
    scaler = torch.amp.GradScaler("cuda", enabled=(device.type == "cuda"))

    log_path = os.path.join(save_dir, "train_log_kp_seg.csv")
    log_f = open(log_path, "w", newline="", encoding="utf-8")
    log_w = csv.writer(log_f)
    log_w.writerow(["epoch", "train_loss", "train_bce", "train_dice",
                    "val_loss", "val_mIoU"] +
                   [f"val_iou_{c}" for c in classes] +
                   ["lr", "elapsed_s"])
    log_f.flush()

    best_path = os.path.join(save_dir, "best_seg.pt")
    classes_path = os.path.join(save_dir, "kp_classes.json")
    # Persist class list separately for tools that don't want to
    # crack open the checkpoint payload.
    with open(classes_path, "w", encoding="utf-8") as f:
        json.dump(classes, f, indent=2)

    best_miou = -1.0
    start = time.time()
    step  = 0

    for ep in range(1, epochs + 1):
        # --- Train ---
        model.train()
        tr_loss = tr_bce = tr_dice = 0.0
        n_seen = 0
        for img_t, mask_t in tr_dl:
            img_t, mask_t = img_t.to(device), mask_t.to(device)
            optim.zero_grad(set_to_none=True)
            with torch.amp.autocast("cuda", enabled=(device.type == "cuda")):
                logits = model(img_t)
                loss, bce_part, dice_part = bce_dice_loss(
                    logits, mask_t, bce_w, dice_w)
            scaler.scale(loss).backward()
            scaler.step(optim)
            scaler.update()
            sched.step()
            step += 1
            bs = img_t.shape[0]
            tr_loss += loss.item() * bs
            tr_bce  += bce_part.item() * bs
            tr_dice += dice_part.item() * bs
            n_seen  += bs
        tr_loss /= max(1, n_seen)
        tr_bce  /= max(1, n_seen)
        tr_dice /= max(1, n_seen)

        # --- Val ---
        model.eval()
        va_loss = 0.0; va_seen = 0
        iou_sum = torch.zeros(num_classes, device=device)
        iou_n   = 0
        with torch.no_grad():
            for img_t, mask_t in va_dl:
                img_t, mask_t = img_t.to(device), mask_t.to(device)
                with torch.amp.autocast("cuda", enabled=(device.type == "cuda")):
                    logits = model(img_t)
                    loss, _, _ = bce_dice_loss(logits, mask_t, bce_w, dice_w)
                bs = img_t.shape[0]
                va_loss += loss.item() * bs
                va_seen += bs
                probs = torch.sigmoid(logits.float())
                iou_sum += compute_iou(probs, mask_t.float(), val_thresh)
                iou_n   += 1
        va_loss /= max(1, va_seen)
        iou_per_class = (iou_sum / max(1, iou_n)).detach().cpu().tolist()
        mIoU = float(np.mean(iou_per_class)) if iou_per_class else 0.0

        elapsed = time.time() - start
        cur_lr  = sched.get_last_lr()[0]
        log_w.writerow([ep, f"{tr_loss:.4f}", f"{tr_bce:.4f}", f"{tr_dice:.4f}",
                        f"{va_loss:.4f}", f"{mIoU:.4f}"] +
                       [f"{x:.4f}" for x in iou_per_class] +
                       [f"{cur_lr:.2e}", f"{elapsed:.1f}"])
        log_f.flush()

        log_fn(f"  Epoch {ep:3d}/{epochs} | "
               f"train {tr_loss:.4f} (bce {tr_bce:.4f} dice {tr_dice:.4f}) | "
               f"val {va_loss:.4f} | mIoU {mIoU:.4f} | "
               f"per-class " + " ".join(
                   f"{c}={iou_per_class[i]:.3f}"
                   for i, c in enumerate(classes)) +
               f" | lr {cur_lr:.2e}")

        # --- Checkpoint best ---
        if mIoU > best_miou:
            best_miou = mIoU
            payload = {
                "state_dict": model.state_dict(),
                "config"    : {
                    "backbone"   : cfg.KP_SEG_BACKBONE,
                    "num_classes": num_classes,
                    "classes"    : classes,
                    "input_size" : input_size,
                    "disk_radius": disk_radius,
                },
                "best_miou" : best_miou,
                "epoch"     : ep,
            }
            torch.save(payload, best_path)
            log_fn(f"    + best mIoU {best_miou:.4f} -> {best_path}")

        if save_every and (ep % save_every == 0):
            ck = os.path.join(save_dir, f"kp_seg_ep{ep:04d}.pt")
            torch.save({"state_dict": model.state_dict(),
                        "epoch": ep, "mIoU": mIoU}, ck)

    log_f.close()
    total = str(datetime.timedelta(seconds=int(time.time() - start)))
    log_fn(f"\n  Keypoint-seg training complete | Time: {total}")
    log_fn(f"  Best mIoU    : {best_miou:.4f}")
    log_fn(f"  Best ckpt    : {best_path}")
    log_fn(f"  Classes JSON : {classes_path}")
    return best_path
