# # ============================================================
# # polyline_model/train_polyline_s1.py
# # Stage 1 — Polyline vertex heatmap detector
# # Same architecture as keypoint model, trained on polyline vertices
# # ============================================================

# import os
# import sys
# import json
# import time
# import datetime
# import csv

# import torch
# import torch.nn as nn
# import torch.nn.functional as F
# from torch.utils.data import Dataset, DataLoader
# import torchvision.transforms.functional as TF
# import numpy as np
# from PIL import Image

# sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
# import config as cfg

# # Reuse KeypointHeatmapModel — identical architecture
# sys.path.insert(0, os.path.join(
#     os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
#     "keypoint_model"
# ))
# from train_keypoint import KeypointHeatmapModel


# # ============================================================
# # --- Dataset ---
# # ============================================================

# class PolylineVertexDataset(Dataset):
#     """
#     Loads polyline annotations and builds heatmaps at vertex locations.
#     Each vertex of each polyline is a positive location in the heatmap.
#     """

#     def __init__(self, coco_json_path, img_dir, input_size,
#                  sigma, augment=True):
#         self.img_dir    = img_dir
#         self.input_size = input_size
#         self.sigma      = sigma
#         self.augment    = augment

#         with open(coco_json_path) as f:
#             data = json.load(f)

#         self.images_map = {img["id"]: img for img in data["images"]}

#         from collections import defaultdict
#         anns_by_img = defaultdict(list)
#         for ann in data["annotations"]:
#             if ann.get("shape_type") == "polyline":
#                 seg = ann.get("segmentation", [])
#                 if seg and isinstance(seg, list) and len(seg[0]) >= 4:
#                     anns_by_img[ann["image_id"]].append(ann)

#         self.samples = [
#             (iid, anns)
#             for iid, anns in anns_by_img.items()
#             if iid in self.images_map
#         ]

#     def __len__(self):
#         return len(self.samples)

#     def __getitem__(self, idx):
#         image_id, anns = self.samples[idx]
#         img_info       = self.images_map[image_id]
#         img_path       = os.path.join(self.img_dir, img_info["file_name"])

#         image          = Image.open(img_path).convert("RGB")
#         orig_w, orig_h = image.size
#         image          = image.resize(
#             (self.input_size, self.input_size), Image.BILINEAR
#         )

#         # Collect all polyline vertices
#         vertices = []
#         for ann in anns:
#             flat = ann["segmentation"][0]   # [x1,y1,x2,y2,...]
#             for i in range(0, len(flat), 2):
#                 sx = flat[i]     * self.input_size / orig_w
#                 sy = flat[i + 1] * self.input_size / orig_h
#                 vertices.append((sx, sy))

#         # Build heatmap at vertex locations
#         heatmap = self._build_heatmap(vertices, self.input_size, self.sigma)

#         # Augment — horizontal flip only (preserves vertex validity)
#         if self.augment and torch.rand(1).item() > 0.5:
#             image   = TF.hflip(image)
#             heatmap = np.fliplr(heatmap).copy()

#         image   = TF.to_tensor(image)
#         image   = TF.normalize(image, cfg.PIXEL_MEAN, cfg.PIXEL_STD)
#         heatmap = torch.tensor(heatmap, dtype=torch.float32).unsqueeze(0)

#         return image, heatmap

#     def _build_heatmap(self, vertices, size, sigma):
#         heatmap = np.zeros((size, size), dtype=np.float32)
#         for (cx, cy) in vertices:
#             x0 = max(0, int(cx) - 3 * sigma)
#             x1 = min(size, int(cx) + 3 * sigma + 1)
#             y0 = max(0, int(cy) - 3 * sigma)
#             y1 = min(size, int(cy) + 3 * sigma + 1)
#             for y in range(y0, y1):
#                 for x in range(x0, x1):
#                     d2  = (x - cx) ** 2 + (y - cy) ** 2
#                     val = np.exp(-d2 / (2 * sigma ** 2))
#                     if val > heatmap[y, x]:
#                         heatmap[y, x] = val
#         return heatmap


# # ============================================================
# # --- Training Loop ---
# # ============================================================

# def train(log_fn=print):
#     log_fn("\n" + "─" * 50)
#     log_fn("  POLYLINE MODEL — Stage 1: Vertex Heatmap")
#     log_fn("─" * 50)

#     os.makedirs(cfg.POLYLINE_SAVE_DIR, exist_ok=True)
#     device = torch.device(cfg.DEVICE if torch.cuda.is_available()
#                           else "cpu")

#     full_dataset = PolylineVertexDataset(
#         coco_json_path = cfg.POLYLINE_JSON,
#         img_dir        = cfg.IMG_DIR,
#         input_size     = cfg.INPUT_SIZE,
#         sigma          = cfg.POLY_S1_HEATMAP_SIGMA,
#         augment        = True,
#     )

#     n_val   = max(1, int(len(full_dataset) * cfg.VAL_RATIO))
#     n_train = len(full_dataset) - n_val
#     train_ds, val_ds = torch.utils.data.random_split(
#         full_dataset, [n_train, n_val],
#         generator=torch.Generator().manual_seed(cfg.RANDOM_SEED)
#     )
#     val_ds.dataset.augment = False

#     train_loader = DataLoader(train_ds, batch_size=cfg.POLY_S1_BATCH_SIZE,
#                               shuffle=True,  num_workers=cfg.NUM_WORKERS,
#                               pin_memory=True)
#     val_loader   = DataLoader(val_ds,   batch_size=cfg.POLY_S1_BATCH_SIZE,
#                               shuffle=False, num_workers=cfg.NUM_WORKERS,
#                               pin_memory=True)

#     log_fn(f"  Train samples : {n_train}")
#     log_fn(f"  Val samples   : {n_val}")

#     model = KeypointHeatmapModel(
#         backbone   = cfg.POLY_S1_BACKBONE,
#         pretrained = cfg.POLY_S1_PRETRAINED
#     ).to(device)

#     total_p = sum(p.numel() for p in model.parameters())
#     log_fn(f"  Parameters    : {total_p/1e6:.1f}M")

#     optimizer = torch.optim.AdamW(
#         model.parameters(),
#         lr           = cfg.POLY_S1_LR,
#         weight_decay = cfg.POLY_S1_WEIGHT_DECAY
#     )
#     scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
#         optimizer, T_max=cfg.POLY_S1_EPOCHS
#     )

#     log_path  = os.path.join(cfg.POLYLINE_SAVE_DIR, "s1_train_log.csv")
#     best_val  = float("inf")
#     best_path = os.path.join(cfg.POLYLINE_SAVE_DIR, "s1_best.pt")
#     start     = time.time()

#     with open(log_path, "w", newline="") as f:
#         csv.writer(f).writerow(
#             ["epoch", "train_loss", "val_loss", "epoch_time"]
#         )

#     for epoch in range(1, cfg.POLY_S1_EPOCHS + 1):
#         t0 = time.time()

#         model.train()
#         train_loss = 0.0
#         for images, heatmaps in train_loader:
#             images   = images.to(device)
#             heatmaps = heatmaps.to(device)
#             pred     = model(images)
#             pred_up  = F.interpolate(pred, size=heatmaps.shape[2:],
#                                      mode="bilinear", align_corners=False)
#             loss     = F.mse_loss(pred_up.sigmoid(), heatmaps)
#             optimizer.zero_grad()
#             loss.backward()
#             optimizer.step()
#             train_loss += loss.item()
#         train_loss /= len(train_loader)

#         model.eval()
#         val_loss = 0.0
#         with torch.no_grad():
#             for images, heatmaps in val_loader:
#                 images   = images.to(device)
#                 heatmaps = heatmaps.to(device)
#                 pred     = model(images)
#                 pred_up  = F.interpolate(pred, size=heatmaps.shape[2:],
#                                          mode="bilinear", align_corners=False)
#                 loss     = F.mse_loss(pred_up.sigmoid(), heatmaps)
#                 val_loss += loss.item()
#         val_loss /= len(val_loader)

#         scheduler.step()
#         ep_time = time.time() - t0

#         if val_loss < best_val:
#             best_val = val_loss
#             torch.save({"epoch": epoch,
#                         "model_state_dict": model.state_dict(),
#                         "val_loss": best_val}, best_path)

#         if epoch % cfg.POLY_S1_CHECKPOINT_EVERY == 0:
#             ckpt = os.path.join(
#                 cfg.POLYLINE_SAVE_DIR, f"s1_checkpoint_ep{epoch}.pt"
#             )
#             torch.save({"epoch": epoch,
#                         "model_state_dict": model.state_dict()}, ckpt)

#         with open(log_path, "a", newline="") as f:
#             csv.writer(f).writerow(
#                 [epoch, f"{train_loss:.6f}",
#                  f"{val_loss:.6f}", f"{ep_time:.1f}"]
#             )

#         if epoch % 10 == 0 or epoch == 1:
#             log_fn(f"  Epoch {epoch:>4}/{cfg.POLY_S1_EPOCHS} | "
#                    f"train {train_loss:.4f} | "
#                    f"val {val_loss:.4f} | "
#                    f"{ep_time:.1f}s")

#     elapsed = str(datetime.timedelta(seconds=int(time.time() - start)))
#     log_fn(f"\n  Polyline Stage 1 training complete")
#     log_fn(f"  Time       : {elapsed}")
#     log_fn(f"  Best model → {best_path}")

#     return best_path


# ============================================================
# polyline_model/train_polyline_s1.py
# Stage 1 — Polyline vertex heatmap — reads from CVAT XML
# ============================================================

import os
import sys
import time
import datetime
import csv

import torch
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms.functional as TF
import numpy as np
from PIL import Image

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import config as cfg

sys.path.insert(0, os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
    "keypoint_model"
))
from train_keypoint import KeypointHeatmapModel, _build_heatmap


# ============================================================
# --- Dataset ---
# ============================================================

class PolylineVertexDataset(Dataset):

    def __init__(self, images, img_dir, input_size, sigma, augment=True):
        self.img_dir    = img_dir
        self.input_size = input_size
        self.sigma      = sigma
        self.augment    = augment
        # Keep only images that have polyline annotations
        self.samples    = [img for img in images if img["polylines"]]

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        img_info = self.samples[idx]
        img_path = os.path.join(self.img_dir, img_info["name"])

        image    = Image.open(img_path).convert("RGB")
        orig_w, orig_h = image.size
        image    = image.resize((self.input_size, self.input_size),
                                Image.BILINEAR)

        # All polyline vertices scaled to input_size
        vertices = []
        for pl in img_info["polylines"]:
            for (px, py) in pl["points"]:
                sx = px * self.input_size / orig_w
                sy = py * self.input_size / orig_h
                vertices.append((sx, sy))

        heatmap = _build_heatmap(vertices, self.input_size, self.sigma)

        if self.augment and torch.rand(1).item() > 0.5:
            image   = TF.hflip(image)
            heatmap = np.fliplr(heatmap).copy()

        image   = TF.to_tensor(image)
        image   = TF.normalize(image, cfg.PIXEL_MEAN, cfg.PIXEL_STD)
        heatmap = torch.tensor(heatmap, dtype=torch.float32).unsqueeze(0)
        return image, heatmap


# ============================================================
# --- Training loop ---
# ============================================================

def train(images, log_fn=print):
    """
    Args:
        images : parsed image list from parse_cvat_xml()
        log_fn : logging function from master_train.py
    """
    log_fn("\n" + "─" * 50)
    log_fn("  POLYLINE MODEL — Stage 1: Vertex Heatmap")
    log_fn("─" * 50)

    os.makedirs(cfg.POLYLINE_SAVE_DIR, exist_ok=True)
    device = torch.device(cfg.DEVICE if torch.cuda.is_available()
                          else "cpu")

    full_dataset = PolylineVertexDataset(
        images     = images,
        img_dir    = cfg.IMG_DIR,
        input_size = cfg.INPUT_SIZE,
        sigma      = cfg.POLY_S1_HEATMAP_SIGMA,
        augment    = True,
    )

    n_val   = max(1, int(len(full_dataset) * cfg.VAL_RATIO))
    n_train = len(full_dataset) - n_val
    train_ds, val_ds = torch.utils.data.random_split(
        full_dataset, [n_train, n_val],
        generator=torch.Generator().manual_seed(cfg.RANDOM_SEED)
    )
    val_ds.dataset.augment = False

    train_loader = DataLoader(train_ds, batch_size=cfg.POLY_S1_BATCH_SIZE,
                              shuffle=True,  num_workers=cfg.NUM_WORKERS,
                              pin_memory=True)
    val_loader   = DataLoader(val_ds,   batch_size=cfg.POLY_S1_BATCH_SIZE,
                              shuffle=False, num_workers=cfg.NUM_WORKERS,
                              pin_memory=True)

    log_fn(f"  Train samples : {n_train} | Val samples: {n_val}")

    model = KeypointHeatmapModel(
        backbone=cfg.POLY_S1_BACKBONE,
        pretrained=cfg.POLY_S1_PRETRAINED
    ).to(device)
    log_fn(f"  Parameters    : "
           f"{sum(p.numel() for p in model.parameters())/1e6:.1f}M")

    optimizer = torch.optim.AdamW(
        model.parameters(), lr=cfg.POLY_S1_LR,
        weight_decay=cfg.POLY_S1_WEIGHT_DECAY
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=cfg.POLY_S1_EPOCHS
    )

    log_path  = os.path.join(cfg.POLYLINE_SAVE_DIR, "s1_train_log.csv")
    best_val  = float("inf")
    best_path = os.path.join(cfg.POLYLINE_SAVE_DIR, "s1_best.pt")
    start     = time.time()

    with open(log_path, "w", newline="") as f:
        csv.writer(f).writerow(
            ["epoch", "train_loss", "val_loss", "epoch_time"]
        )

    for epoch in range(1, cfg.POLY_S1_EPOCHS + 1):
        t0 = time.time()

        model.train()
        train_loss = 0.0
        for imgs, hms in train_loader:
            imgs = imgs.to(device)
            hms  = hms.to(device)
            pred = model(imgs)
            pred = F.interpolate(pred, size=hms.shape[2:],
                                 mode="bilinear", align_corners=False)
            loss = F.mse_loss(pred.sigmoid(), hms)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            train_loss += loss.item()
        train_loss /= len(train_loader)

        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for imgs, hms in val_loader:
                imgs = imgs.to(device)
                hms  = hms.to(device)
                pred = model(imgs)
                pred = F.interpolate(pred, size=hms.shape[2:],
                                     mode="bilinear", align_corners=False)
                val_loss += F.mse_loss(pred.sigmoid(), hms).item()
        val_loss /= len(val_loader)
        scheduler.step()
        ep_time = time.time() - t0

        if val_loss < best_val:
            best_val = val_loss
            torch.save({"epoch": epoch,
                        "model_state_dict": model.state_dict(),
                        "val_loss": best_val}, best_path)

        if epoch % cfg.POLY_S1_CHECKPOINT_EVERY == 0:
            torch.save({"epoch": epoch,
                        "model_state_dict": model.state_dict()},
                       os.path.join(cfg.POLYLINE_SAVE_DIR,
                                    f"s1_checkpoint_ep{epoch}.pt"))

        with open(log_path, "a", newline="") as f:
            csv.writer(f).writerow(
                [epoch, f"{train_loss:.6f}",
                 f"{val_loss:.6f}", f"{ep_time:.1f}"]
            )

        if epoch % 10 == 0 or epoch == 1:
            log_fn(f"  Epoch {epoch:>4}/{cfg.POLY_S1_EPOCHS} | "
                   f"train {train_loss:.4f} | val {val_loss:.4f} | "
                   f"{ep_time:.1f}s")

    elapsed = str(datetime.timedelta(seconds=int(time.time() - start)))
    log_fn(f"\n  Polyline S1 complete | Time: {elapsed}")
    log_fn(f"  Best model → {best_path}")
    return best_path