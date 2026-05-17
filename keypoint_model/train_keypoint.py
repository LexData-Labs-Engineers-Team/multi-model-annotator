# # ============================================================
# # keypoint_model/train_keypoint.py
# # Heatmap-based keypoint detector using ResNet encoder-decoder
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
# from torchvision import models
# from torchvision.models import ResNet50_Weights, ResNet18_Weights
# import torchvision.transforms.functional as TF
# import numpy as np
# from PIL import Image

# sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
# import config as cfg


# # ============================================================
# # --- Model: ResNet encoder + lightweight decoder ---
# # ============================================================

# class KeypointHeatmapModel(nn.Module):
#     """
#     ResNet encoder + decoder that outputs a single-channel heatmap.
#     Heatmap peaks = keypoint locations.
#     """

#     def __init__(self, backbone="resnet50", pretrained=True):
#         super().__init__()

#         if backbone == "resnet50":
#             weights  = ResNet50_Weights.DEFAULT if pretrained else None
#             base     = models.resnet50(weights=weights)
#             enc_ch   = [64, 256, 512, 1024, 2048]
#         else:
#             weights  = ResNet18_Weights.DEFAULT if pretrained else None
#             base     = models.resnet18(weights=weights)
#             enc_ch   = [64, 64, 128, 256, 512]

#         # Encoder layers
#         self.enc0 = nn.Sequential(base.conv1, base.bn1, base.relu)
#         self.pool = base.maxpool
#         self.enc1 = base.layer1
#         self.enc2 = base.layer2
#         self.enc3 = base.layer3
#         self.enc4 = base.layer4

#         # Decoder — progressive upsampling with skip connections
#         self.dec4 = self._dec_block(enc_ch[4], 256)
#         self.dec3 = self._dec_block(256 + enc_ch[3], 128)
#         self.dec2 = self._dec_block(128 + enc_ch[2], 64)
#         self.dec1 = self._dec_block(64  + enc_ch[1], 32)
#         self.dec0 = self._dec_block(32  + enc_ch[0], 16)

#         # Final 1x1 conv → single heatmap channel
#         self.head = nn.Sequential(
#             nn.Conv2d(16, 16, 3, padding=1),
#             nn.ReLU(inplace=True),
#             nn.Conv2d(16, 1, 1),
#         )

#     def _dec_block(self, in_ch, out_ch):
#         return nn.Sequential(
#             nn.Conv2d(in_ch, out_ch, 3, padding=1),
#             nn.BatchNorm2d(out_ch),
#             nn.ReLU(inplace=True),
#             nn.Conv2d(out_ch, out_ch, 3, padding=1),
#             nn.BatchNorm2d(out_ch),
#             nn.ReLU(inplace=True),
#         )

#     def forward(self, x):
#         # Encoder
#         e0 = self.enc0(x)           # /2
#         p  = self.pool(e0)          # /4
#         e1 = self.enc1(p)           # /4
#         e2 = self.enc2(e1)          # /8
#         e3 = self.enc3(e2)          # /16
#         e4 = self.enc4(e3)          # /32

#         # Decoder with skip connections
#         d = self.dec4(e4)
#         d = F.interpolate(d, size=e3.shape[2:], mode="bilinear",
#                           align_corners=False)
#         d = self.dec3(torch.cat([d, e3], dim=1))

#         d = F.interpolate(d, size=e2.shape[2:], mode="bilinear",
#                           align_corners=False)
#         d = self.dec2(torch.cat([d, e2], dim=1))

#         d = F.interpolate(d, size=e1.shape[2:], mode="bilinear",
#                           align_corners=False)
#         d = self.dec1(torch.cat([d, e1], dim=1))

#         d = F.interpolate(d, size=e0.shape[2:], mode="bilinear",
#                           align_corners=False)
#         d = self.dec0(torch.cat([d, e0], dim=1))

#         d = F.interpolate(d, scale_factor=2, mode="bilinear",
#                           align_corners=False)
#         heatmap = self.head(d)      # (B, 1, H, W)
#         return heatmap


# # ============================================================
# # --- Dataset ---
# # ============================================================

# class KeypointDataset(Dataset):

#     def __init__(self, coco_json_path, img_dir, input_size,
#                  sigma, augment=True):
#         self.img_dir    = img_dir
#         self.input_size = input_size
#         self.sigma      = sigma
#         self.augment    = augment

#         with open(coco_json_path) as f:
#             data = json.load(f)

#         self.images_map = {img["id"]: img for img in data["images"]}

#         # Collect images that have keypoint annotations
#         from collections import defaultdict
#         anns_by_img = defaultdict(list)
#         for ann in data["annotations"]:
#             kps = ann.get("keypoints", [])
#             if kps and len(kps) >= 3:
#                 anns_by_img[ann["image_id"]].append(ann)

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

#         image  = Image.open(img_path).convert("RGB")
#         orig_w, orig_h = image.size
#         image  = image.resize((self.input_size, self.input_size),
#                                Image.BILINEAR)

#         # Collect all keypoint locations
#         kp_locations = []
#         for ann in anns:
#             kps  = ann.get("keypoints", [])
#             n_kp = len(kps) // 3
#             for k in range(n_kp):
#                 x = kps[k * 3]
#                 y = kps[k * 3 + 1]
#                 v = kps[k * 3 + 2]
#                 if v > 0:
#                     # Scale to input_size
#                     sx = x * self.input_size / orig_w
#                     sy = y * self.input_size / orig_h
#                     kp_locations.append((sx, sy))

#         # Build Gaussian heatmap
#         heatmap = self._build_heatmap(kp_locations, self.input_size,
#                                       self.sigma)

#         # Augment
#         if self.augment and torch.rand(1).item() > 0.5:
#             image   = TF.hflip(image)
#             heatmap = np.fliplr(heatmap).copy()

#         # To tensor
#         image   = TF.to_tensor(image)
#         image   = TF.normalize(image, cfg.PIXEL_MEAN, cfg.PIXEL_STD)
#         heatmap = torch.tensor(heatmap, dtype=torch.float32).unsqueeze(0)

#         return image, heatmap

#     def _build_heatmap(self, kp_locations, size, sigma):
#         heatmap = np.zeros((size, size), dtype=np.float32)
#         for (cx, cy) in kp_locations:
#             x0 = max(0, int(cx) - 3 * sigma)
#             x1 = min(size, int(cx) + 3 * sigma + 1)
#             y0 = max(0, int(cy) - 3 * sigma)
#             y1 = min(size, int(cy) + 3 * sigma + 1)
#             for y in range(y0, y1):
#                 for x in range(x0, x1):
#                     d2 = (x - cx) ** 2 + (y - cy) ** 2
#                     val = np.exp(-d2 / (2 * sigma ** 2))
#                     if val > heatmap[y, x]:
#                         heatmap[y, x] = val
#         return heatmap


# # ============================================================
# # --- Training Loop ---
# # ============================================================

# def train(log_fn=print):
#     log_fn("\n" + "─" * 50)
#     log_fn("  KEYPOINT MODEL — ResNet Heatmap")
#     log_fn("─" * 50)

#     os.makedirs(cfg.KEYPOINT_SAVE_DIR, exist_ok=True)
#     device = torch.device(cfg.DEVICE if torch.cuda.is_available()
#                           else "cpu")

#     # Dataset
#     from sklearn.model_selection import train_test_split
#     full_dataset = KeypointDataset(
#         coco_json_path = cfg.KEYPOINT_JSON,
#         img_dir        = cfg.IMG_DIR,
#         input_size     = cfg.INPUT_SIZE,
#         sigma          = cfg.KP_HEATMAP_SIGMA,
#         augment        = True,
#     )

#     n_val   = max(1, int(len(full_dataset) * cfg.VAL_RATIO))
#     n_train = len(full_dataset) - n_val
#     train_ds, val_ds = torch.utils.data.random_split(
#         full_dataset, [n_train, n_val],
#         generator=torch.Generator().manual_seed(cfg.RANDOM_SEED)
#     )
#     val_ds.dataset.augment = False

#     train_loader = DataLoader(train_ds, batch_size=cfg.KP_BATCH_SIZE,
#                               shuffle=True,  num_workers=cfg.NUM_WORKERS,
#                               pin_memory=True)
#     val_loader   = DataLoader(val_ds,   batch_size=cfg.KP_BATCH_SIZE,
#                               shuffle=False, num_workers=cfg.NUM_WORKERS,
#                               pin_memory=True)

#     log_fn(f"  Train samples : {n_train}")
#     log_fn(f"  Val samples   : {n_val}")

#     # Model
#     model = KeypointHeatmapModel(
#         backbone   = cfg.KP_BACKBONE,
#         pretrained = cfg.KP_PRETRAINED
#     ).to(device)

#     total_p = sum(p.numel() for p in model.parameters())
#     log_fn(f"  Parameters    : {total_p/1e6:.1f}M")

#     optimizer = torch.optim.AdamW(
#         model.parameters(),
#         lr           = cfg.KP_LR,
#         weight_decay = cfg.KP_WEIGHT_DECAY
#     )
#     scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
#         optimizer, T_max=cfg.KP_EPOCHS
#     )

#     log_path  = os.path.join(cfg.KEYPOINT_SAVE_DIR, "train_log.csv")
#     best_val  = float("inf")
#     best_path = os.path.join(cfg.KEYPOINT_SAVE_DIR, "best.pt")
#     start     = time.time()

#     with open(log_path, "w", newline="") as f:
#         csv.writer(f).writerow(
#             ["epoch", "train_loss", "val_loss", "epoch_time"]
#         )

#     for epoch in range(1, cfg.KP_EPOCHS + 1):
#         t0 = time.time()

#         # Train
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

#         # Validate
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

#         # Save best
#         if val_loss < best_val:
#             best_val = val_loss
#             torch.save({"epoch": epoch,
#                         "model_state_dict": model.state_dict(),
#                         "val_loss": best_val}, best_path)

#         # Checkpoint
#         if epoch % cfg.KP_CHECKPOINT_EVERY == 0:
#             ckpt_path = os.path.join(
#                 cfg.KEYPOINT_SAVE_DIR, f"checkpoint_ep{epoch}.pt"
#             )
#             torch.save({"epoch": epoch,
#                         "model_state_dict": model.state_dict()}, ckpt_path)

#         # Log
#         with open(log_path, "a", newline="") as f:
#             csv.writer(f).writerow(
#                 [epoch, f"{train_loss:.6f}",
#                  f"{val_loss:.6f}", f"{ep_time:.1f}"]
#             )

#         if epoch % 10 == 0 or epoch == 1:
#             log_fn(f"  Epoch {epoch:>4}/{cfg.KP_EPOCHS} | "
#                    f"train {train_loss:.4f} | "
#                    f"val {val_loss:.4f} | "
#                    f"{ep_time:.1f}s")

#     elapsed = str(datetime.timedelta(seconds=int(time.time() - start)))
#     log_fn(f"\n  Keypoint model training complete")
#     log_fn(f"  Time       : {elapsed}")
#     log_fn(f"  Best model → {best_path}")

#     return best_path


# ============================================================
# keypoint_model/train_keypoint.py
# Heatmap keypoint detector — reads directly from CVAT XML
# ============================================================

import os
import sys
import time
import datetime
import csv
import random

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torchvision import models
from torchvision.models import ResNet50_Weights, ResNet18_Weights
import torchvision.transforms.functional as TF
import numpy as np
from PIL import Image

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import config as cfg


# ============================================================
# --- Model ---
# ============================================================

class KeypointHeatmapModel(nn.Module):

    def __init__(self, backbone="resnet50", pretrained=True):
        super().__init__()

        if backbone == "resnet50":
            weights = ResNet50_Weights.DEFAULT if pretrained else None
            base    = models.resnet50(weights=weights)
            enc_ch  = [64, 256, 512, 1024, 2048]
        else:
            weights = ResNet18_Weights.DEFAULT if pretrained else None
            base    = models.resnet18(weights=weights)
            enc_ch  = [64, 64, 128, 256, 512]

        self.enc0 = nn.Sequential(base.conv1, base.bn1, base.relu)
        self.pool = base.maxpool
        self.enc1 = base.layer1
        self.enc2 = base.layer2
        self.enc3 = base.layer3
        self.enc4 = base.layer4

        self.dec4 = self._dec_block(enc_ch[4], 256)
        self.dec3 = self._dec_block(256 + enc_ch[3], 128)
        self.dec2 = self._dec_block(128 + enc_ch[2], 64)
        self.dec1 = self._dec_block(64  + enc_ch[1], 32)
        self.dec0 = self._dec_block(32  + enc_ch[0], 16)
        self.head = nn.Sequential(
            nn.Conv2d(16, 16, 3, padding=1), nn.ReLU(inplace=True),
            nn.Conv2d(16, 1, 1),
        )

    def _dec_block(self, in_ch, out_ch):
        return nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 3, padding=1),
            nn.BatchNorm2d(out_ch), nn.ReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, 3, padding=1),
            nn.BatchNorm2d(out_ch), nn.ReLU(inplace=True),
        )

    def forward(self, x):
        e0 = self.enc0(x)
        p  = self.pool(e0)
        e1 = self.enc1(p)
        e2 = self.enc2(e1)
        e3 = self.enc3(e2)
        e4 = self.enc4(e3)
        d  = self.dec4(e4)
        d  = F.interpolate(d, size=e3.shape[2:],
                           mode="bilinear", align_corners=False)
        d  = self.dec3(torch.cat([d, e3], dim=1))
        d  = F.interpolate(d, size=e2.shape[2:],
                           mode="bilinear", align_corners=False)
        d  = self.dec2(torch.cat([d, e2], dim=1))
        d  = F.interpolate(d, size=e1.shape[2:],
                           mode="bilinear", align_corners=False)
        d  = self.dec1(torch.cat([d, e1], dim=1))
        d  = F.interpolate(d, size=e0.shape[2:],
                           mode="bilinear", align_corners=False)
        d  = self.dec0(torch.cat([d, e0], dim=1))
        d  = F.interpolate(d, scale_factor=2,
                           mode="bilinear", align_corners=False)
        return self.head(d)


# ============================================================
# --- Dataset — reads from parsed XML image list ---
# ============================================================

class KeypointDataset(Dataset):

    def __init__(self, images, img_dir, input_size, sigma, augment=True):
        self.img_dir    = img_dir
        self.input_size = input_size
        self.sigma      = sigma
        self.augment    = augment

        # Keep only images that have keypoint annotations
        self.samples = [
            img for img in images if img["keypoints"]
        ]

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        img_info = self.samples[idx]
        img_path = os.path.join(self.img_dir, img_info["name"])

        image    = Image.open(img_path).convert("RGB")
        orig_w, orig_h = image.size
        image    = image.resize((self.input_size, self.input_size),
                                Image.BILINEAR)

        # Scale keypoint locations to input_size
        kp_locations = []
        for kp in img_info["keypoints"]:
            sx = kp["x"] * self.input_size / orig_w
            sy = kp["y"] * self.input_size / orig_h
            kp_locations.append((sx, sy))

        heatmap = _build_heatmap(kp_locations, self.input_size, self.sigma)

        if self.augment and torch.rand(1).item() > 0.5:
            image   = TF.hflip(image)
            heatmap = np.fliplr(heatmap).copy()

        image   = TF.to_tensor(image)
        image   = TF.normalize(image, cfg.PIXEL_MEAN, cfg.PIXEL_STD)
        heatmap = torch.tensor(heatmap, dtype=torch.float32).unsqueeze(0)
        return image, heatmap


def _build_heatmap(kp_locations, size, sigma):
    heatmap = np.zeros((size, size), dtype=np.float32)
    for (cx, cy) in kp_locations:
        x0 = max(0, int(cx) - 3 * sigma)
        x1 = min(size, int(cx) + 3 * sigma + 1)
        y0 = max(0, int(cy) - 3 * sigma)
        y1 = min(size, int(cy) + 3 * sigma + 1)
        for y in range(y0, y1):
            for x in range(x0, x1):
                d2  = (x - cx) ** 2 + (y - cy) ** 2
                val = np.exp(-d2 / (2 * sigma ** 2))
                if val > heatmap[y, x]:
                    heatmap[y, x] = val
    return heatmap


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
    log_fn("  KEYPOINT MODEL — ResNet Heatmap")
    log_fn("─" * 50)

    os.makedirs(cfg.KEYPOINT_SAVE_DIR, exist_ok=True)
    device = torch.device(cfg.DEVICE if torch.cuda.is_available()
                          else "cpu")

    full_dataset = KeypointDataset(
        images     = images,
        img_dir    = cfg.IMG_DIR,
        input_size = cfg.INPUT_SIZE,
        sigma      = cfg.KP_HEATMAP_SIGMA,
        augment    = True,
    )

    n_val   = max(1, int(len(full_dataset) * cfg.VAL_RATIO))
    n_train = len(full_dataset) - n_val
    train_ds, val_ds = torch.utils.data.random_split(
        full_dataset, [n_train, n_val],
        generator=torch.Generator().manual_seed(cfg.RANDOM_SEED)
    )
    val_ds.dataset.augment = False

    train_loader = DataLoader(train_ds, batch_size=cfg.KP_BATCH_SIZE,
                              shuffle=True,  num_workers=cfg.NUM_WORKERS,
                              pin_memory=True)
    val_loader   = DataLoader(val_ds,   batch_size=cfg.KP_BATCH_SIZE,
                              shuffle=False, num_workers=cfg.NUM_WORKERS,
                              pin_memory=True)

    log_fn(f"  Train samples : {n_train} | Val samples: {n_val}")

    model = KeypointHeatmapModel(
        backbone=cfg.KP_BACKBONE, pretrained=cfg.KP_PRETRAINED
    ).to(device)
    log_fn(f"  Parameters    : "
           f"{sum(p.numel() for p in model.parameters())/1e6:.1f}M")

    optimizer = torch.optim.AdamW(
        model.parameters(), lr=cfg.KP_LR,
        weight_decay=cfg.KP_WEIGHT_DECAY
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=cfg.KP_EPOCHS
    )

    log_path  = os.path.join(cfg.KEYPOINT_SAVE_DIR, "train_log.csv")
    best_val  = float("inf")
    best_path = os.path.join(cfg.KEYPOINT_SAVE_DIR, "best.pt")
    start     = time.time()

    with open(log_path, "w", newline="") as f:
        csv.writer(f).writerow(
            ["epoch", "train_loss", "val_loss", "epoch_time"]
        )

    for epoch in range(1, cfg.KP_EPOCHS + 1):
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

        if epoch % cfg.KP_CHECKPOINT_EVERY == 0:
            torch.save({"epoch": epoch,
                        "model_state_dict": model.state_dict()},
                       os.path.join(cfg.KEYPOINT_SAVE_DIR,
                                    f"checkpoint_ep{epoch}.pt"))

        with open(log_path, "a", newline="") as f:
            csv.writer(f).writerow(
                [epoch, f"{train_loss:.6f}",
                 f"{val_loss:.6f}", f"{ep_time:.1f}"]
            )

        if epoch % 10 == 0 or epoch == 1:
            log_fn(f"  Epoch {epoch:>4}/{cfg.KP_EPOCHS} | "
                   f"train {train_loss:.4f} | val {val_loss:.4f} | "
                   f"{ep_time:.1f}s")

    elapsed = str(datetime.timedelta(seconds=int(time.time() - start)))
    log_fn(f"\n  Keypoint training complete | Time: {elapsed}")
    log_fn(f"  Best model → {best_path}")
    return best_path