# # ============================================================
# # polyline_model/train_polyline_s2.py
# # Stage 2 — Edge connectivity MLP
# # Given pairs of detected vertices, predict if they are connected
# # ============================================================

# import os
# import sys
# import json
# import time
# import datetime
# import csv
# import random

# import torch
# import torch.nn as nn
# import torch.nn.functional as F
# from torch.utils.data import Dataset, DataLoader
# import numpy as np

# sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
# import config as cfg


# # ============================================================
# # --- Model: Edge connectivity MLP ---
# # ============================================================

# class EdgeMLP(nn.Module):
#     """
#     Given two vertex locations (x1,y1,x2,y2) + distance features,
#     predicts whether they should be connected by a polyline edge.

#     Input features per pair:
#         x1, y1, x2, y2          — normalized coordinates of both vertices
#         dx, dy                   — relative offset
#         dist                     — euclidean distance
#         angle                    — direction angle (sin, cos)
#     Total: 9 features
#     """

#     def __init__(self, hidden_dim=128):
#         super().__init__()
#         self.net = nn.Sequential(
#             nn.Linear(9, hidden_dim),
#             nn.ReLU(inplace=True),
#             nn.Dropout(0.2),
#             nn.Linear(hidden_dim, hidden_dim),
#             nn.ReLU(inplace=True),
#             nn.Dropout(0.2),
#             nn.Linear(hidden_dim, hidden_dim // 2),
#             nn.ReLU(inplace=True),
#             nn.Linear(hidden_dim // 2, 1),
#         )
#         self._init_weights()

#     def _init_weights(self):
#         for m in self.modules():
#             if isinstance(m, nn.Linear):
#                 nn.init.xavier_uniform_(m.weight)
#                 nn.init.zeros_(m.bias)

#     def forward(self, x):
#         return self.net(x).squeeze(-1)   # (B,) logits


# def build_edge_features(x1, y1, x2, y2):
#     """
#     Build 9-dimensional feature vector for a vertex pair.
#     All coordinates are normalized [0, 1].
#     """
#     dx    = x2 - x1
#     dy    = y2 - y1
#     dist  = np.sqrt(dx ** 2 + dy ** 2)
#     angle = np.arctan2(dy, dx)
#     return np.array([
#         x1, y1, x2, y2,
#         dx, dy, dist,
#         np.sin(angle), np.cos(angle)
#     ], dtype=np.float32)


# # ============================================================
# # --- Dataset ---
# # ============================================================

# class EdgeDataset(Dataset):
#     """
#     Builds positive/negative vertex pair examples from polyline annotations.
#     Positive pairs: consecutive vertices in the same polyline.
#     Negative pairs: non-consecutive vertices or vertices from different polylines.
#     Negative pairs filtered to be within MAX_DIST to keep the task hard.
#     """

#     def __init__(self, coco_json_path, img_dir, max_dist,
#                  neg_ratio=3.0, seed=42):
#         random.seed(seed)
#         np.random.seed(seed)

#         self.max_dist  = max_dist
#         self.neg_ratio = neg_ratio

#         with open(coco_json_path) as f:
#             data = json.load(f)

#         images_map = {img["id"]: img for img in data["images"]}
#         self.pairs  = []   # (features, label)

#         # Process each image's polyline annotations
#         from collections import defaultdict
#         anns_by_img = defaultdict(list)
#         for ann in data["annotations"]:
#             if ann.get("shape_type") == "polyline":
#                 seg = ann.get("segmentation", [])
#                 if seg and len(seg[0]) >= 4:
#                     anns_by_img[ann["image_id"]].append(ann)

#         for image_id, anns in anns_by_img.items():
#             img_info = images_map.get(image_id)
#             if not img_info:
#                 continue
#             w = img_info.get("width", 1)
#             h = img_info.get("height", 1)

#             # Extract all polylines as lists of normalized vertices
#             polylines = []
#             for ann in anns:
#                 flat = ann["segmentation"][0]
#                 pts  = []
#                 for i in range(0, len(flat), 2):
#                     pts.append((flat[i] / w, flat[i + 1] / h))
#                 if len(pts) >= 2:
#                     polylines.append(pts)

#             if not polylines:
#                 continue

#             # Positive pairs — consecutive vertices
#             pos_pairs = []
#             for pts in polylines:
#                 for i in range(len(pts) - 1):
#                     x1, y1 = pts[i]
#                     x2, y2 = pts[i + 1]
#                     feat   = build_edge_features(x1, y1, x2, y2)
#                     pos_pairs.append(feat)

#             # All vertices across all polylines
#             all_verts = [pt for pts in polylines for pt in pts]

#             # Positive vertex set for exclusion
#             pos_set = set()
#             for pts in polylines:
#                 for i in range(len(pts) - 1):
#                     pos_set.add((i, i + 1))

#             # Negative pairs — non-consecutive, within max_dist
#             neg_pairs = []
#             n_neg_needed = int(len(pos_pairs) * neg_ratio)
#             attempts = 0
#             while len(neg_pairs) < n_neg_needed and attempts < 10000:
#                 attempts += 1
#                 i = random.randrange(len(all_verts))
#                 j = random.randrange(len(all_verts))
#                 if i == j:
#                     continue
#                 x1, y1 = all_verts[i]
#                 x2, y2 = all_verts[j]
#                 dist = np.sqrt((x2 - x1) ** 2 + (y2 - y1) ** 2)
#                 if dist > self.max_dist:
#                     continue
#                 # Check it is not a positive pair
#                 feat = build_edge_features(x1, y1, x2, y2)
#                 neg_pairs.append(feat)

#             for feat in pos_pairs:
#                 self.pairs.append((feat, 1.0))
#             for feat in neg_pairs:
#                 self.pairs.append((feat, 0.0))

#         random.shuffle(self.pairs)
#         n_pos = sum(1 for _, l in self.pairs if l == 1.0)
#         n_neg = sum(1 for _, l in self.pairs if l == 0.0)
#         print(f"    Edge pairs — positive: {n_pos} | negative: {n_neg}")

#     def __len__(self):
#         return len(self.pairs)

#     def __getitem__(self, idx):
#         feat, label = self.pairs[idx]
#         return (
#             torch.tensor(feat,  dtype=torch.float32),
#             torch.tensor(label, dtype=torch.float32)
#         )


# # ============================================================
# # --- Training Loop ---
# # ============================================================

# def train(log_fn=print):
#     log_fn("\n" + "─" * 50)
#     log_fn("  POLYLINE MODEL — Stage 2: Edge Connectivity")
#     log_fn("─" * 50)

#     os.makedirs(cfg.POLYLINE_SAVE_DIR, exist_ok=True)
#     device = torch.device(cfg.DEVICE if torch.cuda.is_available()
#                           else "cpu")

#     full_dataset = EdgeDataset(
#         coco_json_path = cfg.POLYLINE_JSON,
#         img_dir        = cfg.IMG_DIR,
#         max_dist       = cfg.POLY_S2_MAX_DIST,
#         seed           = cfg.RANDOM_SEED,
#     )

#     if len(full_dataset) == 0:
#         log_fn("  No edge pairs found — skipping Stage 2.")
#         return None

#     n_val   = max(1, int(len(full_dataset) * cfg.VAL_RATIO))
#     n_train = len(full_dataset) - n_val
#     train_ds, val_ds = torch.utils.data.random_split(
#         full_dataset, [n_train, n_val],
#         generator=torch.Generator().manual_seed(cfg.RANDOM_SEED)
#     )

#     train_loader = DataLoader(train_ds, batch_size=cfg.POLY_S2_BATCH_SIZE,
#                               shuffle=True,  num_workers=cfg.NUM_WORKERS)
#     val_loader   = DataLoader(val_ds,   batch_size=cfg.POLY_S2_BATCH_SIZE,
#                               shuffle=False, num_workers=cfg.NUM_WORKERS)

#     log_fn(f"  Train pairs : {n_train}")
#     log_fn(f"  Val pairs   : {n_val}")

#     model = EdgeMLP(hidden_dim=cfg.POLY_S2_HIDDEN_DIM).to(device)
#     log_fn(f"  Parameters  : "
#            f"{sum(p.numel() for p in model.parameters()):,}")

#     optimizer = torch.optim.AdamW(
#         model.parameters(),
#         lr           = cfg.POLY_S2_LR,
#         weight_decay = cfg.POLY_S2_WEIGHT_DECAY
#     )
#     scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
#         optimizer, T_max=cfg.POLY_S2_EPOCHS
#     )

#     log_path  = os.path.join(cfg.POLYLINE_SAVE_DIR, "s2_train_log.csv")
#     best_val  = float("inf")
#     best_path = os.path.join(cfg.POLYLINE_SAVE_DIR, "s2_best.pt")
#     start     = time.time()

#     with open(log_path, "w", newline="") as f:
#         csv.writer(f).writerow(
#             ["epoch", "train_loss", "val_loss",
#              "val_acc", "epoch_time"]
#         )

#     for epoch in range(1, cfg.POLY_S2_EPOCHS + 1):
#         t0 = time.time()

#         model.train()
#         train_loss = 0.0
#         for feats, labels in train_loader:
#             feats  = feats.to(device)
#             labels = labels.to(device)
#             logits = model(feats)
#             loss   = F.binary_cross_entropy_with_logits(logits, labels)
#             optimizer.zero_grad()
#             loss.backward()
#             optimizer.step()
#             train_loss += loss.item()
#         train_loss /= len(train_loader)

#         model.eval()
#         val_loss = 0.0
#         correct  = 0
#         total    = 0
#         with torch.no_grad():
#             for feats, labels in val_loader:
#                 feats  = feats.to(device)
#                 labels = labels.to(device)
#                 logits = model(feats)
#                 loss   = F.binary_cross_entropy_with_logits(logits, labels)
#                 val_loss += loss.item()
#                 preds    = (logits.sigmoid() > 0.5).float()
#                 correct += (preds == labels).sum().item()
#                 total   += len(labels)
#         val_loss /= len(val_loader)
#         val_acc   = correct / max(total, 1)

#         scheduler.step()
#         ep_time = time.time() - t0

#         if val_loss < best_val:
#             best_val = val_loss
#             torch.save({"epoch": epoch,
#                         "model_state_dict": model.state_dict(),
#                         "val_loss": best_val}, best_path)

#         if epoch % cfg.POLY_S2_CHECKPOINT_EVERY == 0:
#             ckpt = os.path.join(
#                 cfg.POLYLINE_SAVE_DIR, f"s2_checkpoint_ep{epoch}.pt"
#             )
#             torch.save({"epoch": epoch,
#                         "model_state_dict": model.state_dict()}, ckpt)

#         with open(log_path, "a", newline="") as f:
#             csv.writer(f).writerow(
#                 [epoch, f"{train_loss:.6f}", f"{val_loss:.6f}",
#                  f"{val_acc:.4f}", f"{ep_time:.1f}"]
#             )

#         if epoch % 10 == 0 or epoch == 1:
#             log_fn(f"  Epoch {epoch:>4}/{cfg.POLY_S2_EPOCHS} | "
#                    f"train {train_loss:.4f} | "
#                    f"val {val_loss:.4f} | "
#                    f"acc {val_acc*100:.1f}% | "
#                    f"{ep_time:.1f}s")

#     elapsed = str(datetime.timedelta(seconds=int(time.time() - start)))
#     log_fn(f"\n  Polyline Stage 2 training complete")
#     log_fn(f"  Time       : {elapsed}")
#     log_fn(f"  Best model → {best_path}")

#     return best_path


# ============================================================
# polyline_model/train_polyline_s2.py
# Stage 2 — Edge connectivity MLP — reads from CVAT XML
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
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import config as cfg


# ============================================================
# --- Model ---
# ============================================================

class EdgeMLP(nn.Module):
    def __init__(self, hidden_dim=128):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(9, hidden_dim), nn.ReLU(inplace=True),
            nn.Dropout(0.2),
            nn.Linear(hidden_dim, hidden_dim), nn.ReLU(inplace=True),
            nn.Dropout(0.2),
            nn.Linear(hidden_dim, hidden_dim // 2), nn.ReLU(inplace=True),
            nn.Linear(hidden_dim // 2, 1),
        )
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                nn.init.zeros_(m.bias)

    def forward(self, x):
        return self.net(x).squeeze(-1)


def build_edge_features(x1, y1, x2, y2):
    dx    = x2 - x1
    dy    = y2 - y1
    dist  = np.sqrt(dx ** 2 + dy ** 2)
    angle = np.arctan2(dy, dx)
    return np.array([
        x1, y1, x2, y2, dx, dy, dist,
        np.sin(angle), np.cos(angle)
    ], dtype=np.float32)


# ============================================================
# --- Dataset — reads from parsed XML image list ---
# ============================================================

class EdgeDataset(Dataset):

    def __init__(self, images, max_dist, neg_ratio=3.0, seed=42):
        random.seed(seed)
        np.random.seed(seed)
        self.pairs = []

        for img_info in images:
            if not img_info["polylines"]:
                continue

            w = img_info["width"]
            h = img_info["height"]
            if w == 0 or h == 0:
                continue

            # Normalize all polyline vertices
            polylines = []
            for pl in img_info["polylines"]:
                pts = [
                    (px / w, py / h)
                    for (px, py) in pl["points"]
                ]
                if len(pts) >= 2:
                    polylines.append(pts)

            if not polylines:
                continue

            # Positive pairs — consecutive vertices
            pos_pairs = []
            for pts in polylines:
                for i in range(len(pts) - 1):
                    x1, y1 = pts[i]
                    x2, y2 = pts[i + 1]
                    pos_pairs.append(
                        build_edge_features(x1, y1, x2, y2)
                    )

            # All vertices
            all_verts = [pt for pts in polylines for pt in pts]

            # Negative pairs — within max_dist, not consecutive
            neg_pairs  = []
            n_needed   = int(len(pos_pairs) * neg_ratio)
            attempts   = 0
            while len(neg_pairs) < n_needed and attempts < 10000:
                attempts += 1
                i = random.randrange(len(all_verts))
                j = random.randrange(len(all_verts))
                if i == j:
                    continue
                x1, y1 = all_verts[i]
                x2, y2 = all_verts[j]
                dist = np.sqrt((x2-x1)**2 + (y2-y1)**2)
                if dist > max_dist:
                    continue
                neg_pairs.append(
                    build_edge_features(x1, y1, x2, y2)
                )

            for feat in pos_pairs:
                self.pairs.append((feat, 1.0))
            for feat in neg_pairs:
                self.pairs.append((feat, 0.0))

        random.shuffle(self.pairs)
        n_pos = sum(1 for _, l in self.pairs if l == 1.0)
        n_neg = len(self.pairs) - n_pos
        print(f"    Edge pairs — positive: {n_pos} | negative: {n_neg}")

    def __len__(self):
        return len(self.pairs)

    def __getitem__(self, idx):
        feat, label = self.pairs[idx]
        return (
            torch.tensor(feat,  dtype=torch.float32),
            torch.tensor(label, dtype=torch.float32),
        )


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
    log_fn("  POLYLINE MODEL — Stage 2: Edge Connectivity")
    log_fn("─" * 50)

    os.makedirs(cfg.POLYLINE_SAVE_DIR, exist_ok=True)
    device = torch.device(cfg.DEVICE if torch.cuda.is_available()
                          else "cpu")

    full_dataset = EdgeDataset(
        images    = images,
        max_dist  = cfg.POLY_S2_MAX_DIST,
        seed      = cfg.RANDOM_SEED,
    )

    if len(full_dataset) == 0:
        log_fn("  No edge pairs found — skipping Stage 2.")
        return None

    n_val   = max(1, int(len(full_dataset) * cfg.VAL_RATIO))
    n_train = len(full_dataset) - n_val
    train_ds, val_ds = torch.utils.data.random_split(
        full_dataset, [n_train, n_val],
        generator=torch.Generator().manual_seed(cfg.RANDOM_SEED)
    )

    train_loader = DataLoader(train_ds, batch_size=cfg.POLY_S2_BATCH_SIZE,
                              shuffle=True,  num_workers=cfg.NUM_WORKERS)
    val_loader   = DataLoader(val_ds,   batch_size=cfg.POLY_S2_BATCH_SIZE,
                              shuffle=False, num_workers=cfg.NUM_WORKERS)

    log_fn(f"  Train pairs : {n_train} | Val pairs: {n_val}")

    model = EdgeMLP(hidden_dim=cfg.POLY_S2_HIDDEN_DIM).to(device)
    log_fn(f"  Parameters  : "
           f"{sum(p.numel() for p in model.parameters()):,}")

    optimizer = torch.optim.AdamW(
        model.parameters(), lr=cfg.POLY_S2_LR,
        weight_decay=cfg.POLY_S2_WEIGHT_DECAY
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=cfg.POLY_S2_EPOCHS
    )

    log_path  = os.path.join(cfg.POLYLINE_SAVE_DIR, "s2_train_log.csv")
    best_val  = float("inf")
    best_path = os.path.join(cfg.POLYLINE_SAVE_DIR, "s2_best.pt")
    start     = time.time()

    with open(log_path, "w", newline="") as f:
        csv.writer(f).writerow(
            ["epoch", "train_loss", "val_loss", "val_acc", "epoch_time"]
        )

    for epoch in range(1, cfg.POLY_S2_EPOCHS + 1):
        t0 = time.time()

        model.train()
        train_loss = 0.0
        for feats, labels in train_loader:
            feats  = feats.to(device)
            labels = labels.to(device)
            loss   = F.binary_cross_entropy_with_logits(
                model(feats), labels
            )
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            train_loss += loss.item()
        train_loss /= len(train_loader)

        model.eval()
        val_loss = 0.0
        correct  = 0
        total    = 0
        with torch.no_grad():
            for feats, labels in val_loader:
                feats  = feats.to(device)
                labels = labels.to(device)
                logits = model(feats)
                val_loss += F.binary_cross_entropy_with_logits(
                    logits, labels
                ).item()
                preds    = (logits.sigmoid() > 0.5).float()
                correct += (preds == labels).sum().item()
                total   += len(labels)
        val_loss /= len(val_loader)
        val_acc   = correct / max(total, 1)
        scheduler.step()
        ep_time = time.time() - t0

        if val_loss < best_val:
            best_val = val_loss
            torch.save({"epoch": epoch,
                        "model_state_dict": model.state_dict(),
                        "val_loss": best_val}, best_path)

        if epoch % cfg.POLY_S2_CHECKPOINT_EVERY == 0:
            torch.save({"epoch": epoch,
                        "model_state_dict": model.state_dict()},
                       os.path.join(cfg.POLYLINE_SAVE_DIR,
                                    f"s2_checkpoint_ep{epoch}.pt"))

        with open(log_path, "a", newline="") as f:
            csv.writer(f).writerow(
                [epoch, f"{train_loss:.6f}", f"{val_loss:.6f}",
                 f"{val_acc:.4f}", f"{ep_time:.1f}"]
            )

        if epoch % 10 == 0 or epoch == 1:
            log_fn(f"  Epoch {epoch:>4}/{cfg.POLY_S2_EPOCHS} | "
                   f"train {train_loss:.4f} | val {val_loss:.4f} | "
                   f"acc {val_acc*100:.1f}% | {ep_time:.1f}s")

    elapsed = str(datetime.timedelta(seconds=int(time.time() - start)))
    log_fn(f"\n  Polyline S2 complete | Time: {elapsed}")
    log_fn(f"  Best model → {best_path}")
    return best_path