# # ============================================================
# # tag_model/train_tag.py
# # Multi-label tag classifier using EfficientNet-B0
# # ============================================================

# import os
# import sys
# import json
# import time
# import datetime
# import csv
# from collections import defaultdict

# import torch
# import torch.nn as nn
# import torch.nn.functional as F
# from torch.utils.data import Dataset, DataLoader
# from torchvision import models
# from torchvision.models import EfficientNet_B0_Weights
# import torchvision.transforms.functional as TF
# import torchvision.transforms as T
# import numpy as np
# from PIL import Image

# sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
# import config as cfg


# # ============================================================
# # --- Model ---
# # ============================================================

# class TagClassifier(nn.Module):
#     """
#     EfficientNet-B0 backbone + multi-label classification head.
#     Each output neuron represents one tag — sigmoid → present/absent.
#     """

#     def __init__(self, num_tags, pretrained=True):
#         super().__init__()
#         weights  = EfficientNet_B0_Weights.DEFAULT if pretrained else None
#         base     = models.efficientnet_b0(weights=weights)
#         in_feat  = base.classifier[1].in_features
#         base.classifier = nn.Identity()
#         self.backbone   = base
#         self.head       = nn.Sequential(
#             nn.Dropout(0.3),
#             nn.Linear(in_feat, 256),
#             nn.ReLU(inplace=True),
#             nn.Dropout(0.2),
#             nn.Linear(256, num_tags),
#         )

#     def forward(self, x):
#         feats  = self.backbone(x)
#         logits = self.head(feats)
#         return logits   # (B, num_tags) — raw logits


# # ============================================================
# # --- Dataset ---
# # ============================================================

# class TagDataset(Dataset):
#     """
#     Loads tag annotations from COCO JSON.
#     Tags are stored either as:
#       (a) image-level via img["tags"] — list of tag strings
#       (b) annotation-level via ann["shape_type"] == "tag"

#     Supports both. Builds a multi-label binary vector per image.
#     """

#     def __init__(self, coco_json_path, img_dir, input_size,
#                  tag_names, augment=True):
#         self.img_dir   = img_dir
#         self.input_size= input_size
#         self.tag_names = tag_names
#         self.tag_to_idx= {t: i for i, t in enumerate(tag_names)}
#         self.augment   = augment

#         with open(coco_json_path) as f:
#             data = json.load(f)

#         cat_id_to_name = {
#             c["id"]: c["name"] for c in data.get("categories", [])
#         }
#         images_map = {img["id"]: img for img in data["images"]}

#         # Collect tags per image
#         tags_by_img = defaultdict(set)

#         # Image-level tags
#         for img in data["images"]:
#             for tag in img.get("tags", []):
#                 if tag in self.tag_to_idx:
#                     tags_by_img[img["id"]].add(tag)

#         # Annotation-level tags
#         for ann in data.get("annotations", []):
#             if ann.get("shape_type") == "tag":
#                 cat_id = ann.get("category_id")
#                 name   = cat_id_to_name.get(cat_id, "")
#                 if name in self.tag_to_idx:
#                     tags_by_img[ann["image_id"]].add(name)

#         # Build samples — only images with at least one tag
#         self.samples = []
#         for image_id, tags in tags_by_img.items():
#             if image_id not in images_map:
#                 continue
#             img_info = images_map[image_id]
#             label    = torch.zeros(len(tag_names), dtype=torch.float32)
#             for tag in tags:
#                 label[self.tag_to_idx[tag]] = 1.0
#             self.samples.append((img_info["file_name"], label))

#         self.train_transform = T.Compose([
#             T.Resize((input_size, input_size)),
#             T.RandomHorizontalFlip(),
#             T.ColorJitter(0.2, 0.2, 0.2),
#             T.ToTensor(),
#             T.Normalize(cfg.PIXEL_MEAN, cfg.PIXEL_STD),
#         ])
#         self.val_transform = T.Compose([
#             T.Resize((input_size, input_size)),
#             T.ToTensor(),
#             T.Normalize(cfg.PIXEL_MEAN, cfg.PIXEL_STD),
#         ])

#     def __len__(self):
#         return len(self.samples)

#     def __getitem__(self, idx):
#         fname, label = self.samples[idx]
#         img_path     = os.path.join(self.img_dir, fname)
#         image        = Image.open(img_path).convert("RGB")
#         transform    = self.train_transform if self.augment \
#                        else self.val_transform
#         return transform(image), label


# def get_tag_names(coco_json_path):
#     """
#     Extracts all unique tag names from the COCO JSON.
#     Considers both category names where shape_type == 'tag'
#     and image-level tag fields.
#     """
#     with open(coco_json_path) as f:
#         data = json.load(f)

#     cat_id_to_name = {c["id"]: c["name"] for c in data.get("categories", [])}
#     tag_names      = set()

#     for ann in data.get("annotations", []):
#         if ann.get("shape_type") == "tag":
#             cat_id = ann.get("category_id")
#             name   = cat_id_to_name.get(cat_id, "")
#             if name:
#                 tag_names.add(name)

#     for img in data.get("images", []):
#         for tag in img.get("tags", []):
#             tag_names.add(tag)

#     return sorted(tag_names)


# # ============================================================
# # --- Training Loop ---
# # ============================================================

# def train(log_fn=print):
#     log_fn("\n" + "─" * 50)
#     log_fn("  TAG MODEL — EfficientNet-B0 Multi-label")
#     log_fn("─" * 50)

#     os.makedirs(cfg.TAG_SAVE_DIR, exist_ok=True)
#     device = torch.device(cfg.DEVICE if torch.cuda.is_available()
#                           else "cpu")

#     tag_names = get_tag_names(cfg.TAG_JSON)
#     log_fn(f"  Tags found  : {tag_names}")

#     if not tag_names:
#         log_fn("  No tags found in JSON — skipping tag model.")
#         return None

#     full_dataset = TagDataset(
#         coco_json_path = cfg.TAG_JSON,
#         img_dir        = cfg.IMG_DIR,
#         input_size     = cfg.INPUT_SIZE,
#         tag_names      = tag_names,
#         augment        = True,
#     )

#     n_val   = max(1, int(len(full_dataset) * cfg.VAL_RATIO))
#     n_train = len(full_dataset) - n_val
#     train_ds, val_ds = torch.utils.data.random_split(
#         full_dataset, [n_train, n_val],
#         generator=torch.Generator().manual_seed(cfg.RANDOM_SEED)
#     )
#     val_ds.dataset.augment = False

#     train_loader = DataLoader(train_ds, batch_size=cfg.TAG_BATCH_SIZE,
#                               shuffle=True,  num_workers=cfg.NUM_WORKERS,
#                               pin_memory=True)
#     val_loader   = DataLoader(val_ds,   batch_size=cfg.TAG_BATCH_SIZE,
#                               shuffle=False, num_workers=cfg.NUM_WORKERS,
#                               pin_memory=True)

#     log_fn(f"  Train samples : {n_train}")
#     log_fn(f"  Val samples   : {n_val}")
#     log_fn(f"  Num tags      : {len(tag_names)}")

#     model = TagClassifier(
#         num_tags   = len(tag_names),
#         pretrained = cfg.TAG_PRETRAINED
#     ).to(device)

#     total_p = sum(p.numel() for p in model.parameters())
#     log_fn(f"  Parameters    : {total_p/1e6:.1f}M")

#     optimizer = torch.optim.AdamW(
#         model.parameters(),
#         lr           = cfg.TAG_LR,
#         weight_decay = cfg.TAG_WEIGHT_DECAY
#     )
#     scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
#         optimizer, T_max=cfg.TAG_EPOCHS
#     )

#     log_path  = os.path.join(cfg.TAG_SAVE_DIR, "train_log.csv")
#     best_val  = float("inf")
#     best_path = os.path.join(cfg.TAG_SAVE_DIR, "best.pt")
#     start     = time.time()

#     # Save tag names alongside model for inference
#     tag_meta_path = os.path.join(cfg.TAG_SAVE_DIR, "tag_names.json")
#     with open(tag_meta_path, "w") as f:
#         json.dump(tag_names, f)

#     with open(log_path, "w", newline="") as f:
#         csv.writer(f).writerow(
#             ["epoch", "train_loss", "val_loss",
#              "val_f1", "epoch_time"]
#         )

#     for epoch in range(1, cfg.TAG_EPOCHS + 1):
#         t0 = time.time()

#         model.train()
#         train_loss = 0.0
#         for images, labels in train_loader:
#             images = images.to(device)
#             labels = labels.to(device)
#             logits = model(images)
#             loss   = F.binary_cross_entropy_with_logits(logits, labels)
#             optimizer.zero_grad()
#             loss.backward()
#             optimizer.step()
#             train_loss += loss.item()
#         train_loss /= len(train_loader)

#         model.eval()
#         val_loss = 0.0
#         all_preds  = []
#         all_labels = []
#         with torch.no_grad():
#             for images, labels in val_loader:
#                 images = images.to(device)
#                 labels = labels.to(device)
#                 logits = model(images)
#                 loss   = F.binary_cross_entropy_with_logits(logits, labels)
#                 val_loss   += loss.item()
#                 preds       = (logits.sigmoid() > cfg.TAG_SCORE_THRESH).float()
#                 all_preds.append(preds.cpu())
#                 all_labels.append(labels.cpu())
#         val_loss /= len(val_loader)

#         # Compute F1 across all tags
#         all_preds  = torch.cat(all_preds,  dim=0)
#         all_labels = torch.cat(all_labels, dim=0)
#         tp   = (all_preds * all_labels).sum().item()
#         fp   = (all_preds * (1 - all_labels)).sum().item()
#         fn   = ((1 - all_preds) * all_labels).sum().item()
#         prec = tp / max(tp + fp, 1)
#         rec  = tp / max(tp + fn, 1)
#         f1   = 2 * prec * rec / max(prec + rec, 1e-6)

#         scheduler.step()
#         ep_time = time.time() - t0

#         if val_loss < best_val:
#             best_val = val_loss
#             torch.save({
#                 "epoch"           : epoch,
#                 "model_state_dict": model.state_dict(),
#                 "tag_names"       : tag_names,
#                 "val_loss"        : best_val
#             }, best_path)

#         if epoch % cfg.TAG_CHECKPOINT_EVERY == 0:
#             ckpt = os.path.join(
#                 cfg.TAG_SAVE_DIR, f"checkpoint_ep{epoch}.pt"
#             )
#             torch.save({"epoch": epoch,
#                         "model_state_dict": model.state_dict(),
#                         "tag_names": tag_names}, ckpt)

#         with open(log_path, "a", newline="") as f:
#             csv.writer(f).writerow(
#                 [epoch, f"{train_loss:.6f}", f"{val_loss:.6f}",
#                  f"{f1:.4f}", f"{ep_time:.1f}"]
#             )

#         if epoch % 10 == 0 or epoch == 1:
#             log_fn(f"  Epoch {epoch:>4}/{cfg.TAG_EPOCHS} | "
#                    f"train {train_loss:.4f} | "
#                    f"val {val_loss:.4f} | "
#                    f"F1 {f1*100:.1f}% | "
#                    f"{ep_time:.1f}s")

#     elapsed = str(datetime.timedelta(seconds=int(time.time() - start)))
#     log_fn(f"\n  Tag model training complete")
#     log_fn(f"  Time       : {elapsed}")
#     log_fn(f"  Best model → {best_path}")
#     log_fn(f"  Tag names  → {tag_meta_path}")

#     return best_path


# ============================================================
# tag_model/train_tag.py
# Multi-label tag classifier — reads from CVAT XML
# ============================================================

import os
import sys
import json
import time
import datetime
import csv

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torchvision import models
from torchvision.models import EfficientNet_B0_Weights
import torchvision.transforms as T
from PIL import Image

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from multi_model_annotator import config as cfg


# ============================================================
# --- Model ---
# ============================================================

class TagClassifier(nn.Module):

    def __init__(self, num_tags, pretrained=True):
        super().__init__()
        weights       = EfficientNet_B0_Weights.DEFAULT if pretrained \
                        else None
        base          = models.efficientnet_b0(weights=weights)
        in_feat       = base.classifier[1].in_features
        base.classifier = nn.Identity()
        self.backbone = base
        self.head     = nn.Sequential(
            nn.Dropout(0.3),
            nn.Linear(in_feat, 256), nn.ReLU(inplace=True),
            nn.Dropout(0.2),
            nn.Linear(256, num_tags),
        )

    def forward(self, x):
        return self.head(self.backbone(x))


# ============================================================
# --- Dataset — reads from parsed XML image list ---
# ============================================================

class TagDataset(Dataset):

    def __init__(self, images, img_dir, input_size,
                 tag_names, augment=True):
        self.img_dir  = img_dir
        self.tag_names= tag_names
        self.tag_to_idx = {t: i for i, t in enumerate(tag_names)}
        self.augment  = augment

        self.train_tf = T.Compose([
            T.Resize((input_size, input_size)),
            T.RandomHorizontalFlip(),
            T.ColorJitter(0.2, 0.2, 0.2),
            T.ToTensor(),
            T.Normalize(cfg.PIXEL_MEAN, cfg.PIXEL_STD),
        ])
        self.val_tf   = T.Compose([
            T.Resize((input_size, input_size)),
            T.ToTensor(),
            T.Normalize(cfg.PIXEL_MEAN, cfg.PIXEL_STD),
        ])

        # Build samples — images that have at least one tag
        self.samples = []
        for img_info in images:
            if not img_info["tags"]:
                continue
            label = torch.zeros(len(tag_names), dtype=torch.float32)
            for tag in img_info["tags"]:
                idx = self.tag_to_idx.get(tag["label"])
                if idx is not None:
                    label[idx] = 1.0
            if label.sum() > 0:
                self.samples.append((img_info["name"], label))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        fname, label = self.samples[idx]
        image = Image.open(
            os.path.join(self.img_dir, fname)
        ).convert("RGB")
        tf = self.train_tf if self.augment else self.val_tf
        return tf(image), label


def get_tag_names_from_xml(images):
    """Extract all unique tag label names from parsed XML image list."""
    tags = set()
    for img in images:
        for tag in img["tags"]:
            tags.add(tag["label"])
    return sorted(tags)


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
    log_fn("  TAG MODEL — EfficientNet-B0 Multi-label")
    log_fn("─" * 50)

    os.makedirs(cfg.TAG_SAVE_DIR, exist_ok=True)
    device = torch.device(cfg.DEVICE if torch.cuda.is_available()
                          else "cpu")

    tag_names = get_tag_names_from_xml(images)
    log_fn(f"  Tags found  : {tag_names}")

    if not tag_names:
        log_fn("  No tags found — skipping tag model.")
        return None

    full_dataset = TagDataset(
        images     = images,
        img_dir    = cfg.IMG_DIR,
        input_size = cfg.INPUT_SIZE,
        tag_names  = tag_names,
        augment    = True,
    )

    n_val   = max(1, int(len(full_dataset) * cfg.VAL_RATIO))
    n_train = len(full_dataset) - n_val
    train_ds, val_ds = torch.utils.data.random_split(
        full_dataset, [n_train, n_val],
        generator=torch.Generator().manual_seed(cfg.RANDOM_SEED)
    )
    val_ds.dataset.augment = False

    train_loader = DataLoader(train_ds, batch_size=cfg.TAG_BATCH_SIZE,
                              shuffle=True,  num_workers=cfg.NUM_WORKERS,
                              pin_memory=True)
    val_loader   = DataLoader(val_ds,   batch_size=cfg.TAG_BATCH_SIZE,
                              shuffle=False, num_workers=cfg.NUM_WORKERS,
                              pin_memory=True)

    log_fn(f"  Train samples : {n_train} | Val samples: {n_val}")
    log_fn(f"  Num tags      : {len(tag_names)}")

    model = TagClassifier(
        num_tags=len(tag_names), pretrained=cfg.TAG_PRETRAINED
    ).to(device)
    log_fn(f"  Parameters    : "
           f"{sum(p.numel() for p in model.parameters())/1e6:.1f}M")

    optimizer = torch.optim.AdamW(
        model.parameters(), lr=cfg.TAG_LR,
        weight_decay=cfg.TAG_WEIGHT_DECAY
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=cfg.TAG_EPOCHS
    )

    log_path  = os.path.join(cfg.TAG_SAVE_DIR, "train_log.csv")
    best_val  = float("inf")
    best_path = os.path.join(cfg.TAG_SAVE_DIR, "best.pt")
    tag_meta  = os.path.join(cfg.TAG_SAVE_DIR, "tag_names.json")
    start     = time.time()

    with open(tag_meta, "w") as f:
        json.dump(tag_names, f)

    with open(log_path, "w", newline="") as f:
        csv.writer(f).writerow(
            ["epoch", "train_loss", "val_loss", "val_f1", "epoch_time"]
        )

    for epoch in range(1, cfg.TAG_EPOCHS + 1):
        t0 = time.time()

        model.train()
        train_loss = 0.0
        for imgs, labels in train_loader:
            imgs   = imgs.to(device)
            labels = labels.to(device)
            loss   = F.binary_cross_entropy_with_logits(
                model(imgs), labels
            )
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            train_loss += loss.item()
        train_loss /= len(train_loader)

        model.eval()
        val_loss   = 0.0
        all_preds  = []
        all_labels = []
        with torch.no_grad():
            for imgs, labels in val_loader:
                imgs   = imgs.to(device)
                labels = labels.to(device)
                logits = model(imgs)
                val_loss += F.binary_cross_entropy_with_logits(
                    logits, labels
                ).item()
                preds = (logits.sigmoid() > cfg.TAG_SCORE_THRESH).float()
                all_preds.append(preds.cpu())
                all_labels.append(labels.cpu())
        val_loss /= len(val_loader)

        all_preds  = torch.cat(all_preds,  dim=0)
        all_labels = torch.cat(all_labels, dim=0)
        tp   = (all_preds * all_labels).sum().item()
        fp   = (all_preds * (1 - all_labels)).sum().item()
        fn   = ((1 - all_preds) * all_labels).sum().item()
        prec = tp / max(tp + fp, 1)
        rec  = tp / max(tp + fn, 1)
        f1   = 2 * prec * rec / max(prec + rec, 1e-6)

        scheduler.step()
        ep_time = time.time() - t0

        if val_loss < best_val:
            best_val = val_loss
            torch.save({
                "epoch"           : epoch,
                "model_state_dict": model.state_dict(),
                "tag_names"       : tag_names,
                "val_loss"        : best_val,
            }, best_path)

        if epoch % cfg.TAG_CHECKPOINT_EVERY == 0:
            torch.save({"epoch": epoch,
                        "model_state_dict": model.state_dict(),
                        "tag_names": tag_names},
                       os.path.join(cfg.TAG_SAVE_DIR,
                                    f"checkpoint_ep{epoch}.pt"))

        with open(log_path, "a", newline="") as f:
            csv.writer(f).writerow(
                [epoch, f"{train_loss:.6f}", f"{val_loss:.6f}",
                 f"{f1:.4f}", f"{ep_time:.1f}"]
            )

        if epoch % 10 == 0 or epoch == 1:
            log_fn(f"  Epoch {epoch:>4}/{cfg.TAG_EPOCHS} | "
                   f"train {train_loss:.4f} | val {val_loss:.4f} | "
                   f"F1 {f1*100:.1f}% | {ep_time:.1f}s")

    elapsed = str(datetime.timedelta(seconds=int(time.time() - start)))
    log_fn(f"\n  Tag training complete | Time: {elapsed}")
    log_fn(f"  Best model → {best_path}")
    log_fn(f"  Tag names  → {tag_meta}")
    return best_path