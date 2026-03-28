#!/usr/bin/env python3
"""
Train Faster R-CNN baseline vs grid-fused backbone on AWIR, selecting best checkpoint by validation loss.

Example:
  python train_awir_frcnn_grid.py \
    --data_root /blue/azare/zhou.m/awir/data_awir/AWIR \
    --grid_root /blue/azare/zhou.m/object_detection_CTL/awir_sliding_grids \
    --variant dtl \
    --epochs 20 \
    --batch 2 \
    --num_workers 4 \
    --run_dir /blue/azare/zhou.m/awir/fasterrcnn_runs \
    --run_tag baseline_vs_grid_dtl

Notes:
- Grid files are stored in: <grid_root>/<split_name>/*.npz  (split_name: train/val)
- Each npz contains keys like: dtl_grid, dtl_hard_grid, matl_grid, etc.
- Grid filenames may be: Class__STEM.npz or STEM.npz
"""

import os
import glob
import random
import argparse
from typing import Dict, List, Tuple, Optional

import numpy as np
from PIL import Image

import torch
from torch.utils.data import Dataset, DataLoader
import torchvision
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms.functional as TF
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor

from collections import Counter

# -----------------------------
# I/O helpers: save/load
# -----------------------------
def save_checkpoint(path, model, optimizer=None, scheduler=None, epoch=None, extra=None):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    ckpt = {
        "model_state": model.state_dict(),
        "epoch": epoch,
        "extra": extra or {}
    }
    if optimizer is not None:
        ckpt["optim_state"] = optimizer.state_dict()
    if scheduler is not None:
        ckpt["sched_state"] = scheduler.state_dict()
    torch.save(ckpt, path)


# -----------------------------
# Model builders
# -----------------------------
def build_frcnn(num_classes: int):
    m = torchvision.models.detection.fasterrcnn_resnet50_fpn(weights="DEFAULT")
    in_features = m.roi_heads.box_predictor.cls_score.in_features
    m.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)
    return m



# ============================================================
# GridFPNFusion 1: Additive (x + proj(g))
#   - Projects grid to 256 then adds to FPN feature map.
#   - Keeps output channels = fpn_channels (torchvision compatible).
# ============================================================
class GridFPNFusionAdditive(nn.Module):
    def __init__(self, fpn_channels=256, grid_dim=64, levels=(0,1,2,3), debug=False):
        super().__init__()
        self.fusion_type = "additive"
        self.levels = set(str(l) for l in levels)
        self.debug = debug
        self.grid_proj = nn.ModuleDict({
            l: nn.Conv2d(grid_dim, fpn_channels, 1) for l in self.levels
        })

    def forward(self, feats, grid: torch.Tensor):
        out = {}
        hits = 0

        for k, x in feats.items():
            ks = str(k)
            if ks in self.levels:
                hits += 1
                g = F.interpolate(grid, size=x.shape[-2:], mode="bilinear", align_corners=False)
                g = self.grid_proj[ks](g)
                out[k] = x + g
            else:
                out[k] = x

        if self.debug:
            assert hits > 0, f"No fusion applied. feats keys={list(feats.keys())}"
        return out


# ============================================================
# GridFPNFusion 2: Residual Concat (x + proj(cat(x,g)))
#   - Concats x and grid along channels then 1x1 back to 256.
#   - Residual makes this a stable drop-in.
# ============================================================
class GridFPNFusionResidualConcat(nn.Module):
    def __init__(self, fpn_channels=256, grid_dim=64, levels=(0,1,2,3), debug=False):
        super().__init__()
        self.fusion_type = "residual_concat"
        self.levels = set(str(l) for l in levels)
        self.debug = debug
        self.post = nn.ModuleDict({
            l: nn.Conv2d(fpn_channels + grid_dim, fpn_channels, 1) for l in self.levels
        })

    def forward(self, feats, grid: torch.Tensor):
        out = {}
        hits = 0

        for k, x in feats.items():
            ks = str(k)
            if ks in self.levels:
                hits += 1
                g = F.interpolate(grid, size=x.shape[-2:], mode="bilinear", align_corners=False)
                y = torch.cat([x, g], dim=1)       # [B, 256 + D, H, W]
                out[k] = x + self.post[ks](y)      # [B, 256, H, W]
            else:
                out[k] = x

        if self.debug:
            assert hits > 0, f"No fusion applied. feats keys={list(feats.keys())}"
        return out


# ============================================================
# GridFPNFusion 3: FiLM Gating ((1+γ)*x + β) from grid
#   - Grid produces per-pixel gamma/beta maps for feature modulation.
#   - Uses tanh for γ to limit scale; still fully differentiable.
# ============================================================
class GridFPNFusionFiLM(nn.Module):
    def __init__(self, fpn_channels=256, grid_dim=64, levels=(0,1,2,3), debug=False):
        super().__init__()
        self.fusion_type = "film"
        self.levels = set(str(l) for l in levels)
        self.debug = debug
        self.film = nn.ModuleDict({
            l: nn.Conv2d(grid_dim, 2 * fpn_channels, 1) for l in self.levels
        })

    def forward(self, feats, grid: torch.Tensor):
        out = {}
        hits = 0

        for k, x in feats.items():
            ks = str(k)
            if ks in self.levels:
                hits += 1
                g = F.interpolate(grid, size=x.shape[-2:], mode="bilinear", align_corners=False)
                gb = self.film[ks](g)
                gamma, beta = torch.chunk(gb, chunks=2, dim=1)  # each [B,256,H,W]

                # scale modulation in a bounded way
                gamma = torch.tanh(gamma)  # [-1,1]
                out[k] = (1.0 + gamma) * x + beta
            else:
                out[k] = x

        if self.debug:
            assert hits > 0, f"No fusion applied. feats keys={list(feats.keys())}"
        return out


# ============================================================
# GridFPNFusion 4: Spatial Mask (x * (1 + m)) from grid
#   - Grid produces a single-channel per-pixel mask.
#   - Mask is sigmoid -> [0,1], then used as multiplicative emphasis.
# ============================================================
class GridFPNFusionSpatialMask(nn.Module):
    def __init__(self, fpn_channels=256, grid_dim=64, levels=(0,1,2,3), debug=False):
        super().__init__()
        self.fusion_type = "spatial_mask"
        self.levels = set(str(l) for l in levels)
        self.debug = debug
        self.mask = nn.ModuleDict({
            l: nn.Conv2d(grid_dim, 1, 1) for l in self.levels
        })
    def forward(self, feats, grid: torch.Tensor):
        out = {}
        hits = 0

        for k, x in feats.items():
            ks = str(k)
            if ks in self.levels:
                hits += 1
                g = F.interpolate(grid, size=x.shape[-2:], mode="bilinear", align_corners=False)
                m = torch.sigmoid(self.mask[ks](g))   # [B,1,H,W] in [0,1]
                out[k] = x * (1.0 + m)               # broadcast over channels
            else:
                out[k] = x

        if self.debug:
            assert hits > 0, f"No fusion applied. feats keys={list(feats.keys())}"
        return out


# ============================================================
# Backbone wrapper (unchanged API)
#   - Set a grid per batch via set_grid(grid) before forward
#   - Clears cached grid after use for safety
# ============================================================
class BackboneWithCachedGrid(nn.Module):
    def __init__(self, backbone: nn.Module, fusion: nn.Module):
        super().__init__()
        self.backbone = backbone
        self.fusion = fusion
        self.out_channels = backbone.out_channels
        self._cached_grid = None
        self._printed_fusion = False

    def set_grid(self, grid: torch.Tensor):
        self._cached_grid = grid

    def forward(self, x: torch.Tensor):

        if not self._printed_fusion:
            print(f"[FPN Fusion] Using fusion type: {self.fusion.fusion_type}")
            self._printed_fusion = True


        feats = self.backbone(x)

        if self._cached_grid is None:
            raise RuntimeError("Grid not set. Call model.backbone.set_grid(grid) before forward.")

        out = self.fusion(feats, self._cached_grid)
        self._cached_grid = None
        return out
    

    
    
def wrap_frcnn_backbone_with_grid_fusion(frcnn_model, grid_dim, fusion_type="additive", levels=(0,1,2,3), debug=False):
    """
    frcnn_model: torchvision FasterRCNN returned by build_frcnn(...)
    grid_dim: D in your grid [B,D,Gh,Gw]
    fusion_type: "additive" | "residual_concat" | "film" | "spatial_mask"
    """
    fpn_channels = frcnn_model.backbone.out_channels  # typically 256

    if fusion_type == "additive":
        fusion = GridFPNFusionAdditive(fpn_channels=fpn_channels, grid_dim=grid_dim, levels=levels, debug=debug)
    elif fusion_type == "residual_concat":
        fusion = GridFPNFusionResidualConcat(fpn_channels=fpn_channels, grid_dim=grid_dim, levels=levels, debug=debug)
    elif fusion_type == "film":
        fusion = GridFPNFusionFiLM(fpn_channels=fpn_channels, grid_dim=grid_dim, levels=levels, debug=debug)
    elif fusion_type == "spatial_mask":
        fusion = GridFPNFusionSpatialMask(fpn_channels=fpn_channels, grid_dim=grid_dim, levels=levels, debug=debug)
    else:
        raise ValueError(f"Unknown fusion_type '{fusion_type}'")

    frcnn_model.backbone = BackboneWithCachedGrid(frcnn_model.backbone, fusion=fusion)
    return frcnn_model




# def build_frcnn_with_grid(num_classes: int, grid_dim: int):
#     m = build_frcnn(num_classes)
#     m.backbone = BackboneWithCachedGrid(m.backbone, fpn_channels=256, grid_dim=grid_dim)
#     return m


# -----------------------------
# Dataset utilities
# -----------------------------
def yolo_to_xyxy(line: str, W: int, H: int):
    parts = line.strip().split()
    if len(parts) < 5:
        return None
    cx, cy, bw, bh = map(float, parts[1:5])

    x1 = (cx - bw / 2.0) * W
    y1 = (cy - bh / 2.0) * H
    x2 = (cx + bw / 2.0) * W
    y2 = (cy + bh / 2.0) * H

    x1 = max(0.0, min(x1, W - 1))
    y1 = max(0.0, min(y1, H - 1))
    x2 = max(0.0, min(x2, W - 1))
    y2 = max(0.0, min(y2, H - 1))
    if x2 <= x1 or y2 <= y1:
        return None
    return [x1, y1, x2, y2]


def load_grid_npz(feature_npz: str, grid_key: str) -> torch.Tensor:
    z = np.load(feature_npz, allow_pickle=True)
    if grid_key not in z:
        raise KeyError(f"{feature_npz}: missing key '{grid_key}'. Available keys: {list(z.keys())}")
    feat = z[grid_key].astype(np.float32)  # (Gh, Gw, D)
    return torch.from_numpy(feat).permute(2, 0, 1).contiguous()  # (D, Gh, Gw)


class AWIRBaseDataset(Dataset):
    def __init__(self, split_items: List[Tuple[str, str, str]], class_to_id: Dict[str, int], image_size=(640, 640)):
        self.items = split_items
        self.class_to_id = class_to_id
        self.image_size = image_size

    def __len__(self):
        return len(self.items)

    def __getitem__(self, idx: int):
        img_path, label_path, cname = self.items[idx]

        img = Image.open(img_path).convert("RGB")
        img = img.resize((self.image_size[1], self.image_size[0]))  # (W,H)
        W, H = img.size

        boxes, labels = [], []
        with open(label_path, "r") as f:
            for line in f:
                if not line.strip():
                    continue
                box = yolo_to_xyxy(line, W, H)
                if box is None:
                    continue
                boxes.append(box)
                labels.append(self.class_to_id[cname])

        boxes = torch.tensor(boxes, dtype=torch.float32)
        labels = torch.tensor(labels, dtype=torch.int64)

        target = {
            "boxes": boxes,
            "labels": labels,
            "image_id": torch.tensor([idx]),
            "iscrowd": torch.zeros((len(labels),), dtype=torch.int64),
            "area": ((boxes[:, 2] - boxes[:, 0]) * (boxes[:, 3] - boxes[:, 1]))
            if len(labels) else torch.tensor([], dtype=torch.float32),
        }

        img_t = TF.to_tensor(img)
        return img_t, target, img_path


class AWIRGridDataset(AWIRBaseDataset):
    def __init__(
        self,
        split_items: List[Tuple[str, str, str]],
        class_to_id: Dict[str, int],
        grid_root: str,
        split_name: str,
        variant: str = "dtl",
        image_size=(640, 640),
        emb_dim: Optional[int] = None,
        grid_key: Optional[str] = None,
    ):
        super().__init__(split_items, class_to_id=class_to_id, image_size=image_size)
        self.grid_dir = os.path.join(grid_root, split_name)
        if not os.path.isdir(self.grid_dir):
            raise FileNotFoundError(f"Grid dir not found: {self.grid_dir}")

        self.variant = variant
        self.grid_key = grid_key if grid_key is not None else f"{variant}_grid"
        self.emb_dim = emb_dim

        self.grid_index = self._build_grid_index(self.grid_dir)

    def _build_grid_index(self, grid_dir: str) -> Dict[str, str]:
        idx = {}
        files = glob.glob(os.path.join(grid_dir, "*.npz"))
        if len(files) == 0:
            raise RuntimeError(f"No .npz files found in {grid_dir}")

        for p in files:
            fname = os.path.splitext(os.path.basename(p))[0]
            stem = fname.split("__", 1)[1] if "__" in fname else fname
            if stem not in idx:
                idx[stem] = p
        return idx

    def __getitem__(self, idx: int):
        img_t, target, img_path = super().__getitem__(idx)
        stem = os.path.splitext(os.path.basename(img_path))[0]

        feature_npz = self.grid_index.get(stem, None)
        if feature_npz is None:
            matches = [p for s, p in self.grid_index.items() if stem in s]
            if len(matches) == 1:
                feature_npz = matches[0]
            elif len(matches) > 1:
                raise FileNotFoundError(f"Multiple grid matches for stem '{stem}': {matches[:5]} ...")
            else:
                raise FileNotFoundError(f"No grid file found for stem '{stem}' in {self.grid_dir}")

        grid = load_grid_npz(feature_npz, self.grid_key)  # [D,Gh,Gw]
        if self.emb_dim is not None and grid.shape[0] != self.emb_dim:
            raise RuntimeError(f"{feature_npz}: grid D={grid.shape[0]} != emb_dim={self.emb_dim}")

        return img_t, target, grid


def collate_base(batch):
    images, targets, paths = zip(*batch)
    return list(images), list(targets), list(paths)


def collate_grid(batch):
    images, targets, grids = zip(*batch)
    return list(images), list(targets), torch.stack(grids, dim=0)


# -----------------------------
# Metrics + loss
# -----------------------------
def box_iou_xyxy(boxes1: torch.Tensor, boxes2: torch.Tensor):
    if boxes1.numel() == 0 or boxes2.numel() == 0:
        return torch.zeros((boxes1.shape[0], boxes2.shape[0]))
    x11, y11, x12, y12 = boxes1[:, 0:1], boxes1[:, 1:2], boxes1[:, 2:3], boxes1[:, 3:4]
    x21, y21, x22, y22 = boxes2[:, 0], boxes2[:, 1], boxes2[:, 2], boxes2[:, 3]
    xi1 = torch.maximum(x11, x21); yi1 = torch.maximum(y11, y21)
    xi2 = torch.minimum(x12, x22); yi2 = torch.minimum(y12, y22)
    inter = torch.clamp(xi2 - xi1, min=0) * torch.clamp(yi2 - yi1, min=0)
    area1 = (x12 - x11) * (y12 - y11)
    area2 = (x22 - x21) * (y22 - y21)
    union = area1 + area2 - inter
    return inter / torch.clamp(union, min=1e-8)


@torch.no_grad()
def eval_overall_pr(model, loader, device, iou_thresh=0.5, score_thresh=0.5, uses_grid=False):
    model.eval()
    TP = FP = FN = 0

    for batch in loader:
        if uses_grid:
            images, targets, grids = batch
            grids = grids.to(device)
            model.backbone.set_grid(grids)
        else:
            images, targets, paths = batch

        images = [img.to(device) for img in images]
        outputs = model(images)

        for out, tgt in zip(outputs, targets):
            pb = out["boxes"].detach().cpu()
            ps = out["scores"].detach().cpu()
            pl = out["labels"].detach().cpu()
            keep = ps >= score_thresh
            pb, pl = pb[keep], pl[keep]

            gb = tgt["boxes"].detach().cpu()
            gl = tgt["labels"].detach().cpu()
            matched = torch.zeros(len(gl), dtype=torch.bool)

            for i in range(len(pl)):
                idxs = torch.where((gl == pl[i]) & (~matched))[0]
                if len(idxs) == 0:
                    FP += 1
                    continue
                ious = box_iou_xyxy(pb[i:i + 1], gb[idxs]).squeeze(0)
                j = torch.argmax(ious)
                if float(ious[j]) >= iou_thresh:
                    TP += 1
                    matched[idxs[j]] = True
                else:
                    FP += 1

            FN += int((~matched).sum().item())

    precision = TP / max(1, TP + FP)
    recall = TP / max(1, TP + FN)
    f1 = 2 * precision * recall / max(1e-8, precision + recall)
    return {"precision": precision, "recall": recall, "f1": f1, "TP": TP, "FP": FP, "FN": FN}


@torch.no_grad()
def eval_per_class_pr(model, loader, device, class_ids, iou_thresh=0.5, score_thresh=0.5, uses_grid=False):
    model.eval()
    TP = {c: 0 for c in class_ids}
    FP = {c: 0 for c in class_ids}
    FN = {c: 0 for c in class_ids}

    for batch in loader:
        if uses_grid:
            images, targets, grids = batch
            grids = grids.to(device)
            model.backbone.set_grid(grids)
        else:
            images, targets, paths = batch

        images = [img.to(device) for img in images]
        outputs = model(images)

        for out, tgt in zip(outputs, targets):
            pb = out["boxes"].detach().cpu()
            ps = out["scores"].detach().cpu()
            pl = out["labels"].detach().cpu()
            keep = ps >= score_thresh
            pb, pl = pb[keep], pl[keep]

            gb = tgt["boxes"].detach().cpu()
            gl = tgt["labels"].detach().cpu()

            for c in class_ids:
                pidx = torch.where(pl == c)[0]
                gidx = torch.where(gl == c)[0]
                preds = pb[pidx]
                gts = gb[gidx]
                matched = torch.zeros(len(gts), dtype=torch.bool)

                for i in range(len(preds)):
                    if len(gts) == 0:
                        FP[c] += 1
                        continue
                    avail = torch.where(~matched)[0]
                    if len(avail) == 0:
                        FP[c] += 1
                        continue
                    ious = box_iou_xyxy(preds[i:i + 1], gts[avail]).squeeze(0)
                    j = torch.argmax(ious)
                    if float(ious[j]) >= iou_thresh:
                        TP[c] += 1
                        matched[avail[j]] = True
                    else:
                        FP[c] += 1

                FN[c] += int((~matched).sum().item())

    metrics = {}
    for c in class_ids:
        tp, fp, fn = TP[c], FP[c], FN[c]
        prec = tp / max(1, tp + fp)
        rec = tp / max(1, tp + fn)
        f1 = 2 * prec * rec / max(1e-8, prec + rec)
        metrics[c] = {"precision": prec, "recall": rec, "f1": f1, "tp": tp, "fp": fp, "fn": fn}
    return metrics


# -----------------------------
# Training
# -----------------------------
from torch.optim import SGD
from torch.optim.lr_scheduler import StepLR


def make_optim(model, lr=0.005):
    params = [p for p in model.parameters() if p.requires_grad]
    opt = SGD(params, lr=lr, momentum=0.9, weight_decay=0.0005)
    sch = StepLR(opt, step_size=3, gamma=0.1)
    return opt, sch


def train_one_epoch(model, loader, optimizer, device, uses_grid=False):
    model.train()
    total = 0.0
    for batch in loader:
        if uses_grid:
            images, targets, grids = batch
            grids = grids.to(device)
            model.backbone.set_grid(grids)
        else:
            images, targets, paths = batch

        images = [img.to(device) for img in images]
        targets = [{k: v.to(device) for k, v in t.items()} for t in targets]

        loss_dict = model(images, targets)
        loss = sum(v for v in loss_dict.values())

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        total += float(loss.item())

    return total / max(1, len(loader))


@torch.no_grad()
def compute_val_loss(model, loader, device, uses_grid=False):
    """
    Torchvision detection returns losses only in train() mode when targets are provided.
    """
    model.train()
    total = 0.0
    n = 0

    for batch in loader:
        if uses_grid:
            images, targets, grids = batch
            grids = grids.to(device)
            model.backbone.set_grid(grids)
        else:
            images, targets, paths = batch

        images = [img.to(device) for img in images]
        targets = [{k: v.to(device) for k, v in t.items()} for t in targets]

        loss_dict = model(images, targets)
        loss = sum(v for v in loss_dict.values())

        total += float(loss.item())
        n += 1

    return total / max(1, n)


# -----------------------------
# Split builder
# -----------------------------
def build_train_val_test_items(
    root,
    class_names,
    seed=42,
    test_frac=0.2,
    new_val_frac_from_train=0.25,
):
    """
    Returns: train_items, val_items, test_items

    Behavior (matches your request):
    1) test_items = the ORIGINAL random 20% split of the FULL dataset (not stratified)
       - This corresponds to your previous "val" split and is treated as HELD-OUT TEST.
    2) val_items = a NEW validation set sampled from the remaining pool (80%)
       - Stratified by class between train_items and val_items
       - Size is new_val_frac_from_train of the remaining pool (default 0.25 so ~20% of total)
    3) train_items = remaining samples after val_items is removed
       - Stratified by class

    Deterministic + disjoint.
    """

    assert 0.0 < test_frac < 1.0, "test_frac must be in (0,1)"
    assert 0.0 < new_val_frac_from_train < 1.0, "new_val_frac_from_train must be in (0,1)"

    import os, glob, random
    rng = random.Random(seed)

    # -------------------------------------------------
    # 1) Collect ALL items (not per-class split yet)
    # -------------------------------------------------
    all_items = []
    for cname in class_names:
        cdir = os.path.join(root, cname)
        imgs = sorted(
            glob.glob(os.path.join(cdir, "*.jpg")) +
            glob.glob(os.path.join(cdir, "*.JPG"))
        )
        for img_path in imgs:
            stem = os.path.splitext(os.path.basename(img_path))[0]
            if stem.endswith("_R"):
                continue
            label_path = os.path.splitext(img_path)[0] + ".txt"
            if os.path.exists(label_path):
                all_items.append((img_path, label_path, cname))

    # -------------------------------------------------
    # 2) HELD-OUT TEST = random split of full dataset
    #    (exactly like your original code did)
    # -------------------------------------------------
    rng.shuffle(all_items)
    n_total = len(all_items)
    n_test = int(n_total * test_frac)

    test_items = all_items[:n_test]
    train_pool = all_items[n_test:]   # pool used to create stratified train/val

    # -------------------------------------------------
    # 3) NEW VAL from train_pool, STRATIFIED by class
    # -------------------------------------------------
    per_class_pool = {c: [] for c in class_names}
    for item in train_pool:
        per_class_pool[item[2]].append(item)

    train_items, val_items = [], []

    for cname, items in per_class_pool.items():
        if len(items) == 0:
            continue

        rng.shuffle(items)
        n = len(items)
        n_val_c = int(round(n * new_val_frac_from_train))

        val_c = items[:n_val_c]
        train_c = items[n_val_c:]

        val_items.extend(val_c)
        train_items.extend(train_c)

    # -------------------------------------------------
    # 4) Final shuffle (randomize order)
    # -------------------------------------------------
    rng.shuffle(train_items)
    rng.shuffle(val_items)
    rng.shuffle(test_items)

    return all_items, train_items, val_items, test_items


def infer_grid_dim(grid_root: str, variant: str) -> int:
    key = f"{variant}_grid"
    train_dir = os.path.join(grid_root, "train")
    candidates = glob.glob(os.path.join(train_dir, "*.npz"))
    if len(candidates) == 0:
        raise RuntimeError(f"No npz files found in {train_dir} to infer grid dim.")
    z = np.load(candidates[0], allow_pickle=True)
    if key not in z:
        raise KeyError(f"{candidates[0]} missing '{key}'. Available keys: {list(z.keys())}")
    return int(z[key].shape[-1])


# -----------------------------
# Main
# -----------------------------
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_root", type=str, required=True, help="AWIR root with class subdirs.")
    parser.add_argument("--grid_root", type=str, required=True, help="Root containing train/ and val/ npz grids.")

    parser.add_argument(
        "--variants",
        type=str,
        default="dtl,dtl_hard,matl",
        help="Comma-separated list of grid variants to train (expects <variant>_grid key inside each npz)."
    )

    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--batch", type=int, default=2)
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--lr", type=float, default=0.005)

    parser.add_argument("--val_frac", type=float, default=0.25)
    parser.add_argument("--seed", type=int, default=42)

    parser.add_argument("--image_size", type=int, nargs=2, default=[640, 640], help="H W")
    parser.add_argument("--run_dir", type=str, required=True)
    parser.add_argument("--run_tag", type=str, required=True)  # single run folder
    parser.add_argument("--test_frac", type=float, default=0.2)
    
    parser.add_argument("--fusion", type=str, default='additive')
    
    # "additive" | "residual_concat" | "film" | "spatial_mask"


    args = parser.parse_args()

    print("\n=== Run arguments ===")
    for k, v in vars(args).items():
        print(f"{k:>12}: {v}")
    print("=====================\n")
    
    variants = [v.strip() for v in args.variants.split(",") if v.strip()]
    if len(variants) == 0:
        raise ValueError("No variants provided. Use --variants dtl,dtl_hard,matl")

    class_names = ["Cow", "Horse", "Deer"]
    class_to_id = {c: i + 1 for i, c in enumerate(class_names)}
    id_to_class = {v: k for k, v in class_to_id.items()}
    class_ids = sorted(id_to_class.keys())

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}")

    # Same train/val split for everything
    all_items, train_items, val_items, test_items = build_train_val_test_items(
        root=args.data_root,
        class_names=class_names,
        seed=args.seed,
        test_frac=args.test_frac,
        new_val_frac_from_train=args.val_frac,
    )

    print("Total:", len(all_items), Counter([c for _, _, c in all_items]))
    print("Train:", len(train_items), Counter([c for _, _, c in train_items]))
    print("Val:  ", len(val_items),   Counter([c for _, _, c in val_items]))
    print("Test: ", len(test_items),  Counter([c for _, _, c in test_items]))

    img_h, img_w = int(args.image_size[0]), int(args.image_size[1])

    # Baseline loaders (reused)
    train_base_loader = DataLoader(
        AWIRBaseDataset(train_items, class_to_id, image_size=(img_h, img_w)),
        batch_size=args.batch, shuffle=True,
        num_workers=args.num_workers, collate_fn=collate_base, pin_memory=True
    )
    val_base_loader = DataLoader(
        AWIRBaseDataset(val_items, class_to_id, image_size=(img_h, img_w)),
        batch_size=args.batch, shuffle=False,
        num_workers=args.num_workers, collate_fn=collate_base, pin_memory=True
    )

    # Single run folder
    run_path = os.path.join(args.run_dir, args.run_tag)
    os.makedirs(run_path, exist_ok=True)
    print(f"Saving checkpoints to: {run_path}")

    num_classes = 1 + len(class_names)

    # -------------------------
    # Build + train BASELINE once
    # -------------------------
    baseline_model = build_frcnn(num_classes).to(device)
    opt_base, sch_base = make_optim(baseline_model, lr=args.lr)
    best_base_loss = float("inf")

    # -------------------------
    # Build GRID models + loaders (one per variant)
    # -------------------------
    grid_models = {}
    grid_opts = {}
    grid_schs = {}
    grid_best_loss = {}
    train_grid_loaders = {}
    val_grid_loaders = {}
    grid_dims = {}

    for v in variants:
        gd = infer_grid_dim(args.grid_root, v)
        grid_dims[v] = gd
        print(f"Inferred grid_dim[{v}]={gd} from key {v}_grid")

        train_grid_loaders[v] = DataLoader(
            AWIRGridDataset(train_items, class_to_id, grid_root=args.grid_root, split_name="train",
                            variant=v, image_size=(img_h, img_w), emb_dim=gd),
            batch_size=args.batch, shuffle=True,
            num_workers=args.num_workers, collate_fn=collate_grid, pin_memory=True
        )
        val_grid_loaders[v] = DataLoader(
            AWIRGridDataset(val_items, class_to_id, grid_root=args.grid_root, split_name="val",
                            variant=v, image_size=(img_h, img_w), emb_dim=gd),
            batch_size=args.batch, shuffle=False,
            num_workers=args.num_workers, collate_fn=collate_grid, pin_memory=True
        )

        # m = build_frcnn_with_grid(num_classes, grid_dim=gd).to(device)
        
        
        m = build_frcnn(num_classes).to(device)
        m = wrap_frcnn_backbone_with_grid_fusion(
            m,
            grid_dim=gd,
            fusion_type=args.fusion,          # "additive" | "residual_concat" | "film" | "spatial_mask"
            levels=(0,1,2,3),
            debug=False
        ).to(device)

        opt, sch = make_optim(m, lr=args.lr)

        grid_models[v] = m
        grid_opts[v] = opt
        grid_schs[v] = sch
        grid_best_loss[v] = float("inf")

    # -------------------------
    # Training loop: one epoch trains baseline + all grid variants
    # -------------------------
    for e in range(args.epochs):
        print("\n" + "=" * 90)
        print(f"Epoch {e+1}/{args.epochs}")
        print("=" * 90)

        # ---- Train baseline ----
        train_loss_base = train_one_epoch(baseline_model, train_base_loader, opt_base, device, uses_grid=False)
        sch_base.step()

        # ---- Train each grid model ----
        train_loss_grid = {}
        for v in variants:
            train_loss_grid[v] = train_one_epoch(
                grid_models[v], train_grid_loaders[v], grid_opts[v], device, uses_grid=True
            )
            grid_schs[v].step()

        # ---- Val loss (selection criterion) ----
        val_loss_base = compute_val_loss(baseline_model, val_base_loader, device, uses_grid=False)

        val_loss_grid = {}
        for v in variants:
            val_loss_grid[v] = compute_val_loss(grid_models[v], val_grid_loaders[v], device, uses_grid=True)

        print(f"Baseline  train loss: {train_loss_base:.4f} | val loss: {val_loss_base:.4f}")
        for v in variants:
            print(f"{v:<9} train loss: {train_loss_grid[v]:.4f} | val loss: {val_loss_grid[v]:.4f}")

        # ---- Metrics (optional) ----
        base_overall = eval_overall_pr(baseline_model, val_base_loader, device, uses_grid=False)
        print(f"Baseline  val: P={base_overall['precision']:.3f} R={base_overall['recall']:.3f} F1={base_overall['f1']:.3f}")

        grid_overall = {}
        for v in variants:
            grid_overall[v] = eval_overall_pr(grid_models[v], val_grid_loaders[v], device, uses_grid=True)
            o = grid_overall[v]
            print(f"{v:<9} val: P={o['precision']:.3f} R={o['recall']:.3f} F1={o['f1']:.3f}")

#         # -------------------------
#         # Save per-epoch checkpoints
#         # -------------------------
#         save_checkpoint(
#             os.path.join(run_path, f"baseline_epoch{e+1}.pt"),
#             baseline_model, opt_base, sch_base, epoch=e+1,
#             extra={"type": "baseline", "num_classes": num_classes, "class_names": class_names,
#                    "train_loss": train_loss_base, "val_loss": val_loss_base}
#         )

#         for v in variants:
#             save_checkpoint(
#                 os.path.join(run_path, f"{v}_grid_epoch{e+1}.pt"),
#                 grid_models[v], grid_opts[v], grid_schs[v], epoch=e+1,
#                 extra={"type": "grid", "variant": v, "num_classes": num_classes, "class_names": class_names,
#                        "grid_dim": grid_dims[v], "train_loss": train_loss_grid[v], "val_loss": val_loss_grid[v]}
#             )

        # -------------------------
        # Save best-by-val-loss checkpoints
        # -------------------------
        if val_loss_base < best_base_loss:
            best_base_loss = val_loss_base
            save_checkpoint(
                os.path.join(run_path, "baseline_best_by_val_loss.pt"),
                baseline_model, opt_base, sch_base, epoch=e+1,
                extra={"type": "baseline", "best_metric": "val_loss", "val_loss": val_loss_base,
                       "class_names": class_names, "num_classes": num_classes}
            )

        for v in variants:
            if val_loss_grid[v] < grid_best_loss[v]:
                grid_best_loss[v] = val_loss_grid[v]
                save_checkpoint(
                    os.path.join(run_path, f"{v}_grid_best_by_val_loss.pt"),
                    grid_models[v], grid_opts[v], grid_schs[v], epoch=e+1,
                    extra={"type": "grid", "variant": v, "best_metric": "val_loss",
                           "val_loss": val_loss_grid[v], "grid_dim": grid_dims[v],
                           "class_names": class_names, "num_classes": num_classes}
                )

        # -------------------------
        # Per-class summary table (optional)
        # -------------------------
        print("\nPer-class F1 (baseline vs grids):")
        base_pc = eval_per_class_pr(baseline_model, val_base_loader, device, class_ids, uses_grid=False)
        grid_pc = {v: eval_per_class_pr(grid_models[v], val_grid_loaders[v], device, class_ids, uses_grid=True)
                   for v in variants}

        header = f"{'Class':<10} | {'Base F1':>7} " + " ".join([f"{v[:6]+' F1':>8}" for v in variants])
        print(header)
        print("-" * len(header))
        for cid in class_ids:
            row = f"{id_to_class[cid]:<10} | {base_pc[cid]['f1']:>7.3f} "
            for v in variants:
                row += f"{grid_pc[v][cid]['f1']:>8.3f} "
            print(row)

        print("\nBest val losses so far:")
        print(f"  baseline: {best_base_loss:.6f}")
        for v in variants:
            print(f"  {v:<8}: {grid_best_loss[v]:.6f}")

    print("\nDone.")
    print(f"Final best baseline val loss: {best_base_loss:.6f}")
    for v in variants:
        print(f"Final best {v} grid val loss: {grid_best_loss[v]:.6f}")


if __name__ == "__main__":
    main()