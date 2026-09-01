"""
Train the full-frame InterField baseline on the grayscale cache, with only the
backbone as a knob. Whole frame resized to 224 -> backbone -> MLP -> (2,21,3)
camera-frame field (mm). This mirrors the official baseline, just allows swapping
ResNet-50 (official) for a larger ResNet-101.
"""
import argparse
import json
import time

import cv2
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torchvision import models
from torchvision.models import ResNet50_Weights, ResNet101_Weights, ResNet152_Weights, ConvNeXt_Tiny_Weights

IMAGENET_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
IMAGENET_STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)
NUM_JOINTS = 21
NUM_HANDS = 2

ALL_SUBJECTS = ["ASC023", "SPI102", "LWA828", "YZH016", "XXI103",
                "PCW023", "MHA016", "MMO925", "XYZ109", "LYA722"]


def subject_of(sample_id):
    return sample_id.split("/")[0]


def make_backbone(name, pretrained=True):
    if name == "resnet50":
        weights = ResNet50_Weights.IMAGENET1K_V2 if pretrained else None
        return models.resnet50(weights=weights), 2048
    if name == "resnet101":
        weights = ResNet101_Weights.IMAGENET1K_V2 if pretrained else None
        return models.resnet101(weights=weights), 2048
    if name == "resnet152":
        weights = ResNet152_Weights.IMAGENET1K_V1 if pretrained else None
        return models.resnet152(weights=weights), 2048
    if name == "convnext_tiny":
        weights = ConvNeXt_Tiny_Weights.IMAGENET1K_V1 if pretrained else None
        return models.convnext_tiny(weights=weights), 768
    raise ValueError(f"unknown backbone {name}")


class FullFrameInterFieldModel(nn.Module):
    def __init__(self, backbone="resnet50", pretrained=True, hidden=512):
        super().__init__()
        self.backbone, feat_dim = make_backbone(backbone, pretrained)
        self.backbone_name = backbone
        if backbone.startswith("resnet"):
            self.backbone.fc = nn.Identity()
        else:  # convnext
            self.backbone.classifier = nn.Identity()
        self.head = nn.Sequential(
            nn.Linear(feat_dim, hidden), nn.ReLU(inplace=True), nn.Dropout(0.5),
            nn.Linear(hidden, hidden), nn.ReLU(inplace=True), nn.Dropout(0.5),
            nn.Linear(hidden, NUM_HANDS * NUM_JOINTS * 3),
        )

    def forward(self, images):
        feat = self.backbone(images)
        if not self.backbone_name.startswith("resnet"):
            feat = feat.flatten(1)  # convnext outputs (B, C, 1, 1) -> (B, C)
        return self.head(feat).view(-1, NUM_HANDS, NUM_JOINTS, 3)  # (B,2,21,3)


def preprocess(gray):
    img = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)  # gray -> 3ch (official baseline)
    img = cv2.resize(img, (224, 224), interpolation=cv2.INTER_AREA)
    img = img.astype(np.float32) / 255.0
    arr = np.transpose(img, (2, 0, 1))
    mean = IMAGENET_MEAN[:, None, None]
    std = IMAGENET_STD[:, None, None]
    return ((arr - mean) / std).astype(np.float32)


class FullFrameCacheDataset(Dataset):
    def __init__(self, cache_path, subjects=None):
        self.samples = []
        with open(cache_path) as f:
            for line in f:
                line = line.strip()
                if line:
                    self.samples.append(json.loads(line))
        subject_set = set(subjects) if subjects is not None else None
        self.items = []
        for i, s in enumerate(self.samples):
            if subject_set is not None and subject_of(s["sample_id"]) not in subject_set:
                continue
            self.items.append(i)

    def __len__(self):
        return len(self.items)

    def __getitem__(self, idx):
        s = self.samples[self.items[idx]]
        gray = cv2.imread(s["image"], cv2.IMREAD_GRAYSCALE)
        x = preprocess(gray)
        target = np.asarray(s["field_cam_mm"], np.float32)  # (2,21,3) mm
        mask = np.asarray(s["mask"], np.float32)            # (2,)
        return torch.from_numpy(x), torch.from_numpy(target), torch.from_numpy(mask)


def collate_fn(batch):
    x = torch.stack([b[0] for b in batch])
    t = torch.stack([b[1] for b in batch])
    m = torch.stack([b[2] for b in batch])
    return x, t, m


@torch.no_grad()
def val_ade(model, loader, device):
    model.eval()
    tot, n = 0.0, 0
    for x, t, m in loader:
        x, t, m = x.to(device), t.to(device), m.to(device)
        pred = model(x)
        err = torch.linalg.norm(pred - t, dim=-1).mean(dim=-1)  # (B,2)
        tot += float((err * m).sum())
        n += int(m.sum())
    return tot / max(n, 1)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cache", type=str, required=True)
    ap.add_argument("--val-subjects", nargs="*", default=["XYZ109", "LYA722"],
                    help="held-out subjects; pass empty to train on all")
    ap.add_argument("--backbone", type=str, default="resnet50",
                    choices=["resnet50", "resnet101", "resnet152", "convnext_tiny"])
    ap.add_argument("--epochs", type=int, default=20)
    ap.add_argument("--batch-size", type=int, default=256)
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--weight-decay", type=float, default=5e-4)
    ap.add_argument("--left-weight", type=float, default=1.0,
                    help="loss weight on the LEFT hand (right hand = 1.0)")
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--out", type=str, default="checkpoints/full_r50.pt")
    ap.add_argument("--resume", type=str, default=None,
                    help="checkpoint to resume from (loads model weights, restarts lr schedule)")
    args = ap.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    train_all = not args.val_subjects
    if train_all:
        train_subjects = None
        print("training on ALL subjects (no val)")
    else:
        train_subjects = [s for s in ALL_SUBJECTS if s not in args.val_subjects]
        print(f"train {train_subjects}\nval   {args.val_subjects}")

    tr = FullFrameCacheDataset(args.cache, subjects=train_subjects)
    tl = DataLoader(tr, batch_size=args.batch_size, shuffle=True, num_workers=args.workers,
                    pin_memory=True, drop_last=True, collate_fn=collate_fn,
                    persistent_workers=args.workers > 0)
    vl = None
    if not train_all:
        va = FullFrameCacheDataset(args.cache, subjects=args.val_subjects)
        vl = DataLoader(va, batch_size=args.batch_size, shuffle=False, num_workers=args.workers,
                        pin_memory=True, collate_fn=collate_fn,
                        persistent_workers=args.workers > 0)
        print(f"train frames {len(tr)} | val frames {len(va)}")
    else:
        print(f"train frames {len(tr)}")

    model = FullFrameInterFieldModel(backbone=args.backbone, pretrained=True).to(device)
    if torch.cuda.device_count() > 1:
        model = nn.DataParallel(model)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    start_epoch = 0
    best = float("inf")
    if args.resume:
        ckpt = torch.load(args.resume, map_location=device)
        base = model.module if hasattr(model, "module") else model
        base.load_state_dict(ckpt["model_state"])
        start_epoch = ckpt.get("epoch", 0)
        print(f"resumed from epoch {start_epoch} (lr re-initialized to {args.lr})")
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=args.epochs - start_epoch)

    import os
    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    for epoch in range(start_epoch + 1, args.epochs + 1):
        model.train()
        t0 = time.time()
        run, seen = 0.0, 0
        for it, (x, t, m) in enumerate(tl):
            x, t, m = x.to(device), t.to(device), m.to(device)
            opt.zero_grad(set_to_none=True)
            pred = model(x)
            err = torch.linalg.norm(pred - t, dim=-1).mean(dim=-1)  # (B,2)
            hand_w = torch.tensor([args.left_weight, 1.0], device=device)
            loss = (err * m * hand_w).sum() / (m * hand_w).sum().clamp_min(1.0)
            loss.backward()
            opt.step()
            run += float(loss.item()) * x.size(0)
            seen += x.size(0)
            if it % 50 == 0:
                print(f"  ep{epoch} it{it}/{len(tl)} loss {loss.item():.2f} mm", flush=True)
        sched.step()

        ade = None if train_all else val_ade(model, vl, device)
        if ade is None:
            print(f"[epoch {epoch}] train {run/max(seen,1):.2f} mm | {time.time()-t0:.0f}s", flush=True)
        else:
            print(f"[epoch {epoch}] train {run/max(seen,1):.2f} mm | val {ade:.2f} mm | {time.time()-t0:.0f}s", flush=True)

        state = (model.module if hasattr(model, "module") else model).state_dict()
        ckpt = {"model_state": state, "epoch": epoch, "val_ade": ade, "backbone": args.backbone,
                "optimizer_state": opt.state_dict(), "scheduler_state": sched.state_dict()}
        torch.save(ckpt, args.out + ".last.pt")
        if train_all:
            torch.save(ckpt, args.out)
        elif ade < best:
            best = ade
            torch.save(ckpt, args.out)
            print(f"  -> new best val {best:.2f} mm, saved {args.out}")

    print(f"done. best val {best:.2f} mm" if not train_all else f"done (all subjects) -> {args.out}")


if __name__ == "__main__":
    main()
