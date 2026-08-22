"""
Train the joint-conditioned InterField model for the SHOW3D challenge.

Data flow per frame:
    image (RGB) -> gray -> resize 224 -> 3ch -> ImageNet normalize
    GT hand joints: world mm -> camera mm (subtract translation) -> meters
    GT field labels: world mm -> camera mm (rotate only) -> millimeters (target)

The model regresses the field in the CAMERA frame; at predict time we rotate the
output back to world with R_world_from_camera (rotation only, no translation).
"""
import argparse
import json
import os

import cv2
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

from model import JointInterFieldModel

# SHOW3D API (installed / on PYTHONPATH)
from show3d.interaction_field import Show3DInteractionFieldDataset

IMAGENET_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
IMAGENET_STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)


def joints_world_to_camera_mm(joints_world_mm, R_wc, t_wc):
    """Points: subtract translation, then rotate (row-vector convention)."""
    return (joints_world_mm - t_wc[None, :]) @ R_wc


def field_world_to_camera_mm(field_world_mm, R_wc):
    """Vectors: rotate only (translation cancels)."""
    return field_world_mm @ R_wc


def preprocess_image(image_rgb):
    """RGB uint8 (H,W,3) -> (3,224,224) float32, ImageNet-normalized gray."""
    gray = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2GRAY)
    gray = cv2.resize(gray, (224, 224))
    img = np.stack([gray, gray, gray], axis=0).astype(np.float32) / 255.0
    img = (img - IMAGENET_MEAN[:, None, None]) / IMAGENET_STD[:, None, None]
    return img


class TrainDataset(Dataset):
    def __init__(self, root, manifest, joint_noise_mm=0.0):
        self.ds = Show3DInteractionFieldDataset(
            root,
            manifest,
            load_labels=True,
            decode_images=True,
            multiview=False,
        )
        self.joint_noise_mm = joint_noise_mm

    def __len__(self):
        return len(self.ds)

    def __getitem__(self, idx):
        ex = self.ds[idx]
        view = ex.views["headset0"]
        calib = view.calibration

        # Calibration may be None on synthesized frames -> skip via a valid flag
        if calib is None or calib.t_world_from_camera is None:
            return None

        R_wc = np.asarray(calib.t_world_from_camera[:3, :3], dtype=np.float64)
        t_wc = np.asarray(calib.t_world_from_camera[:3, 3], dtype=np.float64)

        image = preprocess_image(view.image)  # (3,224,224)

        # Build joints (input) and field targets (supervision) for both hands
        joints_cam_m = np.zeros((2, 21, 3), dtype=np.float32)
        field_cam_mm = np.zeros((2, 21, 3), dtype=np.float32)
        mask = np.zeros((2,), dtype=np.float32)

        hands = [ex.frame_data.left_hand, ex.frame_data.right_hand]
        fields = [None, None]
        if ex.labels is not None:
            fields = [ex.labels.left_to_object, ex.labels.right_to_object]

        # Skip frames that have no field label for EITHER hand (no supervision)
        if fields[0] is None and fields[1] is None:
            return None

        for h in range(2):
            hand = hands[h]
            field_world = fields[h]
            if hand is None or field_world is None:
                continue

            joints_world_mm = np.asarray(hand.landmarks_world_mm, dtype=np.float64)
            field_world_mm = np.asarray(field_world, dtype=np.float64)

            joints_cam_mm = joints_world_to_camera_mm(joints_world_mm, R_wc, t_wc)
            field_cam_mm[h] = field_world_to_camera_mm(field_world_mm, R_wc).astype(
                np.float32
            )

            # Add noise to simulate WiLoR joint error (train/val robustness)
            if self.joint_noise_mm > 0:
                joints_cam_mm = joints_cam_mm + np.random.randn(
                    21, 3
                ).astype(np.float64) * self.joint_noise_mm

            joints_cam_m[h] = (joints_cam_mm / 1000.0).astype(np.float32)  # m
            mask[h] = 1.0

        return (
            torch.from_numpy(image),
            torch.from_numpy(joints_cam_m),
            torch.from_numpy(field_cam_mm),
            torch.from_numpy(mask),
        )


def collate_fn(batch):
    batch = [b for b in batch if b is not None]
    if len(batch) == 0:
        return None
    images, joints, targets, masks = zip(*batch)
    return (
        torch.stack(images),
        torch.stack(joints),
        torch.stack(targets),
        torch.stack(masks),
    )


class CachedDataset(Dataset):
    """Reads the pre-extracted cache (grayscale JPEG + camera-frame joints/field)."""

    def __init__(self, cache_path, joint_noise_mm=0.0):
        self.samples = []
        with open(cache_path) as f:
            for line in f:
                line = line.strip()
                if line:
                    self.samples.append(json.loads(line))
        self.joint_noise_mm = joint_noise_mm

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        s = self.samples[idx]
        gray = cv2.imread(s["image"], cv2.IMREAD_GRAYSCALE)
        gray = cv2.resize(gray, (224, 224))
        img = np.stack([gray, gray, gray], axis=0).astype(np.float32) / 255.0
        img = (img - IMAGENET_MEAN[:, None, None]) / IMAGENET_STD[:, None, None]

        joints = np.asarray(s["joints_cam_m"], dtype=np.float32)
        target = np.asarray(s["field_cam_mm"], dtype=np.float32)
        mask = np.asarray(s["mask"], dtype=np.float32)

        if self.joint_noise_mm > 0:
            noise = np.random.randn(2, 21, 3).astype(np.float32) * (
                self.joint_noise_mm / 1000.0
            )
            joints = joints + noise * mask[:, None, None]

        return (
            torch.from_numpy(img),
            torch.from_numpy(joints),
            torch.from_numpy(target),
            torch.from_numpy(mask),
        )


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", type=str, required=True)
    ap.add_argument("--manifest", type=str, required=True)
    ap.add_argument("--out", type=str, default="checkpoints/joint_interfield.pt")
    ap.add_argument("--epochs", type=int, default=20)
    ap.add_argument("--batch-size", type=int, default=32)
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--weight-decay", type=float, default=5e-4)
    ap.add_argument("--joint-noise-mm", type=float, default=3.0,
                    help="gaussian noise on GT joints to mimic WiLoR error")
    ap.add_argument("--resume", action="store_true",
                    help="resume training from the checkpoint at --out")
    ap.add_argument("--num-workers", type=int, default=2,
                    help="DataLoader workers (lower if OOM)")
    ap.add_argument("--cache", type=str, default=None,
                    help="path to train_cache.jsonl from extract_frames.py (if set, skip MP4 decoding)")
    args = ap.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    if args.cache:
        dataset = CachedDataset(args.cache, joint_noise_mm=args.joint_noise_mm)
    else:
        dataset = TrainDataset(
            args.root, args.manifest, joint_noise_mm=args.joint_noise_mm
        )
    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        collate_fn=collate_fn,
        drop_last=True,
    )

    model = JointInterFieldModel(pretrained=True).to(device)
    model = nn.DataParallel(model)

    optimizer = torch.optim.AdamW(
        model.parameters(), lr=args.lr, weight_decay=args.weight_decay
    )

    start_epoch = 0
    if args.resume and os.path.exists(args.out):
        ckpt = torch.load(args.out, map_location=device)
        if isinstance(ckpt, dict) and "model" in ckpt:
            # new format {model, optimizer, epoch}
            model.module.load_state_dict(ckpt["model"])
            optimizer.load_state_dict(ckpt["optimizer"])
            start_epoch = ckpt["epoch"] + 1
        else:
            # old format: bare state_dict (only model weights)
            model.module.load_state_dict(ckpt)
            start_epoch = 0
        print(f"resumed from epoch {start_epoch}")

    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)

    for epoch in range(start_epoch, args.epochs):
        model.train()
        total_loss, total_n = 0.0, 0
        for step, batch in enumerate(loader):
            if batch is None:
                continue
            images, joints, targets, masks = [b.to(device) for b in batch]

            pred = model(images, joints)  # (B,2,21,3) mm camera
            per_joint = torch.norm(pred - targets, dim=-1)  # (B,2,21)
            per_hand = per_joint.mean(dim=-1)               # (B,2)
            loss = (per_hand * masks).sum() / masks.sum().clamp_min(1)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item() * masks.sum().item()
            total_n += masks.sum().item()

            if step % 100 == 0:
                print(
                    f"epoch {epoch} step {step} loss {loss.item():.3f} "
                    f"ade_mm {loss.item():.1f}"
                )

        avg = total_loss / max(total_n, 1)
        print(f"=== epoch {epoch} avg ADE (mm) {avg:.2f} ===")

        torch.save(
            {
                "model": model.module.state_dict(),
                "optimizer": optimizer.state_dict(),
                "epoch": epoch,
            },
            args.out,
        )

    print(f"saved to {args.out}")


if __name__ == "__main__":
    main()
