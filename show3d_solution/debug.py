"""
Quick smoke test: load one sample, check field names / coordinate transforms,
run the model forward, and compute a loss. Run BEFORE full training.

Expected value ranges (sanity check):
    image   : ImageNet-normalized, roughly [-2.5, 2.5]
    joints  : meters, max abs ~ 0.1-0.3 (hand size)
    target  : mm, max abs ~ tens of mm (hand-to-object offset)
    mask    : [0,0] / [1,0] / [0,1] / [1,1]
"""
import sys

sys.path.insert(0, "/home/yixuan.wang/zhongmou.ji/show3d_solution")
sys.path.insert(0, "/home/yixuan.wang/zhongmou.ji/SHOW3D-dataset-api")

import argparse

import numpy as np
import torch

from train import TrainDataset, collate_fn
from model import JointInterFieldModel

ap = argparse.ArgumentParser()
ap.add_argument("--root", required=True)
ap.add_argument("--manifest", required=True,
                help="FRAME-level manifest (generate_frame_manifest.py output)")
args = ap.parse_args()

print("building dataset...")
ds = TrainDataset(args.root, args.manifest, joint_noise_mm=3.0)
print(f"dataset size: {len(ds)}")

# Find a valid sample (skip None = bad calibration)
item = None
for i in range(min(500, len(ds))):
    cand = ds[i]
    if cand is not None:
        item = cand
        print(f"got valid sample at index {i}")
        break

if item is None:
    print("ERROR: no valid sample in first 500 — check calibration handling")
    sys.exit(1)

img, joints, target, mask = item
print(f"image   : {tuple(img.shape)} {img.dtype}  range [{img.min():.2f}, {img.max():.2f}]")
print(f"joints  : {tuple(joints.shape)} (m)   max abs {joints.abs().max():.3f}")
print(f"target  : {tuple(target.shape)} (mm)  max abs {target.abs().max():.1f}")
print(f"mask    : {mask.tolist()}")

# Model forward
model = JointInterFieldModel(pretrained=False)
out = model(img.unsqueeze(0), joints.unsqueeze(0))
print(f"model out: {tuple(out.shape)}")

# Loss
per_joint = torch.norm(out - target.unsqueeze(0), dim=-1)
per_hand = per_joint.mean(dim=-1)
loss = (per_hand * mask.unsqueeze(0)).sum() / mask.sum().clamp_min(1)
print(f"loss     : {loss.item():.2f} mm")

print("\nOK: smoke test passed (check the value ranges above look sane)")
