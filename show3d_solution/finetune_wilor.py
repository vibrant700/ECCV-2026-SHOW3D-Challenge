"""
Fine-tune WiLoR's reconstructor on SHOW3D hand-joint annotations (CACHE version).

Reads the pre-extracted cache (train_cache.jsonl) instead of decoding MP4s, which
is tens of times faster. The GT joints are the camera-frame joints already stored
in the cache (meters), so Umeyama aligns WiLoR joints (crop camera, m) to GT
(camera, m) with scale ~1.

Output: a finetuned WiLoR checkpoint whose SHOW3D hand joints match GT closely.
"""
import argparse
import json
import os
import sys

import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

# --- WiLoR ---
sys.path.insert(0, "/home/yixuan.wang/zhongmou.ji/WiLoR")
from wilor.models import WiLoR, load_wilor
from wilor.datasets.vitdet_dataset import ViTDetDataset
from ultralytics import YOLO


def umeyama(X, Y):
    """Optimal similarity transform minimizing ||s*R*X + t - Y||.
    X, Y: (N, 3) numpy. Returns R(3,3), t(3), s(scalar)."""
    X = np.asarray(X, dtype=np.float64)
    Y = np.asarray(Y, dtype=np.float64)
    mx, my = X.mean(0), Y.mean(0)
    Xc, Yc = X - mx, Y - my
    var_x = (Xc ** 2).sum()
    cov = Xc.T @ Yc
    U, S, Vt = np.linalg.svd(cov)
    d = np.sign(np.linalg.det(Vt.T @ U.T))
    D = np.diag([1.0, 1.0, d])
    R = Vt.T @ D @ U.T
    s = np.trace(np.diag(S) @ D) / (var_x + 1e-12)
    t = my - s * R @ mx
    return R, t, s


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cache", required=True,
                    help="path to train_cache.jsonl from extract_frames.py")
    ap.add_argument("--wilor-ckpt", default="/home/yixuan.wang/zhongmou.ji/WiLoR/pretrained_models/wilor_final.ckpt")
    ap.add_argument("--wilor-cfg", default="/home/yixuan.wang/zhongmou.ji/WiLoR/pretrained_models/model_config.yaml")
    ap.add_argument("--detector-pt", default="/home/yixuan.wang/zhongmou.ji/WiLoR/pretrained_models/detector.pt")
    ap.add_argument("--out", default="checkpoints/wilor_finetuned.pt")
    ap.add_argument("--epochs", type=int, default=20)
    ap.add_argument("--lr", type=float, default=1e-5)
    ap.add_argument("--batch-hands", type=int, default=8)
    ap.add_argument("--dp", action="store_true",
                    help="wrap model in DataParallel (use CUDA_VISIBLE_DEVICES=1,2)")
    ap.add_argument("--resume", action="store_true",
                    help="resume training from the checkpoint at --out")
    args = ap.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    wilor_root = os.path.dirname(os.path.dirname(args.wilor_ckpt))
    cwd = os.getcwd()
    os.chdir(wilor_root)
    try:
        model, model_cfg = load_wilor(args.wilor_ckpt, args.wilor_cfg)
    finally:
        os.chdir(cwd)
    model = model.to(device).train()
    if args.dp:
        model = nn.DataParallel(model)
    detector = YOLO(args.detector_pt).to(device)

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr)

    start_epoch = 0
    if args.resume and os.path.exists(args.out):
        ckpt = torch.load(args.out, map_location=device)
        base = model.module if args.dp else model
        if isinstance(ckpt, dict) and "model" in ckpt:
            base.load_state_dict(ckpt["model"])
            optimizer.load_state_dict(ckpt["optimizer"])
            start_epoch = ckpt["epoch"] + 1
        else:
            base.load_state_dict(ckpt)
        print(f"resumed from epoch {start_epoch}", flush=True)

    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)

    # Load cache samples into memory
    samples = []
    with open(args.cache) as f:
        for line in f:
            line = line.strip()
            if line:
                samples.append(json.loads(line))
    print(f"loaded {len(samples)} cache samples", flush=True)

    def collect_batch(start_idx):
        patches, rights, gts = [], [], []
        idx = start_idx
        while len(patches) < args.batch_hands and idx < len(samples):
            s = samples[idx]
            idx += 1
            joints_gt = np.asarray(s["joints_cam_m"], np.float32)  # (2,21,3) camera m
            mask = np.asarray(s["mask"], np.float32)                # (2,)
            boxes = s.get("boxes", [])  # pre-detected [[x1,y1,x2,y2,is_right], ...]
            if not boxes:
                continue

            gray = cv2.imread(s["image"], cv2.IMREAD_GRAYSCALE)
            if gray is None:
                continue
            image_bgr = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)

            boxes_arr = np.array([b[:4] for b in boxes])
            rights_arr = np.array([b[4] for b in boxes])
            vd = ViTDetDataset(model_cfg, image_bgr, boxes_arr, rights_arr, rescale_factor=2.0)
            for i in range(len(vd)):
                item = vd[i]
                side = int(item["right"])
                if mask[side] > 0:
                    patches.append(item["img"])          # numpy (3,256,256)
                    rights.append(side)
                    gts.append(joints_gt[side])          # (21,3) camera m
        return patches, rights, gts, idx

    for epoch in range(start_epoch, args.epochs):
        model.train()
        total_loss, total_hands, n_batch = 0.0, 0, 0
        idx = 0
        while idx < len(samples):
            patches, rights, gts, idx = collect_batch(idx)
            if not patches:
                continue

            img_batch = torch.stack([torch.from_numpy(p).to(device) for p in patches])
            out = model({"img": img_batch})
            joints = out["pred_keypoints_3d"]  # (B,21,3) meters, differentiable

            loss, cnt = 0.0, 0
            for b in range(joints.shape[0]):
                side = rights[b]
                j = joints[b].clone()
                j[:, 0] = (2 * side - 1) * j[:, 0]  # undo left-hand flip
                gt = gts[b]  # (21,3) camera m

                R, t, s = umeyama(j.detach().cpu().numpy(), gt)
                R_t = torch.tensor(R, device=device, dtype=j.dtype)
                t_t = torch.tensor(t, device=device, dtype=j.dtype)
                s_t = torch.tensor(s, device=device, dtype=j.dtype)

                aligned = s_t * (j @ R_t.T) + t_t
                gt_t = torch.tensor(gt, device=device, dtype=j.dtype)
                per_joint = torch.norm(aligned - gt_t, dim=-1)  # (21,) meters
                loss = loss + per_joint.mean() * 1000.0          # mm
                cnt += 1

            if cnt == 0:
                continue

            loss = loss / cnt
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item() * cnt
            total_hands += cnt
            n_batch += 1

            if n_batch % 50 == 0:
                print(f"epoch {epoch} batch {n_batch} loss {total_loss/max(total_hands,1):.3f} mm",
                      flush=True)

        avg = total_loss / max(total_hands, 1)
        print(f"=== epoch {epoch} avg aligned-joint loss {avg:.3f} mm ===", flush=True)
        sd = model.module.state_dict() if args.dp else model.state_dict()
        torch.save(
            {"model": sd, "optimizer": optimizer.state_dict(), "epoch": epoch},
            args.out,
        )
        torch.save(sd, f"{args.out}.epoch{epoch}")

    print(f"saved to {args.out}")


if __name__ == "__main__":
    main()
