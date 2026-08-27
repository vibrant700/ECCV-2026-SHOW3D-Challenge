"""
Visualize: WiLoR predicted joints vs GT joints.

Top row: original image with WiLoR joints projected onto it (2D).
Bottom row: 3D comparison after Umeyama alignment (y-flipped so "up" matches image up).

Both sides are brought into the SAME (OpenPose) joint order before comparing, so
the skeleton connections and per-joint error are anatomically correct.

Joint-order note (this is the key fix):
    - WiLoR's `pred_keypoints_3d` is ALREADY OpenPose order
      (wilor/models/mano_wrapper.py reorders MANO -> OpenPose):
        0=wrist, 1-4=thumb, 5-8=index, 9-12=middle, 13-16=ring, 17-20=pinky
    - SHOW3D's `landmarks_world_mm` is UmeTrack/HOT3D order:
        0-4=fingertips(thumb,index,middle,ring,pinky), 5=wrist,
        6-7=thumb(MCP,IP), 8-10=index, 11-13=middle, 14-16=ring,
        17-19=pinky, 20=palm center
    We reorder the GT into OpenPose order via GT_TO_OPENPOSE below.
"""
import argparse
import json
import os
import sys

import cv2
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch

sys.path.insert(0, "/home/yixuan.wang/zhongmou.ji/WiLoR")
from wilor.models import load_wilor
from wilor.datasets.vitdet_dataset import ViTDetDataset
from wilor.utils import recursive_to
from wilor.utils.renderer import cam_crop_to_full
from ultralytics import YOLO


# GT (UmeTrack/HOT3D) index -> OpenPose index.
# gt_openpose[i] = gt_umetrack[GT_TO_OPENPOSE[i]]
GT_TO_OPENPOSE = [
    5,    #  0 wrist            <- UmeTrack 5
    20,   #  1 thumb CMC        <- UmeTrack 20 palm center (approx, no exact CMC)
    6,    #  2 thumb MCP        <- UmeTrack 6
    7,    #  3 thumb IP         <- UmeTrack 7
    0,    #  4 thumb tip        <- UmeTrack 0
    8,    #  5 index MCP        <- UmeTrack 8
    9,    #  6 index PIP        <- UmeTrack 9
    10,   #  7 index DIP        <- UmeTrack 10
    1,    #  8 index tip        <- UmeTrack 1
    11,   #  9 middle MCP       <- UmeTrack 11
    12,   # 10 middle PIP       <- UmeTrack 12
    13,   # 11 middle DIP       <- UmeTrack 13
    2,    # 12 middle tip       <- UmeTrack 2
    14,   # 13 ring MCP         <- UmeTrack 14
    15,   # 14 ring PIP         <- UmeTrack 15
    16,   # 15 ring DIP         <- UmeTrack 16
    3,    # 16 ring tip         <- UmeTrack 3
    17,   # 17 pinky MCP        <- UmeTrack 17
    18,   # 18 pinky PIP        <- UmeTrack 18
    19,   # 19 pinky DIP        <- UmeTrack 19
    4,    # 20 pinky tip        <- UmeTrack 4
]


def umeyama(X, Y):
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


# OpenPose skeleton: wrist -> each finger chain
BONES = [(0,1),(1,2),(2,3),(3,4),
         (0,5),(5,6),(6,7),(7,8),
         (0,9),(9,10),(10,11),(11,12),
         (0,13),(13,14),(14,15),(15,16),
         (0,17),(17,18),(18,19),(19,20)]


def finger_color(i):
    if i == 0:
        return "gray"      # wrist
    if 1 <= i <= 4:
        return "red"       # thumb
    if 5 <= i <= 8:
        return "orange"    # index
    if 9 <= i <= 12:
        return "green"     # middle
    if 13 <= i <= 16:
        return "blue"      # ring
    return "purple"        # pinky


def cam_crop_to_full_np(cam_bbox, box_center, box_size, img_size, focal_length):
    """Numpy version of wilor.utils.renderer.cam_crop_to_full.
    cam_bbox = [scale, tx, ty] (weak-perspective crop camera)."""
    img_w, img_h = float(img_size[0]), float(img_size[1])
    cx, cy, b = float(box_center[0]), float(box_center[1]), float(box_size)
    w_2, h_2 = img_w / 2.0, img_h / 2.0
    bs = b * cam_bbox[0] + 1e-9
    tz = 2.0 * focal_length / bs
    tx = (2.0 * (cx - w_2) / bs) + cam_bbox[1]
    ty = (2.0 * (cy - h_2) / bs) + cam_bbox[2]
    return np.array([tx, ty, tz], dtype=np.float64)


def project_full_img_np(points, cam_trans, focal_length, img_res):
    """Project camera-frame 3D points (m) to full-image pixels."""
    cx, cy = float(img_res[0]) / 2.0, float(img_res[1]) / 2.0
    p = points + cam_trans[None, :]
    p = p / p[:, 2:3]
    x = focal_length * p[:, 0] + cx
    y = focal_length * p[:, 1] + cy
    return np.stack([x, y], axis=1)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cache", required=True, help="train_cache.jsonl")
    ap.add_argument("--index", type=int, default=0, help="which cache sample")
    ap.add_argument("--wilor-ckpt", default="/home/yixuan.wang/zhongmou.ji/WiLoR/pretrained_models/wilor_final.ckpt")
    ap.add_argument("--wilor-cfg", default="/home/yixuan.wang/zhongmou.ji/WiLoR/pretrained_models/model_config.yaml")
    ap.add_argument("--detector-pt", default="/home/yixuan.wang/zhongmou.ji/WiLoR/pretrained_models/detector.pt")
    ap.add_argument("--wilor-finetuned", type=str, default=None)
    ap.add_argument("--out", default="viz_compare.png")
    args = ap.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    samples = []
    with open(args.cache) as f:
        for line in f:
            if line.strip():
                samples.append(json.loads(line))
    s = samples[args.index]
    joints_gt_umetrack = np.asarray(s["joints_cam_m"], np.float32)  # (2,21,3) UmeTrack order
    mask = np.asarray(s["mask"], np.float32)
    print(f"sample index {args.index}, mask={mask.tolist()}")

    gray = cv2.imread(s["image"], cv2.IMREAD_GRAYSCALE)
    image_bgr = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)

    wilor_root = os.path.dirname(os.path.dirname(args.wilor_ckpt))
    cwd = os.getcwd(); os.chdir(wilor_root)
    try:
        model, model_cfg = load_wilor(args.wilor_ckpt, args.wilor_cfg)
    finally:
        os.chdir(cwd)
    if args.wilor_finetuned:
        ft = torch.load(args.wilor_finetuned, map_location="cpu")
        if isinstance(ft, dict) and "model" in ft:
            ft = ft["model"]
        model.load_state_dict(ft)
    model = model.to(device).eval()
    detector = YOLO(args.detector_pt).to(device)

    detections = detector(image_bgr, conf=0.3, verbose=False)[0]
    boxes, is_right = [], []
    for det in detections:
        box = det.boxes.data.cpu().squeeze().numpy()
        boxes.append(box[:4].tolist())
        is_right.append(det.boxes.cls.cpu().squeeze().item())
    if not boxes:
        print("no hands detected")
        return

    vd = ViTDetDataset(model_cfg, image_bgr, np.stack(boxes), np.stack(is_right),
                       rescale_factor=2.0)
    loader = torch.utils.data.DataLoader(vd, batch_size=len(boxes), shuffle=False)

    fig = plt.figure(figsize=(5 * len(boxes), 10))
    for batch in loader:
        batch = recursive_to(batch, device)
        with torch.no_grad():
            out = model(batch)
        joints = out["pred_keypoints_3d"]  # (B,21,3) m, OpenPose order

        # crop -> full camera params (per demo.py)
        pred_cam = out["pred_cam"].cpu().numpy().copy()          # (B,3) [scale,tx,ty]
        mult = (2 * batch["right"] - 1).cpu().numpy()            # +1 right, -1 left
        pred_cam[:, 1] = mult * pred_cam[:, 1]                    # undo flip on tx
        box_center = batch["box_center"].cpu().numpy()
        box_size = batch["box_size"].cpu().numpy()
        img_size = batch["img_size"].cpu().numpy()
        scaled_focal = model_cfg.EXTRA.FOCAL_LENGTH / model_cfg.MODEL.IMAGE_SIZE * img_size.max()

        for b in range(joints.shape[0]):
            side = int(batch["right"][b].cpu().numpy())
            if mask[side] == 0:
                continue
            j = joints[b].cpu().numpy().copy()          # OpenPose order already
            j[:, 0] = (2 * side - 1) * j[:, 0]          # undo left-hand flip
            gt = joints_gt_umetrack[side][GT_TO_OPENPOSE]  # -> OpenPose order

            # ---- 2D: project WiLoR joints onto the original image ----
            cam_t_full = cam_crop_to_full_np(pred_cam[b], box_center[b], box_size[b],
                                             img_size[b], scaled_focal)
            kpts2d = project_full_img_np(j, cam_t_full, scaled_focal, img_size[b])

            ax_img = fig.add_subplot(2, len(boxes), b + 1)
            ax_img.imshow(image_bgr[:, :, ::-1])
            for a, bb in BONES:
                c = finger_color(bb)
                ax_img.plot([kpts2d[a,0], kpts2d[bb,0]], [kpts2d[a,1], kpts2d[bb,1]],
                            c=c, lw=2)
            for i in range(21):
                c = finger_color(i)
                ax_img.scatter(kpts2d[i,0], kpts2d[i,1], c=c, s=40, zorder=5)
                ax_img.text(kpts2d[i,0], kpts2d[i,1], f"{i}", fontsize=6, color=c)
            ax_img.set_title(f"{'right' if side==1 else 'left'} hand (WiLoR on image)")
            ax_img.axis("off")

            # ---- 3D: WiLoR-aligned vs GT (y-flipped so up == image up) ----
            R, t, s = umeyama(j, gt)
            aligned = s * (j @ R.T) + t  # WiLoR aligned to GT

            ax = fig.add_subplot(2, len(boxes), len(boxes) + b + 1, projection="3d")
            for a, bb in BONES:
                c = finger_color(bb)
                ax.plot([gt[a,0], gt[bb,0]], [-gt[a,1], -gt[bb,1]], [gt[a,2], gt[bb,2]],
                        c=c, lw=2, ls="-")
                ax.plot([aligned[a,0], aligned[bb,0]],
                        [-aligned[a,1], -aligned[bb,1]],
                        [aligned[a,2], aligned[bb,2]],
                        c=c, lw=2, ls="--")

            for i in range(21):
                c = finger_color(i)
                ax.scatter(gt[i,0], -gt[i,1], gt[i,2], c=c, s=50, marker="o")
                ax.scatter(aligned[i,0], -aligned[i,1], aligned[i,2],
                           c=c, s=50, marker="x")
                ax.text(gt[i,0], -gt[i,1], gt[i,2], f"{i}", color="black", fontsize=7)

            err = np.linalg.norm(aligned - gt, axis=1) * 1000.0  # mm per joint
            ax.set_title(f"mean err {err.mean():.1f}mm (o=GT, x=WiLoR)")
            ax.view_init(elev=-90, azim=-90)

    plt.tight_layout()
    plt.savefig(args.out, dpi=100)
    print(f"saved to {args.out}")


if __name__ == "__main__":
    main()
