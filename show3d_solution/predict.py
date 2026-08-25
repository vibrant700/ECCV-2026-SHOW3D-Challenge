"""
Predict + produce a SHOW3D challenge submission using the joint-conditioned model.

Pipeline per frame:
    image -> WiLoR detect + reconstruct -> 21 hand joints (camera m)
          -> JointInterFieldModel -> field vectors (camera mm)
          -> rotate to world (R_world_from_camera^T, rotation only)
          -> write predictions.jsonl

NOTE on the WiLoR joint coordinate transform:
    WiLoR's `pred_keypoints_3d` is in the crop camera frame (meters, hand-centered).
    The regression model uses a root-relative joint encoding, so the hand-internal
    geometry (relative to wrist) is what matters most; the wrist's absolute camera
    position is secondary. We therefore feed the crop-frame joints directly as an
    approximation of the camera frame. If you need exact metric alignment, calibrate
    this against SHOW3D GT `landmarks_world_mm` (see train.py's transform) and
    adjust the offset below.
"""
import argparse
import json
import os
import sys
from pathlib import Path

import cv2
import numpy as np
import torch

# --- WiLoR imports (adjust if not on PYTHONPATH) ---
sys.path.insert(0, "/home/yixuan.wang/zhongmou.ji/WiLoR")
from wilor.models import WiLoR, load_wilor
from wilor.datasets.vitdet_dataset import ViTDetDataset
from wilor.utils import recursive_to
from ultralytics import YOLO

# --- SHOW3D API ---
from show3d.interaction_field import (
    PredictionRecord,
    Show3DInteractionFieldDataset,
    write_submission_jsonl,
)

from model import JointInterFieldModel

IMAGENET_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
IMAGENET_STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)


def preprocess_image(image_rgb):
    """RGB uint8 (H,W,3) -> (3,224,224) float32, ImageNet-normalized gray."""
    gray = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2GRAY)
    gray = cv2.resize(gray, (224, 224))
    img = np.stack([gray, gray, gray], axis=0).astype(np.float32) / 255.0
    img = (img - IMAGENET_MEAN[:, None, None]) / IMAGENET_STD[:, None, None]
    return img


class WiLoRJoints:
    """Run WiLoR detector + reconstructor, return per-hand 21 joints (camera m)."""

    def __init__(self, wilor_ckpt, wilor_cfg, detector_pt, device, finetuned=None):
        import os
        # WiLoR's load_wilor resolves './mano_data/...' relative to CWD, so
        # temporarily chdir into the WiLoR root while loading.
        wilor_root = os.path.dirname(os.path.dirname(wilor_ckpt))
        cwd = os.getcwd()
        os.chdir(wilor_root)
        try:
            self.model, self.model_cfg = load_wilor(
                checkpoint_path=wilor_ckpt, cfg_path=wilor_cfg
            )
        finally:
            os.chdir(cwd)
        if finetuned is not None:
            ft = torch.load(finetuned, map_location="cpu")
            if isinstance(ft, dict) and "model" in ft:
                ft = ft["model"]
            self.model.load_state_dict(ft)
            print(f"loaded finetuned WiLoR weights from {finetuned}")
        self.model = self.model.to(device).eval()
        self.detector = YOLO(detector_pt).to(device)
        self.device = device

    @torch.no_grad()
    def __call__(self, image_bgr):
        """image_bgr: (H,W,3) uint8 BGR. Returns (joints_m, is_right) for each hand."""
        detections = self.detector(image_bgr, conf=0.1, verbose=False)[0]

        bboxes, is_right = [], []
        for det in detections:
            box = det.boxes.data.cpu().squeeze().numpy()
            bboxes.append(box[:4].tolist())
            is_right.append(det.boxes.cls.cpu().squeeze().item())

        if len(bboxes) == 0:
            return [], []

        boxes = np.stack(bboxes)
        right = np.stack(is_right)
        dataset = ViTDetDataset(
            self.model_cfg, image_bgr, boxes, right, rescale_factor=2.0
        )
        loader = torch.utils.data.DataLoader(dataset, batch_size=16, shuffle=False)

        all_joints, all_right = [], []
        for batch in loader:
            batch = recursive_to(batch, self.device)
            out = self.model(batch)
            joints = out["pred_keypoints_3d"]  # (B,21,3) meters, crop frame
            r = batch["right"].cpu().numpy()
            for n in range(joints.shape[0]):
                j = joints[n].detach().cpu().numpy().copy()
                # undo the left-hand horizontal flip applied by ViTDetDataset
                j[:, 0] = (2 * r[n] - 1) * j[:, 0]
                all_joints.append(j)
                all_right.append(r[n])
        return all_joints, all_right


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", type=str, default=None,
                    help="data root (required unless --test-cache is used)")
    ap.add_argument("--manifest", type=str, default=None,
                    help="frame manifest (required unless --test-cache is used)")
    ap.add_argument("--checkpoint", type=str, required=True,
                    help="trained JointInterFieldModel checkpoint")
    ap.add_argument("--wilor-ckpt", type=str,
                    default="/home/yixuan.wang/zhongmou.ji/WiLoR/pretrained_models/wilor_final.ckpt")
    ap.add_argument("--wilor-cfg", type=str,
                    default="/home/yixuan.wang/zhongmou.ji/WiLoR/pretrained_models/model_config.yaml")
    ap.add_argument("--detector-pt", type=str,
                    default="/home/yixuan.wang/zhongmou.ji/WiLoR/pretrained_models/detector.pt")
    ap.add_argument("--wilor-finetuned", type=str, default=None,
                    help="optional finetuned WiLoR checkpoint (state_dict) to load")
    ap.add_argument("--out", type=str, default="predictions.jsonl")
    ap.add_argument("--start", type=int, default=0,
                    help="inclusive start index (for sharding across GPUs)")
    ap.add_argument("--end", type=int, default=-1,
                    help="exclusive end index; -1 = all")
    ap.add_argument("--test-cache", type=str, default=None,
                    help="test_cache.jsonl from extract_test.py (if set, skip MP4 decoding)")
    args = ap.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Load regression model (strip DataParallel "module." prefix if present)
    model = JointInterFieldModel(pretrained=False).to(device).eval()
    ckpt = torch.load(args.checkpoint, map_location=device)
    if isinstance(ckpt, dict) and "model" in ckpt:
        ckpt = ckpt["model"]  # new checkpoint format {model, optimizer, epoch}
    if any(k.startswith("module.") for k in ckpt):
        ckpt = {k.replace("module.", "", 1): v for k, v in ckpt.items()}
    model.load_state_dict(ckpt)

    # Load WiLoR
    wilor = WiLoRJoints(args.wilor_ckpt, args.wilor_cfg, args.detector_pt, device,
                        finetuned=args.wilor_finetuned)

    # Load test data: from test_cache.jsonl if provided, else decode MP4s
    use_cache = args.test_cache is not None
    if use_cache:
        test_samples = []
        with open(args.test_cache) as f:
            for line in f:
                line = line.strip()
                if line:
                    test_samples.append(json.loads(line))
        data_len = len(test_samples)
    else:
        dataset = Show3DInteractionFieldDataset(
            args.root, args.manifest, load_labels=False, decode_images=True, multiview=False
        )
        data_len = len(dataset)

    records = []
    end = data_len if args.end < 0 else args.end
    for idx in range(args.start, min(end, data_len)):
        if use_cache:
            s = test_samples[idx]
            sample_id = s["sample_id"]
            if s["R_wc"] is None:
                records.append(PredictionRecord(
                    sample_id=sample_id,
                    fields={
                        "left_to_object": np.zeros((21, 3), dtype=np.float64),
                        "right_to_object": np.zeros((21, 3), dtype=np.float64),
                    },
                ))
                continue
            R_wc = np.asarray(s["R_wc"], np.float64)
            gray = cv2.imread(s["image"], cv2.IMREAD_GRAYSCALE)
            image_bgr = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)
            image_rgb = image_bgr[:, :, ::-1]  # grayscale replicated to 3ch
        else:
            ex = dataset[idx]
            view = ex.views["headset0"]
            calib = view.calibration
            if calib is None or calib.t_world_from_camera is None:
                records.append(PredictionRecord(
                    sample_id=ex.sample.sample_id,
                    fields={
                        "left_to_object": np.zeros((21, 3), dtype=np.float64),
                        "right_to_object": np.zeros((21, 3), dtype=np.float64),
                    },
                ))
                continue
            sample_id = ex.sample.sample_id
            R_wc = np.asarray(calib.t_world_from_camera[:3, :3], dtype=np.float64)
            image_rgb = view.image  # (H,W,3) uint8 RGB
            image_bgr = image_rgb[:, :, ::-1]  # RGB -> BGR for WiLoR/YOLO

        # WiLoR joints (camera m) per detected hand
        joints_m, is_right = wilor(image_bgr)

        joints_cam = np.zeros((2, 21, 3), dtype=np.float32)
        for j_m, r in zip(joints_m, is_right):
            hand_idx = 1 if r else 0
            joints_cam[hand_idx] = j_m.astype(np.float32)

        img = preprocess_image(image_rgb)
        img_t = torch.from_numpy(img).unsqueeze(0).to(device)
        joints_t = torch.from_numpy(joints_cam).unsqueeze(0).to(device)

        with torch.no_grad():
            field_cam = model(img_t, joints_t)[0].cpu().numpy()  # (2,21,3) mm camera

        field_world = field_cam @ R_wc.T  # (2,21,3) mm world

        fields = {
            "left_to_object": field_world[0].astype(np.float64),
            "right_to_object": field_world[1].astype(np.float64),
        }

        records.append(PredictionRecord(sample_id=sample_id, fields=fields))

        if idx % 500 == 0:
            print(f"processed {idx}/{data_len}", flush=True)

    write_submission_jsonl(args.out, records)
    print(f"wrote {args.out} with {len(records)} frames")


if __name__ == "__main__":
    main()
