"""
Predict the interaction field with the full-frame model on the test cache,
and write a SHOW3D submission (predictions.jsonl).

Whole frame -> resize 224 -> backbone -> (2,21,3) camera-frame field (mm)
           -> rotate to world with R_world_from_camera^T -> submission.

Frames with missing extrinsic (R_wc None) are emitted as all-zeros (recall 1.0).
"""
import argparse
import json
import sys

import cv2
import numpy as np
import torch

sys.path.insert(0, "/home/yixuan.wang/zhongmou.ji/SHOW3D-dataset-api")

from train_full import FullFrameInterFieldModel, preprocess


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--test-cache", type=str, required=True, help="test_cache.jsonl")
    ap.add_argument("--checkpoint", type=str, required=True)
    ap.add_argument("--out", type=str, default="predictions.jsonl")
    ap.add_argument("--start", type=int, default=0)
    ap.add_argument("--end", type=int, default=-1)
    args = ap.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    ckpt = torch.load(args.checkpoint, map_location=device)
    backbone = ckpt.get("backbone", "resnet50")
    model = FullFrameInterFieldModel(backbone=backbone, pretrained=False).to(device)
    sd = ckpt.get("model_state", ckpt)
    if any(k.startswith("module.") for k in sd):
        sd = {k.replace("module.", "", 1): v for k, v in sd.items()}
    model.load_state_dict(sd)
    model.eval()

    samples = []
    with open(args.test_cache) as f:
        for line in f:
            line = line.strip()
            if line:
                samples.append(json.loads(line))

    from show3d.interaction_field import PredictionRecord, write_submission_jsonl

    zeros = np.zeros((21, 3), dtype=np.float64)
    records = []
    end = len(samples) if args.end < 0 else args.end
    for idx in range(args.start, min(end, len(samples))):
        s = samples[idx]
        sid = s["sample_id"]
        if s["R_wc"] is None:
            records.append(PredictionRecord(
                sample_id=sid,
                fields={"left_to_object": zeros.copy(), "right_to_object": zeros.copy()},
            ))
            continue
        R_wc = np.asarray(s["R_wc"], np.float64)

        gray = cv2.imread(s["image"], cv2.IMREAD_GRAYSCALE)
        if gray is None:
            records.append(PredictionRecord(
                sample_id=sid,
                fields={"left_to_object": zeros.copy(), "right_to_object": zeros.copy()},
            ))
            continue

        x = preprocess(gray)
        x_t = torch.from_numpy(x).unsqueeze(0).to(device)
        with torch.no_grad():
            pred = model(x_t)[0].cpu().numpy()  # (2,21,3) mm camera

        field_world = pred @ R_wc.T  # (2,21,3) mm world
        records.append(PredictionRecord(sample_id=sid, fields={
            "left_to_object": field_world[0].astype(np.float64),
            "right_to_object": field_world[1].astype(np.float64),
        }))

        if idx % 500 == 0:
            print(f"processed {idx}/{len(samples)}", flush=True)

    write_submission_jsonl(args.out, records)
    print(f"wrote {args.out} with {len(records)} frames")


if __name__ == "__main__":
    main()
