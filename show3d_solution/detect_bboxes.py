"""
One-time pre-detection: run YOLO on every cache sample and store the hand bboxes.

The finetune script then reads these boxes directly (skipping YOLO at train time,
which removes a CPU bottleneck). Output is a JSONL with one line per sample:
    {image, joints_cam_m, mask, boxes: [[x1,y1,x2,y2,is_right], ...]}
"""
import argparse
import json
import sys

import cv2
import numpy as np
from ultralytics import YOLO


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cache", required=True, help="train_cache.jsonl")
    ap.add_argument("--detector-pt",
                    default="/home/yixuan.wang/zhongmou.ji/WiLoR/pretrained_models/detector.pt")
    ap.add_argument("--out", default="cache/finetune_bbox.jsonl")
    args = ap.parse_args()

    detector = YOLO(args.detector_pt)

    samples = []
    with open(args.cache) as f:
        for line in f:
            if line.strip():
                samples.append(json.loads(line))
    print(f"loaded {len(samples)} samples", flush=True)

    out_lines = []
    for i, s in enumerate(samples):
        gray = cv2.imread(s["image"], cv2.IMREAD_GRAYSCALE)
        if gray is None:
            continue
        image_bgr = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)

        detections = detector(image_bgr, conf=0.3, verbose=False)[0]
        boxes = []
        for det in detections:
            box = det.boxes.data.cpu().squeeze().numpy()
            cls = det.boxes.cls.cpu().squeeze().item()
            boxes.append([float(box[0]), float(box[1]), float(box[2]), float(box[3]), int(cls)])

        out_lines.append(json.dumps({
            "image": s["image"],
            "joints_cam_m": s["joints_cam_m"],
            "mask": s["mask"],
            "boxes": boxes,
        }))

        if i % 1000 == 0:
            print(f"processed {i}/{len(samples)}", flush=True)

    with open(args.out, "w") as f:
        for line in out_lines:
            f.write(line + "\n")
    print(f"wrote {len(out_lines)} to {args.out}")


if __name__ == "__main__":
    main()
