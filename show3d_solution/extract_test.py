"""
One-time pre-extraction of the TEST set: decode every test frame once, save it as
a grayscale JPEG, and store the camera calibration (R_world_from_camera) needed
by predict.py to rotate fields back to world space.

This runs sequentially (no USB IO storm), then predict.py reads local JPEGs.
Output:
    <out-dir>/images/XXXX.jpg      grayscale JPEG per frame
    <out-dir>/test_cache.jsonl     {sample_id, image, R_wc}
"""
import argparse
import json
import os
import sys

import cv2
import numpy as np

sys.path.insert(0, "/home/yixuan.wang/zhongmou.ji/SHOW3D-dataset-api")
from show3d.interaction_field import Show3DInteractionFieldDataset


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", required=True)
    ap.add_argument("--manifest", required=True, help="test_manifest_5fps_202607.jsonl")
    ap.add_argument("--out-dir", default="test_cache")
    args = ap.parse_args()

    ds = Show3DInteractionFieldDataset(
        args.root, args.manifest, load_labels=False, decode_images=True, multiview=False
    )

    img_dir = os.path.join(args.out_dir, "images")
    os.makedirs(img_dir, exist_ok=True)

    out_lines = []
    for i in range(len(ds)):
        ex = ds[i]
        view = ex.views["headset0"]
        calib = view.calibration
        if calib is None or calib.t_world_from_camera is None:
            out_lines.append(json.dumps({
                "sample_id": ex.sample.sample_id, "image": None, "R_wc": None
            }))
            continue

        R_wc = np.asarray(calib.t_world_from_camera[:3, :3], np.float64)
        gray = cv2.cvtColor(view.image, cv2.COLOR_RGB2GRAY)
        img_path = os.path.join(img_dir, f"{i:08d}.jpg")
        cv2.imwrite(img_path, gray, [cv2.IMWRITE_JPEG_QUALITY, 90])

        out_lines.append(json.dumps({
            "sample_id": ex.sample.sample_id,
            "image": img_path,
            "R_wc": R_wc.tolist(),
        }))

        if i % 500 == 0:
            print(f"processed {i}/{len(ds)}", flush=True)

    cache_path = os.path.join(args.out_dir, "test_cache.jsonl")
    with open(cache_path, "w") as f:
        for line in out_lines:
            f.write(line + "\n")
    print(f"wrote {len(out_lines)} to {cache_path}")


if __name__ == "__main__":
    main()
