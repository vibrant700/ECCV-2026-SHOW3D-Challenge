"""
One-time preprocessing: decode every training frame once, save it as a grayscale
JPEG, and materialize the camera-frame joints + field labels into a JSONL cache.

After this, train.py reads the cache directly (fast, no MP4 decoding).

Output:
    <out-dir>/images/XXXX.jpg          grayscale JPEG per valid frame
    <out-dir>/train_cache.jsonl        one line per valid frame:
        {sample_id, image, joints_cam_m (2,21,3), field_cam_mm (2,21,3), mask (2,)}
"""
import argparse
import json
import os

import cv2
import numpy as np


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", required=True)
    ap.add_argument("--manifest", required=True,
                    help="frame-level manifest (e.g. frames/train_frames.jsonl)")
    ap.add_argument("--out-dir", default="cache")
    ap.add_argument("--start", type=int, default=0)
    ap.add_argument("--end", type=int, default=-1,
                    help="exclusive end index; -1 = all")
    args = ap.parse_args()

    import sys
    sys.path.insert(0, "/home/yixuan.wang/zhongmou.ji/SHOW3D-dataset-api")
    from show3d.interaction_field import Show3DInteractionFieldDataset

    ds = Show3DInteractionFieldDataset(
        args.root, args.manifest, load_labels=True, decode_images=True, multiview=False
    )

    img_dir = os.path.join(args.out_dir, "images")
    os.makedirs(img_dir, exist_ok=True)

    end = len(ds) if args.end < 0 else args.end
    out_lines = []

    for i in range(args.start, min(end, len(ds))):
        ex = ds[i]
        view = ex.views["headset0"]
        calib = view.calibration
        if calib is None or calib.t_world_from_camera is None:
            continue

        R_wc = np.asarray(calib.t_world_from_camera[:3, :3], dtype=np.float64)
        t_wc = np.asarray(calib.t_world_from_camera[:3, 3], dtype=np.float64)

        # field labels (vectors: rotate only) -> camera mm
        field_cam_mm = np.zeros((2, 21, 3), dtype=np.float32)
        mask = np.zeros((2,), dtype=np.float32)
        if ex.labels is not None:
            for h, fw in enumerate([ex.labels.left_to_object, ex.labels.right_to_object]):
                if fw is not None:
                    fw = np.asarray(fw, dtype=np.float64)
                    field_cam_mm[h] = (fw @ R_wc).astype(np.float32)
                    mask[h] = 1.0

        if mask.sum() == 0:
            continue  # no field label for either hand -> skip

        # joints (points: subtract translation) -> camera meters
        joints_cam_m = np.zeros((2, 21, 3), dtype=np.float32)
        for h, hand in enumerate([ex.frame_data.left_hand, ex.frame_data.right_hand]):
            if hand is not None and hand.landmarks_world_mm is not None:
                jw = np.asarray(hand.landmarks_world_mm, dtype=np.float64)
                jc = (jw - t_wc[None, :]) @ R_wc
                joints_cam_m[h] = (jc / 1000.0).astype(np.float32)

        # save grayscale image
        gray = cv2.cvtColor(view.image, cv2.COLOR_RGB2GRAY)
        img_path = os.path.join(img_dir, f"{i:08d}.jpg")
        cv2.imwrite(img_path, gray, [cv2.IMWRITE_JPEG_QUALITY, 90])

        out_lines.append(
            json.dumps(
                {
                    "sample_id": ex.sample.sample_id,
                    "image": img_path,
                    "joints_cam_m": joints_cam_m.tolist(),
                    "field_cam_mm": field_cam_mm.tolist(),
                    "mask": mask.tolist(),
                }
            )
        )

        if i % 1000 == 0:
            print(f"processed {i}/{len(ds)}, valid {len(out_lines)}", flush=True)

    cache_path = os.path.join(args.out_dir, "train_cache.jsonl")
    with open(cache_path, "w") as f:
        for line in out_lines:
            f.write(line + "\n")
    print(f"wrote {len(out_lines)} samples to {cache_path}")


if __name__ == "__main__":
    main()
