"""
Expand the recording-level train_manifest into a frame-level manifest
(each row has sample_id + frame_index) that Show3DInteractionFieldDataset accepts.

Source videos are 60 fps; we sample every `step` frames (step=12 => 5 fps, matching
the test manifest's frame numbering: 0, 12, 24, ...).
"""
import argparse
import json
from pathlib import Path

import cv2


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", type=str, required=True)
    ap.add_argument("--manifest", type=str, required=True,
                    help="recording-level train_manifest_202607.jsonl")
    ap.add_argument("--fps", type=float, default=5.0)
    ap.add_argument("--out", type=str, default="frames/train_frames.jsonl")
    args = ap.parse_args()

    root = Path(args.root)
    step = max(1, round(60.0 / args.fps))  # source is 60 fps

    rows = []
    with open(args.manifest) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            scene = json.loads(line)
            subject = scene["subject_id"]
            scene_id = scene["scene_id"]
            object_alias = scene.get("object_alias")

            video = root / "scenes" / subject / scene_id / "headset0.mp4"
            if not video.exists():
                print(f"WARN: missing {video}")
                continue

            cap = cv2.VideoCapture(str(video))
            n_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            cap.release()

            for fi in range(0, n_frames, step):
                rows.append({
                    "sample_id": f"{subject}/{scene_id}:{fi:06d}",
                    "subject_id": subject,
                    "scene_id": scene_id,
                    "frame_index": fi,
                    "object_alias": object_alias,
                })

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w") as f:
        for r in rows:
            f.write(json.dumps(r) + "\n")

    print(f"wrote {len(rows)} frames to {out}")


if __name__ == "__main__":
    main()
