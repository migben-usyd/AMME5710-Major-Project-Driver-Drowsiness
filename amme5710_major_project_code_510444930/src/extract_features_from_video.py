"""
Extract per-frame EAR/MAR/head-pose features from a video file and write them
to a CSV in the format src/train.py expects

Used both to build the labeled training/evaluation set (point it at each
labeled clip, pass --label and a unique --clip-id, append to the same CSV) and
to turn benchmark-dataset videos (NTHU-DDD, YawDD, or your own recordings)
into the common feature format the rest of the pipeline works with.

Usage:
    python src/extract_features_from_video.py --video data/videos/alert/clip01.mp4 \
        --clip-id alert_clip01 --label 0 --out data/features_labeled.csv --append
"""

import argparse
import os
import sys

import cv2
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from features import FaceFeatureExtractor


def process_video(video_path: str, clip_id: str, label: int):
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise SystemExit(f"Could not open video: {video_path}")

    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    extractor = FaceFeatureExtractor()

    rows = []
    frame_idx = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        timestamp_ms = int(frame_idx * (1000.0 / fps))
        feats = extractor.process(frame, timestamp_ms=timestamp_ms)
        rows.append({
            "clip_id": clip_id,
            "frame_idx": frame_idx,
            "timestamp_ms": timestamp_ms,
            "face_detected": feats.face_detected,
            "ear_left": feats.ear_left,
            "ear_right": feats.ear_right,
            "ear_avg": feats.ear_avg,
            "mar": feats.mar,
            "pitch": feats.pitch,
            "yaw": feats.yaw,
            "roll": feats.roll,
            "pitch_pnp": feats.pitch_pnp,
            "yaw_pnp": feats.yaw_pnp,
            "roll_pnp": feats.roll_pnp,
            "face_skin_overlap": feats.face_skin_overlap,
            "label": label,
            "fps": fps,
        })
        frame_idx += 1

    extractor.close()
    cap.release()
    return pd.DataFrame(rows)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--video", required=True, help="Path to input video file")
    ap.add_argument("--clip-id", required=True, help="Unique id for this clip (used for grouped CV splits)")
    ap.add_argument("--label", type=int, required=True, choices=[0, 1],
                     help="0 = alert clip, 1 = drowsy clip (per-frame; use the dataset's own labels if finer-grained)")
    ap.add_argument("--out", default="data/features_labeled.csv")
    ap.add_argument("--append", action="store_true", help="Append to --out if it already exists")
    args = ap.parse_args()

    df = process_video(args.video, args.clip_id, args.label)
    n_detected = int(df["face_detected"].sum())
    print(f"Processed {len(df)} frames from {args.video}: face detected in {n_detected} "
          f"({100.0 * n_detected / max(1, len(df)):.1f}%)")

    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    if args.append and os.path.exists(args.out):
        df.to_csv(args.out, mode="a", header=False, index=False)
    else:
        df.to_csv(args.out, index=False)
    print(f"Wrote features to {args.out}")


if __name__ == "__main__":
    main()
