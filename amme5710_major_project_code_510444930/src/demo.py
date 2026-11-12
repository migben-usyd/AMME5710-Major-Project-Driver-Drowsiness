"""
Run the full pipeline (feature extraction -> rule-based baseline, optionally
-> learned classifier) on a video and write an annotated output video plus a
per-frame CSV log, for qualitative figures in the report and for sanity-
checking the pipeline visually.

Usage:
    python src/demo.py --video data/videos/drowsy/drowsy_00.mp4 --out results/demo_drowsy_00.mp4
    python src/demo.py --video data/videos/alert/alert_00.mp4 --out results/demo_alert_00.mp4 \
        --model results/model.joblib
"""

import argparse
import os
import sys

import cv2
import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from features import FaceFeatureExtractor, FrameFeatures
from baseline import BaselineDetector, BaselineConfig, AlertLevel
from windowing import WindowFeaturizer

LEVEL_COLORS = {
    AlertLevel.ALERT: (60, 200, 60),
    AlertLevel.DISTRACTED: (0, 165, 255),
    AlertLevel.DROWSY: (0, 90, 255),
    AlertLevel.MICROSLEEP: (0, 0, 255),
    AlertLevel.NO_FACE: (128, 128, 128),
}


def draw_overlay(frame, feats: FrameFeatures, alert_level: AlertLevel, clf_proba=None):
    out = frame.copy()
    color = LEVEL_COLORS.get(alert_level, (255, 255, 255))

    if feats.face_detected and feats.face_box is not None:
        x, y, w, h = feats.face_box
        cv2.rectangle(out, (x, y), (x + w, y + h), color, 2)

        for eye_box in (feats.left_eye_box, feats.right_eye_box):
            if eye_box is not None:
                ex, ey, ew, eh = eye_box
                cv2.rectangle(out, (ex, ey), (ex + ew, ey + eh), (255, 200, 0), 1)

        for pupil in (feats.left_pupil, feats.right_pupil):
            if pupil is not None:
                cx, cy, r = pupil
                cv2.circle(out, (int(cx), int(cy)), int(r), (0, 255, 255), 1)

        if feats.mouth_box is not None:
            mx, my, mw, mh = feats.mouth_box
            cv2.rectangle(out, (mx, my), (mx + mw, my + mh), (255, 0, 255), 1)

        # Our own landmark-style corner points (eye inner/outer corners,
        # mouth left/right corners) -- drawn as filled dots distinct from
        # the ROI boxes above, to make clear these are localised points,
        # not just detector bounding boxes.
        for corners in (feats.left_eye_corners, feats.right_eye_corners, feats.mouth_corners):
            if corners is not None:
                for (cx, cy) in corners:
                    cv2.circle(out, (int(cx), int(cy)), 3, (0, 0, 255), -1)
                    cv2.circle(out, (int(cx), int(cy)), 3, (255, 255, 255), 1)

        if feats.pose_points is not None:
            for px, py in feats.pose_points:
                cv2.circle(out, (int(px), int(py)), 2, (0, 255, 0), -1)

    # Text panel
    lines = [f"STATE: {alert_level.name}"]
    if feats.ear_avg is not None:
        lines.append(f"EAR: {feats.ear_avg:.3f}")
    if feats.mar is not None:
        lines.append(f"MAR: {feats.mar:.3f}")
    if feats.pitch is not None and feats.yaw is not None:
        lines.append(f"pitch: {feats.pitch:.1f}  yaw: {feats.yaw:.1f}")
    if clf_proba is not None:
        lines.append(f"P(drowsy): {clf_proba:.2f}")

    panel_h = 24 * len(lines) + 10
    cv2.rectangle(out, (0, 0), (260, panel_h), (0, 0, 0), -1)
    for i, line in enumerate(lines):
        cv2.putText(out, line, (8, 22 + 24 * i), cv2.FONT_HERSHEY_SIMPLEX, 0.55, color, 1, cv2.LINE_AA)

    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--video", required=True)
    ap.add_argument("--out", default="results/demo_output.mp4")
    ap.add_argument("--csv-out", default=None, help="Optional path for the per-frame log CSV")
    ap.add_argument("--model", default=None, help="Optional path to a trained model.joblib for live classifier output")
    ap.add_argument("--window-sec", type=float, default=1.5)
    args = ap.parse_args()

    cap = cv2.VideoCapture(args.video)
    if not cap.isOpened():
        raise SystemExit(f"Could not open video: {args.video}")

    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(args.out, fourcc, fps, (width, height))

    extractor = FaceFeatureExtractor()
    baseline = BaselineDetector(fps=fps, config=BaselineConfig())

    clf = None
    featurizer = None
    if args.model:
        from classifier import DrowsyClassifier
        clf = DrowsyClassifier.load(args.model)
        window_frames = max(1, int(round(args.window_sec * fps)))
        featurizer = WindowFeaturizer(window_size_frames=window_frames)

    rows = []
    frame_idx = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        timestamp_ms = int(frame_idx * (1000.0 / fps))
        feats = extractor.process(frame, timestamp_ms=timestamp_ms)
        state = baseline.update(feats)

        clf_proba = None
        if clf is not None and featurizer is not None:
            vec = featurizer.push(feats)
            if vec is not None:
                clf_proba = clf.predict_proba(vec)

        annotated = draw_overlay(frame, feats, state.alert_level, clf_proba)
        writer.write(annotated)

        rows.append({
            "frame_idx": frame_idx,
            "timestamp_ms": timestamp_ms,
            "face_detected": feats.face_detected,
            "ear_avg": feats.ear_avg,
            "mar": feats.mar,
            "pitch": feats.pitch,
            "yaw": feats.yaw,
            "roll": feats.roll,
            "alert_level": state.alert_level.name,
            "clf_proba_drowsy": clf_proba,
        })
        frame_idx += 1

    extractor.close()
    cap.release()
    writer.release()

    print(f"Processed {frame_idx} frames -> {args.out}")

    csv_out = args.csv_out or (os.path.splitext(args.out)[0] + "_log.csv")
    pd.DataFrame(rows).to_csv(csv_out, index=False)
    print(f"Wrote per-frame log to {csv_out}")

    levels, counts = np.unique([r["alert_level"] for r in rows], return_counts=True)
    print("\nAlert-level distribution over this clip:")
    for lvl, cnt in zip(levels, counts):
        print(f"  {lvl}: {cnt} frames ({100.0 * cnt / frame_idx:.1f}%)")


if __name__ == "__main__":
    main()
