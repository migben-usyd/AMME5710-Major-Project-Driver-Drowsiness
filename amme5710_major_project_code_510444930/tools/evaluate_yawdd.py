"""
External generalisation evaluation on the public YawDD dataset.
"""
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

import numpy as np
import pandas as pd
import joblib

from train import build_windowed_dataset, evaluate

FEATURES_CSV = os.path.join(os.path.dirname(__file__), "..", "data", "yawdd_features.csv")
MODEL_PATH = os.path.join(os.path.dirname(__file__), "..", "results", "model.joblib")
USABLE_DETECTION_THRESHOLD = 0.40  # min per-clip face-detection rate to count as "usable"


def main():
    df = pd.read_csv(FEATURES_CSV)

    # Report per-clip face detection rate -- the key generalisation finding:
    # Haar-cascade detection, tuned implicitly on our own recording setup,
    # does not transfer uniformly to YawDD's different camera angle/lighting.
    print("=== Per-clip face detection rate (YawDD) ===")
    det = df.groupby("clip_id")["face_detected"].mean().sort_values()
    for cid, rate in det.items():
        print(f"  {cid}: {rate * 100:.1f}%")
    print(f"\nOverall mean detection rate across {det.shape[0]} YawDD clips: "
          f"{df['face_detected'].mean() * 100:.1f}%")
    print("(compare: our own 14 clips ranged ~37%-100%, mean ~90%+)\n")

    X, y, groups, baseline_preds = build_windowed_dataset(df, window_sec=1.5)
    print(f"Built {len(X)} windowed samples from {len(set(groups))} YawDD clips "
          f"({int(y.sum())} yawning windows, {len(y) - int(y.sum())} non-yawning windows)\n")

    model = joblib.load(MODEL_PATH)
    clf_preds = model["pipeline"].predict(X)

    evaluate(y, baseline_preds, "Baseline on YawDD (external, all clips, no retraining)")
    evaluate(y, clf_preds, "Classifier (trained only on our 14 clips) on YawDD (external, all clips)")

    # Restrict to clips with usable face detection for a fairer end-to-end
    # read of detector performance once the pipeline can actually see a face.
    usable_clips = set(det[det >= USABLE_DETECTION_THRESHOLD].index)
    mask = np.array([g in usable_clips for g in groups])
    print(f"\n--- Restricted to {len(usable_clips)} clips with "
          f">={USABLE_DETECTION_THRESHOLD * 100:.0f}% face detection ({mask.sum()} windows) ---")
    if mask.sum() > 0:
        evaluate(y[mask], baseline_preds[mask], "Baseline on YawDD (usable-detection subset)")
        evaluate(y[mask], clf_preds[mask], "Classifier on YawDD (usable-detection subset)")


if __name__ == "__main__":
    main()
