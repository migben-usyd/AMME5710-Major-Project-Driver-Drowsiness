"""
Train the learned drowsiness classifier from a per-frame, labeled CSV.

Expected input CSV columns (produced by extract_features_from_video.py, one row
per processed frame, concatenated across all clips you've labeled):
    clip_id, frame_idx, timestamp_ms, face_detected,
    ear_left, ear_right, ear_avg, mar, pitch, yaw, roll, label, fps

`label` is per-frame or per-clip ground truth: 0 = alert, 1 = drowsy.
`fps` is the source video's frame rate (used to size the rolling window).

    python src/train.py --csv data/features_labeled.csv --model svm --window-sec 1.5
"""

import argparse
import json
import os
import sys

import numpy as np
import pandas as pd
from sklearn.model_selection import GroupKFold
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score, confusion_matrix,
)

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from features import FrameFeatures
from windowing import WindowFeaturizer, FEATURE_NAMES
from classifier import build_model, DrowsyClassifier, LABEL_ALERT, LABEL_DROWSY
from baseline import BaselineDetector, BaselineConfig, AlertLevel


def build_windowed_dataset(df: pd.DataFrame, window_sec: float):
    X, y, groups = [], [], []
    baseline_preds = []

    for clip_id, clip_df in df.groupby("clip_id"):
        clip_df = clip_df.sort_values("frame_idx")
        fps = float(clip_df["fps"].iloc[0])
        window_frames = max(1, int(round(window_sec * fps)))

        featurizer = WindowFeaturizer(window_size_frames=window_frames)
        baseline = BaselineDetector(fps=fps, config=BaselineConfig())

        def _none_if_nan(v):
            # CSV round-tripping turns a missing (None) feature into NaN,
            # not None -- normalize back so "is None" checks in baseline.py
            # and the NaN-aware aggregation in windowing.py both see the
            # same "not available" value they'd see from a live extractor.
            return None if (v is None or (isinstance(v, float) and np.isnan(v))) else v

        label_window = []
        for _, row in clip_df.iterrows():
            feats = FrameFeatures(
                face_detected=bool(row["face_detected"]),
                ear_left=_none_if_nan(row.get("ear_left")), ear_right=_none_if_nan(row.get("ear_right")),
                ear_avg=_none_if_nan(row.get("ear_avg")), mar=_none_if_nan(row.get("mar")),
                pitch=_none_if_nan(row.get("pitch")), yaw=_none_if_nan(row.get("yaw")),
                roll=_none_if_nan(row.get("roll")),
            )
            vec = featurizer.push(feats)
            state = baseline.update(feats)
            label_window.append(row["label"])

            if vec is not None:
                majority_label = int(round(np.mean(label_window[-window_frames:])))
                X.append(vec)
                y.append(majority_label)
                groups.append(clip_id)
                baseline_preds.append(
                    1 if state.alert_level in (AlertLevel.DROWSY, AlertLevel.MICROSLEEP) else 0
                )

    return np.array(X), np.array(y), np.array(groups), np.array(baseline_preds)


def evaluate(y_true, y_pred, name: str) -> dict:
    metrics = {
        "accuracy": accuracy_score(y_true, y_pred),
        "precision": precision_score(y_true, y_pred, zero_division=0),
        "recall": recall_score(y_true, y_pred, zero_division=0),
        "f1": f1_score(y_true, y_pred, zero_division=0),
        "confusion_matrix": confusion_matrix(y_true, y_pred).tolist(),
    }
    print(f"\n=== {name} ===")
    for k, v in metrics.items():
        if k != "confusion_matrix":
            print(f"  {k}: {v:.4f}")
    print(f"  confusion_matrix [[TN, FP], [FN, TP]]: {metrics['confusion_matrix']}")
    return metrics


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--csv", required=True, help="Path to labeled per-frame feature CSV")
    ap.add_argument("--model", choices=["svm", "mlp"], default="svm")
    ap.add_argument("--window-sec", type=float, default=1.5, help="Rolling window length in seconds")
    ap.add_argument("--folds", type=int, default=5, help="Number of GroupKFold splits (by clip)")
    ap.add_argument("--out-model", default="results/model.joblib")
    ap.add_argument("--out-metrics", default="results/classifier_metrics.json")
    args = ap.parse_args()

    df = pd.read_csv(args.csv)
    required_cols = {"clip_id", "frame_idx", "face_detected", "ear_avg", "mar", "pitch", "yaw", "label", "fps"}
    missing = required_cols - set(df.columns)
    if missing:
        raise SystemExit(f"CSV is missing required columns: {sorted(missing)}")

    X, y, groups, baseline_preds = build_windowed_dataset(df, args.window_sec)
    n_clips = len(set(groups))
    print(f"Built {len(X)} windowed samples from {n_clips} clips "
          f"({int(y.sum())} drowsy windows, {int(len(y) - y.sum())} alert windows).")

    n_splits = min(args.folds, n_clips) if n_clips >= 2 else 1
    if n_splits < 2:
        raise SystemExit(
            "Need at least 2 distinct clips (clip_id values) for a clip-grouped "
            "train/test split. Label more clips before training."
        )

    gkf = GroupKFold(n_splits=n_splits)
    all_true, all_pred_clf, all_pred_base = [], [], []

    for fold, (train_idx, test_idx) in enumerate(gkf.split(X, y, groups)):
        model = build_model(args.model)
        model.fit(X[train_idx], y[train_idx])
        pred = model.predict(X[test_idx])
        all_true.extend(y[test_idx])
        all_pred_clf.extend(pred)
        all_pred_base.extend(baseline_preds[test_idx])
        print(f"Fold {fold+1}/{n_splits}: test clips = "
              f"{sorted(set(groups[test_idx]))}, n={len(test_idx)}")

    clf_metrics = evaluate(all_true, all_pred_clf, f"Learned classifier ({args.model}, {n_splits}-fold CV)")
    base_metrics = evaluate(all_true, all_pred_base, "Rule-based baseline (same windows)")

    # Refit on all data for the deployable model
    final_model = build_model(args.model)
    final_model.fit(X, y)
    clf = DrowsyClassifier(pipeline=final_model, feature_names=FEATURE_NAMES)
    os.makedirs(os.path.dirname(args.out_model) or ".", exist_ok=True)
    clf.save(args.out_model)
    print(f"\nSaved final model (trained on all {len(X)} windows) to {args.out_model}")

    with open(args.out_metrics, "w") as f:
        json.dump({
            "model": args.model,
            "window_sec": args.window_sec,
            "n_windows": len(X),
            "n_clips": n_clips,
            "n_folds": n_splits,
            "classifier": clf_metrics,
            "baseline": base_metrics,
        }, f, indent=2)
    print(f"Saved metrics to {args.out_metrics}")


if __name__ == "__main__":
    main()
