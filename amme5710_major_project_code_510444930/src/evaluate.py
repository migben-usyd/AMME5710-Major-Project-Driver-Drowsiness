"""
Turn src/train.py's raw metrics into the comparison table/plots/false-alarm-
rate/latency numbers the report's Experimental Setup and Results sections need.

Two independent pieces:
  1. --metrics results/classifier_metrics.json (from train.py): renders a
     baseline-vs-classifier comparison bar chart and confusion-matrix heatmaps,
     and adds False Alarm Rate (FAR = FP / (FP + TN)) to the printed table --
     a safety-relevant number accuracy alone hides.
  2. --latency-video path/to/clip.mp4: measures real-world per-frame
     processing latency (feature extraction + baseline update) to support
     a real-time feasibility claim in the report.

Usage:
    python src/evaluate.py --metrics results/classifier_metrics.json
    python src/evaluate.py --metrics results/classifier_metrics.json --latency-video data/videos/sample.mp4
"""

import argparse
import json
import os
import sys
import time

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))


def false_alarm_rate(confusion_matrix) -> float:
    (tn, fp), (fn, tp) = confusion_matrix
    return fp / (fp + tn) if (fp + tn) > 0 else 0.0


def plot_confusion(cm, title: str, out_path: str):
    cm = np.array(cm)
    fig, ax = plt.subplots(figsize=(3.2, 3.0))
    im = ax.imshow(cm, cmap="Blues")
    ax.set_xticks([0, 1]); ax.set_yticks([0, 1])
    ax.set_xticklabels(["Alert", "Drowsy"]); ax.set_yticklabels(["Alert", "Drowsy"])
    ax.set_xlabel("Predicted"); ax.set_ylabel("Actual")
    ax.set_title(title, fontsize=10)
    for i in range(2):
        for j in range(2):
            ax.text(j, i, str(cm[i, j]), ha="center", va="center",
                    color="white" if cm[i, j] > cm.max() / 2 else "black")
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def plot_comparison(base: dict, clf: dict, out_path: str):
    metrics = ["accuracy", "precision", "recall", "f1", "far"]
    labels = ["Accuracy", "Precision", "Recall", "F1", "False Alarm Rate"]
    base_vals = [base[m] for m in metrics]
    clf_vals = [clf[m] for m in metrics]

    x = np.arange(len(metrics))
    width = 0.35
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.bar(x - width/2, base_vals, width, label="Rule-based baseline")
    ax.bar(x + width/2, clf_vals, width, label="Learned classifier")
    ax.set_xticks(x); ax.set_xticklabels(labels, rotation=15)
    ax.set_ylim(0, 1.05)
    ax.set_ylabel("Score")
    ax.set_title("Baseline vs. learned classifier")
    ax.legend()
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def run_metrics_report(metrics_path: str, out_dir: str):
    with open(metrics_path) as f:
        payload = json.load(f)

    base = dict(payload["baseline"])
    clf = dict(payload["classifier"])
    base["far"] = false_alarm_rate(base["confusion_matrix"])
    clf["far"] = false_alarm_rate(clf["confusion_matrix"])

    print(f"Windows: {payload['n_windows']} from {payload['n_clips']} clips, "
          f"{payload['n_folds']}-fold grouped CV, model={payload['model']}, "
          f"window={payload['window_sec']}s\n")
    print(f"{'Metric':<20}{'Baseline':>12}{'Classifier':>12}")
    for name, key in [("Accuracy", "accuracy"), ("Precision", "precision"),
                       ("Recall", "recall"), ("F1", "f1"), ("False Alarm Rate", "far")]:
        print(f"{name:<20}{base[key]:>12.3f}{clf[key]:>12.3f}")

    os.makedirs(out_dir, exist_ok=True)
    plot_confusion(base["confusion_matrix"], "Baseline", os.path.join(out_dir, "confusion_baseline.png"))
    plot_confusion(clf["confusion_matrix"], "Classifier", os.path.join(out_dir, "confusion_classifier.png"))
    plot_comparison(base, clf, os.path.join(out_dir, "metrics_comparison.png"))
    print(f"\nSaved confusion_baseline.png, confusion_classifier.png, "
          f"metrics_comparison.png to {out_dir}/")


def run_latency_benchmark(video_path: str, out_dir: str, max_frames: int = 300):
    import cv2
    from features import FaceFeatureExtractor
    from baseline import BaselineDetector

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise SystemExit(f"Could not open video: {video_path}")
    fps_meta = cap.get(cv2.CAP_PROP_FPS) or 30.0

    extractor = FaceFeatureExtractor()
    baseline = BaselineDetector(fps=fps_meta)

    frame_times_ms = []
    n = 0
    while n < max_frames:
        ret, frame = cap.read()
        if not ret:
            break
        t0 = time.perf_counter()
        feats = extractor.process(frame, timestamp_ms=int(n * (1000.0 / fps_meta)))
        baseline.update(feats)
        frame_times_ms.append((time.perf_counter() - t0) * 1000.0)
        n += 1

    extractor.close()
    cap.release()

    arr = np.array(frame_times_ms)
    result = {
        "n_frames": int(n),
        "mean_ms_per_frame": float(arr.mean()),
        "p95_ms_per_frame": float(np.percentile(arr, 95)),
        "max_ms_per_frame": float(arr.max()),
        "achievable_fps": float(1000.0 / arr.mean()) if arr.mean() > 0 else None,
    }
    print(f"\nLatency over {n} frames from {video_path}:")
    for k, v in result.items():
        print(f"  {k}: {v}")

    os.makedirs(out_dir, exist_ok=True)
    with open(os.path.join(out_dir, "latency_benchmark.json"), "w") as f:
        json.dump(result, f, indent=2)
    print(f"Saved latency_benchmark.json to {out_dir}/")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--metrics", help="Path to classifier_metrics.json from train.py")
    ap.add_argument("--latency-video", help="Video file to benchmark per-frame processing latency on")
    ap.add_argument("--out-dir", default="results")
    ap.add_argument("--max-frames", type=int, default=300)
    args = ap.parse_args()

    if not args.metrics and not args.latency_video:
        raise SystemExit("Pass --metrics and/or --latency-video (see --help)")

    if args.metrics:
        run_metrics_report(args.metrics, args.out_dir)
    if args.latency_video:
        run_latency_benchmark(args.latency_video, args.out_dir, args.max_frames)


if __name__ == "__main__":
    main()
