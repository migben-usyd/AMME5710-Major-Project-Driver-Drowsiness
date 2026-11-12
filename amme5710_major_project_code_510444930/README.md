# Driver Drowsiness & Attention Monitoring (AMME5710 Major Project)

Classical computer-vision pipeline for detecting driver drowsiness and
distraction from webcam video

## Pipeline

1. **Face localisation** -- Haar-cascade boosted classifier over Haar-like
   features via integral images
2. **Eye localisation** -- geoemetric with no second detector: a
   fixed band of the face box is taken as each eye's ROI.
3. **Eye-corner landmarks** -- Canny edges within each eye ROI, projected
   onto the horizontal axis, give inner/outer corner points.
4. **Pupil/iris detection** -- Circular Hough Transform (`cv2.HoughCircles`)
   on each eye ROI.
5. **Eye openness** -- standard deviation of pixel intensity within the
   (blurred) eye ROI
6. **Mouth localisation + corners + openness** -- CIELAB colour space,
   Otsu threshold on the $a^*$ channel to segment lip tissue, then a second
   Otsu threshold on brightness for the open-mouth cavity (MAR-equivalent).
7. **Head pose** -- a closed-form geometric formula (roll/yaw/pitch) over
   the self-detected eye/mouth corner points, cross-checked against
   `cv2.solvePnP` fit to the same points.

## Usage

```bash
# 1. Record labeled clips (needs a real webcam; run locally)
python tools/record_clips.py

# 2. Extract features from each clip into one CSV
python src/extract_features_from_video.py --video data/videos/alert/alert_00.mp4 \
    --clip-id alert_00 --label 0 --out data/features_labeled.csv
python src/extract_features_from_video.py --video data/videos/drowsy/drowsy_00.mp4 \
    --clip-id drowsy_00 --label 1 --out data/features_labeled.csv --append
# ... repeat for all clips (--append after the first) ...

# 3. Train + cross-validate the classifier, compare against the baseline
python src/train.py --csv data/features_labeled.csv --model mlp

# 4. Generate comparison plots / false-alarm-rate table / latency benchmark
python src/evaluate.py --metrics results/classifier_metrics.json \
    --latency-video data/videos/alert/alert_00.mp4

# 5. Annotated demo video for qualitative figures
python src/demo.py --video data/videos/drowsy/drowsy_00.mp4 \
    --out results/demo_drowsy_00.mp4 --model results/model.joblib

# 6. External generalisation test on the public YawDD dataset (no retraining)
#    -- edit the SRC path at the top of the script to point at your local
#    copy of YawDD before running.
bash tools/extract_yawdd_features.sh
python tools/evaluate_yawdd.py
```

`data/features_labeled.csv` and `data/yawdd_features.csv` (the outputs of
steps 2 and 6) and `results/model.joblib` (the output of step 3) are
included in this package already, so steps 3-6 can be re-run directly
without re-extracting features from the raw video.