"""
Guided webcam recording session for the labeled dataset

Walks through a fixed shot list of 14 short clips with on-screen prompts and
a countdown, saving each to data/videos/<label>/<clip_id>.mp4.

Usage:
    pip install opencv-python
    python tools/record_clips.py
"""

import os
import time

import cv2

OUT_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data", "videos")
CLIP_SECONDS = 12
COUNTDOWN_SECONDS = 3
FPS = 30

# (clip_id, folder/label, on-screen instruction)
SHOT_LIST = [
    ("alert_00", "alert", "Normal sitting. Relaxed, attentive posture."),
    ("alert_01", "alert", "Normal sitting. Slight head movement."),
    ("alert_02", "alert", "Normal sitting. Dimmer lighting."),
    ("alert_03", "alert", "Normal sitting. Natural talking / blinking."),
    ("alert_04", "alert", "Normal sitting. Slight left/right head turn."),
    ("alert_05", "alert", "Normal sitting. Normal attentive baseline."),
    ("distracted_00", "alert", "Looking away to side and back."),
    ("distracted_01", "alert", "Head down repeated (phone checking)"),
    ("drowsy_00", "drowsy", "Closing eyes slowly for ~3 seconds and repeat."),
    ("drowsy_01", "drowsy", "Yawning: mouth opening wide repeat"),
    ("drowsy_02", "drowsy", "Slow heavy blinking"),
    ("drowsy_03", "drowsy", "Microsleep: Eyes closed for seconds and head drooping."),
    ("drowsy_04", "drowsy", "Slow blinks, yawning & eyes closed."),
    ("drowsy_05", "drowsy", "Slow blinks, head fighting sleep."),
]


def record_clip(cap, out_path: str, instruction: str, fps: float):
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(out_path, fourcc, fps, (width, height))

    # Countdown
    t0 = time.time()
    while time.time() - t0 < COUNTDOWN_SECONDS:
        ret, frame = cap.read()
        if not ret:
            continue
        remaining = COUNTDOWN_SECONDS - (time.time() - t0)
        disp = frame.copy()
        cv2.putText(disp, instruction, (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
        cv2.putText(disp, f"Starting in {remaining:.1f}s...", (20, 80), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)
        cv2.imshow("record_clips", disp)
        if cv2.waitKey(1) == 27:
            return False

    # Record
    t0 = time.time()
    while time.time() - t0 < CLIP_SECONDS:
        ret, frame = cap.read()
        if not ret:
            continue
        writer.write(frame)
        remaining = CLIP_SECONDS - (time.time() - t0)
        disp = frame.copy()
        cv2.putText(disp, instruction, (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        cv2.putText(disp, f"REC  {remaining:.1f}s left", (20, 80), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)
        cv2.imshow("record_clips", disp)
        if cv2.waitKey(1) == 27:
            break

    writer.release()
    return True


def main():
    for _, label, _ in SHOT_LIST:
        os.makedirs(os.path.join(OUT_DIR, label), exist_ok=True)

    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        raise SystemExit("Could not open webcam")
    cap.set(cv2.CAP_PROP_FPS, FPS)

    print(f"Recording {len(SHOT_LIST)} clips, {CLIP_SECONDS}s each. Press ESC in the preview window to skip/stop.\n")

    for clip_id, label, instruction in SHOT_LIST:
        out_path = os.path.join(OUT_DIR, label, f"{clip_id}.mp4")
        print(f"[{clip_id}] ({label}) {instruction}")
        input("  Press Enter when ready...")
        ok = record_clip(cap, out_path, instruction, cap.get(cv2.CAP_PROP_FPS) or FPS)
        print(f"  Saved -> {out_path}" if ok else "  Skipped/stopped early.")

    cap.release()
    cv2.destroyAllWindows()
    print("\nDone. Clips are under data/videos/alert/ and data/videos/drowsy/.")


if __name__ == "__main__":
    main()
