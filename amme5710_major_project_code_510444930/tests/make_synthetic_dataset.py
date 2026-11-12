"""
Generate a small synthetic sanity-check dataset of fake "face" images with
controllable eye/mouth openness, for unit-testing src/features.py's
isolated eye/mouth-openness functions
"""

import os
import sys

import cv2
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "src"))
from features import _eye_openness_and_pupil, _mouth_openness

OUT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "synthetic_crops")


def make_eye_crop(open_amount: float, size=(60, 40)) -> np.ndarray:
    """open_amount in [0, 1]: 0 = fully closed (thin dark line), 1 = fully open
    (large dark circle/ellipse for pupil+iris)."""
    w, h = size
    img = np.full((h, w), 200, dtype=np.uint8)  # bright skin-ish background
    cy = h // 2
    half_height = max(1, int(open_amount * h * 0.4))
    cv2.ellipse(img, (w // 2, cy), (w // 4, half_height), 0, 0, 360, 40, -1)
    return img


def make_mouth_crop(open_amount: float, size=(80, 50)) -> np.ndarray:
    w, h = size
    img = np.full((h, w), 180, dtype=np.uint8)
    cy = h // 2
    half_height = max(1, int(open_amount * h * 0.45))
    cv2.ellipse(img, (w // 2, cy), (w // 3, half_height), 0, 0, 360, 30, -1)
    return img


def main():
    os.makedirs(OUT_DIR, exist_ok=True)

    print("Eye openness ratio vs. synthetic open_amount:")
    for amt in [0.0, 0.2, 0.5, 0.8, 1.0]:
        crop = make_eye_crop(amt)
        cv2.imwrite(os.path.join(OUT_DIR, f"eye_{amt:.1f}.png"), crop)
        box = (0, 0, crop.shape[1], crop.shape[0])
        _, ratio = _eye_openness_and_pupil(crop, box)
        print(f"  open_amount={amt:.1f} -> measured openness_ratio={ratio:.3f}")

    print("\nMouth openness ratio vs. synthetic open_amount:")
    for amt in [0.0, 0.2, 0.5, 0.8, 1.0]:
        crop = make_mouth_crop(amt)
        cv2.imwrite(os.path.join(OUT_DIR, f"mouth_{amt:.1f}.png"), crop)
        box = (0, 0, crop.shape[1], crop.shape[0])
        ratio = _mouth_openness(crop, box)
        print(f"  open_amount={amt:.1f} -> measured openness_ratio={ratio:.3f}")

    print(f"\nSaved crop images to {OUT_DIR}/ for visual inspection.")


if __name__ == "__main__":
    main()
