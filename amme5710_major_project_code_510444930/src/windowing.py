"""
Sliding-window feature aggregation shared by both the baseline detector context
and the learned classifier.

Per-frame features (EAR, MAR, pitch, yaw) are noisy; classifier works on
summary statistics over a short rolling window (e.g. 1-2 seconds of frames)
"""

from collections import deque
from dataclasses import dataclass
from typing import Deque, List, Optional

import numpy as np

from features import FrameFeatures

#Keep train.py and classifier.py in sync with this list
FEATURE_NAMES = [
    "ear_mean", "ear_std", "ear_min",
    "mar_mean", "mar_max",
    "perclos",
    "pitch_mean", "pitch_std",
    "yaw_mean", "yaw_std",
]


@dataclass
class WindowFeaturizer:
    window_size_frames: int
    ear_closed_threshold: float = 23.0

    def __post_init__(self):
        self._ear: Deque[float] = deque(maxlen=self.window_size_frames)
        self._mar: Deque[float] = deque(maxlen=self.window_size_frames)
        self._pitch: Deque[float] = deque(maxlen=self.window_size_frames)
        self._yaw: Deque[float] = deque(maxlen=self.window_size_frames)

    def reset(self):
        self._ear.clear()
        self._mar.clear()
        self._pitch.clear()
        self._yaw.clear()

    def push(self, feats: FrameFeatures) -> Optional[np.ndarray]:
        """Add one frame's features. Returns the window feature vector once the
        window is full"""
        if feats.face_detected:
            self._ear.append(feats.ear_avg if feats.ear_avg is not None else np.nan)
            self._mar.append(feats.mar if feats.mar is not None else np.nan)
            self._pitch.append(feats.pitch if feats.pitch is not None else np.nan)
            self._yaw.append(feats.yaw if feats.yaw is not None else np.nan)

        if len(self._ear) < self.window_size_frames:
            return None

        return self.current_vector()

    def current_vector(self) -> Optional[np.ndarray]:
        if not self._ear:
            return None
        ear = np.array(self._ear, dtype=float)
        mar = np.array(self._mar, dtype=float)
        pitch = np.array(self._pitch, dtype=float)
        yaw = np.array(self._yaw, dtype=float)

        # A NaN here means no face/eye ROI was available that frame
        eye_closed_mask = np.isnan(ear) | (ear < self.ear_closed_threshold)
        perclos = float(np.mean(eye_closed_mask))

        vec = np.array([
            np.nanmean(ear), np.nanstd(ear), np.nanmin(ear),
            np.nanmean(mar), np.nanmax(mar),
            perclos,
            np.nanmean(pitch), np.nanstd(pitch),
            np.nanmean(yaw), np.nanstd(yaw),
        ], dtype=float)
        return np.nan_to_num(vec, nan=0.0)

    def is_full(self) -> bool:
        return len(self._ear) >= self.window_size_frames
