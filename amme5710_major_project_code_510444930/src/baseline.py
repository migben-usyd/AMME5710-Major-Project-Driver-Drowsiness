from collections import deque
from dataclasses import dataclass, field
from enum import Enum
from typing import Deque, Optional

import numpy as np

from features import FrameFeatures


class AlertLevel(str, Enum):
    ALERT = "ALERT"
    DISTRACTED = "DISTRACTED"       # head turned away / looking down-and-off-road
    DROWSY = "DROWSY"               # elevated PERCLOS / frequent long blinks
    MICROSLEEP = "MICROSLEEP"       # single eye closure beyond critical duration
    NO_FACE = "NO_FACE"


@dataclass
class BaselineConfig:
    ear_calibration_window_s: float = 5.0   # trailing window for the rolling mean/std reference
    ear_relative_margin: float = 1.0        # eye closed if ear_avg is this many std-devs below the window mean
    mar_threshold: float = 0.70          # above this => mouth considered open (yawn)
    yawn_min_duration_s: float = 1.0     # sustained MAR above threshold to count as yawn
    perclos_window_s: float = 30.0       # rolling window for PERCLOS
    perclos_drowsy_threshold: float = 0.15   # PERCLOS >= 15% over window => drowsy
    microsleep_duration_s: float = 1.5   # continuous eye closure => microsleep alert
    pitch_down_threshold: float = 15.0   # degrees, head nodding down
    yaw_away_threshold: float = 25.0     # degrees, head turned away from forward
    distraction_min_duration_s: float = 1.0


@dataclass
class BaselineState:
    alert_level: AlertLevel = AlertLevel.NO_FACE
    perclos: float = 0.0
    eye_closed_duration_s: float = 0.0
    is_yawning: bool = False
    is_head_nodding: bool = False
    is_looking_away: bool = False
    ear_avg: Optional[float] = None
    mar: Optional[float] = None


class BaselineDetector:
    """Stateful, frame-by-frame rule-based detector. Call `update()` once per frame."""

    def __init__(self, fps: float, config: BaselineConfig = None):
        self.fps = fps
        self.cfg = config or BaselineConfig()

        window_len = max(1, int(self.cfg.perclos_window_s * fps))
        self._eye_closed_history: Deque[bool] = deque(maxlen=window_len)

        calib_len = max(1, int(self.cfg.ear_calibration_window_s * fps))
        self._ear_history: Deque[float] = deque(maxlen=calib_len)

        self._consec_closed_frames = 0
        self._consec_yawn_frames = 0
        self._consec_nod_frames = 0
        self._consec_away_frames = 0

    def _frames_to_seconds(self, n_frames: int) -> float:
        return n_frames / self.fps if self.fps > 0 else 0.0

    def update(self, feats: FrameFeatures) -> BaselineState:
        cfg = self.cfg

        if not feats.face_detected:
            # Decay counters when face is lost rather than resetting instantly,
            # so a single dropped frame doesn't erase an in-progress event.
            self._consec_closed_frames = 0
            self._consec_yawn_frames = 0
            self._consec_nod_frames = 0
            self._consec_away_frames = 0
            return BaselineState(alert_level=AlertLevel.NO_FACE)

        if feats.ear_avg is None:
            eye_closed = True
        elif len(self._ear_history) < 5:
            eye_closed = False  # not enough history yet: assume alert rather than flagging closed
        else:
            window_mean = float(np.mean(self._ear_history))
            window_std = max(float(np.std(self._ear_history)), 1e-6)
            eye_closed = (window_mean - feats.ear_avg) / window_std > cfg.ear_relative_margin

        if feats.ear_avg is not None:
            self._ear_history.append(feats.ear_avg)

        self._eye_closed_history.append(eye_closed)
        self._consec_closed_frames = self._consec_closed_frames + 1 if eye_closed else 0

        perclos = (
            sum(self._eye_closed_history) / len(self._eye_closed_history)
            if self._eye_closed_history else 0.0
        )

        mouth_open = feats.mar is not None and feats.mar > cfg.mar_threshold
        self._consec_yawn_frames = self._consec_yawn_frames + 1 if mouth_open else 0
        is_yawning = self._frames_to_seconds(self._consec_yawn_frames) >= cfg.yawn_min_duration_s

        head_nodding_now = feats.pitch is not None and abs(feats.pitch) >= cfg.pitch_down_threshold
        self._consec_nod_frames = self._consec_nod_frames + 1 if head_nodding_now else 0
        is_head_nodding = self._frames_to_seconds(self._consec_nod_frames) >= cfg.distraction_min_duration_s

        looking_away_now = feats.yaw is not None and abs(feats.yaw) >= cfg.yaw_away_threshold
        self._consec_away_frames = self._consec_away_frames + 1 if looking_away_now else 0
        is_looking_away = self._frames_to_seconds(self._consec_away_frames) >= cfg.distraction_min_duration_s

        eye_closed_duration_s = self._frames_to_seconds(self._consec_closed_frames)

        # Priority: microsleep (safety-critical) > drowsy (PERCLOS trend) > distracted > alert
        if eye_closed_duration_s >= cfg.microsleep_duration_s:
            level = AlertLevel.MICROSLEEP
        elif perclos >= cfg.perclos_drowsy_threshold or is_yawning:
            level = AlertLevel.DROWSY
        elif is_head_nodding or is_looking_away:
            level = AlertLevel.DISTRACTED
        else:
            level = AlertLevel.ALERT

        return BaselineState(
            alert_level=level,
            perclos=perclos,
            eye_closed_duration_s=eye_closed_duration_s,
            is_yawning=is_yawning,
            is_head_nodding=is_head_nodding,
            is_looking_away=is_looking_away,
            ear_avg=feats.ear_avg,
            mar=feats.mar,
        )
