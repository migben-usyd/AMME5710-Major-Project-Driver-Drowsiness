"""
Classical, self-built facial feature extraction for driver drowsiness monitoring.
"""

import os
from dataclasses import dataclass
from typing import Optional, Tuple

import cv2
import numpy as np

DATA_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data")
FACE_CASCADE_PATH = os.path.join(DATA_DIR, "haarcascade_frontalface_default.xml")

Box = Tuple[int, int, int, int]  # (x, y, w, h) in full-frame pixel coordinates
Point = Tuple[float, float]

# Generic 3D face model for the solvePnP cross-check
MODEL_POINTS_3D = np.array([
    (0.0, 0.0, 0.0),          # nose tip
    (0.0, -330.0, -65.0),     # chin
    (-225.0, 170.0, -135.0),  # left eye centre (subject's left, image right)
    (225.0, 170.0, -135.0),   # right eye centre
    (-150.0, -150.0, -125.0), # left mouth corner
    (150.0, -150.0, -125.0),  # right mouth corner
], dtype=np.float64)

# Calibrated from this project's own alert-frame data: the typical
# (eye-line-to-mouth-line distance) / (interocular distance) ratio for a
# forward-facing, neutral head pose on this camera setup.

NEUTRAL_EYE_MOUTH_RATIO = 0.653
PITCH_SCALE_DEG = 60.0   # maps ratio deviation to a degree-like scale
YAW_SCALE_DEG = 90.0     # maps symmetry ratio [-1, 1] to degrees


@dataclass
class FrameFeatures:
    face_detected: bool
    ear_left: Optional[float] = None
    ear_right: Optional[float] = None
    ear_avg: Optional[float] = None
    mar: Optional[float] = None
    # Primary (geometric formula) head pose -- used by baseline/classifier.
    pitch: Optional[float] = None
    yaw: Optional[float] = None
    roll: Optional[float] = None
    # Secondary cross-check head pose (solvePnP), kept only for validating
    # the geometric estimate above -- not used by baseline/classifier.
    pitch_pnp: Optional[float] = None
    yaw_pnp: Optional[float] = None
    roll_pnp: Optional[float] = None
    # Geometry kept for drawing / debugging / report figures, all in
    # full-frame pixel coordinates.
    face_box: Optional[Box] = None
    face_skin_overlap: Optional[float] = None  # IoU of cascade box vs skin-colour blob
    left_eye_box: Optional[Box] = None
    right_eye_box: Optional[Box] = None
    left_eye_corners: Optional[Tuple[Point, Point]] = None   # (inner, outer)
    right_eye_corners: Optional[Tuple[Point, Point]] = None
    left_pupil: Optional[Tuple[int, int, int]] = None   # (cx, cy, r) or None
    right_pupil: Optional[Tuple[int, int, int]] = None
    mouth_box: Optional[Box] = None
    mouth_corners: Optional[Tuple[Point, Point]] = None  # (left, right)
    pose_points: Optional[np.ndarray] = None  # (6, 2) points fed to solvePnP


def _box_area(b: Box) -> int:
    return b[2] * b[3]


def _iou(a: Box, b: Box) -> float:
    ax, ay, aw, ah = a
    bx, by, bw, bh = b
    ix0, iy0 = max(ax, bx), max(ay, by)
    ix1, iy1 = min(ax + aw, bx + bw), min(ay + ah, by + bh)
    iw, ih = max(0, ix1 - ix0), max(0, iy1 - iy0)
    inter = iw * ih
    union = aw * ah + bw * bh - inter
    return inter / union if union > 0 else 0.0


def _skin_face_box(color_frame: np.ndarray) -> Optional[Box]:
    """Classical skin-colour segmentation (YCbCr thresholding), largest
    contour, as a cross-check against the Haar-cascade face box."""
    ycrcb = cv2.cvtColor(color_frame, cv2.COLOR_BGR2YCrCb)
    lower = np.array([0, 133, 77], dtype=np.uint8)
    upper = np.array([255, 173, 127], dtype=np.uint8)
    mask = cv2.inRange(ycrcb, lower, upper)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, np.ones((5, 5), np.uint8))
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, np.ones((9, 9), np.uint8))
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return None
    largest = max(contours, key=cv2.contourArea)
    if cv2.contourArea(largest) < 0.01 * color_frame.shape[0] * color_frame.shape[1]:
        return None
    x, y, w, h = cv2.boundingRect(largest)
    return (int(x), int(y), int(w), int(h))


def _eye_roi_from_face(face_box: Box, side: str) -> Box:
    """Geometric ROI heuristic (no eye detector): eyes sit in the upper-
    middle band of the face box, left half / right half by image position."""
    x, y, w, h = face_box
    # Narrower vertical band than a naive "upper-middle" guess: wider bands
    # pull in the eyebrow, whose own dark edge response barely changes with
    # eyelid state and swamps the eyelid-closure signal we actually want
    # (see Discussion -- found by inspecting ROI crops directly).
    y0, y1 = y + int(0.30 * h), y + int(0.47 * h)
    if side == "left":   # image-left half
        x0, x1 = x + int(0.08 * w), x + int(0.50 * w)
    else:                # image-right half
        x0, x1 = x + int(0.50 * w), x + int(0.92 * w)
    return (x0, y0, x1 - x0, y1 - y0)


def _eye_openness_pupil_and_corners(gray_frame: np.ndarray, eye_box: Box):
    """Eye corners from Canny edges + a horizontal intensity-projection
    profile; eye-openness from the ROI's own intensity variance.

    Returns (pupil (cx, cy, r) or None, openness_index, (inner_corner,
    outer_corner) in full-frame coords or None).
    """
    x, y, w, h = eye_box
    if w < 8 or h < 6:
        return None, 0.0, None
    # gray_frame is already globally contrast-equalised by the caller; a
    # second, local equalizeHist here was found (in the same diagnostic
    # process) to distort the ROI statistics this openness measure depends
    # on, so the ROI is used as-is beyond blurring.
    roi = gray_frame[y:y + h, x:x + w]
    blurred = cv2.GaussianBlur(roi, (5, 5), 0)

    openness_ratio = float(blurred.std())

    edges = cv2.Canny(blurred, 40, 120)

    row_profile = edges.sum(axis=1).astype(np.float64)
    row_thresh = 0.15 * row_profile.max() if row_profile.max() > 0 else 0
    rows_active = np.where(row_profile > row_thresh)[0]

    # Horizontal extent -> eye corners: columns with significant edge density.
    col_profile = edges.sum(axis=0).astype(np.float64)
    col_thresh = 0.15 * col_profile.max() if col_profile.max() > 0 else 0
    cols_active = np.where(col_profile > col_thresh)[0]
    corners = None
    if len(cols_active) >= 2:
        left_col, right_col = int(cols_active.min()), int(cols_active.max())
        mid_row = int(np.median(rows_active)) if len(rows_active) > 0 else h // 2
        corners = ((x + left_col, y + mid_row), (x + right_col, y + mid_row))

    # Pupil via Circular Hough Transform (unchanged from the proposal).
    pupil = None
    circles = cv2.HoughCircles(
        blurred, cv2.HOUGH_GRADIENT, dp=1, minDist=max(w, h),
        param1=50, param2=15, minRadius=max(2, w // 8), maxRadius=max(3, w // 2),
    )
    if circles is not None and len(circles[0]) > 0:
        cands = np.round(circles[0, :]).astype(int)
        cx0, cy0 = w / 2.0, h / 2.0
        best = min(cands, key=lambda c: (c[0] - cx0) ** 2 + (c[1] - cy0) ** 2)
        pupil = (x + int(best[0]), y + int(best[1]), int(best[2]))

    return pupil, openness_ratio, corners


def _mouth_box_and_corners(color_frame: np.ndarray, face_box: Box):
    """Lower-face ROI + CIELAB a*-channel lip segmentation (reddish lip
    tissue vs. skin), Otsu-thresholded. Returns (mouth_box, (left_corner,
    right_corner)) or (fallback_box, None)."""
    x, y, w, h = face_box
    H, W = color_frame.shape[:2]
    mx0, mx1 = x + int(0.18 * w), x + int(0.82 * w)
    my0, my1 = y + int(0.55 * h), y + int(0.95 * h)
    mx0, mx1 = max(0, mx0), min(W, mx1)
    my0, my1 = max(0, my0), min(H, my1)
    if mx1 <= mx0 or my1 <= my0:
        return None, None

    roi = color_frame[my0:my1, mx0:mx1]
    lab = cv2.cvtColor(roi, cv2.COLOR_BGR2LAB)
    a_channel = lab[:, :, 1]  # green(low)-red(high) axis; lips skew red
    a_blur = cv2.GaussianBlur(a_channel, (5, 5), 0)
    _, lip_mask = cv2.threshold(a_blur, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    lip_mask = cv2.morphologyEx(lip_mask, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))

    contours, _ = cv2.findContours(lip_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    fallback_box = (mx0, my0, mx1 - mx0, my1 - my0)
    if not contours:
        return fallback_box, None

    largest = max(contours, key=cv2.contourArea)
    if cv2.contourArea(largest) < 0.03 * (mx1 - mx0) * (my1 - my0):
        return fallback_box, None

    bx, by, bw, bh = cv2.boundingRect(largest)
    pts = largest.reshape(-1, 2)
    left_pt = pts[np.argmin(pts[:, 0])]
    right_pt = pts[np.argmax(pts[:, 0])]
    left_corner = (mx0 + float(left_pt[0]), my0 + float(left_pt[1]))
    right_corner = (mx0 + float(right_pt[0]), my0 + float(right_pt[1]))
    return (mx0 + bx, my0 + by, bw, bh), (left_corner, right_corner)


def _mouth_openness(gray_frame: np.ndarray, mouth_box: Box) -> float:
    """Brightness (Otsu) threshold inside the mouth box finds the dark
    open-mouth cavity; its vertical extent over the mouth width is the
    MAR-equivalent openness ratio."""
    x, y, w, h = mouth_box
    if w < 4 or h < 4:
        return 0.0
    roi = gray_frame[y:y + h, x:x + w]
    roi_blur = cv2.GaussianBlur(roi, (5, 5), 0)
    _, dark_mask = cv2.threshold(roi_blur, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    band_w = max(1, w // 2)
    cx = w // 2
    band = dark_mask[:, max(0, cx - band_w // 2):min(w, cx + band_w // 2)]
    rows = np.where(band.any(axis=1))[0]
    extent = (rows.max() - rows.min() + 1) if len(rows) > 0 else 0
    return extent / float(w)


def _geometric_head_pose(left_eye_c: Point, right_eye_c: Point, nose_tip: Point,
                          mouth_left: Point, mouth_right: Point) -> Tuple[float, float, float]:
    """Head pose from geometric formulas on our own detected points, as
    proposed: eye-line angle for roll, eye-to-nose symmetry for yaw, and
    eye-mouth triangle vertical displacement for pitch."""
    lx, ly = left_eye_c
    rx, ry = right_eye_c
    interocular = float(np.hypot(rx - lx, ry - ly)) or 1.0

    # Roll: angle of the eye-corner line from horizontal.
    roll = float(np.degrees(np.arctan2(ry - ly, rx - lx)))

    # Yaw: left/right eye-to-nose-bridge distance symmetry.
    dist_left = float(np.hypot(nose_tip[0] - lx, nose_tip[1] - ly))
    dist_right = float(np.hypot(nose_tip[0] - rx, nose_tip[1] - ry))
    symmetry = (dist_right - dist_left) / (dist_right + dist_left + 1e-6)
    yaw = symmetry * YAW_SCALE_DEG

    # Pitch: vertical displacement of the eye-line/mouth-line "triangle",
    # normalised by interocular distance, relative to this dataset's
    # calibrated neutral ratio.
    eyes_mid_y = (ly + ry) / 2.0
    mouth_mid_y = (mouth_left[1] + mouth_right[1]) / 2.0
    eye_mouth_dist = mouth_mid_y - eyes_mid_y
    ratio = eye_mouth_dist / interocular
    pitch = (ratio - NEUTRAL_EYE_MOUTH_RATIO) * PITCH_SCALE_DEG

    return pitch, yaw, roll


def _solvepnp_head_pose(image_points: np.ndarray, image_shape) -> tuple:
    """Secondary cross-check: cv2.solvePnP against a generic 3D face model,
    fed by our own six detected 2D points."""
    h, w = image_shape[:2]
    focal_length = w
    center = (w / 2, h / 2)
    camera_matrix = np.array([
        [focal_length, 0, center[0]],
        [0, focal_length, center[1]],
        [0, 0, 1],
    ], dtype=np.float64)
    dist_coeffs = np.zeros((4, 1))

    success, rotation_vec, _ = cv2.solvePnP(
        MODEL_POINTS_3D, image_points.astype(np.float64), camera_matrix, dist_coeffs,
        flags=cv2.SOLVEPNP_ITERATIVE,
    )
    if not success:
        return None, None, None

    rotation_mat, _ = cv2.Rodrigues(rotation_vec)
    proj_matrix = np.hstack((rotation_mat, np.zeros((3, 1))))
    euler_angles = cv2.decomposeProjectionMatrix(proj_matrix)[6]
    pitch, yaw, roll = [float(a) for a in euler_angles.flatten()]

    # decomposeProjectionMatrix's Euler convention reports a forward-facing
    # pose as ~180 degrees, not ~0, for this solvePnP setup -- wrap into
    # [-90, 90] so "forward" reads as ~0 (see Discussion).
    def _wrap(a):
        if a > 90:
            return a - 180
        if a < -90:
            return a + 180
        return a

    return _wrap(pitch), _wrap(yaw), _wrap(roll)


class FaceFeatureExtractor:
    #Classical-CV face/eye/mouth detector and feature extractor.

    def __init__(self, face_cascade_path: str = FACE_CASCADE_PATH):
        if not os.path.exists(face_cascade_path):
            raise FileNotFoundError(f"Face cascade not found at {face_cascade_path}")
        self.face_cascade = cv2.CascadeClassifier(face_cascade_path)

    def _detect_face(self, gray: np.ndarray) -> Optional[Box]:
        faces = self.face_cascade.detectMultiScale(
            gray, scaleFactor=1.2, minNeighbors=5, minSize=(60, 60)
        )
        if len(faces) == 0:
            return None
        faces = sorted(faces, key=_box_area, reverse=True)
        return tuple(int(v) for v in faces[0])

    def process(self, frame_bgr: np.ndarray, timestamp_ms: Optional[int] = None) -> FrameFeatures:
        gray = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2GRAY)
        gray_eq = cv2.equalizeHist(gray)

        face_box = self._detect_face(gray_eq)
        if face_box is None:
            return FrameFeatures(face_detected=False)

        skin_box = _skin_face_box(frame_bgr)
        face_skin_overlap = _iou(face_box, skin_box) if skin_box is not None else None

        left_eye_box = _eye_roi_from_face(face_box, "left")
        right_eye_box = _eye_roi_from_face(face_box, "right")

        left_pupil, ear_left, left_eye_corners = _eye_openness_pupil_and_corners(gray_eq, left_eye_box)
        right_pupil, ear_right, right_eye_corners = _eye_openness_pupil_and_corners(gray_eq, right_eye_box)
        ear_avg = None
        if ear_left is not None and ear_right is not None:
            ear_avg = (ear_left + ear_right) / 2.0
        elif ear_left is not None:
            ear_avg = ear_left
        elif ear_right is not None:
            ear_avg = ear_right

        mouth_box, mouth_corners = _mouth_box_and_corners(frame_bgr, face_box)
        mar = _mouth_openness(gray_eq, mouth_box) if mouth_box is not None else None

        pitch = yaw = roll = None
        pitch_pnp = yaw_pnp = roll_pnp = None
        pose_points = None
        if left_eye_corners is not None and right_eye_corners is not None and mouth_corners is not None:
            left_eye_c = ((left_eye_corners[0][0] + left_eye_corners[1][0]) / 2.0,
                          (left_eye_corners[0][1] + left_eye_corners[1][1]) / 2.0)
            right_eye_c = ((right_eye_corners[0][0] + right_eye_corners[1][0]) / 2.0,
                           (right_eye_corners[0][1] + right_eye_corners[1][1]) / 2.0)
            mouth_left_c, mouth_right_c = mouth_corners
            eyes_mid = ((left_eye_c[0] + right_eye_c[0]) / 2.0, (left_eye_c[1] + right_eye_c[1]) / 2.0)
            mouth_top_mid = ((mouth_left_c[0] + mouth_right_c[0]) / 2.0, (mouth_left_c[1] + mouth_right_c[1]) / 2.0)
            nose_tip = (
                eyes_mid[0] + 0.45 * (mouth_top_mid[0] - eyes_mid[0]),
                eyes_mid[1] + 0.45 * (mouth_top_mid[1] - eyes_mid[1]),
            )
            fx, fy, fw, fh = face_box
            chin = (fx + fw / 2.0, fy + fh)

            # Primary: geometric formulas.
            pitch, yaw, roll = _geometric_head_pose(left_eye_c, right_eye_c, nose_tip, mouth_left_c, mouth_right_c)

            # Secondary cross-check: solvePnP on the same points.
            pose_points = np.array([
                nose_tip, chin, left_eye_c, right_eye_c, mouth_left_c, mouth_right_c,
            ], dtype=np.float64)
            pitch_pnp, yaw_pnp, roll_pnp = _solvepnp_head_pose(pose_points, frame_bgr.shape)

        return FrameFeatures(
            face_detected=True,
            ear_left=ear_left, ear_right=ear_right, ear_avg=ear_avg,
            mar=mar,
            pitch=pitch, yaw=yaw, roll=roll,
            pitch_pnp=pitch_pnp, yaw_pnp=yaw_pnp, roll_pnp=roll_pnp,
            face_box=face_box, face_skin_overlap=face_skin_overlap,
            left_eye_box=left_eye_box, right_eye_box=right_eye_box,
            left_eye_corners=left_eye_corners, right_eye_corners=right_eye_corners,
            left_pupil=left_pupil, right_pupil=right_pupil,
            mouth_box=mouth_box, mouth_corners=mouth_corners,
            pose_points=pose_points,
        )

    def close(self):
        pass  # no persistent resources to release
