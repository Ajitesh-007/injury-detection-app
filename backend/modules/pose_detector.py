"""
Pose Detection Module — Robust Rewrite
=======================================
Uses MediaPipe Tasks PoseLandmarker. Falls back gracefully if the model
file is missing or the Tasks API fails. The detector is NEVER instantiated
at import time — use `get_pose_detector()` lazy singleton instead.
"""

from __future__ import annotations

import logging
import math
import os
import time
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np

from config import (
    FATIGUE_ANGLE_DRIFT_THRESHOLD,
    FATIGUE_WINDOW_SECONDS,
    MODELS_DIR,
    POSE_CONFIDENCE_THRESHOLD,
    POSE_TRACKING_CONFIDENCE,
    SAFE_ANGLE_RANGES,
    SUDDEN_ANGLE_CHANGE_THRESHOLD,
)

logger = logging.getLogger(__name__)

MODEL_PATH = os.path.join(MODELS_DIR, "pose_landmarker_heavy.task")

# ─── Data classes ────────────────────────────────────────────────────────

@dataclass
class JointAngle:
    name: str
    angle: float
    side: str
    is_safe: bool = True
    threshold_exceeded_by: float = 0.0


@dataclass
class PostureAlert:
    joint: str
    side: str
    message: str
    severity: str   # "warning" | "danger"
    angle: float
    safe_min: float
    safe_max: float


@dataclass
class PoseAnalysis:
    keypoints: Dict[str, Tuple[float, float, float]]
    joint_angles: List[JointAngle]
    asymmetry_scores: Dict[str, float]
    fatigue_score: float
    overall_pose_risk: float
    skeleton_connections: List[Tuple[str, str]]
    landmarks_normalized: list
    posture_alerts: List[PostureAlert] = field(default_factory=list)
    issues: List[str] = field(default_factory=list)


# ─── Constants ───────────────────────────────────────────────────────────

LANDMARK_NAMES = {
    0: "nose", 1: "left_eye_inner", 2: "left_eye", 3: "left_eye_outer",
    4: "right_eye_inner", 5: "right_eye", 6: "right_eye_outer",
    7: "left_ear", 8: "right_ear", 9: "mouth_left", 10: "mouth_right",
    11: "left_shoulder", 12: "right_shoulder", 13: "left_elbow",
    14: "right_elbow", 15: "left_wrist", 16: "right_wrist",
    17: "left_pinky", 18: "right_pinky", 19: "left_index",
    20: "right_index", 21: "left_thumb", 22: "right_thumb",
    23: "left_hip", 24: "right_hip", 25: "left_knee", 26: "right_knee",
    27: "left_ankle", 28: "right_ankle", 29: "left_heel",
    30: "right_heel", 31: "left_foot_index", 32: "right_foot_index",
}

ANGLE_DEFINITIONS = [
    ("knee",     "left",   "left_hip",     "left_knee",     "left_ankle"),
    ("knee",     "right",  "right_hip",    "right_knee",    "right_ankle"),
    ("elbow",    "left",   "left_shoulder","left_elbow",    "left_wrist"),
    ("elbow",    "right",  "right_shoulder","right_elbow",  "right_wrist"),
    ("shoulder", "left",   "left_elbow",   "left_shoulder", "left_hip"),
    ("shoulder", "right",  "right_elbow",  "right_shoulder","right_hip"),
    ("hip",      "left",   "left_shoulder","left_hip",      "left_knee"),
    ("hip",      "right",  "right_shoulder","right_hip",    "right_knee"),
    ("spine",    "center", "left_shoulder","left_hip",      "left_knee"),
]

SKELETON_CONNECTIONS = [
    ("left_shoulder",  "right_shoulder"),
    ("left_shoulder",  "left_elbow"),   ("left_elbow",  "left_wrist"),
    ("right_shoulder", "right_elbow"),  ("right_elbow", "right_wrist"),
    ("left_shoulder",  "left_hip"),     ("right_shoulder","right_hip"),
    ("left_hip",       "right_hip"),
    ("left_hip",       "left_knee"),    ("left_knee",   "left_ankle"),
    ("right_hip",      "right_knee"),   ("right_knee",  "right_ankle"),
]


# ─── PoseDetector ────────────────────────────────────────────────────────

class PoseDetector:
    """Real-time pose detection. Initialises lazily; returns None on failure."""

    def __init__(self):
        self._available = False
        self._use_legacy = False
        self._landmarker = None
        self._legacy_pose = None
        self._baseline_angles: Optional[Dict[str, float]] = None
        self._angle_history: List[Tuple[float, Dict[str, float]]] = []
        self._prev_angles: Dict[str, float] = {}
        self._start_time = time.time()
        self._init_detector()

    def _init_detector(self):
        """Try Tasks API first, fall back to legacy mp.solutions.pose."""
        # ── Try MediaPipe Tasks API ──────────────────────────────────────
        if os.path.isfile(MODEL_PATH):
            try:
                import mediapipe as mp
                from mediapipe.tasks.python import BaseOptions
                from mediapipe.tasks.python.vision import (
                    PoseLandmarker,
                    PoseLandmarkerOptions,
                    RunningMode,
                )
                opts = PoseLandmarkerOptions(
                    base_options=BaseOptions(model_asset_path=MODEL_PATH),
                    running_mode=RunningMode.IMAGE,
                    num_poses=1,
                    min_pose_detection_confidence=POSE_CONFIDENCE_THRESHOLD,
                    min_tracking_confidence=POSE_TRACKING_CONFIDENCE,
                )
                self._landmarker = PoseLandmarker.create_from_options(opts)
                self._available = True
                self._use_legacy = False
                logger.info("PoseDetector: using MediaPipe Tasks API")
                return
            except Exception as exc:
                logger.warning(f"Tasks API init failed: {exc} — trying legacy API")

        # ── Fall back to legacy mp.solutions.pose ────────────────────────
        try:
            import mediapipe as mp
            self._legacy_pose = mp.solutions.pose.Pose(
                static_image_mode=True,
                model_complexity=2,
                min_detection_confidence=POSE_CONFIDENCE_THRESHOLD,
                min_tracking_confidence=POSE_TRACKING_CONFIDENCE,
            )
            self._available = True
            self._use_legacy = True
            logger.info("PoseDetector: using legacy MediaPipe Pose API")
        except Exception as exc:
            logger.error(f"PoseDetector unavailable: {exc}")
            self._available = False

    # ─── Public API ──────────────────────────────────────────────────────

    def analyze_frame(self, frame: np.ndarray) -> Optional[PoseAnalysis]:
        if not self._available:
            return None
        try:
            if self._use_legacy:
                return self._analyze_legacy(frame)
            return self._analyze_tasks(frame)
        except Exception as exc:
            logger.debug(f"Pose analysis error (non-fatal): {exc}")
            return None

    def reset(self):
        self._baseline_angles = None
        self._angle_history.clear()
        self._prev_angles.clear()
        self._start_time = time.time()

    # ─── Tasks API path ──────────────────────────────────────────────────

    def _analyze_tasks(self, frame: np.ndarray) -> Optional[PoseAnalysis]:
        import mediapipe as mp
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
        results = self._landmarker.detect(mp_image)

        if not results.pose_landmarks:
            return None

        landmarks = results.pose_landmarks[0]
        keypoints = self._extract_keypoints_tasks(landmarks)
        lm_normalized = [(lm.x, lm.y, lm.z, lm.visibility) for lm in landmarks]
        return self._build_analysis(keypoints, lm_normalized)

    def _extract_keypoints_tasks(self, landmarks) -> Dict:
        kp: Dict[str, Tuple[float, float, float]] = {}
        for idx, name in LANDMARK_NAMES.items():
            if idx < len(landmarks):
                lm = landmarks[idx]
                if lm.visibility > 0.15:
                    kp[name] = (lm.x, lm.y, lm.visibility)
        return kp

    # ─── Legacy API path ─────────────────────────────────────────────────

    def _analyze_legacy(self, frame: np.ndarray) -> Optional[PoseAnalysis]:
        import mediapipe as mp
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = self._legacy_pose.process(rgb)

        if not results.pose_landmarks:
            return None

        landmarks = results.pose_landmarks.landmark
        kp: Dict[str, Tuple[float, float, float]] = {}
        for idx, name in LANDMARK_NAMES.items():
            if idx < len(landmarks):
                lm = landmarks[idx]
                if lm.visibility > 0.15:
                    kp[name] = (lm.x, lm.y, lm.visibility)

        lm_normalized = [
            (lm.x, lm.y, lm.z, lm.visibility) for lm in landmarks
        ]
        return self._build_analysis(kp, lm_normalized)

    # ─── Shared analysis pipeline ────────────────────────────────────────

    def _build_analysis(self, keypoints: Dict, lm_normalized: list) -> PoseAnalysis:
        joint_angles  = self._compute_all_angles(keypoints)
        asymmetry     = self._detect_asymmetry(joint_angles)
        fatigue       = self._compute_fatigue(joint_angles)
        posture_alerts = self._detect_abnormal_posture(joint_angles)

        issues: List[str] = []
        overall_risk = self._compute_pose_risk(joint_angles, asymmetry, fatigue, posture_alerts, issues)

        return PoseAnalysis(
            keypoints=keypoints,
            joint_angles=joint_angles,
            asymmetry_scores=asymmetry,
            fatigue_score=fatigue,
            overall_pose_risk=overall_risk,
            skeleton_connections=SKELETON_CONNECTIONS,
            landmarks_normalized=lm_normalized,
            posture_alerts=posture_alerts,
            issues=issues,
        )

    # ─── Geometry helpers ────────────────────────────────────────────────

    @staticmethod
    def _angle_between(a, b, c) -> float:
        ba = (a[0] - b[0], a[1] - b[1])
        bc = (c[0] - b[0], c[1] - b[1])
        dot = ba[0]*bc[0] + ba[1]*bc[1]
        mag = math.sqrt(ba[0]**2 + ba[1]**2) * math.sqrt(bc[0]**2 + bc[1]**2)
        if mag == 0:
            return 0.0
        return math.degrees(math.acos(max(-1.0, min(1.0, dot / mag))))

    def _compute_all_angles(self, kp: Dict) -> List[JointAngle]:
        angles: List[JointAngle] = []
        for name, side, pa, vertex, pc in ANGLE_DEFINITIONS:
            if pa in kp and vertex in kp and pc in kp:
                angle = self._angle_between(kp[pa][:2], kp[vertex][:2], kp[pc][:2])
                angles.append(JointAngle(name=name, angle=round(angle, 1), side=side))
        return angles

    def _detect_asymmetry(self, angles: List[JointAngle]) -> Dict[str, float]:
        left  = {a.name: a.angle for a in angles if a.side == "left"}
        right = {a.name: a.angle for a in angles if a.side == "right"}
        return {
            joint: round(abs(left[joint] - right[joint]), 1)
            for joint in left if joint in right
        }

    def _detect_abnormal_posture(self, angles: List[JointAngle]) -> List[PostureAlert]:
        alerts: List[PostureAlert] = []
        current: Dict[str, float] = {}

        for ja in angles:
            key = f"{ja.name}_{ja.side}"
            current[key] = ja.angle

            if ja.name in SAFE_ANGLE_RANGES:
                safe_min, safe_max = SAFE_ANGLE_RANGES[ja.name]
                side_label = f" ({ja.side})" if ja.side != "center" else ""

                if ja.angle < safe_min:
                    sev = "danger" if ja.angle < safe_min * 0.5 else "warning"
                    msg = (f"⚠️ {ja.name.title()}{side_label} at {ja.angle:.0f}° — "
                           f"below safe min {safe_min}°! Possible hyperextension.")
                    alerts.append(PostureAlert(ja.name, ja.side, msg, sev,
                                               ja.angle, safe_min, safe_max))
                    ja.is_safe = False
                    ja.threshold_exceeded_by = safe_min - ja.angle

                elif ja.angle > safe_max:
                    sev = "danger" if ja.angle > safe_max + 10 else "warning"
                    msg = (f"⚠️ {ja.name.title()}{side_label} at {ja.angle:.0f}° — "
                           f"above safe max {safe_max}°! Unnatural position.")
                    alerts.append(PostureAlert(ja.name, ja.side, msg, sev,
                                               ja.angle, safe_min, safe_max))
                    ja.is_safe = False
                    ja.threshold_exceeded_by = ja.angle - safe_max

            if key in self._prev_angles:
                change = abs(ja.angle - self._prev_angles[key])
                if change > SUDDEN_ANGLE_CHANGE_THRESHOLD:
                    side_label = f" ({ja.side})" if ja.side != "center" else ""
                    msg = (f"🚨 Sudden {ja.name.title()}{side_label} movement! "
                           f"Changed {change:.0f}° in one frame. Possible injury!")
                    alerts.append(PostureAlert(ja.name, ja.side, msg, "danger",
                                               ja.angle, 0, 0))

        self._prev_angles = current
        return alerts

    def _compute_fatigue(self, angles: List[JointAngle]) -> float:
        now = time.time()
        angle_dict = {f"{a.name}_{a.side}": a.angle for a in angles}

        if self._baseline_angles is None:
            self._baseline_angles = angle_dict.copy()

        self._angle_history.append((now, angle_dict))
        cutoff = now - FATIGUE_WINDOW_SECONDS
        self._angle_history = [(t, a) for t, a in self._angle_history if t >= cutoff]

        if len(self._angle_history) < 5:
            return 0.0

        drifts = [
            abs(angle_dict[k] - self._baseline_angles[k])
            for k in self._baseline_angles if k in angle_dict
        ]
        if not drifts:
            return 0.0
        avg_drift = sum(drifts) / len(drifts)
        return round(min(100.0, (avg_drift / FATIGUE_ANGLE_DRIFT_THRESHOLD) * 100), 1)

    def _compute_pose_risk(self, angles, asymmetry, fatigue, posture_alerts, issues) -> float:
        risk = 0.0
        for a in angles:
            if a.name == "knee" and a.angle < 40:
                risk += 25; issues.append(f"Dangerous {a.side} knee angle: {a.angle}°")
            elif a.name == "spine" and a.angle < 120:
                risk += 30; issues.append(f"Excessive spinal flexion: {a.angle}°")
            elif a.name == "shoulder" and a.angle > 170:
                risk += 20; issues.append(f"Shoulder hyperextension ({a.side}): {a.angle}°")

        for joint, diff in asymmetry.items():
            if diff > 15:
                risk += 10; issues.append(f"High {joint} asymmetry: {diff}°")

        if fatigue > 50:
            risk += fatigue * 0.2; issues.append(f"Fatigue detected: {fatigue:.0f}%")

        for alert in posture_alerts:
            risk += 30 if alert.severity == "danger" else 15
            issues.append(alert.message)

        return min(100.0, round(risk, 1))

    def __del__(self):
        try:
            if self._landmarker:
                self._landmarker.close()
        except Exception:
            pass
        try:
            if self._legacy_pose:
                self._legacy_pose.close()
        except Exception:
            pass


# ─── Lazy Singleton ──────────────────────────────────────────────────────

_pose_detector: Optional[PoseDetector] = None

def get_pose_detector() -> PoseDetector:
    global _pose_detector
    if _pose_detector is None:
        _pose_detector = PoseDetector()
    return _pose_detector
