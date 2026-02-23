"""
Object Speed & Impact Analysis Module
======================================
Tracks fast-moving objects using background subtraction + contour detection.
Lazy singleton: use get_object_tracker() instead of instantiating directly.
"""

import math
import logging
from collections import deque
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np

from config import (
    BALL_MAX_CONTOUR_AREA,
    BALL_MIN_CONTOUR_AREA,
    FRAME_RATE,
    PIXELS_PER_METER,
)

logger = logging.getLogger(__name__)


@dataclass
class TrackedObject:
    position: Tuple[int, int]
    speed_kmh: float
    acceleration: float
    direction: float
    contour_area: float


@dataclass
class ObjectAnalysis:
    objects_detected: int
    primary_object: Optional[TrackedObject]
    impact_risk: float
    impact_zone: str
    closest_body_distance: float
    speed_alert: bool
    issues: List[str] = field(default_factory=list)


_ZERO_OBJ = ObjectAnalysis(
    objects_detected=0, primary_object=None,
    impact_risk=0, impact_zone="none",
    closest_body_distance=float("inf"), speed_alert=False,
)

BODY_ZONES = {
    "nose": "head", "left_eye": "head", "right_eye": "head",
    "left_ear": "head", "right_ear": "head",
    "left_shoulder": "torso", "right_shoulder": "torso",
    "left_hip": "torso", "right_hip": "torso",
    "left_elbow": "arm", "right_elbow": "arm",
    "left_wrist": "arm", "right_wrist": "arm",
    "left_knee": "leg", "right_knee": "leg",
    "left_ankle": "leg", "right_ankle": "leg",
}


class ObjectTracker:
    """Track fast-moving objects and assess impact risk."""

    def __init__(self, pixels_per_meter: float = PIXELS_PER_METER, fps: int = FRAME_RATE):
        self.ppm = pixels_per_meter
        self.fps = fps
        self._prev_gray: Optional[np.ndarray] = None
        self._position_history: deque = deque(maxlen=30)
        self._bg_subtractor = cv2.createBackgroundSubtractorMOG2(
            history=60, varThreshold=50, detectShadows=False
        )

    def analyze_frame(
        self,
        frame: np.ndarray,
        body_keypoints: Optional[Dict[str, Tuple[float, float, float]]] = None,
        frame_width: int = 640,
        frame_height: int = 480,
    ) -> ObjectAnalysis:
        try:
            return self._run(frame, body_keypoints, frame_width, frame_height)
        except Exception as exc:
            logger.debug(f"Object tracker error (non-fatal): {exc}")
            return _ZERO_OBJ

    def _run(self, frame, body_keypoints, frame_width, frame_height):
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        fg_mask = self._bg_subtractor.apply(frame)
        fg_mask = cv2.morphologyEx(fg_mask, cv2.MORPH_OPEN,  np.ones((5, 5), np.uint8))
        fg_mask = cv2.morphologyEx(fg_mask, cv2.MORPH_CLOSE, np.ones((7, 7), np.uint8))

        contours, _ = cv2.findContours(fg_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        candidates = []
        for c in contours:
            area = cv2.contourArea(c)
            if BALL_MIN_CONTOUR_AREA <= area <= BALL_MAX_CONTOUR_AREA:
                M = cv2.moments(c)
                if M["m00"] > 0:
                    cx = int(M["m10"] / M["m00"])
                    cy = int(M["m01"] / M["m00"])
                    candidates.append((cx, cy, area))

        if not candidates:
            self._prev_gray = gray
            return _ZERO_OBJ

        cx, cy, area = max(candidates, key=lambda c: c[2])
        self._position_history.append((cx, cy))
        speed_kmh, acceleration, direction = self._compute_kinematics()

        tracked = TrackedObject(
            position=(cx, cy),
            speed_kmh=round(speed_kmh, 1),
            acceleration=round(acceleration, 1),
            direction=round(direction, 1),
            contour_area=area,
        )

        issues: List[str] = []
        impact_risk, impact_zone, closest = self._assess_impact(
            cx, cy, speed_kmh, body_keypoints, frame_width, frame_height, issues
        )
        self._prev_gray = gray

        return ObjectAnalysis(
            objects_detected=len(candidates),
            primary_object=tracked,
            impact_risk=round(impact_risk, 1),
            impact_zone=impact_zone,
            closest_body_distance=round(closest, 1),
            speed_alert=speed_kmh > 80,
            issues=issues,
        )

    def _compute_kinematics(self) -> Tuple[float, float, float]:
        if len(self._position_history) < 2:
            return 0.0, 0.0, 0.0
        p1, p2 = self._position_history[-2], self._position_history[-1]
        dx, dy = p2[0]-p1[0], p2[1]-p1[1]
        speed_kmh = (math.sqrt(dx**2+dy**2) / self.ppm) * self.fps * 3.6
        direction  = math.degrees(math.atan2(dy, dx)) % 360
        acceleration = 0.0
        if len(self._position_history) >= 3:
            p0 = self._position_history[-3]
            prev_speed = (math.sqrt((p1[0]-p0[0])**2+(p1[1]-p0[1])**2)/self.ppm)*self.fps*3.6
            acceleration = (speed_kmh - prev_speed) * self.fps
        return speed_kmh, acceleration, direction

    def _assess_impact(self, ox, oy, speed, kp, fw, fh, issues) -> Tuple[float, str, float]:
        if not kp:
            return 0.0, "none", float("inf")
        closest, closest_zone = float("inf"), "none"
        for name, (nx, ny, vis) in kp.items():
            if vis < 0.5 or name not in BODY_ZONES:
                continue
            dist = math.sqrt((ox - nx*fw)**2 + (oy - ny*fh)**2)
            if dist < closest:
                closest, closest_zone = dist, BODY_ZONES[name]

        prox   = max(0, 1.0 - closest/200)
        spd    = min(1.0, speed/150)
        weight = {"head":1.5,"torso":1.0,"arm":0.7,"leg":0.8,"none":0}.get(closest_zone,0.5)
        risk   = prox * spd * weight * 100

        if risk > 50:
            issues.append(f"Object at {speed:.0f} km/h near {closest_zone} ({closest:.0f}px)")
        if speed > 120:
            issues.append(f"Extreme object speed: {speed:.0f} km/h")
        return min(100.0, risk), closest_zone, closest

    def reset(self):
        self._position_history.clear()
        self._prev_gray = None


# ─── Lazy Singleton ──────────────────────────────────────────────────────

_object_tracker: Optional[ObjectTracker] = None

def get_object_tracker() -> ObjectTracker:
    global _object_tracker
    if _object_tracker is None:
        _object_tracker = ObjectTracker()
    return _object_tracker
