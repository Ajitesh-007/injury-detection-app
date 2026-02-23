"""
Facial Expression & Physiological Indicator Module — Robust Rewrite
====================================================================
Uses MediaPipe Tasks FaceLandmarker. Falls back gracefully if the model
file is missing. Never initialised at import time — use get_face_analyzer().
"""

from __future__ import annotations

import logging
import math
import os
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np

from config import (
    FACE_CONFIDENCE_THRESHOLD,
    MODELS_DIR,
    PAIN_EXPRESSION_THRESHOLD,
    SKIN_STRESS_PALENESS_THRESHOLD,
    SKIN_STRESS_REDNESS_THRESHOLD,
)

logger = logging.getLogger(__name__)

MODEL_PATH = os.path.join(MODELS_DIR, "face_landmarker.task")

# Key landmark indices
LEFT_BROW_TOP, LEFT_BROW_BOTTOM   = 70,  63
RIGHT_BROW_TOP, RIGHT_BROW_BOTTOM = 300, 293
LEFT_EYE_TOP,  LEFT_EYE_BOTTOM    = 159, 145
RIGHT_EYE_TOP, RIGHT_EYE_BOTTOM   = 386, 374
MOUTH_TOP, MOUTH_BOTTOM           = 13,  14
MOUTH_LEFT, MOUTH_RIGHT           = 61,  291
NOSE_TIP, NOSE_BRIDGE             = 1,   6
LEFT_CHEEK_LANDMARKS  = [123, 147, 213, 192]
RIGHT_CHEEK_LANDMARKS = [352, 376, 433, 416]


@dataclass
class FacialAnalysis:
    pain_score: float
    stress_score: float
    skin_redness: float
    skin_paleness: float
    overall_facial_stress: float
    face_detected: bool
    indicators: List[str] = field(default_factory=list)


_ZERO_FACE = FacialAnalysis(
    pain_score=0, stress_score=0, skin_redness=0,
    skin_paleness=0, overall_facial_stress=0, face_detected=False,
)


class FaceAnalyzer:
    """Analyze facial expressions and physiological indicators."""

    def __init__(self):
        self._available = False
        self._landmarker = None
        self._baseline_skin: Optional[np.ndarray] = None
        self._init_detector()

    def _init_detector(self):
        if not os.path.isfile(MODEL_PATH):
            logger.warning("FaceAnalyzer: model file not found — face analysis disabled")
            return
        try:
            import mediapipe as mp
            from mediapipe.tasks.python import BaseOptions
            from mediapipe.tasks.python.vision import (
                FaceLandmarker,
                FaceLandmarkerOptions,
                RunningMode,
            )
            opts = FaceLandmarkerOptions(
                base_options=BaseOptions(model_asset_path=MODEL_PATH),
                running_mode=RunningMode.IMAGE,
                num_faces=1,
                min_face_detection_confidence=FACE_CONFIDENCE_THRESHOLD,
                min_face_presence_confidence=FACE_CONFIDENCE_THRESHOLD,
            )
            self._landmarker = FaceLandmarker.create_from_options(opts)
            self._available = True
            logger.info("FaceAnalyzer: ready")
        except Exception as exc:
            logger.warning(f"FaceAnalyzer init failed: {exc} — face analysis disabled")

    def analyze_frame(self, frame: np.ndarray) -> FacialAnalysis:
        if not self._available:
            return _ZERO_FACE
        try:
            return self._run(frame)
        except Exception as exc:
            logger.debug(f"Face analysis error (non-fatal): {exc}")
            return _ZERO_FACE

    def _run(self, frame: np.ndarray) -> FacialAnalysis:
        import mediapipe as mp
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        mp_img = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
        results = self._landmarker.detect(mp_img)

        if not results.face_landmarks:
            return _ZERO_FACE

        landmarks = results.face_landmarks[0]
        h, w = frame.shape[:2]
        pts: Dict[int, Tuple[float, float]] = {
            i: (lm.x * w, lm.y * h) for i, lm in enumerate(landmarks)
        }

        indicators: List[str] = []
        pain   = self._compute_pain(pts, indicators)
        stress = self._compute_stress(pts, indicators)
        redness, paleness = self._analyze_skin(frame, pts, h, w, indicators)
        overall = self._overall(pain, stress, redness, paleness)

        return FacialAnalysis(
            pain_score=round(pain, 1),
            stress_score=round(stress, 1),
            skin_redness=round(redness, 3),
            skin_paleness=round(paleness, 3),
            overall_facial_stress=round(overall, 1),
            face_detected=True,
            indicators=indicators,
        )

    # ─── Scoring helpers ─────────────────────────────────────────────────

    def _compute_pain(self, pts: Dict, indicators: List[str]) -> float:
        score = 0.0
        ref = self._dist(pts.get(NOSE_TIP, (0,0)), pts.get(NOSE_BRIDGE, (0,0)))
        if ref < 1:
            return 0.0

        avg_brow = (
            self._dist(pts.get(LEFT_BROW_TOP,(0,0)), pts.get(LEFT_BROW_BOTTOM,(0,0)))
            + self._dist(pts.get(RIGHT_BROW_TOP,(0,0)), pts.get(RIGHT_BROW_BOTTOM,(0,0)))
        ) / (2 * ref)
        if avg_brow < 0.25:
            score += 35; indicators.append("Brow furrow detected")

        avg_eye = (
            self._dist(pts.get(LEFT_EYE_TOP,(0,0)), pts.get(LEFT_EYE_BOTTOM,(0,0)))
            + self._dist(pts.get(RIGHT_EYE_TOP,(0,0)), pts.get(RIGHT_EYE_BOTTOM,(0,0)))
        ) / (2 * ref)
        if avg_eye < 0.08:
            score += 35; indicators.append("Eye squeeze (pain indicator)")

        mouth_v = self._dist(pts.get(MOUTH_TOP,(0,0)), pts.get(MOUTH_BOTTOM,(0,0))) / ref
        mouth_h = self._dist(pts.get(MOUTH_LEFT,(0,0)), pts.get(MOUTH_RIGHT,(0,0))) / ref
        if mouth_v < 0.05 and mouth_h > 0.4:
            score += 30; indicators.append("Mouth compression detected")

        return min(100.0, score)

    def _compute_stress(self, pts: Dict, indicators: List[str]) -> float:
        score = 0.0
        ref = self._dist(pts.get(NOSE_TIP,(0,0)), pts.get(NOSE_BRIDGE,(0,0)))
        if ref < 1:
            return 0.0

        mouth_w = self._dist(pts.get(MOUTH_LEFT,(0,0)), pts.get(MOUTH_RIGHT,(0,0))) / ref
        if mouth_w > 0.6:
            score += 40; indicators.append("Jaw tension / grimace")

        lb = self._dist(pts.get(LEFT_BROW_TOP,(0,0)), pts.get(LEFT_BROW_BOTTOM,(0,0)))
        rb = self._dist(pts.get(RIGHT_BROW_TOP,(0,0)), pts.get(RIGHT_BROW_BOTTOM,(0,0)))
        if abs(lb - rb) / ref > 0.1:
            score += 30; indicators.append("Brow asymmetry (stress indicator)")

        le = self._dist(pts.get(LEFT_EYE_TOP,(0,0)), pts.get(LEFT_EYE_BOTTOM,(0,0))) / ref
        re = self._dist(pts.get(RIGHT_EYE_TOP,(0,0)), pts.get(RIGHT_EYE_BOTTOM,(0,0))) / ref
        if abs(le - re) > 0.04:
            score += 30; indicators.append("Eye asymmetry")

        return min(100.0, score)

    def _analyze_skin(self, frame, pts, h, w, indicators) -> Tuple[float, float]:
        try:
            colors = []
            for idxs in [LEFT_CHEEK_LANDMARKS, RIGHT_CHEEK_LANDMARKS]:
                region = []
                for idx in idxs:
                    if idx in pts:
                        px, py = int(pts[idx][0]), int(pts[idx][1])
                        px, py = max(0, min(w-1, px)), max(0, min(h-1, py))
                        region.append(frame[py, px])
                if region:
                    colors.append(np.mean(region, axis=0))
            if not colors:
                return 0.0, 0.0

            avg = np.mean(colors, axis=0)
            b, g, r = float(avg[0]), float(avg[1]), float(avg[2])

            if self._baseline_skin is None:
                self._baseline_skin = avg.copy()
                return 0.0, 0.0

            br = float(self._baseline_skin[2])
            redness  = max(0.0, (r - br) / br) if br > 0 else 0.0
            bright   = (r + g + b) / 3
            bb       = float(np.mean(self._baseline_skin))
            paleness = max(0.0, (bb - bright) / bb) if bb > 0 else 0.0

            if redness > SKIN_STRESS_REDNESS_THRESHOLD:
                indicators.append(f"Skin redness: {redness:.0%}")
            if paleness > SKIN_STRESS_PALENESS_THRESHOLD:
                indicators.append(f"Skin paleness: {paleness:.0%}")
            return redness, paleness
        except Exception:
            return 0.0, 0.0

    @staticmethod
    def _overall(pain, stress, redness, paleness) -> float:
        skin = min(100, (redness + paleness) * 100)
        return min(100.0, pain * 0.4 + stress * 0.3 + skin * 0.3)

    @staticmethod
    def _dist(a: Tuple[float, float], b: Tuple[float, float]) -> float:
        return math.sqrt((a[0]-b[0])**2 + (a[1]-b[1])**2)

    def reset(self):
        self._baseline_skin = None

    def __del__(self):
        try:
            if self._landmarker:
                self._landmarker.close()
        except Exception:
            pass


# ─── Lazy Singleton ──────────────────────────────────────────────────────

_face_analyzer: Optional[FaceAnalyzer] = None

def get_face_analyzer() -> FaceAnalyzer:
    global _face_analyzer
    if _face_analyzer is None:
        _face_analyzer = FaceAnalyzer()
    return _face_analyzer
