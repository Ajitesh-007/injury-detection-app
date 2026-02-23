"""
Central configuration for the Injury Detection & Risk Prediction System.
All thresholds, constants, and tunable parameters live here.
"""

import os

# ─── Alert Levels ────────────────────────────────────────────────────────
ALERT_GREEN  = "GREEN"
ALERT_YELLOW = "YELLOW"
ALERT_RED    = "RED"

# ─── Risk Score Thresholds ───────────────────────────────────────────────
YELLOW_THRESHOLD = 35   # risk score above this → YELLOW
RED_THRESHOLD    = 70   # risk score above this → RED

# ─── Pose Detection ─────────────────────────────────────────────────────
POSE_CONFIDENCE_THRESHOLD  = 0.5
POSE_TRACKING_CONFIDENCE   = 0.5
FACE_CONFIDENCE_THRESHOLD  = 0.3

# ─── Fatigue Detection ──────────────────────────────────────────────────
FATIGUE_WINDOW_SECONDS      = 60
FATIGUE_ANGLE_DRIFT_THRESHOLD = 8   # degrees of drift that indicate fatigue

# ─── Object Tracking ────────────────────────────────────────────────────
PIXELS_PER_METER     = 200
FRAME_RATE           = 30
BALL_MIN_CONTOUR_AREA = 100
BALL_MAX_CONTOUR_AREA = 5000

# ─── Facial Stress ──────────────────────────────────────────────────────
PAIN_EXPRESSION_THRESHOLD    = 0.6
SKIN_STRESS_REDNESS_THRESHOLD = 0.4
SKIN_STRESS_PALENESS_THRESHOLD = 0.3

# ─── Prediction Engine ──────────────────────────────────────────────────
SYNTHETIC_SAMPLES_PER_SPORT = 3000   # reduced for faster startup
MODEL_RANDOM_STATE           = 42
N_ESTIMATORS                 = 80    # reduced for faster startup

# ─── Alert System ───────────────────────────────────────────────────────
ALERT_HISTORY_MAX    = 100
ALERT_COOLDOWN_SECONDS = 3

# ─── Supported Sports ───────────────────────────────────────────────────
SUPPORTED_SPORTS = ["football", "cricket", "weightlifting", "generic"]

# ─── WebSocket ───────────────────────────────────────────────────────────
WS_FRAME_SKIP = 1   # process every frame (frontend sends at ~6-12 fps)

# ─── Frame Processing ────────────────────────────────────────────────────
PROCESS_FRAME_WIDTH  = 320
PROCESS_FRAME_HEIGHT = 240
SECONDARY_ANALYSIS_INTERVAL = 3   # run face/object every Nth processed frame

# ─── CORS ────────────────────────────────────────────────────────────────
CORS_ORIGINS = ["*"]

# ─── Abnormal Posture Detection ──────────────────────────────────────────
SAFE_ANGLE_RANGES = {
    "knee":     (10, 180),
    "elbow":    (10, 180),
    "shoulder": (0,  175),
    "hip":      (15, 180),
    "spine":    (100, 180),
}
SUDDEN_ANGLE_CHANGE_THRESHOLD = 40   # degrees change in one frame

# ─── Model Files ─────────────────────────────────────────────────────────
MODELS_DIR = os.path.join(os.path.dirname(__file__), "models_data")

MODEL_URLS = {
    "pose_landmarker_heavy.task": (
        "https://storage.googleapis.com/mediapipe-models/"
        "pose_landmarker/pose_landmarker_heavy/float16/latest/pose_landmarker_heavy.task"
    ),
    "face_landmarker.task": (
        "https://storage.googleapis.com/mediapipe-models/"
        "face_landmarker/face_landmarker/float16/latest/face_landmarker.task"
    ),
}
