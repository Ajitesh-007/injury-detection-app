"""
API Routes — Complete Rewrite
==============================
REST endpoints + WebSocket for real-time injury analysis.

Key improvements over the previous version:
- NO module-level ML object instantiation (all lazy singletons)
- Per-frame exception isolation: a bad frame never kills the WebSocket
- /api/status endpoint to check model readiness
- Clean async warmup function called from lifespan
"""

from __future__ import annotations

import asyncio
import base64
import json
import logging
import time
from typing import Optional

import cv2
import numpy as np
from fastapi import APIRouter, WebSocket, WebSocketDisconnect
from starlette.websockets import WebSocketState
from pydantic import BaseModel

from config import (
    PROCESS_FRAME_HEIGHT,
    PROCESS_FRAME_WIDTH,
    SECONDARY_ANALYSIS_INTERVAL,
    WS_FRAME_SKIP,
)

logger = logging.getLogger(__name__)

router = APIRouter()


# ─── Request / Response Models ───────────────────────────────────────────

class FrameRequest(BaseModel):
    image_base64: str
    sport: str = "generic"
    frame_width: int = 640
    frame_height: int = 480


class AnalysisResponse(BaseModel):
    pose_risk: float = 0
    facial_stress: float = 0
    object_risk: float = 0
    injury_probability: float = 0
    injury_type: str = "Unknown"
    time_horizon: str = "long-term"
    alert_level: str = "GREEN"
    alert_message: str = ""
    contributing_factors: list = []
    recommended_action: str = ""
    joint_angles: dict = {}
    asymmetry: dict = {}
    fatigue_score: float = 0
    skeleton_landmarks: list = []
    face_detected: bool = False
    object_speed: float = 0
    issues: list = []
    posture_alerts: list = []


# ─── Lazy accessor helpers ────────────────────────────────────────────────

def _pose():
    from modules.pose_detector import get_pose_detector
    return get_pose_detector()

def _face():
    from modules.face_analyzer import get_face_analyzer
    return get_face_analyzer()

def _obj():
    from modules.object_tracker import get_object_tracker
    return get_object_tracker()

def _alert():
    return _get_alert_system()

_alert_system_instance = None
def _get_alert_system():
    global _alert_system_instance
    if _alert_system_instance is None:
        from modules.alert_system import AlertSystem
        _alert_system_instance = AlertSystem()
    return _alert_system_instance


# ─── REST Endpoints ──────────────────────────────────────────────────────

@router.get("/api/health")
async def health():
    return {"status": "healthy", "service": "InjuryGuard AI", "version": "2.0.0"}


@router.get("/api/status")
async def status():
    """Check model readiness."""
    import os
    from config import MODELS_DIR
    pose_ok = os.path.isfile(os.path.join(MODELS_DIR, "pose_landmarker_lite.task"))
    face_ok = os.path.isfile(os.path.join(MODELS_DIR, "face_landmarker.task"))
    return {
        "pose_model": "ready" if pose_ok else "downloading",
        "face_model": "ready" if face_ok else "downloading",
        "overall": "ready" if (pose_ok and face_ok) else "initializing",
    }


@router.get("/api/sports")
async def get_sports():
    from models.sport_profiles import list_sports
    return {"sports": list_sports()}


@router.get("/api/alerts/history")
async def get_alert_history():
    return {"alerts": _alert().get_history(50)}


@router.post("/api/analyze-frame", response_model=AnalysisResponse)
async def analyze_frame_endpoint(request: FrameRequest):
    """Analyze a single video frame (REST fallback)."""
    try:
        frame = _decode_frame(request.image_base64)
        if frame is None:
            return AnalysisResponse()
        return _process_frame(frame, request.sport, request.frame_width, request.frame_height)
    except Exception as exc:
        logger.error(f"Frame analysis error: {exc}")
        return AnalysisResponse()


# ─── WebSocket — Real-Time Analysis ──────────────────────────────────────

# Per-session secondary analysis cache
_DEFAULT_SECONDARY = {
    "facial_stress": 0.0,
    "face_detected":  False,
    "face_issues":    [],
    "object_risk":    0.0,
    "object_speed":   0.0,
    "obj_issues":     [],
    "closest_body_distance": float("inf"),
}


@router.websocket("/ws/analyze")
async def websocket_analyze(websocket: WebSocket):
    await websocket.accept()
    frame_count = 0
    sport = "generic"
    secondary = dict(_DEFAULT_SECONDARY)
    logger.info("WebSocket client connected")

    try:
        while True:
            # Receive with a timeout so we don't block forever
            try:
                raw = await asyncio.wait_for(websocket.receive_text(), timeout=30.0)
            except asyncio.TimeoutError:
                # Send a heartbeat ping (keep-alive)
                if websocket.client_state == WebSocketState.CONNECTED:
                    await websocket.send_json({"heartbeat": True})
                continue

            try:
                msg = json.loads(raw)
            except json.JSONDecodeError:
                continue

            # Sport change
            if "sport" in msg:
                sport = msg["sport"]

            # Frame analysis
            if "image_base64" not in msg:
                continue

            frame_count += 1
            if frame_count % WS_FRAME_SKIP != 0:
                continue

            # Decode frame — skip silently on error
            frame = _decode_frame(msg["image_base64"])
            if frame is None:
                continue

            fw = msg.get("frame_width", 640)
            fh = msg.get("frame_height", 480)

            # Process — isolate all errors so we never drop the connection
            try:
                result = _process_frame(frame, sport, fw, fh, frame_count, secondary)
                # Use model_dump (Pydantic v2) or dict (v1) — handle both
                try:
                    payload = result.model_dump()
                except AttributeError:
                    payload = result.dict()
                await websocket.send_json(payload)
            except Exception as exc:
                logger.warning(f"Frame processing error (skipping frame): {exc}")

    except WebSocketDisconnect:
        logger.info("WebSocket client disconnected")
    except Exception as exc:
        logger.error(f"WebSocket fatal error: {exc}")
    finally:
        try:
            await websocket.close()
        except Exception:
            pass


# ─── Frame Processing Pipeline ───────────────────────────────────────────

def _decode_frame(b64: str) -> Optional[np.ndarray]:
    """Decode base64 image string to a BGR numpy array, resized for speed."""
    try:
        if "," in b64:
            b64 = b64.split(",", 1)[1]
        data = base64.b64decode(b64)
        arr  = np.frombuffer(data, np.uint8)
        frame = cv2.imdecode(arr, cv2.IMREAD_COLOR)
        if frame is not None:
            # Preserve aspect ratio while resizing
            h, w = frame.shape[:2]
            max_dim = max(PROCESS_FRAME_WIDTH, PROCESS_FRAME_HEIGHT)
            scale = max_dim / max(h, w)
            new_w, new_h = int(w * scale), int(h * scale)
            frame = cv2.resize(frame, (new_w, new_h))
        return frame
    except Exception as exc:
        logger.debug(f"Frame decode error: {exc}")
        return None


def _process_frame(
    frame: np.ndarray,
    sport: str,
    frame_width: int,
    frame_height: int,
    frame_count: int = 0,
    secondary: Optional[dict] = None,
) -> AnalysisResponse:
    if secondary is None:
        secondary = dict(_DEFAULT_SECONDARY)

    # 1. Pose Detection (every frame)
    pose = _pose().analyze_frame(frame)
    pose_risk, joint_angles_dict, asymmetry_dict = 0.0, {}, {}
    fatigue, skeleton, pose_issues, posture_alerts_data = 0.0, [], [], []

    if pose:
        pose_risk   = pose.overall_pose_risk
        fatigue     = pose.fatigue_score
        skeleton    = pose.landmarks_normalized
        pose_issues = pose.issues
        for ja in pose.joint_angles:
            joint_angles_dict[f"{ja.name}_{ja.side}"] = ja.angle
        asymmetry_dict = pose.asymmetry_scores
        for pa in pose.posture_alerts:
            posture_alerts_data.append({
                "joint":    pa.joint,
                "side":     pa.side,
                "message":  pa.message,
                "severity": pa.severity,
                "angle":    round(pa.angle, 1),
                "safe_min": pa.safe_min,
                "safe_max": pa.safe_max,
            })

    # 2 & 3. Secondary analysis (face + object) — every Nth frame
    run_secondary = (frame_count % SECONDARY_ANALYSIS_INTERVAL == 0) or frame_count == 0

    if run_secondary:
        face = _face().analyze_frame(frame)
        secondary["facial_stress"]  = face.overall_facial_stress
        secondary["face_detected"]  = face.face_detected
        secondary["face_issues"]    = face.indicators

        obj = _obj().analyze_frame(
            frame,
            body_keypoints=pose.keypoints if pose else None,
            frame_width=frame_width,
            frame_height=frame_height,
        )
        secondary["object_risk"]    = obj.impact_risk
        secondary["object_speed"]   = obj.primary_object.speed_kmh if obj.primary_object else 0.0
        secondary["obj_issues"]     = obj.issues
        secondary["closest_body_distance"] = obj.closest_body_distance

    facial_stress  = secondary["facial_stress"]
    face_issues    = secondary["face_issues"]
    object_risk    = secondary["object_risk"]
    object_speed   = secondary["object_speed"]
    obj_issues     = secondary["obj_issues"]
    closest_dist   = secondary["closest_body_distance"]

    # 4. Prediction Engine
    from models.prediction_engine import get_predictor
    predictor  = get_predictor(sport)
    proximity  = 1.0 - (closest_dist / 500) if closest_dist < 500 else 0.0
    prediction = predictor.predict_from_raw(
        joint_angles=joint_angles_dict,
        asymmetry=asymmetry_dict,
        facial_stress=facial_stress,
        object_speed=object_speed,
        impact_proximity=proximity,
        fatigue_score=fatigue,
        time_elapsed=0,
    )

    # 5. Alert System
    all_issues = pose_issues + face_issues + obj_issues + prediction.contributing_factors
    alert = _alert().evaluate(
        pose_risk=pose_risk,
        facial_stress=facial_stress,
        object_risk=object_risk,
        prediction_score=prediction.injury_probability,
        injury_type=prediction.injury_type,
        all_issues=all_issues,
    )

    # Override alert level if we have posture alerts
    eff_level, eff_msg = alert.level, alert.message
    if posture_alerts_data:
        has_danger = any(pa["severity"] == "danger" for pa in posture_alerts_data)
        if has_danger and eff_level != "RED":
            eff_level = "RED"
            eff_msg   = posture_alerts_data[0]["message"]
        elif not has_danger and eff_level == "GREEN":
            eff_level = "YELLOW"
            eff_msg   = posture_alerts_data[0]["message"]

    return AnalysisResponse(
        pose_risk=round(pose_risk, 1),
        facial_stress=round(facial_stress, 1),
        object_risk=round(object_risk, 1),
        injury_probability=round(prediction.injury_probability, 1),
        injury_type=prediction.injury_type,
        time_horizon=prediction.time_horizon,
        alert_level=eff_level,
        alert_message=eff_msg,
        contributing_factors=alert.contributing_factors,
        recommended_action=alert.recommended_action,
        joint_angles=joint_angles_dict,
        asymmetry=asymmetry_dict,
        fatigue_score=round(fatigue, 1),
        skeleton_landmarks=skeleton,
        face_detected=secondary["face_detected"],
        object_speed=round(object_speed, 1),
        issues=all_issues[:10],
        posture_alerts=posture_alerts_data,
    )


# ─── Warmup (called from lifespan) ───────────────────────────────────────

async def warmup():
    """Pre-initialize all singletons in a thread pool."""
    loop = asyncio.get_event_loop()

    def _init_all():
        _pose()
        _face()
        _obj()
        _alert()
        from models.prediction_engine import get_predictor
        get_predictor("generic")
        logger.info("All singletons initialized during warmup")

    await loop.run_in_executor(None, _init_all)
