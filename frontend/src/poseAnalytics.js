/**
 * poseAnalytics.js
 * Computes pose risk metrics directly from MediaPipe Pose landmarks
 * in the browser — no backend needed.
 *
 * MediaPipe Pose landmark indices:
 * 11=left_shoulder  12=right_shoulder
 * 13=left_elbow     14=right_elbow
 * 15=left_wrist     16=right_wrist
 * 23=left_hip       24=right_hip
 * 25=left_knee      26=right_knee
 * 27=left_ankle     28=right_ankle
 */

// Compute angle (degrees) at vertex B, formed by points A-B-C
function angleBetween(a, b, c) {
    const ba = { x: a.x - b.x, y: a.y - b.y }
    const bc = { x: c.x - b.x, y: c.y - b.y }
    const dot = ba.x * bc.x + ba.y * bc.y
    const magBa = Math.sqrt(ba.x ** 2 + ba.y ** 2)
    const magBc = Math.sqrt(bc.x ** 2 + bc.y ** 2)
    if (magBa * magBc === 0) return 0
    const cos = Math.max(-1, Math.min(1, dot / (magBa * magBc)))
    return (Math.acos(cos) * 180) / Math.PI
}

// Check if a landmark has enough visibility
function visible(lm, threshold = 0.35) {
    return lm && lm.visibility > threshold
}

/**
 * Main analysis function.
 * @param {Array} landmarks - raw MediaPipe pose landmark array
 * @returns {Object} - pose_risk, joint_angles, asymmetry, issues, fatigue_score, alert_level, injury_probability
 */
export function analyzePose(landmarks) {
    if (!landmarks || landmarks.length < 29) {
        return null
    }

    const lm = landmarks
    const issues = []
    const jointAngles = {}
    let risk = 0

    // ── Knee Angles ────────────────────────────────────────────
    const lKneeAngle =
        visible(lm[23]) && visible(lm[25]) && visible(lm[27])
            ? angleBetween(lm[23], lm[25], lm[27])
            : null

    const rKneeAngle =
        visible(lm[24]) && visible(lm[26]) && visible(lm[28])
            ? angleBetween(lm[24], lm[26], lm[28])
            : null

    if (lKneeAngle !== null) {
        jointAngles['knee_left'] = Math.round(lKneeAngle)
        if (lKneeAngle < 40) {
            risk += 30
            issues.push(`Left knee dangerously bent: ${Math.round(lKneeAngle)}°`)
        } else if (lKneeAngle < 70) {
            risk += 12
            issues.push(`Left knee deep flexion: ${Math.round(lKneeAngle)}°`)
        }
    }

    if (rKneeAngle !== null) {
        jointAngles['knee_right'] = Math.round(rKneeAngle)
        if (rKneeAngle < 40) {
            risk += 30
            issues.push(`Right knee dangerously bent: ${Math.round(rKneeAngle)}°`)
        } else if (rKneeAngle < 70) {
            risk += 12
            issues.push(`Right knee deep flexion: ${Math.round(rKneeAngle)}°`)
        }
    }

    // ── Knee Asymmetry ──────────────────────────────────────────
    if (lKneeAngle !== null && rKneeAngle !== null) {
        const kneeDiff = Math.abs(lKneeAngle - rKneeAngle)
        if (kneeDiff > 20) {
            risk += 15
            issues.push(`Knee asymmetry: ${Math.round(kneeDiff)}° difference`)
        }
    }

    // ── Elbow Angles ────────────────────────────────────────────
    const lElbowAngle =
        visible(lm[11]) && visible(lm[13]) && visible(lm[15])
            ? angleBetween(lm[11], lm[13], lm[15])
            : null

    const rElbowAngle =
        visible(lm[12]) && visible(lm[14]) && visible(lm[16])
            ? angleBetween(lm[12], lm[14], lm[16])
            : null

    if (lElbowAngle !== null) jointAngles['elbow_left'] = Math.round(lElbowAngle)
    if (rElbowAngle !== null) jointAngles['elbow_right'] = Math.round(rElbowAngle)

    // ── Shoulder Angles ─────────────────────────────────────────
    const lShoulderAngle =
        visible(lm[13]) && visible(lm[11]) && visible(lm[23])
            ? angleBetween(lm[13], lm[11], lm[23])
            : null

    const rShoulderAngle =
        visible(lm[14]) && visible(lm[12]) && visible(lm[24])
            ? angleBetween(lm[14], lm[12], lm[24])
            : null

    if (lShoulderAngle !== null) {
        jointAngles['shoulder_left'] = Math.round(lShoulderAngle)
        if (lShoulderAngle > 160) {
            risk += 20
            issues.push(`Left shoulder hyperextension: ${Math.round(lShoulderAngle)}°`)
        }
    }
    if (rShoulderAngle !== null) {
        jointAngles['shoulder_right'] = Math.round(rShoulderAngle)
        if (rShoulderAngle > 160) {
            risk += 20
            issues.push(`Right shoulder hyperextension: ${Math.round(rShoulderAngle)}°`)
        }
    }

    // ── Hip Angles ──────────────────────────────────────────────
    const lHipAngle =
        visible(lm[11]) && visible(lm[23]) && visible(lm[25])
            ? angleBetween(lm[11], lm[23], lm[25])
            : null

    const rHipAngle =
        visible(lm[12]) && visible(lm[24]) && visible(lm[26])
            ? angleBetween(lm[12], lm[24], lm[26])
            : null

    if (lHipAngle !== null) jointAngles['hip_left'] = Math.round(lHipAngle)
    if (rHipAngle !== null) jointAngles['hip_right'] = Math.round(rHipAngle)

    // ── Spine / Trunk Lean ──────────────────────────────────────
    const spineAngle =
        visible(lm[11]) && visible(lm[23]) && visible(lm[12]) && visible(lm[24])
            ? (() => {
                const midShoulder = {
                    x: (lm[11].x + lm[12].x) / 2,
                    y: (lm[11].y + lm[12].y) / 2,
                    visibility: 1,
                }
                const midHip = {
                    x: (lm[23].x + lm[24].x) / 2,
                    y: (lm[23].y + lm[24].y) / 2,
                    visibility: 1,
                }
                // Angle of spine vs vertical
                const dy = midShoulder.y - midHip.y
                const dx = midShoulder.x - midHip.x
                return Math.abs((Math.atan2(dx, dy) * 180) / Math.PI)
            })()
            : null

    if (spineAngle !== null) {
        jointAngles['spine'] = Math.round(spineAngle)
        if (spineAngle > 25) {
            risk += 20
            issues.push(`Excessive trunk lean: ${Math.round(spineAngle)}°`)
        } else if (spineAngle > 15) {
            risk += 8
            issues.push(`Moderate trunk lean: ${Math.round(spineAngle)}°`)
        }
    }

    // ── Shoulder-Hip Alignment (Lateral Sway) ──────────────────
    if (visible(lm[11]) && visible(lm[12]) && visible(lm[23]) && visible(lm[24])) {
        const shoulderMidX = (lm[11].x + lm[12].x) / 2
        const hipMidX = (lm[23].x + lm[24].x) / 2
        const lateralSway = Math.abs(shoulderMidX - hipMidX)
        if (lateralSway > 0.08) {
            risk += 10
            issues.push('Lateral body sway detected')
        }
    }

    const poseRisk = Math.min(100, Math.round(risk))

    // ── Alert Level ─────────────────────────────────────────────
    let alertLevel = 'GREEN'
    if (poseRisk >= 60) alertLevel = 'RED'
    else if (poseRisk >= 30) alertLevel = 'YELLOW'

    const injuryProbability = Math.min(100, Math.round(poseRisk * 0.85))

    return {
        pose_risk: poseRisk,
        facial_stress: 0,       // face analysis still requires backend
        object_risk: 0,
        object_speed: 0,
        injury_probability: injuryProbability,
        injury_type: issues.length > 0 ? issues[0] : 'None detected',
        time_horizon: 'real-time',
        alert_level: alertLevel,
        alert_message: issues.slice(0, 2).join('. '),
        contributing_factors: issues,
        recommended_action: alertLevel === 'RED'
            ? 'Stop and rest immediately'
            : alertLevel === 'YELLOW'
                ? 'Adjust your posture and slow down'
                : 'Form looks good — keep it up!',
        joint_angles: jointAngles,
        asymmetry: {},
        fatigue_score: 0,
        skeleton_landmarks: [],
        face_detected: false,
        issues,
    }
}
