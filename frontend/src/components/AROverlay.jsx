import { useMemo } from 'react'

export default function AROverlay({ analysis, width = 640, height = 480 }) {
    if (!analysis || !analysis.skeleton_landmarks) return null

    const landmarks = analysis.skeleton_landmarks
    const issues = analysis.issues || []

    // Define skeleton connections (indices based on MediaPipe Pose)
    const CONNECTIONS = [
        [11, 12], // Shoulders
        [11, 13], [13, 15], // Left Arm
        [12, 14], [14, 16], // Right Arm
        [11, 23], [12, 24], // Torso
        [23, 24], // Hips
        [23, 25], [25, 27], // Left Leg
        [24, 26], [26, 28]  // Right Leg
    ]

    // Helper to check if a joint is problematic
    const getJointStatus = (idx) => {
        // Map index to name (simplified)
        const nameMap = {
            11: 'shoulder', 12: 'shoulder',
            13: 'elbow', 14: 'elbow',
            23: 'hip', 24: 'hip',
            25: 'knee', 26: 'knee'
        }
        const name = nameMap[idx]
        if (!name) return 'ok'

        const issueStr = issues.join(' ').toLowerCase()
        if (issueStr.includes(name)) return 'danger'
        return 'ok'
    }

    const getColor = (status) => {
        if (status === 'danger') return '#ef4444' // Red
        return '#00ff9d' // Neon Green
    }

    return (
        <svg
            className="ar-overlay"
            viewBox="0 0 1 1"
            preserveAspectRatio="none"
            style={{
                position: 'absolute',
                top: 0,
                left: 0,
                width: '100%',
                height: '100%',
                pointerEvents: 'none',
                zIndex: 5
            }}
        >
            {/* Draw Connections */}
            {CONNECTIONS.map(([start, end], i) => {
                const p1 = landmarks[start]
                const p2 = landmarks[end]
                // Check visibility
                if (!p1 || !p2 || p1[3] < 0.5 || p2[3] < 0.5) return null

                const status1 = getJointStatus(start)
                const status2 = getJointStatus(end)
                const isBad = status1 === 'danger' || status2 === 'danger'

                return (
                    <line
                        key={i}
                        x1={p1[0]} y1={p1[1]}
                        x2={p2[0]} y2={p2[1]}
                        stroke={isBad ? '#ef4444' : 'rgba(0, 255, 157, 0.6)'}
                        strokeWidth="0.008"
                        strokeLinecap="round"
                    />
                )
            })}

            {/* Draw Joints */}
            {landmarks.map((lm, i) => {
                // Only draw main body joints (11-32)
                if (i < 11 || i > 32 || lm[3] < 0.5) return null

                const status = getJointStatus(i)
                const color = getColor(status)

                return (
                    <circle
                        key={i}
                        cx={lm[0]}
                        cy={lm[1]}
                        r={status === 'danger' ? 0.012 : 0.008}
                        fill={color}
                        stroke="rgba(0,0,0,0.5)"
                        strokeWidth="0.002"
                        className={status === 'danger' ? 'pulse-anim' : ''}
                    />
                )
            })}
        </svg>
    )
}
