/**
 * PlayerStatus.jsx
 * Premium 3D-style body heatmap with glowing joints and animated angle arc cards
 */

export default function PlayerStatus({ analysis }) {
    const joints = analysis?.joint_angles || {}
    const issues = analysis?.issues || []
    const active = Object.keys(joints).length > 0
    const issueStr = issues.join(' ').toLowerCase()

    const JOINTS = [
        { id: 'knee_left', label: 'L Knee', cx: 36, cy: 138, zone: 'knee' },
        { id: 'knee_right', label: 'R Knee', cx: 64, cy: 138, zone: 'knee' },
        { id: 'hip_left', label: 'L Hip', cx: 38, cy: 95, zone: 'hip' },
        { id: 'hip_right', label: 'R Hip', cx: 62, cy: 95, zone: 'hip' },
        { id: 'shoulder_left', label: 'L Shoulder', cx: 28, cy: 50, zone: 'shoulder' },
        { id: 'shoulder_right', label: 'R Shoulder', cx: 72, cy: 50, zone: 'shoulder' },
        { id: 'elbow_left', label: 'L Elbow', cx: 16, cy: 75, zone: 'elbow' },
        { id: 'elbow_right', label: 'R Elbow', cx: 84, cy: 75, zone: 'elbow' },
        { id: 'spine', label: 'Spine', cx: 50, cy: 70, zone: 'spine' },
    ]

    function jointColor(id, zone) {
        const val = joints[id]
        if (!active || val == null) return { ring: '#2a3060', fill: '#1a1f38', glow: 'none' }
        // Check if risky by issue text or threshold
        const risky = issueStr.includes(zone)
            || (zone === 'knee' && val < 70)
            || (zone === 'shoulder' && val > 160)
            || (zone === 'spine' && val > 25)
        if (risky) return { ring: '#ff3b3b', fill: 'rgba(255,59,59,0.25)', glow: '0 0 10px #ff3b3b' }
        if (val < 100) return { ring: '#ffbe2e', fill: 'rgba(255,190,46,0.15)', glow: '0 0 6px #ffbe2e' }
        return { ring: '#00ff9d', fill: 'rgba(0,255,157,0.12)', glow: '0 0 6px #00ff9d' }
    }

    function boneColor(zone1, zone2) {
        const r1 = issueStr.includes(zone1)
        const r2 = issueStr.includes(zone2)
        if (r1 || r2) return '#ff3b3b88'
        if (active) return '#00ff9d44'
        return '#2a3060'
    }

    return (
        <div>
            {/* Status row */}
            <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: '10px' }}>
                <span style={{ fontFamily: 'var(--font-mono)', fontSize: '0.65rem', color: 'var(--text-dim)', letterSpacing: '2px' }}>
                    {active ? `${Object.keys(joints).length} JOINTS` : 'SCANNING...'}
                </span>
                <span style={{
                    padding: '2px 10px', borderRadius: '100px',
                    fontFamily: 'var(--font-hud)', fontSize: '0.65rem', fontWeight: 700, letterSpacing: '2px',
                    background: active ? 'rgba(0,255,157,0.1)' : 'rgba(255,255,255,0.04)',
                    color: active ? '#00ff9d' : 'var(--text-dim)',
                    border: `1px solid ${active ? '#00ff9d44' : '#2a3060'}`,
                }}>
                    {active ? 'LIVE' : 'WAITING'}
                </span>
            </div>

            <div style={{ display: 'flex', gap: '12px', alignItems: 'flex-start' }}>
                {/* ── Body Diagram ── */}
                <div style={{ flexShrink: 0, width: 88 }}>
                    <svg viewBox="0 0 100 185" fill="none" style={{ width: '100%', overflow: 'visible' }}>
                        <defs>
                            <filter id="jglow"><feGaussianBlur stdDeviation="2.5" result="b" /><feMerge><feMergeNode in="b" /><feMergeNode in="SourceGraphic" /></feMerge></filter>
                        </defs>

                        {/* Head */}
                        <circle cx="50" cy="18" r="13"
                            stroke={active ? '#00e5ff88' : '#2a3060'} strokeWidth="2"
                            fill={active ? 'rgba(0,229,255,0.08)' : 'transparent'}
                            filter={active ? 'url(#jglow)' : 'none'}
                        />

                        {/* Bones (lines) */}
                        {/* Neck */}
                        <line x1="50" y1="31" x2="50" y2="46" stroke={boneColor('head', 'spine')} strokeWidth="2.5" strokeLinecap="round" />
                        {/* Shoulders */}
                        <line x1="28" y1="50" x2="72" y2="50" stroke={boneColor('shoulder', 'shoulder')} strokeWidth="2.5" strokeLinecap="round" />
                        {/* Torso */}
                        <line x1="50" y1="50" x2="50" y2="92" stroke={boneColor('spine', 'hip')} strokeWidth="2.5" strokeLinecap="round" />
                        {/* L arm upper */}
                        <line x1="28" y1="50" x2="16" y2="75" stroke={boneColor('shoulder', 'elbow')} strokeWidth="2" strokeLinecap="round" />
                        {/* L arm lower */}
                        <line x1="16" y1="75" x2="10" y2="102" stroke={boneColor('elbow', 'wrist')} strokeWidth="2" strokeLinecap="round" />
                        {/* R arm upper */}
                        <line x1="72" y1="50" x2="84" y2="75" stroke={boneColor('shoulder', 'elbow')} strokeWidth="2" strokeLinecap="round" />
                        {/* R arm lower */}
                        <line x1="84" y1="75" x2="90" y2="102" stroke={boneColor('elbow', 'wrist')} strokeWidth="2" strokeLinecap="round" />
                        {/* Hips */}
                        <line x1="38" y1="92" x2="62" y2="92" stroke={boneColor('hip', 'hip')} strokeWidth="2.5" strokeLinecap="round" />
                        {/* L leg upper */}
                        <line x1="38" y1="92" x2="36" y2="138" stroke={boneColor('hip', 'knee')} strokeWidth="2.5" strokeLinecap="round" />
                        {/* L leg lower */}
                        <line x1="36" y1="138" x2="33" y2="178" stroke={boneColor('knee', 'ankle')} strokeWidth="2" strokeLinecap="round" />
                        {/* R leg upper */}
                        <line x1="62" y1="92" x2="64" y2="138" stroke={boneColor('hip', 'knee')} strokeWidth="2.5" strokeLinecap="round" />
                        {/* R leg lower */}
                        <line x1="64" y1="138" x2="67" y2="178" stroke={boneColor('knee', 'ankle')} strokeWidth="2" strokeLinecap="round" />

                        {/* Joint dots */}
                        {JOINTS.map(j => {
                            const { ring, fill, glow } = jointColor(j.id, j.zone)
                            const val = joints[j.id]
                            return (
                                <g key={j.id} filter="url(#jglow)">
                                    {/* Outer ring */}
                                    <circle cx={j.cx} cy={j.cy} r="7" stroke={ring} strokeWidth="1.5" fill={fill} />
                                    {/* Center dot */}
                                    <circle cx={j.cx} cy={j.cy} r="2.5" fill={ring} />
                                    {/* Angle micro-text */}
                                    {val != null && (
                                        <text x={j.cx} y={j.cy - 10} textAnchor="middle"
                                            fill={ring} fontSize="5.5" fontFamily="var(--font-mono)" fontWeight="bold">
                                            {val}°
                                        </text>
                                    )}
                                </g>
                            )
                        })}
                    </svg>
                </div>

                {/* ── Joint Angle Cards ── */}
                <div style={{ flex: 1, display: 'flex', flexDirection: 'column', gap: '5px' }}>
                    {JOINTS.map(j => {
                        const val = joints[j.id]
                        const { ring } = jointColor(j.id, j.zone)
                        const pct = val != null ? Math.min(100, (val / 180) * 100) : 0

                        return (
                            <div key={j.id} style={{
                                background: '#0b0e1a',
                                border: `1px solid ${val != null ? ring + '33' : '#1a1f38'}`,
                                borderRadius: '6px',
                                padding: '5px 8px',
                                transition: 'border-color 0.3s',
                            }}>
                                <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: '3px' }}>
                                    <span style={{ fontFamily: 'var(--font-mono)', fontSize: '0.6rem', color: 'var(--text-dim)', letterSpacing: '1px' }}>
                                        {j.label.toUpperCase()}
                                    </span>
                                    <span style={{
                                        fontFamily: 'var(--font-hud)', fontSize: '0.85rem', fontWeight: 700,
                                        color: val != null ? ring : '#3a4060',
                                        transition: 'color 0.3s',
                                    }}>
                                        {val != null ? `${val}°` : '—'}
                                    </span>
                                </div>
                                {/* Angle bar */}
                                <div style={{ height: '3px', background: '#1a1f38', borderRadius: '3px', overflow: 'hidden' }}>
                                    <div style={{
                                        height: '100%',
                                        width: `${pct}%`,
                                        background: ring,
                                        boxShadow: val != null ? `0 0 4px ${ring}` : 'none',
                                        borderRadius: '3px',
                                        transition: 'width 0.4s ease, background 0.3s',
                                    }} />
                                </div>
                            </div>
                        )
                    })}
                </div>
            </div>

            {/* Issues list */}
            {issues.length > 0 && (
                <div style={{
                    marginTop: '10px',
                    padding: '8px 10px',
                    background: 'rgba(255,59,59,0.06)',
                    border: '1px solid rgba(255,59,59,0.25)',
                    borderRadius: '6px',
                    maxHeight: '72px',
                    overflowY: 'auto',
                }}>
                    {issues.slice(0, 3).map((iss, i) => (
                        <div key={i} style={{
                            fontFamily: 'var(--font-mono)', fontSize: '0.65rem', color: '#ff6666',
                            padding: '2px 0', borderBottom: i < Math.min(issues.length, 3) - 1 ? '1px solid rgba(255,59,59,0.15)' : 'none',
                        }}>
                            ⚠ {iss}
                        </div>
                    ))}
                </div>
            )}
        </div>
    )
}
