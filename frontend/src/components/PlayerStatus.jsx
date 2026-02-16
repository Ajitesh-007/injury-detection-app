export default function PlayerStatus({ analysis }) {
    const joints = analysis?.joint_angles || {}
    const asymmetry = analysis?.asymmetry || {}
    const issues = analysis?.issues || []

    // Group angles for display
    const displayJoints = [
        { name: 'L Knee', key: 'knee_left' },
        { name: 'R Knee', key: 'knee_right' },
        { name: 'L Hip', key: 'hip_left' },
        { name: 'R Hip', key: 'hip_right' },
        { name: 'L Shoulder', key: 'shoulder_left' },
        { name: 'R Shoulder', key: 'shoulder_right' },
        { name: 'L Elbow', key: 'elbow_left' },
        { name: 'R Elbow', key: 'elbow_right' },
        { name: 'Spine', key: 'spine_center' },
    ]

    return (
        <div style={{ height: '100%', display: 'flex', flexDirection: 'column' }}>
            <div style={{
                display: 'flex', justifyContent: 'space-between', alignItems: 'center',
                marginBottom: '16px', paddingBottom: '12px', borderBottom: '1px solid #333'
            }}>
                <span className="card-title" style={{ margin: 0 }}>Status</span>
                <span className="card-badge" style={{
                    background: 'rgba(6, 182, 212, 0.15)',
                    color: '#06b6d4',
                    fontSize: '0.7rem'
                }}>
                    {Object.keys(joints).length > 0 ? 'ACTIVE' : 'WAITING'}
                </span>
            </div>

            <div style={{ flex: 1, display: 'flex', gap: '12px' }}>
                {/* Body Diagram */}
                <div className="body-diagram" style={{ flex: '0 0 100px', opacity: 0.8 }}>
                    <svg viewBox="0 0 100 180" className="body-svg" fill="none">
                        {/* Head */}
                        <circle cx="50" cy="20" r="12" stroke={getZoneColor('head', issues)} strokeWidth="2" />
                        {/* Neck */}
                        <line x1="50" y1="32" x2="50" y2="42" stroke="#444" strokeWidth="2" />
                        {/* Torso */}
                        <rect x="30" y="42" width="40" height="45" rx="5"
                            stroke={getZoneColor('spine', issues)} strokeWidth="2" />
                        {/* Left arm */}
                        <line x1="30" y1="48" x2="12" y2="70" stroke={getZoneColor('shoulder_left', issues)} strokeWidth="2" />
                        <line x1="12" y1="70" x2="8" y2="95" stroke={getZoneColor('elbow_left', issues)} strokeWidth="2" />
                        {/* Right arm */}
                        <line x1="70" y1="48" x2="88" y2="70" stroke={getZoneColor('shoulder_right', issues)} strokeWidth="2" />
                        <line x1="88" y1="70" x2="92" y2="95" stroke={getZoneColor('elbow_right', issues)} strokeWidth="2" />
                        {/* Left leg */}
                        <line x1="38" y1="87" x2="30" y2="130" stroke={getZoneColor('hip_left', issues)} strokeWidth="2" />
                        <line x1="30" y1="130" x2="25" y2="170" stroke={getZoneColor('knee_left', issues)} strokeWidth="2" />
                        {/* Right leg */}
                        <line x1="62" y1="87" x2="70" y2="130" stroke={getZoneColor('hip_right', issues)} strokeWidth="2" />
                        <line x1="70" y1="130" x2="75" y2="170" stroke={getZoneColor('knee_right', issues)} strokeWidth="2" />
                        {/* Joint dots */}
                        {[
                            [50, 20], [30, 48], [70, 48], [12, 70], [88, 70],
                            [38, 87], [62, 87], [30, 130], [70, 130], [25, 170], [75, 170]
                        ].map(([cx, cy], i) => (
                            <circle key={i} cx={cx} cy={cy} r="3" fill="#06b6d4" opacity="0.5" />
                        ))}
                    </svg>
                </div>

                {/* Joint Values */}
                <div className="joint-readouts" style={{ flex: 1, overflowY: 'auto', fontSize: '0.8rem' }}>
                    {displayJoints.map(({ name, key }) => {
                        const val = joints[key]
                        return (
                            <div key={key} className="joint-item" style={{
                                display: 'flex', justifyContent: 'space-between',
                                borderBottom: '1px solid #222', padding: '4px 0'
                            }}>
                                <span className="joint-name" style={{ color: '#888' }}>{name}</span>
                                <span className="joint-value" style={{
                                    fontFamily: 'var(--font-mono)',
                                    color: getAngleColor(val)
                                }}>
                                    {val != null ? `${val.toFixed(0)}°` : '—'}
                                </span>
                            </div>
                        )
                    })}
                </div>
            </div>

            {/* Asymmetry warnings */}
            {Object.keys(asymmetry).length > 0 && (
                <div style={{ marginTop: '12px', borderTop: '1px solid #333', paddingTop: '8px' }}>
                    <div style={{ fontSize: '0.7rem', color: '#666', marginBottom: '4px', textTransform: 'uppercase' }}>
                        Asymmetry
                    </div>
                    {Object.entries(asymmetry).slice(0, 3).map(([joint, diff]) => (
                        <div key={joint} style={{
                            display: 'flex', justifyContent: 'space-between',
                            padding: '2px 0', fontSize: '0.75rem'
                        }}>
                            <span style={{ textTransform: 'capitalize', color: '#aaa' }}>{joint}</span>
                            <span style={{
                                fontFamily: 'var(--font-mono)',
                                color: diff > 15 ? '#ef4444' : diff > 8 ? '#f59e0b' : '#22c55e'
                            }}>
                                {diff.toFixed(1)}°
                            </span>
                        </div>
                    ))}
                </div>
            )}
        </div>
    )
}

function getZoneColor(zone, issues) {
    const issueStr = issues.join(' ').toLowerCase()
    const zoneLower = zone.toLowerCase().replace('_', ' ')
    if (issueStr.includes(zoneLower) || issueStr.includes(zone.split('_')[0])) {
        return '#ef4444'
    }
    return '#64748b'
}

function getAngleColor(val) {
    if (val == null) return '#64748b'
    if (val < 40 || val > 175) return '#ef4444'
    if (val < 60 || val > 165) return '#f59e0b'
    return '#22c55e'
}
