import { useMemo, useEffect, useRef } from 'react'

const COLORS = {
    GREEN: { main: '#00ff9d', glow: 'rgba(0,255,157,0.5)', label: 'LOW RISK', bg: 'rgba(0,255,157,0.06)' },
    YELLOW: { main: '#ffbe2e', glow: 'rgba(255,190,46,0.5)', label: 'CAUTION', bg: 'rgba(255,190,46,0.06)' },
    RED: { main: '#ff3b3b', glow: 'rgba(255,59,59,0.5)', label: 'HIGH RISK', bg: 'rgba(255,59,59,0.08)' },
}

export default function RiskMeter({ analysis }) {
    const score = analysis?.injury_probability ?? 0
    const level = analysis?.alert_level || 'GREEN'
    const injuryType = analysis?.injury_type || 'Monitoring...'
    const poseRisk = analysis?.pose_risk ?? 0
    const c = COLORS[level] || COLORS.GREEN

    // Animated score with lerp
    const displayRef = useRef(0)
    const animRef = useRef(null)
    const svgScoreRef = useRef(null)

    useEffect(() => {
        const target = score
        const animate = () => {
            const diff = target - displayRef.current
            if (Math.abs(diff) < 0.5) { displayRef.current = target }
            else { displayRef.current += diff * 0.12 }
            if (svgScoreRef.current) {
                svgScoreRef.current.textContent = Math.round(displayRef.current)
            }
            if (Math.abs(diff) > 0.5) animRef.current = requestAnimationFrame(animate)
        }
        animRef.current = requestAnimationFrame(animate)
        return () => cancelAnimationFrame(animRef.current)
    }, [score])

    // SVG arc (180° half-circle gauge)
    const r = 68
    const cx = 90
    const cy = 82
    const arc = Math.PI * r
    const offset = arc - (score / 100) * arc

    // Tick marks
    const ticks = [0, 25, 50, 75, 100]

    return (
        <div style={{
            background: c.bg,
            border: `1px solid ${c.main}22`,
            borderRadius: '10px',
            padding: '12px',
            transition: 'background 0.5s, border-color 0.5s',
        }}>
            {/* Gauge SVG */}
            <div style={{ position: 'relative', textAlign: 'center' }}>
                <svg viewBox="0 0 180 96" style={{ width: '100%', maxWidth: '220px', overflow: 'visible' }}>
                    <defs>
                        <filter id="glow">
                            <feGaussianBlur stdDeviation="3" result="coloredBlur" />
                            <feMerge><feMergeNode in="coloredBlur" /><feMergeNode in="SourceGraphic" /></feMerge>
                        </filter>
                        <linearGradient id="arcGrad" x1="0%" y1="0%" x2="100%" y2="0%">
                            <stop offset="0%" stopColor="#22c55e" />
                            <stop offset="50%" stopColor="#f59e0b" />
                            <stop offset="100%" stopColor="#ef4444" />
                        </linearGradient>
                    </defs>

                    {/* Track */}
                    <path
                        d={`M ${cx - r} ${cy} A ${r} ${r} 0 0 1 ${cx + r} ${cy}`}
                        fill="none" stroke="#1a1f38" strokeWidth="14" strokeLinecap="round"
                    />

                    {/* Gradient color track (background) */}
                    <path
                        d={`M ${cx - r} ${cy} A ${r} ${r} 0 0 1 ${cx + r} ${cy}`}
                        fill="none" stroke="url(#arcGrad)" strokeWidth="14" strokeLinecap="round"
                        strokeOpacity="0.15"
                    />

                    {/* Active fill */}
                    <path
                        d={`M ${cx - r} ${cy} A ${r} ${r} 0 0 1 ${cx + r} ${cy}`}
                        fill="none"
                        stroke={c.main}
                        strokeWidth="14"
                        strokeLinecap="round"
                        strokeDasharray={arc}
                        strokeDashoffset={offset}
                        filter="url(#glow)"
                        style={{ transition: 'stroke-dashoffset 0.4s ease, stroke 0.4s ease' }}
                    />

                    {/* Tick marks */}
                    {ticks.map(t => {
                        const angle = Math.PI - (t / 100) * Math.PI
                        const tx = cx + (r + 10) * Math.cos(angle)
                        const ty = cy - (r + 10) * Math.sin(angle)
                        return (
                            <g key={t}>
                                <circle cx={tx} cy={ty} r="2" fill={t <= score ? c.main : '#333'} style={{ transition: 'fill 0.4s' }} />
                                <text x={tx} y={ty - 5} textAnchor="middle" fill="#444" fontSize="7" fontFamily="var(--font-mono)">{t}</text>
                            </g>
                        )
                    })}

                    {/* Center score */}
                    <text ref={svgScoreRef} x={cx} y={cy - 8} textAnchor="middle"
                        fill={c.main} fontSize="36" fontWeight="800"
                        fontFamily="var(--font-hud)" filter="url(#glow)"
                        style={{ transition: 'fill 0.4s' }}>
                        {Math.round(score)}
                    </text>
                    <text x={cx} y={cy + 6} textAnchor="middle"
                        fill={c.main} fontSize="10" fontFamily="var(--font-mono)" opacity="0.8">
                        RISK %
                    </text>
                </svg>

                {/* Level badge */}
                <div style={{
                    display: 'inline-flex', alignItems: 'center', gap: '6px',
                    padding: '4px 14px',
                    background: `${c.main}18`,
                    border: `1px solid ${c.main}55`,
                    borderRadius: '100px',
                    marginTop: '-4px',
                }}>
                    <span style={{ width: 7, height: 7, borderRadius: '50%', background: c.main, boxShadow: `0 0 6px ${c.main}`, display: 'inline-block' }} />
                    <span style={{ fontFamily: 'var(--font-hud)', fontSize: '0.75rem', fontWeight: 700, letterSpacing: '2px', color: c.main }}>{c.label}</span>
                </div>
            </div>

            {/* Injury type */}
            <div style={{
                marginTop: '10px',
                padding: '8px 10px',
                background: '#0d1021',
                borderRadius: '6px',
                border: '1px solid #1a1f38',
                fontFamily: 'var(--font-mono)',
                fontSize: '0.7rem',
                color: poseRisk > 0 ? c.main : 'var(--text-dim)',
                letterSpacing: '0.5px',
                textAlign: 'center',
                lineHeight: 1.4,
            }}>
                {injuryType.length > 50 ? injuryType.slice(0, 47) + '…' : injuryType}
            </div>

            {/* Sub-metrics row */}
            <div style={{ display: 'flex', gap: '6px', marginTop: '8px' }}>
                <SubStat label="POSE" value={poseRisk} unit="%" color={getColor(poseRisk)} />
                <SubStat label="FACE" value={analysis?.facial_stress ?? 0} unit="%" color={getColor(analysis?.facial_stress)} />
                <SubStat label="FATIGUE" value={analysis?.fatigue_score ?? 0} unit="%" color={getColor(analysis?.fatigue_score)} />
            </div>
        </div>
    )
}

function SubStat({ label, value, unit, color }) {
    return (
        <div style={{
            flex: 1, textAlign: 'center',
            background: '#0b0e1a', borderRadius: '6px',
            padding: '6px 4px',
            border: '1px solid #1a1f38',
        }}>
            <div style={{ fontFamily: 'var(--font-hud)', fontSize: '1rem', fontWeight: 700, color, lineHeight: 1 }}>{Math.round(value || 0)}{unit}</div>
            <div style={{ fontFamily: 'var(--font-mono)', fontSize: '0.55rem', color: 'var(--text-dim)', letterSpacing: '1.5px', marginTop: '3px' }}>{label}</div>
        </div>
    )
}

function getColor(v) {
    if (!v || v < 35) return '#22c55e'
    if (v < 70) return '#f59e0b'
    return '#ef4444'
}
