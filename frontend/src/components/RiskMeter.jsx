import { useMemo } from 'react'

export default function RiskMeter({ analysis }) {
    const score = analysis?.injury_probability || 0
    const level = analysis?.alert_level || 'GREEN'
    const injuryType = analysis?.injury_type || 'None'
    const timeHorizon = analysis?.time_horizon || 'long-term'

    const color = useMemo(() => {
        if (score < 35) return '#22c55e'
        if (score < 70) return '#f59e0b'
        return '#ef4444'
    }, [score])

    // SVG arc math for the gauge
    const radius = 70
    const strokeWidth = 12
    const centerX = 90
    const centerY = 85
    const circumference = Math.PI * radius // half circle
    const offset = circumference - (score / 100) * circumference


    return (
        <div className="risk-meter-container" style={{ padding: '0' }}>
            {/* Gauge */}
            <div className="risk-gauge">
                <svg viewBox="0 0 180 100">
                    {/* Background arc */}
                    <path
                        d="M 20 85 A 70 70 0 0 1 160 85"
                        className="risk-gauge-bg"
                        style={{ stroke: '#333' }}
                    />
                    {/* Filled arc */}
                    <path
                        d="M 20 85 A 70 70 0 0 1 160 85"
                        className="risk-gauge-fill"
                        style={{
                            stroke: color,
                            strokeDasharray: circumference,
                            strokeDashoffset: offset,
                            filter: `drop-shadow(0 0 8px ${color})`
                        }}
                    />
                    {/* Center value */}
                    <text
                        x={centerX}
                        y={centerY - 15}
                        textAnchor="middle"
                        fill={color}
                        fontSize="32"
                        fontWeight="800"
                        fontFamily="var(--font-mono)"
                        style={{ textShadow: `0 0 10px ${color}` }}
                    >
                        {score.toFixed(0)}
                    </text>
                    <text
                        x={centerX + 25}
                        y={centerY - 18}
                        textAnchor="start"
                        fill={color}
                        fontSize="12"
                        fontWeight="400"
                        opacity="0.8"
                    >
                        %
                    </text>
                </svg>
            </div>

            {/* Labels */}
            <div className="risk-value" style={{ textAlign: 'center', marginTop: '-20px' }}>
                <div className="label" style={{ color: color, fontSize: '1.2rem', fontWeight: 'bold' }}>{injuryType}</div>
                <div className="label" style={{ color: 'var(--text-dim)', fontSize: '0.8rem' }}>
                    Horizon: {timeHorizon}
                </div>
            </div>
        </div>
    )
}

function getColor(value) {
    if (!value || value < 35) return '#22c55e'
    if (value < 70) return '#f59e0b'
    return '#ef4444'
}
