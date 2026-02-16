
import { useRef, useEffect } from 'react'

export default function SessionSummary({ stats, onClose }) {
    const modalRef = useRef(null)

    // Animation on mount
    useEffect(() => {
        if (modalRef.current) {
            modalRef.current.style.opacity = 0
            modalRef.current.style.transform = 'scale(0.9)'

            // Trigger reflow
            void modalRef.current.offsetWidth

            modalRef.current.style.transition = 'opacity 0.5s ease, transform 0.5s cubic-bezier(0.175, 0.885, 0.32, 1.275)'
            modalRef.current.style.opacity = 1
            modalRef.current.style.transform = 'scale(1)'
        }
    }, [])

    if (!stats) return null

    const { duration, avgScore, peakRisk, alertCount } = stats

    // Determine Rank
    let rank = "ROOKIE"
    let rankColor = "#f5f5f5"
    const score = parseInt(avgScore)

    if (score >= 95) { rank = "ELITE ATHLETE"; rankColor = "#ffd700"; } // Gold
    else if (score >= 85) { rank = "PRO"; rankColor = "#c0c0c0"; } // Silver
    else if (score >= 70) { rank = "GIFTED"; rankColor = "#cd7f32"; } // Bronze
    else if (score >= 50) { rank = "AMATEUR"; rankColor = "#4ade80"; } // Green

    return (
        <div style={{
            position: 'absolute',
            top: 0, left: 0, width: '100%', height: '100%',
            background: 'rgba(0,0,0,0.85)',
            backdropFilter: 'blur(8px)',
            zIndex: 1000,
            display: 'flex', alignItems: 'center', justifyContent: 'center'
        }}>
            <div
                ref={modalRef}
                style={{
                    background: 'linear-gradient(145deg, #1a1a1a, #0d0d0d)',
                    border: '1px solid #333',
                    borderRadius: '24px',
                    padding: '40px',
                    width: '90%', maxWidth: '500px',
                    textAlign: 'center',
                    boxShadow: '0 20px 50px rgba(0,0,0,0.5), 0 0 30px rgba(0, 229, 255, 0.1)',
                    position: 'relative',
                    color: '#fff'
                }}
            >
                <h2 style={{
                    fontSize: '2rem', marginBottom: '8px',
                    background: 'linear-gradient(to right, #fff, #aaa)',
                    WebkitBackgroundClip: 'text', WebkitTextFillColor: 'transparent'
                }}>
                    SESSION COMPLETE
                </h2>
                <div style={{ color: '#888', marginBottom: '32px', fontSize: '0.9rem', letterSpacing: '1px' }}>
                    PERFORMANCE REPORT
                </div>

                <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: '20px', marginBottom: '32px' }}>
                    <StatBox label="DURATION" value={duration} />
                    <StatBox label="SAFETY SCORE" value={avgScore} color={getColor(avgScore)} suffix="/100" />
                    <StatBox label="PEAK RISK" value={`${peakRisk}%`} color={getRiskColor(peakRisk)} />
                    <StatBox label="ALERTS" value={alertCount} color={alertCount > 0 ? '#ef4444' : '#22c55e'} />
                </div>

                <div style={{
                    borderTop: '1px solid #333', paddingTop: '24px', marginBottom: '32px',
                    display: 'flex', flexDirection: 'column', alignItems: 'center'
                }}>
                    <div style={{ fontSize: '0.8rem', color: '#666', marginBottom: '8px', textTransform: 'uppercase' }}>
                        ACHIEVED RANK
                    </div>
                    <div style={{
                        fontSize: '2.5rem', fontWeight: '900', color: rankColor,
                        textShadow: `0 0 20px ${rankColor}40`, letterSpacing: '2px'
                    }}>
                        {rank}
                    </div>
                </div>

                <button
                    onClick={onClose}
                    style={{
                        background: '#3b82f6', color: 'white', border: 'none',
                        padding: '12px 40px', fontSize: '1rem', fontWeight: 'bold',
                        borderRadius: '12px', cursor: 'pointer',
                        boxShadow: '0 4px 12px rgba(59, 130, 246, 0.3)',
                        transition: 'transform 0.2s'
                    }}
                    onMouseEnter={e => e.target.style.transform = 'translateY(-2px)'}
                    onMouseLeave={e => e.target.style.transform = 'translateY(0)'}
                >
                    CONTINUE
                </button>
            </div>
        </div>
    )
}

function StatBox({ label, value, color = '#fff', suffix = '' }) {
    return (
        <div style={{ background: 'rgba(255,255,255,0.03)', padding: '16px', borderRadius: '16px' }}>
            <div style={{ fontSize: '0.7rem', color: '#666', marginBottom: '4px', fontWeight: 'bold' }}>{label}</div>
            <div style={{ fontSize: '1.5rem', fontWeight: 'bold', color: color }}>
                {value}<span style={{ fontSize: '0.8rem', opacity: 0.6 }}>{suffix}</span>
            </div>
        </div>
    )
}

function getColor(score) {
    if (score >= 90) return '#22c55e'
    if (score >= 70) return '#f59e0b'
    return '#ef4444'
}

function getRiskColor(risk) {
    if (risk < 30) return '#22c55e'
    if (risk < 70) return '#f59e0b'
    return '#ef4444'
}
