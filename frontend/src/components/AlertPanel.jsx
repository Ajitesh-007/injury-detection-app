import { useState } from 'react'

export default function AlertPanel({ alerts }) {
    const [expanded, setExpanded] = useState(null)

    if (alerts.length === 0) {
        return (
            <div className="card">
                <div className="card-header">
                    <span className="card-title">Alerts</span>
                    <span className="card-badge" style={{ background: 'rgba(34, 197, 94, 0.15)', color: '#22c55e' }}>
                        Clear
                    </span>
                </div>
                <div style={{
                    textAlign: 'center', padding: '24px 0',
                    fontSize: '0.85rem', color: '#64748b'
                }}>
                    <div style={{ fontSize: '2rem', marginBottom: '8px', opacity: 0.4 }}>✓</div>
                    No alerts — all clear
                </div>
            </div>
        )
    }


    return (
        <div className="alert-list-container">
            <div className="alert-list">
                {alerts.slice(0, 50).map((alert, idx) => (
                    <div
                        key={alert.id || idx}
                        className={`alert-item ${alert.alert_level?.toLowerCase()}`}
                        onClick={() => setExpanded(expanded === idx ? null : idx)}
                        style={{
                            marginBottom: '10px',
                            padding: '10px',
                            background: '#111',
                            borderLeft: `4px solid ${alert.alert_level === 'RED' ? 'var(--alert-red)' : 'var(--alert-yellow)'}`,
                            borderRadius: '0 4px 4px 0',
                            cursor: 'pointer'
                        }}
                    >
                        <div style={{ display: 'flex', justifyContent: 'space-between', marginBottom: '4px' }}>
                            <span style={{
                                fontWeight: 'bold',
                                color: alert.alert_level === 'RED' ? 'var(--alert-red)' : 'var(--alert-yellow)'
                            }}>
                                {alert.alert_level === 'RED' ? '🛑 DANGER' : '⚠️ WARNING'}
                            </span>
                            <span style={{ fontSize: '0.8rem', color: '#666', fontFamily: 'var(--font-mono)' }}>
                                {alert.timestamp}
                            </span>
                        </div>

                        <div style={{ fontSize: '0.9rem', color: '#ddd' }}>
                            {alert.message}
                        </div>

                        {/* Expanded factors */}
                        {expanded === idx && (
                            <div style={{ marginTop: '10px', padding: '8px', background: '#222', borderRadius: '4px' }}>
                                <div style={{ fontSize: '0.8rem', color: '#aaa', marginBottom: '4px' }}>CONTRIBUTING FACTORS:</div>
                                {alert.contributing_factors?.map((factor, fi) => (
                                    <div key={fi} style={{ fontSize: '0.85rem', color: '#fff', paddingLeft: '8px' }}>• {factor}</div>
                                ))}
                                {alert.recommended_action && (
                                    <div style={{ marginTop: '8px', color: 'var(--secondary)', fontSize: '0.9rem' }}>
                                        💡 {alert.recommended_action}
                                    </div>
                                )}
                            </div>
                        )}
                    </div>
                ))}
            </div>
        </div>
    )
}
