import { useState, useCallback, useRef, useEffect } from 'react'
import VideoFeed from './components/VideoFeed'
import RiskMeter from './components/RiskMeter'
import PlayerStatus from './components/PlayerStatus'
import AlertPanel from './components/AlertPanel'
import SportSelector from './components/SportSelector'
import HistoryChart from './components/HistoryChart'
import SessionSummary from './components/SessionSummary'
import { analyzePose } from './poseAnalytics'
import { speak, PRIORITY, COACH_MESSAGES } from './voiceCoach'
import './App.css'

const API_BASE = import.meta.env.VITE_API_URL || ''
const WS_URL = API_BASE ? API_BASE.replace(/^http/, 'ws') + '/ws/analyze' : null

function App() {
  const [sport, setSport] = useState('generic')
  const [analysis, setAnalysis] = useState(null)
  const [alerts, setAlerts] = useState([])
  const [connected, setConnected] = useState(false)
  const [streaming, setStreaming] = useState(false)
  const [riskHistory, setRiskHistory] = useState([])
  const [sessionStats, setSessionStats] = useState(null)
  const [isMuted, setIsMuted] = useState(false)

  const wsRef = useRef(null)
  const startTimeRef = useRef(null)
  const lastAlertTime = useRef(0)
  const lastGreenTime = useRef(0)
  const prevAlertLevel = useRef('GREEN')
  const isMutedRef = useRef(isMuted)

  // Keep muted ref in sync so callbacks don't go stale
  useEffect(() => { isMutedRef.current = isMuted }, [isMuted])

  // Connect backend WebSocket (optional — for facial/object analysis)
  const connectWS = useCallback(() => {
    if (!WS_URL) return
    if (wsRef.current?.readyState === WebSocket.OPEN) return

    const ws = new WebSocket(WS_URL)
    ws.onopen = () => setConnected(true)
    ws.onmessage = (e) => {
      // Merge backend data into current analysis (backend can enrich readings)
      const data = JSON.parse(e.data)
      setAnalysis(prev => ({ ...prev, facial_stress: data.facial_stress, object_risk: data.object_risk, object_speed: data.object_speed }))
    }
    ws.onclose = () => setConnected(false)
    ws.onerror = () => setConnected(false)
    wsRef.current = ws
  }, [])

  // Send frame to backend (for facial & object analysis)
  const sendFrame = useCallback((base64Frame, width, height) => {
    if (wsRef.current?.readyState === WebSocket.OPEN) {
      wsRef.current.send(JSON.stringify({ image_base64: base64Frame, sport, frame_width: width, frame_height: height }))
    }
  }, [sport])

  // ── Core: handle landmarks from browser MediaPipe ───────────
  const handleLandmarks = useCallback((landmarks) => {
    const result = analyzePose(landmarks)
    if (!result) return

    setAnalysis(prev => ({
      ...result,
      // Keep backend enrichments if available
      facial_stress: prev?.facial_stress || 0,
      object_risk: prev?.object_risk || 0,
      object_speed: prev?.object_speed || 0,
    }))

    // ── Risk history for chart ────────────────────────────────
    setRiskHistory(prev => {
      const next = [...prev, { time: Date.now(), risk: result.pose_risk, level: result.alert_level }]
      return next.slice(-60)
    })

    // ── Alerts log ────────────────────────────────────────────
    if (result.alert_level !== 'GREEN') {
      setAlerts(prev => [
        { ...result, id: Date.now(), timestamp: new Date().toLocaleTimeString() },
        ...prev
      ].slice(0, 50))
    }

    // ── Voice coach ───────────────────────────────────────────
    const now = Date.now()
    const level = result.alert_level

    if (level === 'RED' && now - lastAlertTime.current > 5000) {
      const msg = result.recommended_action || COACH_MESSAGES.RED[0]
      speak(msg, PRIORITY.HIGH, isMutedRef.current)
      lastAlertTime.current = now
    } else if (level === 'YELLOW' && now - lastAlertTime.current > 8000) {
      const msgs = COACH_MESSAGES.YELLOW
      speak(msgs[Math.floor(Math.random() * msgs.length)], PRIORITY.NORMAL, isMutedRef.current)
      lastAlertTime.current = now
    } else if (level === 'GREEN' && now - lastGreenTime.current > 20000) {
      const msgs = COACH_MESSAGES.GREEN
      speak(msgs[Math.floor(Math.random() * msgs.length)], PRIORITY.LOW, isMutedRef.current)
      lastGreenTime.current = now
    }

    prevAlertLevel.current = level
  }, [])

  // ── Toggle streaming ──────────────────────────────────────
  const toggleStreaming = useCallback(() => {
    if (!streaming) {
      if (WS_URL) connectWS()
      setRiskHistory([])
      setAlerts([])
      setSessionStats(null)
      setAnalysis(null)
      startTimeRef.current = Date.now()
      // Welcome message after short delay (let camera open first)
      setTimeout(() => speak(COACH_MESSAGES.START, PRIORITY.NORMAL, isMutedRef.current), 1500)
    } else {
      wsRef.current?.close()
      speak(COACH_MESSAGES.STOP, PRIORITY.NORMAL, isMutedRef.current)

      if (startTimeRef.current && riskHistory.length > 0) {
        const durationSec = Math.floor((Date.now() - startTimeRef.current) / 1000)
        const avgRisk = riskHistory.reduce((a, b) => a + b.risk, 0) / riskHistory.length
        setSessionStats({
          duration: `${Math.floor(durationSec / 60)}m ${durationSec % 60}s`,
          avgScore: Math.max(0, 100 - avgRisk).toFixed(0),
          peakRisk: Math.max(...riskHistory.map(r => r.risk), 0).toFixed(0),
          alertCount: alerts.length,
        })
      }
    }
    setStreaming(prev => !prev)
  }, [streaming, connectWS, riskHistory, alerts.length])

  const handleSportChange = useCallback((newSport) => {
    setSport(newSport)
    if (wsRef.current?.readyState === WebSocket.OPEN) {
      wsRef.current.send(JSON.stringify({ sport: newSport }))
    }
  }, [])

  const alertLevel = analysis?.alert_level || 'GREEN'

  return (
    <div className="app-container">
      {/* ─── Header ──────────────────────────────────────────── */}
      <header className="header">
        <div className="header-brand">
          <div className="header-logo">⚡</div>
          <div>
            <h1>InjuryGuard <span>AI</span></h1>
            <div className="header-sub">Real-time Biomechanical Analysis</div>
          </div>
        </div>
        <div className="header-controls">
          <SportSelector sport={sport} onChange={handleSportChange} />
          <div className={`status-badge ${alertLevel.toLowerCase()}`}>
            <span className="status-dot" />
            {streaming ? alertLevel : (connected ? 'READY' : 'LOCAL')}
          </div>
          <button
            className="mute-btn"
            onClick={() => setIsMuted(p => !p)}
            title={isMuted ? 'Unmute voice coach' : 'Mute voice coach'}
          >
            {isMuted ? '🔇' : '🔊'}
          </button>
        </div>
      </header>

      {/* ─── Main Grid ───────────────────────────────────────── */}
      <main className="main-grid">
        {/* Left: Video */}
        <section className="video-section">
          <div className="video-container">
            <VideoFeed
              streaming={streaming}
              onFrame={sendFrame}
              onLandmarks={handleLandmarks}
              analysis={analysis}
              connected={connected}
              style={{ width: '100%', height: '100%' }}
            />

            {/* HUD overlay */}
            <div className="video-overlay">
              <div className="hud-row">
                <span className="hud-label">MODE</span>
                <span className="hud-value">{sport.toUpperCase()}</span>
              </div>
              <div className="hud-row">
                <span className="hud-label">FPS</span>
                <span className="hud-value">{streaming ? '~20' : '0'}</span>
              </div>
              <div className="hud-row">
                <span className="hud-label">ENGINE</span>
                <span className="hud-value" style={{ color: 'var(--primary)' }}>BROWSER AI</span>
              </div>
            </div>

            {streaming && <div className="scan-line" />}
            {streaming && (
              <>
                <div className="corner corner-tl" />
                <div className="corner corner-tr" />
                <div className="corner corner-bl" />
                <div className="corner corner-br" />
              </>
            )}

            {/* Start */}
            {!streaming && (
              <div className="start-overlay">
                <button className="start-btn" onClick={toggleStreaming}>
                  <span className="start-icon">▶</span>
                  Initialize
                </button>
                <p className="start-hint">Camera · AI Coach · Live Readings</p>
              </div>
            )}

            {/* Stop */}
            {streaming && (
              <button className="stop-btn" onClick={toggleStreaming}>
                ⏹ End Session
              </button>
            )}
          </div>

          {/* Metrics Bar */}
          <div className="metrics-bar">
            <MetricItem label="Pose Risk" value={analysis?.pose_risk} unit="%" />
            <MetricItem label="Facial Stress" value={analysis?.facial_stress} unit="%" />
            <MetricItem label="Impact Risk" value={analysis?.object_risk} unit="%" />
            <MetricItem label="Speed" value={analysis?.object_speed} unit=" km/h" color="#00e5ff" />
          </div>
        </section>

        {/* Right: Panels */}
        <aside className="right-panel">
          <div className="panel-card">
            <h3><span className="panel-icon">📊</span> Live Analysis</h3>
            <RiskMeter analysis={analysis} />
          </div>

          <div className="panel-card">
            <h3><span className="panel-icon">🦴</span> Joint Readings</h3>
            <JointReadings angles={analysis?.joint_angles} />
          </div>

          <div className="panel-card flex-card">
            <h3><span className="panel-icon">🚨</span> Alert Log</h3>
            <div className="alert-scroll">
              <AlertPanel alerts={alerts} />
            </div>
          </div>

          <div className="panel-card">
            <h3><span className="panel-icon">📈</span> Risk Trend</h3>
            <HistoryChart history={riskHistory} />
          </div>
        </aside>
      </main>

      <SessionSummary stats={sessionStats} onClose={() => setSessionStats(null)} />
    </div>
  )
}

// ── Joint Readings panel component ──────────────────────────────
function JointReadings({ angles }) {
  if (!angles || Object.keys(angles).length === 0) {
    return (
      <div style={{ color: 'var(--text-dim)', fontFamily: 'var(--font-mono)', fontSize: '0.75rem', textAlign: 'center', padding: '12px 0' }}>
        Stand in front of camera to detect joints
      </div>
    )
  }

  const labels = {
    knee_left: 'L Knee', knee_right: 'R Knee',
    hip_left: 'L Hip', hip_right: 'R Hip',
    shoulder_left: 'L Shoulder', shoulder_right: 'R Shoulder',
    elbow_left: 'L Elbow', elbow_right: 'R Elbow',
    spine: 'Spine',
  }

  return (
    <div style={{ display: 'flex', flexDirection: 'column', gap: '6px' }}>
      {Object.entries(angles).map(([key, val]) => {
        const label = labels[key] || key
        const isRisky = (key.includes('knee') && val < 70) || (key.includes('shoulder') && val > 160) || (key === 'spine' && val > 25)
        const color = isRisky ? 'var(--alert-red)' : 'var(--primary)'
        return (
          <div key={key} style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center' }}>
            <span style={{ fontFamily: 'var(--font-mono)', fontSize: '0.7rem', color: 'var(--text-dim)', letterSpacing: '1px' }}>{label}</span>
            <div style={{ display: 'flex', alignItems: 'center', gap: '8px' }}>
              {/* Mini angle bar */}
              <div style={{ width: '60px', height: '3px', background: '#1a1f38', borderRadius: '2px', overflow: 'hidden' }}>
                <div style={{ width: `${Math.min(100, (val / 180) * 100)}%`, height: '100%', background: color, borderRadius: '2px', transition: 'width 0.3s' }} />
              </div>
              <span style={{ fontFamily: 'var(--font-hud)', fontSize: '0.85rem', fontWeight: 700, color, minWidth: '38px', textAlign: 'right' }}>{val}°</span>
            </div>
          </div>
        )
      })}
    </div>
  )
}

function MetricItem({ label, value, unit, color }) {
  const c = color || getColor(value)
  return (
    <div className="metric-item">
      <span className="metric-label">{label}</span>
      <span className="metric-value" style={{ color: c }}>
        {value != null ? Number(value).toFixed(0) : '0'}{unit}
      </span>
    </div>
  )
}

function getColor(value) {
  if (!value || value < 35) return '#22c55e'
  if (value < 70) return '#f59e0b'
  return '#ef4444'
}

export default App
