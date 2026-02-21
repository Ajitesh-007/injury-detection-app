import { useState, useCallback, useRef, useEffect } from 'react'
import VideoFeed from './components/VideoFeed'
import RiskMeter from './components/RiskMeter'
import PlayerStatus from './components/PlayerStatus'
import AlertPanel from './components/AlertPanel'
import SportSelector from './components/SportSelector'
import HistoryChart from './components/HistoryChart'
import SessionSummary from './components/SessionSummary'
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
  const [goodStreak, setGoodStreak] = useState(0)
  // Browser-side landmarks from MediaPipe JS
  const [browserLandmarks, setBrowserLandmarks] = useState(null)

  const wsRef = useRef(null)
  const audioRef = useRef(null)
  const startTimeRef = useRef(null)
  const lastCoachTime = useRef(0)

  // Initialize audio
  useEffect(() => {
    try {
      audioRef.current = new Audio('data:audio/wav;base64,UklGRigBAABXQVZFZm10IBAAAAABAAEARKwAAIhYAQACABAAZGF0YQQBAADg/+D/4P/g/+D/4P/g/+D/4P/g/+D/4P/g/8D/wP+g/4D/YP9A/yD/AP/g/sD+oP6A/mD+QP4g/gD+4P3A/aD9gP2A/YD9gP2g/aD9wP3g/QD+IP5A/mD+gP6g/sD+4P4A/yD/QP9g/4D/oP/A/+D/AAAQACAAQABQAGAAYAB4AIgAmACgAKgAsAC4ALgAuACwAKAAlACIAHgAYABQAEAAMAAgABAAAADg/8D/oP+A/2D/QP8g/wD/4P7A/qD+gP5g/kD+IP4A/uD9wP2g/YD9YP1A/SD9AP3g/MD8oPyA/GD8QPwg/AD84PvA+6D7gPtg+0D7')
      audioRef.current.volume = 0.3
    } catch (e) { /* ignore */ }
  }, [])

  // TTS helper
  const speak = useCallback((text, priority = 'normal') => {
    if (isMuted || !window.speechSynthesis) return
    try {
      if (priority === 'high') window.speechSynthesis.cancel()
      else if (window.speechSynthesis.speaking) return

      const utterance = new SpeechSynthesisUtterance(text)
      utterance.rate = 0.95
      utterance.pitch = 1.0
      utterance.volume = 1.0
      const voices = window.speechSynthesis.getVoices()
      const voice = voices.find(v =>
        v.name.includes('Google US English') ||
        v.name.includes('Natural') ||
        v.name.includes('Samantha') ||
        v.lang.startsWith('en-US')
      )
      if (voice) utterance.voice = voice
      window.speechSynthesis.speak(utterance)
    } catch (e) { /* ignore */ }
  }, [isMuted])

  // Connect WebSocket (only if backend URL configured)
  const connectWS = useCallback(() => {
    if (!WS_URL) { setConnected(false); return }
    if (wsRef.current?.readyState === WebSocket.OPEN) return

    const ws = new WebSocket(WS_URL)
    ws.onopen = () => { setConnected(true); console.log('WS connected') }

    ws.onmessage = (event) => {
      const data = JSON.parse(event.data)
      // Merge with browser landmarks
      setAnalysis(prev => ({
        ...data,
        // Keep skeleton from browser MediaPipe if backend didn't send one
        skeleton_landmarks: (data.skeleton_landmarks?.length > 0)
          ? data.skeleton_landmarks
          : (prev?.skeleton_landmarks || [])
      }))

      const risk = data.injury_probability || 0
      if (risk < 20 && data.alert_level === 'GREEN') {
        setGoodStreak(prev => {
          const next = prev + 1
          if (next >= 50 && (Date.now() - lastCoachTime.current) > 15000) {
            const msgs = [
              "Great form, keep it up!",
              "Balance looks perfect. You're in the zone.",
              "Excellent control. This is how pros do it.",
              "Your posture is rock solid. Nice work.",
              "Stay focused, you're doing amazing."
            ]
            speak(msgs[Math.floor(Math.random() * msgs.length)], 'normal')
            lastCoachTime.current = Date.now()
            return 0
          }
          return next
        })
      } else if (risk > 40) {
        setGoodStreak(0)
      }

      setRiskHistory(prev => {
        const next = [...prev, { time: Date.now(), risk: data.injury_probability || 0, level: data.alert_level || 'GREEN' }]
        return next.slice(-60)
      })

      if (data.alert_level && data.alert_level !== 'GREEN') {
        setAlerts(prev => [{ ...data, id: Date.now(), timestamp: new Date().toLocaleTimeString() }, ...prev].slice(0, 50))
        if (data.alert_level === 'RED') {
          if (!isMuted) audioRef.current?.play().catch(() => { })
          speak(data.recommended_action || data.injury_type || "High risk detected", 'high')
        }
      }
    }

    ws.onclose = () => setConnected(false)
    ws.onerror = () => setConnected(false)
    wsRef.current = ws
  }, [speak, isMuted])

  // Send frame to backend
  const sendFrame = useCallback((base64Frame, width, height) => {
    if (wsRef.current?.readyState === WebSocket.OPEN) {
      wsRef.current.send(JSON.stringify({ image_base64: base64Frame, sport, frame_width: width, frame_height: height }))
    }
  }, [sport])

  // Handle browser-side landmarks from MediaPipe JS (always shown even without backend)
  const handleLandmarks = useCallback((landmarks) => {
    setBrowserLandmarks(landmarks)
    // Create a minimal analysis object if no backend connected
    setAnalysis(prev => {
      if (!prev || !connected) {
        return {
          ...(prev || {}),
          skeleton_landmarks: landmarks.map(lm => [lm.x, lm.y, lm.z, lm.visibility]),
          alert_level: prev?.alert_level || 'GREEN',
          injury_probability: prev?.injury_probability || 0,
          pose_risk: prev?.pose_risk || 0,
          facial_stress: prev?.facial_stress || 0,
          object_risk: prev?.object_risk || 0,
          object_speed: prev?.object_speed || 0,
          fatigue_score: prev?.fatigue_score || 0,
          issues: prev?.issues || [],
        }
      }
      return prev
    })
  }, [connected])

  // Toggle streaming
  const toggleStreaming = useCallback(() => {
    if (!streaming) {
      if (WS_URL) connectWS()
      setRiskHistory([])
      setAlerts([])
      setSessionStats(null)
      setBrowserLandmarks(null)
      startTimeRef.current = Date.now()
    } else {
      wsRef.current?.close()
      if (startTimeRef.current && riskHistory.length > 0) {
        const durationSec = Math.floor((Date.now() - startTimeRef.current) / 1000)
        const duration = `${Math.floor(durationSec / 60)}m ${durationSec % 60}s`
        const avgRisk = riskHistory.reduce((a, b) => a + b.risk, 0) / riskHistory.length
        setSessionStats({
          duration,
          avgScore: Math.max(0, 100 - avgRisk).toFixed(0),
          peakRisk: Math.max(...riskHistory.map(r => r.risk), 0).toFixed(0),
          alertCount: alerts.length
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
      {/* ─── Header ─────────────────────────────────────────────── */}
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
            {connected ? alertLevel : (WS_URL ? 'OFFLINE' : 'LOCAL')}
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

      {/* ─── Main Grid ──────────────────────────────────────────── */}
      <main className="main-grid">
        {/* Left: Video Feed */}
        <section className="video-section">
          <div className="video-container">
            <VideoFeed
              streaming={streaming}
              onToggle={toggleStreaming}
              onFrame={sendFrame}
              onLandmarks={handleLandmarks}
              analysis={analysis}
              connected={connected}
              style={{ width: '100%', height: '100%' }}
            />

            {/* Overlay HUD (Top Left) */}
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
                <span className="hud-value" style={{ color: '#00ff9d' }}>BROWSER</span>
              </div>
            </div>

            {/* Live scan line when streaming */}
            {streaming && <div className="scan-line" />}

            {/* Corner brackets for HUD feel */}
            {streaming && (
              <>
                <div className="corner corner-tl" />
                <div className="corner corner-tr" />
                <div className="corner corner-bl" />
                <div className="corner corner-br" />
              </>
            )}

            {/* Start Button */}
            {!streaming && (
              <div className="start-overlay">
                <button className="start-btn" onClick={toggleStreaming}>
                  <span className="start-icon">▶</span>
                  Initialize
                </button>
                <p className="start-hint">Click to start pose detection</p>
              </div>
            )}

            {/* Stop Button */}
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

        {/* Right: Analysis Panel */}
        <aside className="right-panel">
          <div className="panel-card">
            <h3><span className="panel-icon">📊</span> Live Analysis</h3>
            <RiskMeter analysis={analysis} />
          </div>

          <div className="panel-card">
            <h3><span className="panel-icon">🏃</span> Player Status</h3>
            <PlayerStatus analysis={analysis} />
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

      {/* Session Summary Modal */}
      <SessionSummary stats={sessionStats} onClose={() => setSessionStats(null)} />
    </div>
  )
}

function MetricItem({ label, value, unit, color }) {
  const c = color || getColor(value)
  return (
    <div className="metric-item">
      <span className="metric-label">{label}</span>
      <span className="metric-value" style={{ color: c }}>
        {value != null ? value.toFixed(0) : '0'}{unit}
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
