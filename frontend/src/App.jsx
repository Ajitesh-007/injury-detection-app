import { useState, useCallback, useRef, useEffect } from 'react'
import VideoFeed from './components/VideoFeed'
import AROverlay from './components/AROverlay'
import RiskMeter from './components/RiskMeter'
import PlayerStatus from './components/PlayerStatus'
import AlertPanel from './components/AlertPanel'
import SportSelector from './components/SportSelector'
import HistoryChart from './components/HistoryChart'
import SessionSummary from './components/SessionSummary'
import './App.css'

const API_BASE = import.meta.env.VITE_API_URL || 'http://localhost:8000'
const WS_URL = API_BASE.replace(/^http/, 'ws') + '/ws/analyze'
const API_URL = API_BASE

function App() {
  const [sport, setSport] = useState('generic')
  const [analysis, setAnalysis] = useState(null)
  const [alerts, setAlerts] = useState([])
  const [connected, setConnected] = useState(false)
  const [streaming, setStreaming] = useState(false)
  const [riskHistory, setRiskHistory] = useState([])
  const [sessionStats, setSessionStats] = useState(null)
  const [isMuted, setIsMuted] = useState(false)
  const [goodStreak, setGoodStreak] = useState(0) // Tracks consecutive GREEN frames
  const wsRef = useRef(null)
  const audioRef = useRef(null)
  const startTimeRef = useRef(null)
  const lastCoachTime = useRef(0) // To prevent spamming compliments

  // Initialize audio for RED alerts
  useEffect(() => {
    try {
      // Create audio context only once user interacts?
      // For now, simpler HTML5 Audio
      audioRef.current = new Audio('data:audio/wav;base64,UklGRigBAABXQVZFZm10IBAAAAABAAEARKwAAIhYAQACABAAZGF0YQQBAADg/+D/4P/g/+D/4P/g/+D/4P/g/+D/4P/g/8D/wP+g/4D/YP9A/yD/AP/g/sD+oP6A/mD+QP4g/gD+4P3A/aD9gP2A/YD9gP2g/aD9wP3g/QD+IP5A/mD+gP6g/sD+4P4A/yD/QP9g/4D/oP/A/+D/AAAQACAAQABQAGAAYAB4AIgAmACgAKgAsAC4ALgAuACwAKAAlACIAHgAYABQAEAAMAAgABAAAADg/8D/oP+A/2D/QP8g/wD/4P7A/qD+gP5g/kD+IP4A/uD9wP2g/YD9YP1A/SD9AP3g/MD8oPyA/GD8QPwg/AD84PvA+6D7gPtg+0D7')
      audioRef.current.volume = 0.3
    } catch (e) {
      console.error("Audio initialization failed:", e)
    }
  }, [])

  // Text-to-Speech Helper
  const speak = useCallback((text, priority = 'normal') => {
    if (isMuted || !window.speechSynthesis) return

    try {
      // Priority handling: break-in for alerts
      if (priority === 'high') {
        window.speechSynthesis.cancel()
      } else if (window.speechSynthesis.speaking) {
        // Don't overlap normal compliments
        return
      }

      const utterance = new SpeechSynthesisUtterance(text)

      // AI Coach Persona Settings
      utterance.rate = 0.95 // Slightly slower for clarity
      utterance.pitch = 1.0
      utterance.volume = 1.0

      const voices = window.speechSynthesis.getVoices()
      // Try to find a premium/natural English voice
      const preferredVoice = voices.find(v =>
        v.name.includes('Google US English') ||
        v.name.includes('Natural') ||
        v.name.includes('Samantha') ||
        v.name.includes('Microsoft David') ||
        v.lang.startsWith('en-US')
      )

      if (preferredVoice) utterance.voice = preferredVoice

      window.speechSynthesis.speak(utterance)
    } catch (e) {
      console.error("Speech synthesis failed:", e)
    }
  }, [isMuted])

  // Connect WebSocket
  const connectWS = useCallback(() => {
    if (wsRef.current?.readyState === WebSocket.OPEN) return

    const ws = new WebSocket(WS_URL)

    ws.onopen = () => {
      setConnected(true)
      console.log('WebSocket connected')
    }

    ws.onmessage = (event) => {
      const data = JSON.parse(event.data)
      setAnalysis(data)

      // AI Coach Logic: Positive Reinforcement
      const risk = data.injury_probability || 0
      if (risk < 20 && data.alert_level === 'GREEN') {
        setGoodStreak(prev => {
          const next = prev + 1
          // Every 50 "good" frames (~7 seconds), give a compliment
          if (next >= 50 && (Date.now() - lastCoachTime.current) > 15000) {
            const compliments = [
              "Great form, keep it up!",
              "Balance looks perfect. You're in the zone.",
              "Excellent control. This is how pros do it.",
              "Your posture is rock solid. Nice work.",
              "Stay focused, you're doing amazing."
            ]
            const randomMsg = compliments[Math.floor(Math.random() * compliments.length)]
            speak(randomMsg, 'normal')
            lastCoachTime.current = Date.now()
            return 0 // Reset streak for next compliment
          }
          return next
        })
      } else if (risk > 40) {
        setGoodStreak(0) // Reset streak if risk increases
      }

      // Track risk history
      setRiskHistory(prev => {
        const next = [...prev, {
          time: Date.now(),
          risk: data.injury_probability || 0,
          level: data.alert_level || 'GREEN'
        }]
        return next.slice(-60) // keep last 60 data points
      })

      // Add to alerts if not GREEN
      if (data.alert_level && data.alert_level !== 'GREEN') {
        setAlerts(prev => [{
          ...data,
          id: Date.now(),
          timestamp: new Date().toLocaleTimeString()
        }, ...prev].slice(0, 50))

        // Trigger effects
        if (data.alert_level === 'RED') {
          // Play Beep if not muted
          if (!isMuted) {
            audioRef.current?.play().catch(() => { })
          }

          // Voice Coach - Alerts take priority
          const msg = data.recommended_action || data.injury_type || "High risk detected"
          speak(msg, 'high')
        }
      }
    }

    ws.onclose = () => {
      setConnected(false)
      console.log('WebSocket disconnected')
    }

    ws.onerror = () => {
      setConnected(false)
    }

    wsRef.current = ws
  }, [speak])

  // Send frame over WebSocket
  const sendFrame = useCallback((base64Frame, width, height) => {
    if (wsRef.current?.readyState === WebSocket.OPEN) {
      wsRef.current.send(JSON.stringify({
        image_base64: base64Frame,
        sport: sport,
        frame_width: width,
        frame_height: height
      }))
    }
  }, [sport])

  // Toggle streaming with Session Summary logic
  const toggleStreaming = useCallback(() => {
    console.log("Toggle Streaming clicked. Current state:", streaming)
    if (!streaming) {
      // START
      connectWS()
      setRiskHistory([])
      setAlerts([])
      setSessionStats(null)
      startTimeRef.current = Date.now()
    } else {
      // STOP
      wsRef.current?.close()

      // Calculate Stats
      if (startTimeRef.current && riskHistory.length > 0) {
        const durationSec = Math.floor((Date.now() - startTimeRef.current) / 1000)
        const min = Math.floor(durationSec / 60)
        const sec = durationSec % 60
        const duration = `${min}m ${sec}s`

        const avgRisk = riskHistory.reduce((a, b) => a + b.risk, 0) / riskHistory.length
        const avgScore = Math.max(0, 100 - avgRisk).toFixed(0)
        const peakRisk = Math.max(...riskHistory.map(r => r.risk), 0).toFixed(0)

        setSessionStats({
          duration,
          avgScore,
          peakRisk,
          alertCount: alerts.length
        })
      }
    }
    setStreaming(prev => !prev)
  }, [streaming, connectWS, riskHistory, alerts.length])

  // Change sport → send to WS
  const handleSportChange = useCallback((newSport) => {
    setSport(newSport)
    if (wsRef.current?.readyState === WebSocket.OPEN) {
      wsRef.current.send(JSON.stringify({ sport: newSport }))
    }
  }, [])

  const alertLevel = analysis?.alert_level || 'GREEN'

  return (
    <div className="app-container">
      {/* ─── Header ───────────────────────────────────────────── */}
      <header className="header">
        <div style={{ display: 'flex', alignItems: 'center' }}>
          <div className="header-logo">⚡</div>
          <h1>InjuryGuard <span>AI 2.0</span></h1>
        </div>
        <div style={{ display: 'flex', alignItems: 'center', gap: '16px' }}>
          <SportSelector sport={sport} onChange={handleSportChange} />
          <div className={`status-badge ${alertLevel.toLowerCase()}`}>
            {connected ? `● ${alertLevel}` : '○ OFFLINE'}
          </div>
        </div>
      </header>

      {/* ─── Main Grid ────────────────────────────────────────── */}
      <main className="main-grid">
        {/* Left: Video Feed "Hero" Section */}
        <section className="video-section">
          <div className="video-container">
            {/* We pass a custom class or style if needed, but App.css handles .video-feed */}
            <VideoFeed
              streaming={streaming}
              onToggle={toggleStreaming}
              onFrame={sendFrame}
              analysis={analysis}
              connected={connected}
              // Force video component to fill container
              style={{ width: '100%', height: '100%' }}
            />

            {/* DEBUG: Voice Test Button */}
            {!streaming && (
              <button
                onClick={() => speak("Voice system check complete. Ready for analysis.")}
                style={{
                  position: 'absolute', top: '10px', right: '10px',
                  background: '#333', color: '#fff', border: '1px solid #555',
                  padding: '4px 8px', fontSize: '0.7rem', zIndex: 50, cursor: 'pointer'
                }}
              >
                🔊 Test Voice
              </button>
            )}

            {/* AR Skeleton Overlay (Mirrored to match video) */}
            <div style={{ position: 'absolute', top: 0, left: 0, width: '100%', height: '100%', pointerEvents: 'none', transform: 'scaleX(-1)' }}>
              <AROverlay analysis={analysis} />
            </div>


            {/* Overlay Info (Top Left) */}
            <div className="video-overlay">
              <div style={{ display: 'flex', gap: '12px' }}>
                <span>MODE: {sport.toUpperCase()}</span>
                {/* Voice Toggle */}
                <button
                  onClick={() => setIsMuted(p => !p)}
                  style={{
                    background: 'transparent',
                    border: '1px solid rgba(255,255,255,0.3)',
                    borderRadius: '4px',
                    color: isMuted ? '#ef4444' : '#22c55e',
                    padding: '2px 8px',
                    cursor: 'pointer',
                    fontSize: '0.8rem',
                    textTransform: 'uppercase'
                  }}
                >
                  {isMuted ? '🔇 VOICE: OFF' : '🔊 VOICE: ON'}
                </button>
              </div>
              <div>FPS: {analysis ? '30' : '0'}</div>
              <div>TIME: {new Date().toLocaleTimeString()}</div>
            </div>

            {/* ─── CONTROLS (Top Layer) ───────────────────────── */}

            {/* Start Button */}
            {!streaming && (
              <div style={{
                position: 'absolute', top: '50%', left: '50%', transform: 'translate(-50%, -50%)',
                zIndex: 200, pointerEvents: 'auto', textAlign: 'center'
              }}>
                <button
                  onClick={toggleStreaming}
                  style={{
                    background: 'var(--primary)', color: '#000',
                    border: 'none', padding: '16px 32px',
                    fontSize: '1.2rem', fontWeight: 'bold', letterSpacing: '2px',
                    cursor: 'pointer', clipPath: 'polygon(10% 0, 100% 0, 100% 70%, 90% 100%, 0 100%, 0 30%)',
                    textTransform: 'uppercase',
                    boxShadow: '0 0 20px var(--primary-glow)'
                  }}
                >
                  Initialize
                </button>
                <div style={{ marginTop: '12px', fontSize: '0.9rem', color: '#888', textShadow: '0 1px 2px #000' }}>
                  Ready to Connect
                </div>
              </div>
            )}

            {/* Terminate Button */}
            {streaming && (
              <button
                onClick={toggleStreaming}
                style={{
                  position: 'absolute',
                  bottom: '100px', /* Moved up to avoid covering analytics */
                  left: '50%',
                  transform: 'translateX(-50%)',
                  background: 'rgba(255, 42, 42, 0.9)',
                  color: 'white',
                  border: '2px solid white',
                  padding: '12px 32px',
                  fontSize: '1.2rem',
                  fontWeight: 'bold',
                  borderRadius: '50px',
                  cursor: 'pointer',
                  boxShadow: '0 0 20px rgba(255, 0, 0, 0.6)',
                  zIndex: 200,
                  pointerEvents: 'auto',
                  textTransform: 'uppercase'
                }}
              >
                ⏹ Terminate Session
              </button>
            )}
          </div>


          {/* Bottom Bar Metrics (Overlay) */}
          <div className="metrics-bar">
            {/* Added subtle scan line animation via internal style for hackathon vibes */}
            {streaming && (
              <div className="scan-line" style={{
                position: 'absolute', top: 0, left: 0, width: '100%', height: '2px',
                background: 'rgba(0, 229, 255, 0.5)',
                boxShadow: '0 0 10px #00e5ff',
                animation: 'scan 3s linear infinite',
                pointerEvents: 'none',
                zIndex: 5
              }} />
            )}
            <div className="metric-item">
              <span className="metric-label">Pose Risk</span>
              <span className="metric-value" style={{ color: getColor(analysis?.pose_risk) }}>
                {analysis?.pose_risk?.toFixed(0) || '0'}%
              </span>
            </div>
            <div className="metric-item">
              <span className="metric-label">Facial Stress</span>
              <span className="metric-value" style={{ color: getColor(analysis?.facial_stress) }}>
                {analysis?.facial_stress?.toFixed(0) || '0'}%
              </span>
            </div>
            <div className="metric-item">
              <span className="metric-label">Impact Risk</span>
              <span className="metric-value" style={{ color: getColor(analysis?.object_risk) }}>
                {analysis?.object_risk?.toFixed(0) || '0'}%
              </span>
            </div>
            <div className="metric-item">
              <span className="metric-label">Speed</span>
              <span className="metric-value" style={{ color: '#00e5ff' }}>
                {analysis?.object_speed?.toFixed(0) || '0'} <small style={{ fontSize: '1rem' }}>km/h</small>
              </span>
            </div>
          </div>
        </section>

        {/* Right: Analysis Panel */}
        <aside className="right-panel">
          <div className="panel-card">
            <h3>Live Analysis</h3>
            <RiskMeter analysis={analysis} />
          </div>

          <div className="panel-card">
            <h3>Player Status</h3>
            <PlayerStatus analysis={analysis} />
          </div>

          <div className="panel-card" style={{ flex: 1, display: 'flex', flexDirection: 'column' }}>
            <h3>Alert Log</h3>
            <div style={{ flex: 1, overflowY: 'auto' }}>
              <AlertPanel alerts={alerts} />
            </div>
          </div>

          <div className="panel-card">
            <h3>Risk Trend</h3>
            <HistoryChart history={riskHistory} />
          </div>
        </aside>
      </main>

      {/* Session Summary Modal */}
      <SessionSummary stats={sessionStats} onClose={() => setSessionStats(null)} />
    </div>
  )
}

function getColor(value) {
  if (!value || value < 35) return '#22c55e'
  if (value < 70) return '#f59e0b'
  return '#ef4444'
}

export default App
