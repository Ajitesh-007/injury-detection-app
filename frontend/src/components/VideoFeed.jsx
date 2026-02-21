import { useRef, useEffect, useCallback, useState } from 'react'

// MediaPipe Pose connections for drawing the skeleton
const POSE_CONNECTIONS = [
    [11, 12], // Shoulders
    [11, 13], [13, 15], // Left arm
    [12, 14], [14, 16], // Right arm
    [11, 23], [12, 24], // Torso sides
    [23, 24], // Hips
    [23, 25], [25, 27], [27, 29], [27, 31], // Left leg
    [24, 26], [26, 28], [28, 30], [28, 32], // Right leg
]

const JOINT_COLORS = {
    danger: '#ef4444',
    warning: '#f59e0b',
    ok: '#00ff9d',
}

let mediapipeLoaded = false
let loadingPromise = null

function loadMediaPipe() {
    if (mediapipeLoaded) return Promise.resolve()
    if (loadingPromise) return loadingPromise

    loadingPromise = new Promise((resolve, reject) => {
        // Load pose library
        const poseScript = document.createElement('script')
        poseScript.src = 'https://cdn.jsdelivr.net/npm/@mediapipe/pose@0.5.1675469404/pose.js'
        poseScript.crossOrigin = 'anonymous'

        const drawingScript = document.createElement('script')
        drawingScript.src = 'https://cdn.jsdelivr.net/npm/@mediapipe/drawing_utils@0.3.1675466124/drawing_utils.js'
        drawingScript.crossOrigin = 'anonymous'

        let loaded = 0
        const onLoad = () => {
            loaded++
            if (loaded === 2) {
                mediapipeLoaded = true
                resolve()
            }
        }

        poseScript.onload = onLoad
        drawingScript.onload = onLoad
        poseScript.onerror = reject
        drawingScript.onerror = reject

        document.head.appendChild(drawingScript)
        document.head.appendChild(poseScript)
    })

    return loadingPromise
}

export default function VideoFeed({ streaming, onToggle, onFrame, onLandmarks, analysis, connected, style }) {
    const videoRef = useRef(null)
    const canvasRef = useRef(null)  // visible overlay canvas
    const sendCanvasRef = useRef(null)  // hidden canvas for backend frames
    const poseRef = useRef(null)
    const rafRef = useRef(null)
    const frameCountRef = useRef(0)
    const lastSendRef = useRef(0)
    const [hasCamera, setHasCamera] = useState(true)
    const [mpLoading, setMpLoading] = useState(false)
    const [mpReady, setMpReady] = useState(false)

    // Draw skeleton on canvas
    const drawSkeleton = useCallback((canvas, videoEl, landmarks) => {
        const ctx = canvas.getContext('2d')
        canvas.width = videoEl.videoWidth || 640
        canvas.height = videoEl.videoHeight || 480

        // Clear
        ctx.clearRect(0, 0, canvas.width, canvas.height)

        if (!landmarks || landmarks.length === 0) return

        const w = canvas.width
        const h = canvas.height

        const issues = analysis?.issues || []
        const issueStr = issues.join(' ').toLowerCase()
        const riskyJoints = new Set()

        const JOINT_NAME_MAP = {
            11: 'shoulder', 12: 'shoulder',
            13: 'elbow', 14: 'elbow',
            15: 'wrist', 16: 'wrist',
            23: 'hip', 24: 'hip',
            25: 'knee', 26: 'knee',
            27: 'ankle', 28: 'ankle',
        }

        Object.entries(JOINT_NAME_MAP).forEach(([idx, name]) => {
            if (issueStr.includes(name)) riskyJoints.add(Number(idx))
        })

        // Draw connections (bones)
        ctx.lineWidth = Math.max(2, w * 0.004)
        POSE_CONNECTIONS.forEach(([start, end]) => {
            const p1 = landmarks[start]
            const p2 = landmarks[end]
            if (!p1 || !p2 || p1.visibility < 0.2 || p2.visibility < 0.2) return

            const isDanger = riskyJoints.has(start) || riskyJoints.has(end)
            ctx.beginPath()
            ctx.moveTo(p1.x * w, p1.y * h)
            ctx.lineTo(p2.x * w, p2.y * h)

            if (isDanger) {
                ctx.strokeStyle = 'rgba(239, 68, 68, 0.85)'
                ctx.shadowColor = '#ef4444'
                ctx.shadowBlur = 8
            } else {
                ctx.strokeStyle = 'rgba(0, 255, 157, 0.7)'
                ctx.shadowColor = '#00ff9d'
                ctx.shadowBlur = 6
            }
            ctx.stroke()
            ctx.shadowBlur = 0
        })

        // Draw joints (circles)
        landmarks.forEach((lm, i) => {
            if (i < 11 || i > 32 || lm.visibility < 0.2) return

            const x = lm.x * w
            const y = lm.y * h
            const isDanger = riskyJoints.has(i)
            const r = Math.max(4, w * (isDanger ? 0.012 : 0.008))

            ctx.beginPath()
            ctx.arc(x, y, r, 0, Math.PI * 2)

            const color = isDanger ? JOINT_COLORS.danger : JOINT_COLORS.ok
            ctx.fillStyle = color
            ctx.shadowColor = color
            ctx.shadowBlur = isDanger ? 12 : 8
            ctx.fill()

            ctx.strokeStyle = 'rgba(0,0,0,0.6)'
            ctx.lineWidth = 1.5
            ctx.shadowBlur = 0
            ctx.stroke()
        })
    }, [analysis])

    // Main pose processing loop
    const processFrame = useCallback(() => {
        const video = videoRef.current
        const canvas = canvasRef.current
        const pose = poseRef.current

        if (!video || !canvas || !pose || video.paused || video.ended) {
            rafRef.current = requestAnimationFrame(processFrame)
            return
        }

        if (video.readyState < 2) {
            rafRef.current = requestAnimationFrame(processFrame)
            return
        }

        // Send frame to MediaPipe
        pose.send({ image: video }).catch(() => { })

        // Also send to backend every ~200ms (throttled)
        const now = Date.now()
        if (onFrame && now - lastSendRef.current > 200) {
            lastSendRef.current = now
            const sendCanvas = sendCanvasRef.current
            if (sendCanvas) {
                sendCanvas.width = 320  // smaller for speed
                sendCanvas.height = 240
                const ctx2 = sendCanvas.getContext('2d')
                ctx2.drawImage(video, 0, 0, 320, 240)
                const base64 = sendCanvas.toDataURL('image/jpeg', 0.45)
                onFrame(base64, 320, 240)
            }
        }

        rafRef.current = requestAnimationFrame(processFrame)
    }, [onFrame])

    // Initialize MediaPipe Pose
    const initPose = useCallback(async () => {
        setMpLoading(true)
        try {
            await loadMediaPipe()

            const Pose = window.Pose
            if (!Pose) throw new Error('MediaPipe Pose not available')

            const pose = new Pose({
                locateFile: (file) =>
                    `https://cdn.jsdelivr.net/npm/@mediapipe/pose@0.5.1675469404/${file}`
            })

            pose.setOptions({
                modelComplexity: 0,           // 0=Lite, fastest
                smoothLandmarks: true,
                enableSegmentation: false,
                smoothSegmentation: false,
                minDetectionConfidence: 0.5,
                minTrackingConfidence: 0.5,
            })

            pose.onResults((results) => {
                const canvas = canvasRef.current
                const video = videoRef.current
                if (!canvas || !video) return

                const landmarks = results.poseLandmarks
                drawSkeleton(canvas, video, landmarks)

                // Pass raw landmarks to parent for risk display
                if (onLandmarks && landmarks) {
                    onLandmarks(landmarks)
                }
            })

            await pose.initialize()
            poseRef.current = pose
            setMpReady(true)
            setMpLoading(false)

            // Start the RAF loop
            rafRef.current = requestAnimationFrame(processFrame)
        } catch (err) {
            console.error('MediaPipe init failed:', err)
            setMpLoading(false)
        }
    }, [drawSkeleton, onLandmarks, processFrame])

    // Start/stop camera
    useEffect(() => {
        let subscribed = true

        if (streaming) {
            startCamera().then(() => {
                if (!subscribed) { stopCamera(); return }
                initPose()
            })
        } else {
            stopCamera()
        }

        return () => {
            subscribed = false
            stopCamera()
        }
        // eslint-disable-next-line react-hooks/exhaustive-deps
    }, [streaming])

    const startCamera = async () => {
        try {
            const isMobile = /Mobi|Android/i.test(navigator.userAgent)
            const stream = await navigator.mediaDevices.getUserMedia({
                video: {
                    width: { ideal: isMobile ? 480 : 640 },
                    height: { ideal: isMobile ? 360 : 480 },
                    facingMode: 'user',
                }
            })
            if (videoRef.current) {
                videoRef.current.srcObject = stream
                await videoRef.current.play()
            }
            setHasCamera(true)
        } catch (err) {
            console.error('Camera Error:', err)
            setHasCamera(false)
        }
    }

    const stopCamera = () => {
        if (rafRef.current) {
            cancelAnimationFrame(rafRef.current)
            rafRef.current = null
        }
        if (poseRef.current) {
            poseRef.current.close?.()
            poseRef.current = null
        }
        setMpReady(false)
        setMpLoading(false)
        if (videoRef.current?.srcObject) {
            videoRef.current.srcObject.getTracks().forEach(t => t.stop())
            videoRef.current.srcObject = null
        }
    }

    return (
        <div style={{ position: 'relative', width: '100%', height: '100%', background: '#000', ...style }}>
            {streaming && hasCamera ? (
                <>
                    {/* Live video (mirrored for selfie view) */}
                    <video
                        ref={videoRef}
                        autoPlay
                        muted
                        playsInline
                        style={{
                            width: '100%',
                            height: '100%',
                            objectFit: 'contain',
                            transform: 'scaleX(-1)',
                        }}
                    />

                    {/* Skeleton overlay canvas (also mirrored to match video) */}
                    <canvas
                        ref={canvasRef}
                        style={{
                            position: 'absolute',
                            top: 0, left: 0,
                            width: '100%',
                            height: '100%',
                            transform: 'scaleX(-1)',
                            pointerEvents: 'none',
                            zIndex: 5,
                        }}
                    />

                    {/* Loading indicator */}
                    {mpLoading && (
                        <div style={{
                            position: 'absolute', top: '50%', left: '50%',
                            transform: 'translate(-50%, -50%)',
                            color: '#00ff9d', fontFamily: 'monospace',
                            fontSize: '0.85rem', textAlign: 'center',
                            background: 'rgba(0,0,0,0.7)',
                            padding: '12px 20px', borderRadius: '8px',
                            border: '1px solid #00ff9d',
                            zIndex: 20,
                        }}>
                            <div className="mp-spinner" />
                            Loading AI Pose Engine...
                        </div>
                    )}

                    {/* Hidden canvas for backend frames */}
                    <canvas ref={sendCanvasRef} style={{ display: 'none' }} />
                </>
            ) : (
                <div style={{
                    display: 'flex', flexDirection: 'column',
                    alignItems: 'center', justifyContent: 'center',
                    height: '100%',
                }}>
                    <div style={{ fontSize: '4rem', marginBottom: '20px', opacity: 0.3 }}>🎥</div>
                    <p style={{
                        fontFamily: 'var(--font-mono)', color: 'var(--primary)',
                        opacity: 0.7, letterSpacing: '3px', fontSize: '0.85rem'
                    }}>
                        SYSTEM STANDBY
                    </p>
                </div>
            )}
        </div>
    )
}
