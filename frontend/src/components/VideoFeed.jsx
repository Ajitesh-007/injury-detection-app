import { useRef, useEffect, useCallback, useState } from 'react'

export default function VideoFeed({ streaming, onToggle, onFrame, analysis, connected, style }) {
    const videoRef = useRef(null)
    const canvasRef = useRef(null)
    const intervalRef = useRef(null)
    const [hasCamera, setHasCamera] = useState(true)

    // Start/stop camera
    useEffect(() => {
        console.log("VideoFeed effect triggered. Streaming:", streaming)
        let subscribed = true

        if (streaming) {
            startCamera().then(() => {
                if (!subscribed) stopCamera()
            })
        } else {
            stopCamera()
        }
        return () => {
            subscribed = false
            stopCamera()
        }
    }, [streaming])

    const startCamera = async () => {
        console.log("Attempting to start camera...")
        try {
            const stream = await navigator.mediaDevices.getUserMedia({
                video: { width: 640, height: 480 }
            })
            console.log("Camera access granted")
            if (videoRef.current) {
                videoRef.current.srcObject = stream
                await videoRef.current.play()
                console.log("Video playing")
            }
            setHasCamera(true)

            // Start frame capture loop
            intervalRef.current = setInterval(captureAndSend, 150) // ~7 FPS
        } catch (err) {
            console.error('Camera Error:', err)
            // Fallback for demo if no camera
            setHasCamera(false)
            alert("Camera failed to start. Please check permissions.")
        }
    }

    const stopCamera = () => {
        if (intervalRef.current) {
            clearInterval(intervalRef.current)
            intervalRef.current = null
        }
        if (videoRef.current?.srcObject) {
            videoRef.current.srcObject.getTracks().forEach(t => t.stop())
            videoRef.current.srcObject = null
        }
    }

    const captureAndSend = useCallback(() => {
        if (!videoRef.current || !canvasRef.current) return

        const video = videoRef.current
        const canvas = canvasRef.current
        const ctx = canvas.getContext('2d')

        canvas.width = video.videoWidth || 640
        canvas.height = video.videoHeight || 480

        ctx.drawImage(video, 0, 0, canvas.width, canvas.height)

        const base64 = canvas.toDataURL('image/jpeg', 0.7)
        onFrame(base64, canvas.width, canvas.height)
    }, [onFrame])

    return (
        <div style={{ position: 'relative', width: '100%', height: '100%', background: '#000', ...style }}>
            {streaming && hasCamera ? (
                <>
                    <video
                        ref={videoRef}
                        autoPlay
                        muted
                        playsInline
                        style={{
                            width: '100%',
                            height: '100%',
                            objectFit: 'contain',
                            transform: 'scaleX(-1)' // Mirror effect
                        }}
                    />
                    {/* Hidden canvas for frame capture */}
                    <canvas ref={canvasRef} style={{ display: 'none' }} />
                </>
            ) : (
                <div style={{
                    display: 'flex', flexDirection: 'column', alignItems: 'center', justifyContent: 'center',
                    height: '100%', color: '#333'
                }}>
                    <div style={{ fontSize: '4rem', marginBottom: '20px', opacity: 0.5 }}>🎥</div>
                    <p style={{ fontFamily: 'var(--font-mono)', color: 'var(--primary)', opacity: 0.8 }}>
                        SYSTEM STANDBY
                    </p>
                </div>
            )}
        </div>
    )
}
