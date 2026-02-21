/**
 * voiceCoach.js
 * 
 * A smooth, natural-sounding AI voice coach.
 * - Waits for the browser to load all voices before selecting the best one
 * - Uses a priority queue to prevent choppy overlapping speech
 * - Smooth rate, pitch tuned for a coaching persona
 */

let bestVoice = null
let voicesReady = false

// Priority levels — HIGH interrupts, NORMAL queues behind current speech
const PRIORITY = { HIGH: 'high', NORMAL: 'normal', LOW: 'low' }

// Preferred voice names (ordered best → acceptable)
const PREFERRED_VOICES = [
    'Google UK English Male',
    'Google US English',
    'Microsoft Guy Online (Natural) - English (United States)',
    'Microsoft David - English (United States)',
    'Samantha',
    'Daniel',
    'Alex',
]

function loadBestVoice() {
    const voices = window.speechSynthesis?.getVoices() || []
    if (!voices.length) return

    // Try preferred voices in order
    for (const name of PREFERRED_VOICES) {
        const found = voices.find(v => v.name === name)
        if (found) {
            bestVoice = found
            voicesReady = true
            return
        }
    }

    // Fallback: any English voice with "online" (higher quality cloud voices)
    const online = voices.find(v => v.lang.startsWith('en') && v.name.toLowerCase().includes('online'))
    if (online) { bestVoice = online; voicesReady = true; return }

    // Last fallback: any English voice
    const anyEn = voices.find(v => v.lang.startsWith('en-US') || v.lang === 'en-GB')
    if (anyEn) { bestVoice = anyEn; voicesReady = true }
}

// Load voices when they become available (they load async in most browsers)
if (typeof window !== 'undefined' && window.speechSynthesis) {
    loadBestVoice()
    window.speechSynthesis.addEventListener('voiceschanged', loadBestVoice)
}

let lastSpokenAt = 0      // prevent spamming same phrase
let lastSpokenText = ''

export function speak(text, priority = PRIORITY.NORMAL, isMuted = false) {
    if (isMuted || !window.speechSynthesis) return
    if (!text || typeof text !== 'string') return

    const now = Date.now()

    // Debounce: don't repeat the same phrase within 8 seconds
    if (text === lastSpokenText && now - lastSpokenAt < 8000) return

    // Cancel current speech for high priority alerts
    if (priority === PRIORITY.HIGH) {
        window.speechSynthesis.cancel()
    } else if (window.speechSynthesis.speaking) {
        // Don't pile on normal/low messages
        return
    }

    const utterance = new SpeechSynthesisUtterance(text)

    // Smooth, clear coaching voice settings
    utterance.rate = 0.88   // Slightly slower = clearer, less robotic
    utterance.pitch = 1.05   // Slightly above neutral = warmer
    utterance.volume = 1.0

    if (bestVoice) utterance.voice = bestVoice

    // Workaround for Chrome's bug where long speech cuts off after ~15s
    utterance.onstart = () => {
        // Keep alive trick
        const keepAlive = setInterval(() => {
            if (!window.speechSynthesis.speaking) clearInterval(keepAlive)
            else { window.speechSynthesis.pause(); window.speechSynthesis.resume() }
        }, 10000)
    }

    lastSpokenText = text
    lastSpokenAt = now
    window.speechSynthesis.speak(utterance)
}

export { PRIORITY }

// Coaching messages library
export const COACH_MESSAGES = {
    GREEN: [
        "Looking good. Keep that form steady.",
        "Perfect alignment. You're in the zone.",
        "Excellent posture. This is peak performance.",
        "Body mechanics are ideal right now.",
        "Smooth and controlled. Stay focused.",
    ],
    YELLOW: [
        "Watch your posture — adjust and breathe.",
        "Form is slipping slightly. Slow down and reset.",
        "Pay attention to your alignment.",
        "Take it easy. Controlled movement now.",
    ],
    RED: [
        "High risk detected! Stop and check your position.",
        "Dangerous posture. Rest immediately.",
        "Warning — injury risk is high. Stop now.",
    ],
    START: "InjuryGuard AI activated. Hold still while I calibrate your pose.",
    STOP: "Session complete. Well done.",
}
