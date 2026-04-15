/**
 * WebRTC client for OLA Avatar.
 * Handles SDP offer/answer exchange and MediaStream rendering.
 */

let pc = null;
let statsInterval = null;

const videoEl = document.getElementById('avatar-video');
const statusDot = document.getElementById('status-dot');
const statusText = document.getElementById('status-text');
const btnConnect = document.getElementById('btn-connect');
const btnDisconnect = document.getElementById('btn-disconnect');
const statsEl = document.getElementById('stats');

function setStatus(state, text) {
    statusDot.className = 'status-dot ' + state;
    statusText.textContent = text;
}

async function connect() {
    if (pc) {
        console.warn('Already connected');
        return;
    }

    setStatus('connecting', 'Connecting...');
    btnConnect.disabled = true;

    try {
        pc = new RTCPeerConnection({
            iceServers: [{ urls: 'stun:stun.l.google.com:19302' }]
        });

        // Handle incoming media tracks
        pc.ontrack = (event) => {
            console.log('Track received:', event.track.kind);
            if (event.track.kind === 'video') {
                videoEl.srcObject = event.streams[0];
                videoEl.muted = false;
            }
            // Audio track is part of the same stream — browser plays it automatically
        };

        // Monitor connection state
        pc.onconnectionstatechange = () => {
            console.log('Connection state:', pc.connectionState);
            switch (pc.connectionState) {
                case 'connected':
                    setStatus('connected', 'Connected');
                    btnDisconnect.disabled = false;
                    startStats();
                    break;
                case 'disconnected':
                case 'failed':
                    setStatus('error', 'Disconnected');
                    cleanup();
                    break;
                case 'closed':
                    setStatus('', 'Disconnected');
                    cleanup();
                    break;
            }
        };

        pc.oniceconnectionstatechange = () => {
            console.log('ICE state:', pc.iceConnectionState);
        };

        // We need to add transceivers to receive media
        pc.addTransceiver('video', { direction: 'recvonly' });
        pc.addTransceiver('audio', { direction: 'recvonly' });

        // Create offer
        const offer = await pc.createOffer();
        await pc.setLocalDescription(offer);

        // Wait for ICE gathering to complete (or timeout)
        await new Promise((resolve) => {
            if (pc.iceGatheringState === 'complete') {
                resolve();
            } else {
                const checkState = () => {
                    if (pc.iceGatheringState === 'complete') {
                        pc.removeEventListener('icegatheringstatechange', checkState);
                        resolve();
                    }
                };
                pc.addEventListener('icegatheringstatechange', checkState);
                // Timeout after 3s
                setTimeout(resolve, 3000);
            }
        });

        // Send offer to signaling server
        const response = await fetch('/offer', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({
                sdp: pc.localDescription.sdp,
                type: pc.localDescription.type,
            }),
        });

        if (!response.ok) {
            throw new Error(`Signaling failed: ${response.status}`);
        }

        const answer = await response.json();
        await pc.setRemoteDescription(new RTCSessionDescription(answer));

        console.log('WebRTC connection established');
    } catch (err) {
        console.error('Connection failed:', err);
        setStatus('error', 'Connection failed');
        cleanup();
    }
}

function disconnect() {
    if (pc) {
        pc.close();
    }
    cleanup();
}

function cleanup() {
    if (pc) {
        pc.close();
        pc = null;
    }
    videoEl.srcObject = null;
    btnConnect.disabled = false;
    btnDisconnect.disabled = true;
    setStatus('', 'Disconnected');
    stopStats();
}

function startStats() {
    statsInterval = setInterval(async () => {
        if (!pc) return;
        const stats = await pc.getStats();
        let info = [];
        stats.forEach((report) => {
            if (report.type === 'inbound-rtp' && report.kind === 'video') {
                info.push(`Video: ${report.framesPerSecond || '?'} fps, ${report.bytesReceived || 0} bytes`);
            }
            if (report.type === 'inbound-rtp' && report.kind === 'audio') {
                info.push(`Audio: ${report.bytesReceived || 0} bytes`);
            }
        });
        statsEl.textContent = info.join(' | ');
    }, 2000);
}

function stopStats() {
    if (statsInterval) {
        clearInterval(statsInterval);
        statsInterval = null;
    }
    statsEl.textContent = '';
}
