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

// Create a single MediaStream to hold both tracks
const remoteStream = new MediaStream();

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

    // Reset remote stream and attach to video element
    // Ensure video element is NOT muted for audio to work
    videoEl.srcObject = remoteStream;
    videoEl.muted = false;
    videoEl.volume = 1.0;

    // Prime the video element to unlock autoplay
    videoEl.play().catch(e => console.log('[WebRTC] Initial prime play prevented:', e.name));

    try {
        pc = new RTCPeerConnection({});

        // Handle incoming media tracks
        pc.ontrack = (event) => {
            console.log('[WebRTC] Track received:', event.track.kind, event.track.id);
            
            // Add track to our shared remote stream
            remoteStream.addTrack(event.track);
            
            if (event.track.kind === 'video') {
                console.log('[WebRTC] Video track attached to remote stream');
            } else if (event.track.kind === 'audio') {
                console.log('[WebRTC] Audio track attached to remote stream');
                console.log('[Audio] Muted status:', event.track.muted);
                console.log('[Audio] Enabled status:', event.track.enabled);
            }

            // Attempt to play/unmute whenever a track arrives
            videoEl.muted = false;
            videoEl.play()
                .then(() => console.log('[WebRTC] Playback active (Audio/Video combined)'))
                .catch(e => {
                    if (e.name === 'NotAllowedError') {
                        console.warn('[WebRTC] Playback blocked by browser. Click anywhere to unmute.');
                        const unmuteOnGesture = () => {
                            videoEl.muted = false;
                            videoEl.play().then(() => {
                                console.log('[WebRTC] Playback started after user gesture');
                                document.removeEventListener('click', unmuteOnGesture);
                            });
                        };
                        document.addEventListener('click', unmuteOnGesture);
                    } else {
                        console.error('[WebRTC] Playback error:', e);
                    }
                });
        };

        // Monitor connection state
        pc.onconnectionstatechange = () => {
            console.log('[WebRTC] Connection state:', pc.connectionState);
            switch (pc.connectionState) {
                case 'connected':
                    setStatus('connected', 'Connected');
                    btnDisconnect.disabled = false;
                    startStats();
                    break;
                case 'disconnected':
                case 'failed':
                    setStatus('error', pc.connectionState === 'failed' ? 'Connection Failed' : 'Disconnected');
                    cleanup();
                    break;
                case 'closed':
                    setStatus('', 'Disconnected');
                    cleanup();
                    break;
            }
        };

        pc.oniceconnectionstatechange = () => {
            console.log('[WebRTC] ICE connection state:', pc.iceConnectionState);
        };

        pc.onicegatheringstatechange = () => {
            console.log('[WebRTC] ICE gathering state:', pc.iceGatheringState);
        };

        pc.onsignalingstatechange = () => {
            console.log('[WebRTC] Signaling state:', pc.signalingState);
        };

        pc.onicecandidate = (event) => {
            if (event.candidate) {
                console.log('[WebRTC] Local ICE candidate:', event.candidate.type, event.candidate.address, event.candidate.port, event.candidate.protocol);
            }
        };

        // We need to add transceivers to receive media
        pc.addTransceiver('video', { direction: 'recvonly' });
        pc.addTransceiver('audio', { direction: 'recvonly' });

        // Create offer
        const offer = await pc.createOffer();
        await pc.setLocalDescription(offer);
        console.log('[WebRTC] Local description set, gathering ICE candidates...');

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
                setTimeout(resolve, 5000);
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
            const errText = await response.text();
            throw new Error(`Signaling failed: ${response.status} - ${errText}`);
        }

        const answer = await response.json();
        await pc.setRemoteDescription(new RTCSessionDescription(answer));
        console.log('[WebRTC] Remote description set, waiting for connection...');

    } catch (err) {
        console.error('[WebRTC] Connection failed:', err);
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
    remoteStream.getTracks().forEach(track => track.stop());
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
