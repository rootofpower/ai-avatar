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
        // Fetch ICE server config (STUN + TURN) from server
        let iceServers = [{ urls: 'stun:stun.l.google.com:19302' }];
        try {
            const iceResp = await fetch('/ice-config');
            if (iceResp.ok) {
                const iceConfig = await iceResp.json();
                iceServers = iceConfig.iceServers;
                console.log('[WebRTC] ICE servers from server:', JSON.stringify(iceServers));
            }
        } catch (e) {
            console.warn('[WebRTC] Failed to fetch ICE config, using STUN only:', e);
        }

        pc = new RTCPeerConnection({ iceServers });

        const remoteStream = new MediaStream();
        videoEl.srcObject = remoteStream;

        const audioEl = document.getElementById('avatar-audio');

        // Handle incoming media tracks
        pc.ontrack = (event) => {
            console.log('[WebRTC] Track received:', event.track.kind, event.track.id);
            
            if (event.track.kind === 'video') {
                remoteStream.addTrack(event.track);
                videoEl.muted = false;
                videoEl.play().catch(e => console.warn('[WebRTC] Video Autoplay blocked:', e));
                console.log('[WebRTC] Video track added to remote stream');
            } else if (event.track.kind === 'audio') {
                const audioStream = new MediaStream([event.track]);
                audioEl.srcObject = audioStream;
                audioEl.play().catch(e => console.warn('[WebRTC] Audio Autoplay blocked:', e));
                console.log('[WebRTC] Audio track attached to dedicated element');
            }
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
            } else {
                console.log('[WebRTC] ICE candidate gathering done (null candidate)');
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
                console.log('[WebRTC] ICE gathering already complete');
                resolve();
            } else {
                const checkState = () => {
                    if (pc.iceGatheringState === 'complete') {
                        pc.removeEventListener('icegatheringstatechange', checkState);
                        console.log('[WebRTC] ICE gathering completed');
                        resolve();
                    }
                };
                pc.addEventListener('icegatheringstatechange', checkState);
                // Timeout after 5s
                setTimeout(() => {
                    console.warn('[WebRTC] ICE gathering timed out after 5s');
                    resolve();
                }, 5000);
            }
        });

        // Count ICE candidates in the offer
        const candidateCount = (pc.localDescription.sdp.match(/a=candidate:/g) || []).length;
        console.log(`[WebRTC] Sending offer with ${candidateCount} ICE candidates`);

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
        const answerCandidates = (answer.sdp.match(/a=candidate:/g) || []).length;
        console.log(`[WebRTC] Received answer with ${answerCandidates} ICE candidates`);

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
