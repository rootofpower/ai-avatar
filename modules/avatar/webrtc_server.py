
import asyncio
import fractions
import struct
import json
import numpy as np
from typing import Optional

from aiortc import RTCPeerConnection, RTCSessionDescription, MediaStreamTrack, RTCConfiguration, RTCIceServer
from av import VideoFrame, AudioFrame

from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse, HTMLResponse
from fastapi.staticfiles import StaticFiles


class AvatarVideoTrack(MediaStreamTrack):
    """
    Custom video track that pulls frames from the AVMuxer's video queue.
    aiortc handles encoding (VP8/H264) and RTP packetization internally.
    """
    kind = "video"

    def __init__(self, muxer, fps: int = 25):
        super().__init__()
        self._muxer = muxer
        self._fps = fps
        self._pts = 0
        self._time_base = fractions.Fraction(1, fps)
        self._frame_duration = 1.0 / fps

    async def recv(self) -> VideoFrame:
        """Return the next video frame for WebRTC transmission."""
        try:
            entry = await asyncio.wait_for(
                self._muxer.video_queue.get(), timeout=self._frame_duration * 2
            )
            frame_bgr = entry['frame']
            frame_rgb = frame_bgr[:, :, ::-1].copy()
            frame = VideoFrame.from_ndarray(frame_rgb, format='rgb24')
        except (asyncio.TimeoutError, Exception):
            # No frame available — send a black frame to keep the stream alive
            frame = VideoFrame(width=512, height=512, format='rgb24')

        frame.pts = self._pts
        frame.time_base = self._time_base
        self._pts += 1
        return frame


class AvatarAudioTrack(MediaStreamTrack):
    """
    Custom audio track that pulls PCM from the AVMuxer's audio queue.
    aiortc handles Opus encoding automatically.
    """
    kind = "audio"

    def __init__(self, muxer, sample_rate: int = 48000):
        super().__init__()
        self._muxer = muxer
        self._sample_rate = sample_rate
        self._samples_per_frame = 960  # 20ms at 48kHz (Opus standard)
        self._pts = 0
        self._time_base = fractions.Fraction(1, sample_rate)
        self._audio_buffer = bytearray()

    async def recv(self) -> AudioFrame:
        """Return the next audio frame (20ms Opus frame) for WebRTC."""
        needed_bytes = self._samples_per_frame * 2  # 16-bit = 2 bytes/sample

        # Fill buffer from muxer
        while len(self._audio_buffer) < needed_bytes:
            try:
                entry = await asyncio.wait_for(
                    self._muxer.audio_queue.get(), timeout=0.05
                )
                self._audio_buffer.extend(entry['pcm_48k'])
            except (asyncio.TimeoutError, Exception):
                # Fill with silence if no audio available
                silence = b'\x00' * (needed_bytes - len(self._audio_buffer))
                self._audio_buffer.extend(silence)
                break

        # Extract one frame worth of samples
        frame_bytes = bytes(self._audio_buffer[:needed_bytes])
        del self._audio_buffer[:needed_bytes]

        # Convert to av.AudioFrame
        samples = struct.unpack(f'<{self._samples_per_frame}h', frame_bytes)
        audio_array = np.array(samples, dtype=np.int16).reshape(1, -1)

        frame = AudioFrame.from_ndarray(audio_array, format='s16', layout='mono')
        frame.sample_rate = self._sample_rate
        frame.pts = self._pts
        frame.time_base = self._time_base
        self._pts += self._samples_per_frame
        return frame


class WebRTCServer:
    """
    FastAPI-based WebRTC signaling server.
    Serves the client HTML and handles SDP/ICE exchange.
    """

    def __init__(self, muxer, stun_server: str = "stun:stun.l.google.com:19302"):
        self._muxer = muxer
        self._stun_server = stun_server
        self._pcs: set = set()
        self.app = self._create_app()

    def _create_app(self) -> FastAPI:
        app = FastAPI(title="Avatar WebRTC Server")

        @app.on_event("shutdown")
        async def on_shutdown():
            coros = [pc.close() for pc in self._pcs]
            await asyncio.gather(*coros)
            self._pcs.clear()

        @app.get("/", response_class=HTMLResponse)
        async def index():
            """Serve the client page."""
            import os
            client_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(__file__))), "client")
            index_path = os.path.join(client_dir, "index.html")
            if os.path.exists(index_path):
                with open(index_path, 'r') as f:
                    return HTMLResponse(content=f.read())
            return HTMLResponse(content="<h1>Client not found</h1>", status_code=404)

        @app.get("/app.js")
        async def serve_js():
            """Serve the client JavaScript."""
            import os
            client_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(__file__))), "client")
            js_path = os.path.join(client_dir, "app.js")
            if os.path.exists(js_path):
                with open(js_path, 'r') as f:
                    from fastapi.responses import Response
                    return Response(content=f.read(), media_type="application/javascript")
            return JSONResponse({"error": "app.js not found"}, status_code=404)

        @app.post("/offer")
        async def offer(request: Request):
            """Handle WebRTC SDP offer from client."""
            params = await request.json()
            offer_sdp = RTCSessionDescription(sdp=params["sdp"], type=params["type"])

            pc = RTCPeerConnection(configuration=RTCConfiguration(
                iceServers=[RTCIceServer(urls=[self._stun_server])]
            ))
            self._pcs.add(pc)

            @pc.on("connectionstatechange")
            async def on_connectionstatechange():
                print(f"[WebRTC] Connection state: {pc.connectionState}")
                if pc.connectionState == "failed" or pc.connectionState == "closed":
                    await pc.close()
                    self._pcs.discard(pc)

            # Add avatar media tracks
            video_track = AvatarVideoTrack(self._muxer)
            audio_track = AvatarAudioTrack(self._muxer)
            pc.addTrack(video_track)
            pc.addTrack(audio_track)

            await pc.setRemoteDescription(offer_sdp)
            answer = await pc.createAnswer()
            await pc.setLocalDescription(answer)

            return JSONResponse({
                "sdp": pc.localDescription.sdp,
                "type": pc.localDescription.type,
            })

        @app.get("/health")
        async def health():
            return {"status": "ok", "connections": len(self._pcs)}

        return app
