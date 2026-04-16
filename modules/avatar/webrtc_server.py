
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


import cv2
import math
import time as _time


class IdleAnimator:
    """
    Generates idle animation for the avatar when not speaking.

    Two modes:
    1. Driving video (preferred): Loops a short video of natural idle behavior
       (breathing, head movements, blinks). Set via set_driving_video().
    2. Static fallback: Subtle vertical shift on the reference image.

    Pure CPU — no GPU needed.
    """

    def __init__(self, reference_image: Optional[np.ndarray] = None, fps: int = 25):
        self._ref = None
        self._padded = None
        self._fps = fps
        self._size = 512
        self._pad = 8

        # Driving video mode
        self._video_frames: list = []
        self._video_idx: int = 0
        self._video_direction: int = 1  # 1=forward, -1=backward (ping-pong)

        self._start_time = _time.monotonic()
        if reference_image is not None:
            self.set_reference(reference_image)

    def set_reference(self, image: np.ndarray) -> None:
        """Set or update the reference image (BGR, center-cropped to 512x512)."""
        h, w = image.shape[:2]
        if h != self._size or w != self._size:
            # Resize preserving aspect ratio (fit the larger dimension)
            scale = self._size / min(h, w)
            new_w = int(w * scale)
            new_h = int(h * scale)
            image = cv2.resize(image, (new_w, new_h), interpolation=cv2.INTER_LANCZOS4)
            # Center crop to 512x512
            y_off = (new_h - self._size) // 2
            x_off = (new_w - self._size) // 2
            image = image[y_off:y_off + self._size, x_off:x_off + self._size]
        self._ref = image.copy()
        p = self._pad
        self._padded = cv2.copyMakeBorder(
            self._ref, p, p, p, p, cv2.BORDER_REFLECT_101
        )
        self._start_time = _time.monotonic()

    def set_driving_video(self, video_path: str) -> bool:
        """
        Load a short driving video for idle animation.
        All frames are cached in memory for smooth looping.
        Returns True if loaded successfully.
        """
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            print(f"[IdleAnimator] Cannot open driving video: {video_path}")
            return False

        frames = []
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            # Resize to target size
            if frame.shape[0] != self._size or frame.shape[1] != self._size:
                frame = cv2.resize(frame, (self._size, self._size), interpolation=cv2.INTER_LANCZOS4)
            frames.append(frame)
        cap.release()

        if not frames:
            print(f"[IdleAnimator] Driving video is empty: {video_path}")
            return False

        self._video_frames = frames
        self._video_idx = 0
        self._video_direction = 1
        video_fps = cap.get(cv2.CAP_PROP_FPS) or 25
        duration = len(frames) / video_fps
        print(f"[IdleAnimator] Driving video loaded: {len(frames)} frames, {duration:.1f}s (ping-pong loop)")
        return True

    def generate(self) -> Optional[np.ndarray]:
        """Generate one idle-animated frame."""

        # Mode 1: Driving video (preferred)
        if self._video_frames:
            frame = self._video_frames[self._video_idx]
            # Ping-pong: play forward then backward for seamless loop
            self._video_idx += self._video_direction
            if self._video_idx >= len(self._video_frames) - 1:
                self._video_direction = -1
            elif self._video_idx <= 0:
                self._video_direction = 1
            return frame.copy()

        # Mode 2: Static image fallback
        if self._padded is None:
            return None

        t = _time.monotonic() - self._start_time
        p = self._pad
        s = self._size

        dy = -2.0 * math.sin(2 * math.pi * t / 3.8)
        dy += -0.5 * math.sin(2 * math.pi * t / 2.0)

        oy = int(round(p + dy))
        ox = p
        frame = self._padded[oy:oy + s, ox:ox + s].copy()
        return frame


class AvatarVideoTrack(MediaStreamTrack):
    """
    Custom video track that pulls frames from the AVMuxer's video queue.
    When idle (no frames from pipeline), shows driving video or animated reference.
    """
    kind = "video"

    def __init__(self, muxer, fps: int = 25, reference_image: Optional[np.ndarray] = None,
                 driving_video: Optional[str] = None):
        super().__init__()
        self._muxer = muxer
        self._fps = fps
        self._pts = 0
        self._time_base = fractions.Fraction(1, fps)
        self._frame_duration = 1.0 / fps
        self._idle = IdleAnimator(reference_image, fps)
        if driving_video:
            self._idle.set_driving_video(driving_video)

    def set_reference_image(self, image: np.ndarray) -> None:
        self._idle.set_reference(image)

    def set_driving_video(self, video_path: str) -> bool:
        return self._idle.set_driving_video(video_path)

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
            # No pipeline frame — use idle animation
            idle_frame = self._idle.generate()
            if idle_frame is not None:
                frame_rgb = idle_frame[:, :, ::-1].copy()
                frame = VideoFrame.from_ndarray(frame_rgb, format='rgb24')
            else:
                black = np.zeros((512, 512, 3), dtype=np.uint8)
                frame = VideoFrame.from_ndarray(black, format='rgb24')

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
        self._reference_image: Optional[np.ndarray] = None
        self._driving_video: Optional[str] = None
        self.app = self._create_app()

    def set_reference_image(self, image: np.ndarray) -> None:
        """Set the reference image for idle animation."""
        self._reference_image = image.copy()

    def set_driving_video(self, video_path: str) -> None:
        """Set the driving video for idle animation."""
        self._driving_video = video_path

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
            video_track = AvatarVideoTrack(
                self._muxer,
                reference_image=self._reference_image,
                driving_video=self._driving_video,
            )
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
