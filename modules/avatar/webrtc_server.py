
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
                 driving_video: Optional[str] = None, audio_track=None):
        super().__init__()
        self._muxer = muxer
        self._fps = fps
        self._pts = 0
        self._time_base = fractions.Fraction(1, fps)
        self._frame_duration = 1.0 / fps
        self._idle = IdleAnimator(reference_image, fps)
        self._last_speech_frame: Optional[np.ndarray] = None
        self._last_speech_time: float = 0
        self._speech_holdoff: float = 0.5  # hold last frame for 0.5s after speech ends
        self._next_pts_time: Optional[float] = None
        self._audio_track = audio_track
        self._current_speech_frame: Optional[np.ndarray] = None
        self._current_speech_pts: float = -1.0

        if driving_video:
            self._idle.set_driving_video(driving_video)

    def set_reference_image(self, image: np.ndarray) -> None:
        self._idle.set_reference(image)

    def set_driving_video(self, video_path: str) -> bool:
        return self._idle.set_driving_video(video_path)

    async def recv(self) -> VideoFrame:
        """Return the next video frame for WebRTC transmission."""
        
        # --- Pacing logic to prevent drift ---
        if self._next_pts_time is None:
            self._next_pts_time = _time.monotonic()
            
        now = _time.monotonic()
        wait_time = self._next_pts_time - now
        if wait_time > 0:
            await asyncio.sleep(wait_time)
        elif wait_time < -0.1:
            self._next_pts_time = now
            
        self._next_pts_time += self._frame_duration
        # -------------------------------------

        frame_bgr = None
        target_pts = 0.0
        
        if self._audio_track is not None:
            target_pts = self._audio_track.speech_time
            
            if target_pts > 0:
                # Audio is playing, try to catch up with video
                try:
                    while True:
                        if self._current_speech_pts < target_pts:
                            entry = self._muxer.video_queue.get_nowait()
                            self._current_speech_frame = entry['frame']
                            self._current_speech_pts = entry['pts']
                            self._last_speech_time = _time.monotonic()
                        else:
                            break
                except asyncio.QueueEmpty:
                    pass
                
                frame_bgr = self._current_speech_frame
            else:
                self._current_speech_pts = -1.0
                self._current_speech_frame = None

        else:
            # Fallback if audio_track not provided (should not happen)
            try:
                entry = self._muxer.video_queue.get_nowait()
                frame_bgr = entry['frame']
                self._last_speech_time = _time.monotonic()
            except asyncio.QueueEmpty:
                pass

        if frame_bgr is not None:
            frame_rgb = frame_bgr[:, :, ::-1].copy()
            frame = VideoFrame.from_ndarray(frame_rgb, format='rgb24')
        else:
            # Hold last speech frame briefly before returning to idle
            if (self._last_speech_frame is not None and
                    _time.monotonic() - self._last_speech_time < self._speech_holdoff):
                frame_rgb = self._last_speech_frame[:, :, ::-1].copy()
                frame = VideoFrame.from_ndarray(frame_rgb, format='rgb24')
            else:
                self._last_speech_frame = None
                # Use idle animation
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
    Custom audio track that delivers PCM audio for WebRTC.
    Audio is fed directly via push_audio() — decoupled from video inference.

    Includes a pre-buffer delay (PREBUFFER_MS) to let video inference
    produce initial frames before audio playback starts. This prevents
    the abrupt idle→speech transition and improves lip sync.
    """
    kind = "audio"

    PREBUFFER_MS = 600  # accumulate this much audio before starting playback

    def __init__(self, sample_rate: int = 48000):
        super().__init__()
        self._sample_rate = sample_rate
        self._samples_per_frame = 960  # 20ms at 48kHz (Opus standard)
        self._pts = 0
        self._time_base = fractions.Fraction(1, sample_rate)
        self._buffer = bytearray()
        self._lock = asyncio.Lock()
        self._playback_started = False
        self._prebuffer_bytes = int(sample_rate * self.PREBUFFER_MS / 1000) * 2
        self._next_pts_time: Optional[float] = None
        self.speech_time: float = 0.0

    async def push_audio(self, pcm_48k: bytes) -> None:
        """Push resampled 48kHz PCM audio into the buffer."""
        async with self._lock:
            self._buffer.extend(pcm_48k)

    def reset(self) -> None:
        """Reset for a new conversation turn."""
        self._buffer.clear()
        self._playback_started = False
        self._next_pts_time = None
        self.speech_time = 0.0

    async def recv(self) -> AudioFrame:
        """Return the next 20ms audio frame for WebRTC."""
        
        # --- Pacing logic to prevent buffer dumps ---
        if self._next_pts_time is None:
            self._next_pts_time = _time.monotonic()
            
        now = _time.monotonic()
        wait_time = self._next_pts_time - now
        if wait_time > 0:
            await asyncio.sleep(wait_time)
        elif wait_time < -0.1:
            # If we fall too far behind (e.g. CPU stall), reset the clock 
            # to prevent a sudden burst of packets.
            self._next_pts_time = now
            
        # Advance clock by exactly 20ms
        self._next_pts_time += (self._samples_per_frame / self._sample_rate)
        # --------------------------------------------

        needed_bytes = self._samples_per_frame * 2  # 16-bit

        async with self._lock:
            buf_len = len(self._buffer)

            # Pre-buffer: wait until we have enough audio before starting
            if not self._playback_started:
                if buf_len >= self._prebuffer_bytes:
                    self._playback_started = True
                    self.speech_time = 0.0
                else:
                    # Return silence while buffering
                    frame_bytes = b'\x00' * needed_bytes
                    return self._make_frame(frame_bytes)

            # Normal playback
            if buf_len >= needed_bytes:
                frame_bytes = bytes(self._buffer[:needed_bytes])
                del self._buffer[:needed_bytes]
                self.speech_time += (self._samples_per_frame / self._sample_rate)
            else:
                # Buffer exhausted — end of speech
                available = bytes(self._buffer)
                self._buffer.clear()
                self._playback_started = False
                frame_bytes = available + b'\x00' * (needed_bytes - len(available))
                self.speech_time += (len(available) / 2 / self._sample_rate)

        return self._make_frame(frame_bytes)

    def _make_frame(self, frame_bytes: bytes) -> AudioFrame:
        """Build an AudioFrame from raw PCM bytes."""
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
        self._audio_tracks: list = []  # active audio tracks for pushing audio
        self.app = self._create_app()

    def set_reference_image(self, image: np.ndarray) -> None:
        """Set the reference image for idle animation."""
        self._reference_image = image.copy()

    def set_driving_video(self, video_path: str) -> None:
        """Set the driving video for idle animation."""
        self._driving_video = video_path

    async def push_audio(self, pcm_48k: bytes) -> None:
        """Push audio to all connected audio tracks."""
        for track in self._audio_tracks:
            await track.push_audio(pcm_48k)

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
                if pc.connectionState in ("failed", "closed"):
                    await pc.close()
                    self._pcs.discard(pc)
                    if audio_track in self._audio_tracks:
                        self._audio_tracks.remove(audio_track)

            audio_track = AvatarAudioTrack()
            self._audio_tracks.append(audio_track)

            # Add avatar media tracks
            video_track = AvatarVideoTrack(
                self._muxer,
                reference_image=self._reference_image,
                driving_video=self._driving_video,
                audio_track=audio_track,
            )
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
