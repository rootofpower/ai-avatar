
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

from modules.avatar.style_filter import StyleFilter


import cv2
import math
import time as _time


class StateAnimator:
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
        self._state = "idle"
        self._videos = {}
        # Single shared playback cursor (since most states fallback to idle video)
        self._cursor_idx = 0
        self._cursor_dir = 1
        self._cursor_video_key = None  # which video the cursor is currently tracking

        self._start_time = _time.monotonic()
        if reference_image is not None:
            self.set_reference(reference_image)

    def set_reference(self, image: np.ndarray) -> None:
        """Set or update the reference image (BGR, center-cropped to 512x512)."""
        h, w = image.shape[:2]
        if h != self._size or w != self._size:
            scale = self._size / min(h, w)
            new_w = int(w * scale)
            new_h = int(h * scale)
            image = cv2.resize(image, (new_w, new_h), interpolation=cv2.INTER_LANCZOS4)
            y_off = (new_h - self._size) // 2
            x_off = (new_w - self._size) // 2
            image = image[y_off:y_off + self._size, x_off:x_off + self._size]
        self._ref = image.copy()
        p = self._pad
        self._padded = cv2.copyMakeBorder(
            self._ref, p, p, p, p, cv2.BORDER_REFLECT_101
        )
        self._start_time = _time.monotonic()

    def set_driving_video(self, state_name: str, video_path: str) -> bool:
        """
        Load a short driving video for a specific state (e.g., 'idle', 'thinking').
        All frames are cached in memory for smooth looping.
        Returns True if loaded successfully.
        """
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            print(f"[StateAnimator] Cannot open driving video for {state_name}: {video_path}")
            return False

        frames = []
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            # Resize and center crop to maintain aspect ratio
            if frame.shape[0] != self._size or frame.shape[1] != self._size:
                h, w = frame.shape[:2]
                scale = self._size / min(h, w)
                new_w = int(w * scale)
                new_h = int(h * scale)
                frame = cv2.resize(frame, (new_w, new_h), interpolation=cv2.INTER_LANCZOS4)
                y_off = (new_h - self._size) // 2
                x_off = (new_w - self._size) // 2
                frame = frame[y_off:y_off + self._size, x_off:x_off + self._size]
            frames.append(frame)
        cap.release()

        if not frames:
            print(f"[StateAnimator] Driving video for {state_name} is empty: {video_path}")
            return False

        self._videos[state_name] = frames
        video_fps = cap.get(cv2.CAP_PROP_FPS) or 25
        duration = len(frames) / video_fps
        print(f"[StateAnimator] Video for state '{state_name}' loaded: {len(frames)} frames, {duration:.1f}s")
        return True

    def _resolve_video(self, state: str):
        """Resolve which video frames to use for a given state."""
        if state in self._videos:
            return state, self._videos[state]
        if state != "idle" and "idle" in self._videos:
            return "idle", self._videos["idle"]
        return None, None

    def set_state(self, state: str) -> None:
        """Change the current animation state."""
        if state != self._state:
            old_key, _ = self._resolve_video(self._state)
            new_key, _ = self._resolve_video(state)
            print(f"[StateAnimator] State: {self._state} -> {state} (video: {old_key} -> {new_key})")
            self._state = state
            # If we're switching to a genuinely different video, reset cursor
            if new_key != old_key and new_key is not None:
                self._cursor_idx = 0
                self._cursor_dir = 1
                self._cursor_video_key = new_key

    def generate(self) -> Optional[np.ndarray]:
        """Generate one animated frame for the current state."""
        video_key, frames = self._resolve_video(self._state)
        
        if frames:
            # Clamp cursor to valid range
            if self._cursor_idx >= len(frames):
                self._cursor_idx = len(frames) - 1
                self._cursor_dir = -1
            
            frame = frames[self._cursor_idx]
            # Ping-pong: play forward then backward for seamless loop
            self._cursor_idx += self._cursor_dir
            if self._cursor_idx >= len(frames) - 1:
                self._cursor_dir = -1
            elif self._cursor_idx <= 0:
                self._cursor_dir = 1
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
                 driving_videos: Optional[dict] = None, audio_track=None, style_filter: Optional[StyleFilter] = None):
        super().__init__()
        self._muxer = muxer
        self._fps = fps
        self._pts = 0
        self._time_base = fractions.Fraction(1, fps)
        self._frame_duration = 1.0 / fps
        self._animator = StateAnimator(reference_image, fps)
        self._last_speech_frame: Optional[np.ndarray] = None
        self._last_speech_time: float = 0
        self._next_pts_time: Optional[float] = None
        self._audio_track = audio_track
        self._current_speech_frame: Optional[np.ndarray] = None
        self._current_speech_pts: float = -1.0
        self._style_filter = style_filter
        self._fade_out_counter: int = 0
        self._fade_out_length: int = 12  # cross-fade over ~0.5s at 25fps
        
        # Frame interpolation for smooth speech animation
        self._prev_speech_frame: Optional[np.ndarray] = None
        self._interp_new_time: float = 0  # when the newest keyframe arrived
        self._interp_interval: float = 0.1  # expected time between keyframes (~100ms)

        if driving_videos:
            for state_name, path in driving_videos.items():
                self._animator.set_driving_video(state_name, path)

    def reset(self) -> None:
        """Reset speech state for a new conversation turn."""
        self._current_speech_frame = None
        self._current_speech_pts = -1.0
        self._last_speech_frame = None
        self._last_speech_time = 0
        self._fade_out_counter = 0
        self._prev_speech_frame = None
        self._interp_new_time = 0

    def set_reference_image(self, image: np.ndarray) -> None:
        self._animator.set_reference(image)

    def set_driving_video(self, state_name: str, video_path: str) -> bool:
        return self._animator.set_driving_video(state_name, video_path)

    def set_state(self, state: str) -> None:
        self._animator.set_state(state)

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
        
        # Show speech frames when audio is playing OR queued frames exist
        has_queued_frames = not self._muxer.video_queue.empty()
        is_speaking = target_pts > 0 or has_queued_frames or self._current_speech_frame is not None
        
        if is_speaking:
            # Take ONE frame per recv() call — never skip frames
            try:
                entry = self._muxer.video_queue.get_nowait()
                # Save previous frame for interpolation
                if self._current_speech_frame is not None:
                    self._prev_speech_frame = self._current_speech_frame
                    now = _time.monotonic()
                    if self._interp_new_time > 0:
                        self._interp_interval = max(0.04, now - self._interp_new_time)
                    self._interp_new_time = now
                else:
                    self._interp_new_time = _time.monotonic()
                
                self._current_speech_frame = entry['frame']
                self._current_speech_pts = entry['pts']
                self._last_speech_time = _time.monotonic()
                self._last_speech_frame = entry['frame']
            except asyncio.QueueEmpty:
                # No new frame available
                # If audio has stopped AND no frames in queue, end speech mode
                if target_pts <= 0 and not has_queued_frames:
                    if _time.monotonic() - self._last_speech_time > 0.5:
                        self._current_speech_frame = None
                        self._prev_speech_frame = None
                        self._current_speech_pts = -1.0
            
            # Interpolate between previous and current MuseTalk keyframes
            if self._current_speech_frame is not None:
                if self._prev_speech_frame is not None and self._interp_new_time > 0:
                    elapsed = _time.monotonic() - self._interp_new_time
                    alpha = min(1.0, elapsed / self._interp_interval)
                    frame_bgr = cv2.addWeighted(
                        self._prev_speech_frame, 1.0 - alpha,
                        self._current_speech_frame, alpha, 0
                    )
                else:
                    frame_bgr = self._current_speech_frame
            
            self._fade_out_counter = 0

        else:
            # Fully idle — no audio, no queued frames, no held frame
            pass

        if frame_bgr is not None:
            frame_rgb = frame_bgr[:, :, ::-1].copy()
            frame = VideoFrame.from_ndarray(frame_rgb, format='rgb24')
        else:
            # Fade out from last speech frame to idle animation
            idle_frame = self._animator.generate()
            if idle_frame is not None and self._style_filter is not None:
                idle_frame = self._style_filter.apply(idle_frame)
            
            if (self._last_speech_frame is not None and 
                    self._fade_out_counter < self._fade_out_length):
                # Cross-fade: blend last speech frame with idle
                alpha = self._fade_out_counter / self._fade_out_length
                self._fade_out_counter += 1
                if idle_frame is not None:
                    blended = cv2.addWeighted(
                        self._last_speech_frame, 1.0 - alpha, idle_frame, alpha, 0
                    )
                    frame_rgb = blended[:, :, ::-1].copy()
                else:
                    frame_rgb = self._last_speech_frame[:, :, ::-1].copy()
                frame = VideoFrame.from_ndarray(frame_rgb, format='rgb24')
            elif idle_frame is not None:
                self._last_speech_frame = None
                frame_rgb = idle_frame[:, :, ::-1].copy()
                frame = VideoFrame.from_ndarray(frame_rgb, format='rgb24')
            else:
                self._last_speech_frame = None
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

    PREBUFFER_MS = 200  # accumulate this much audio before starting playback

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
        print("[WebRTC] Audio track reset for new turn")

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
                    print(f"[WebRTC] Audio playback started (buffered {buf_len} bytes)")
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
                # Reset speech_time to 0 so video track returns to idle
                self.speech_time = 0.0
                print(f"[WebRTC] Audio playback ended (buffer exhausted)")

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

    def __init__(self, muxer, stun_server: str = "stun:stun.l.google.com:19302",
                 turn_server: Optional[str] = None, turn_username: Optional[str] = None,
                 turn_password: Optional[str] = None):
        self._muxer = muxer
        self._stun_server = stun_server
        self._turn_server = turn_server
        self._turn_username = turn_username
        self._turn_password = turn_password
        self._pcs: set = set()
        self._reference_image: Optional[np.ndarray] = None
        self._driving_videos: dict = {}
        self._audio_tracks: list = []  # active audio tracks for pushing audio
        self._video_tracks: list = []  # active video tracks for reset
        self._style_filter = StyleFilter()

        # Build ICE servers list
        self._ice_servers = [RTCIceServer(urls=[stun_server])]
        if turn_server and turn_username and turn_password:
            self._ice_servers.append(RTCIceServer(
                urls=[turn_server],
                username=turn_username,
                credential=turn_password,
            ))
            # Also add TURNS (TLS) variant on port 443
            turns_url = turn_server.replace("turn:", "turns:", 1)
            if ":80" in turns_url:
                turns_url = turns_url.replace(":80", ":443")
            self._ice_servers.append(RTCIceServer(
                urls=[turns_url],
                username=turn_username,
                credential=turn_password,
            ))
            print(f"[WebRTC] TURN server configured: {turn_server}")
        else:
            print(f"[WebRTC] No TURN server configured (STUN only)")

        self.app = self._create_app()

    def set_reference_image(self, image: np.ndarray) -> None:
        """Set the reference image for idle animation."""
        self._reference_image = image.copy()

    def set_driving_video(self, state_name: str, video_path: str) -> None:
        """Set a driving video for a specific state."""
        self._driving_videos[state_name] = video_path
        for track in self._video_tracks:
            track.set_driving_video(state_name, video_path)

    def set_state(self, state: str) -> None:
        """Set the current animation state for all video tracks."""
        for track in self._video_tracks:
            track.set_state(state)

    async def push_audio(self, pcm_48k: bytes) -> None:
        """Push audio to all connected audio tracks."""
        if not self._audio_tracks:
            print("[WebRTC] WARNING: Received audio from Gemini, but no WebRTC client is connected to play it!")
            return
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

        @app.get("/ice-config")
        async def ice_config():
            """Return ICE server configuration for the client."""
            servers = [{"urls": self._stun_server}]
            if self._turn_server and self._turn_username and self._turn_password:
                servers.append({
                    "urls": self._turn_server,
                    "username": self._turn_username,
                    "credential": self._turn_password,
                })
                # TURNS variant
                turns_url = self._turn_server.replace("turn:", "turns:", 1)
                if ":80" in turns_url:
                    turns_url = turns_url.replace(":80", ":443")
                servers.append({
                    "urls": turns_url,
                    "username": self._turn_username,
                    "credential": self._turn_password,
                })
            return JSONResponse({"iceServers": servers})

        @app.post("/offer")
        async def offer(request: Request):
            """Handle WebRTC SDP offer from client."""
            params = await request.json()
            print(f"[WebRTC] Received offer from client (type={params['type']})")
            offer_sdp = RTCSessionDescription(sdp=params["sdp"], type=params["type"])

            pc = RTCPeerConnection(configuration=RTCConfiguration(
                iceServers=self._ice_servers
            ))
            self._pcs.add(pc)
            ice_info = f"STUN: {self._stun_server}"
            if self._turn_server:
                ice_info += f", TURN: {self._turn_server}"
            print(f"[WebRTC] Created PeerConnection ({ice_info})")

            @pc.on("connectionstatechange")
            async def on_connectionstatechange():
                print(f"[WebRTC] Connection state: {pc.connectionState}")
                if pc.connectionState in ("failed", "closed"):
                    await pc.close()
                    self._pcs.discard(pc)
                    if audio_track in self._audio_tracks:
                        self._audio_tracks.remove(audio_track)
                    if video_track in self._video_tracks:
                        self._video_tracks.remove(video_track)

            @pc.on("iceconnectionstatechange")
            async def on_iceconnectionstatechange():
                print(f"[WebRTC] ICE connection state: {pc.iceConnectionState}")

            @pc.on("icegatheringstatechange")
            async def on_icegatheringstatechange():
                print(f"[WebRTC] ICE gathering state: {pc.iceGatheringState}")

            audio_track = AvatarAudioTrack()
            self._audio_tracks.append(audio_track)

            # Add avatar media tracks
            video_track = AvatarVideoTrack(
                self._muxer,
                reference_image=self._reference_image,
                driving_videos=self._driving_videos,
                audio_track=audio_track,
                style_filter=self._style_filter,
            )
            pc.addTrack(video_track)
            pc.addTrack(audio_track)
            self._video_tracks.append(video_track)
            print(f"[WebRTC] Added video + audio tracks")

            await pc.setRemoteDescription(offer_sdp)
            print(f"[WebRTC] Remote description set")
            answer = await pc.createAnswer()
            await pc.setLocalDescription(answer)
            print(f"[WebRTC] Local description set, waiting for ICE gathering...")

            # Wait for ICE gathering to complete (with timeout)
            if pc.iceGatheringState != "complete":
                ice_done = asyncio.Event()

                @pc.on("icegatheringstatechange")
                async def on_ice_complete():
                    if pc.iceGatheringState == "complete":
                        ice_done.set()

                try:
                    await asyncio.wait_for(ice_done.wait(), timeout=5.0)
                    print(f"[WebRTC] ICE gathering complete")
                except asyncio.TimeoutError:
                    print(f"[WebRTC] ICE gathering timed out (5s) — sending partial candidates")

            # Log offer candidates from client
            offer_lines = [l for l in params["sdp"].split('\n') if 'a=candidate:' in l]
            print(f"[WebRTC] Client offer has {len(offer_lines)} ICE candidates:")
            for line in offer_lines:
                print(f"  [OFFER] {line.strip()}")

            sdp_answer = pc.localDescription.sdp
            answer_lines = [l for l in sdp_answer.split('\n') if 'a=candidate:' in l]
            print(f"[WebRTC] Sending answer with {len(answer_lines)} ICE candidates:")
            for line in answer_lines:
                print(f"  [ANSWER] {line.strip()}")

            return JSONResponse({
                "sdp": pc.localDescription.sdp,
                "type": pc.localDescription.type,
            })

        @app.get("/health")
        async def health():
            return {"status": "ok", "connections": len(self._pcs)}

        return app
