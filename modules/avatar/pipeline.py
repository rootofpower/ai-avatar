
import asyncio
import time
from typing import Optional

from modules.avatar.audio_splitter import AudioSplitter
from modules.avatar.audio_delay_buffer import AudioDelayBuffer
from modules.avatar.audio_feature_extractor import AudioFeatureExtractor
from modules.avatar.visual_frame_generator import VisualFrameGenerator
from modules.avatar.video_encoder import VideoEncoder
from modules.avatar.av_muxer import AVMuxer
from modules.avatar.webrtc_server import WebRTCServer
from modules.avatar.style_filter import StyleFilter


class AvatarPipeline:
    """
    Orchestrates all avatar pipeline components:
    AudioSplitter → AFE → VFG → VideoEncoder → AVMuxer → WebRTC

    Integrates with the existing VoiceAssistant by receiving
    Gemini's streaming audio chunks via feed_audio().
    """

    def __init__(
        self,
        models_dir: str = "./models",
        avatar_image: Optional[str] = None,
        idle_video: Optional[str] = None,
        device: str = "cuda",
        fps: int = 25,
        webrtc_port: int = 8080,
        stun_server: str = "stun:stun.l.google.com:19302",
        turn_server: Optional[str] = None,
        turn_username: Optional[str] = None,
        turn_password: Optional[str] = None,
    ):
        self._models_dir = models_dir
        self._avatar_image = avatar_image
        self._idle_video = idle_video
        self._device = device
        self._fps = fps
        self._webrtc_port = webrtc_port
        self._stun_server = stun_server
        self._turn_server = turn_server
        self._turn_username = turn_username
        self._turn_password = turn_password

        # Pipeline components
        self.splitter = AudioSplitter()
        self.delay_buffer = AudioDelayBuffer()
        self.afe = AudioFeatureExtractor(
            whisper_dir=f"{models_dir}/whisper",
            device=device,
            fps=fps,
        )
        self.vfg = VisualFrameGenerator(
            models_dir=models_dir,
            device=device,
        )
        self.encoder = VideoEncoder(fps=fps)
        self.muxer = AVMuxer(self.delay_buffer)
        self.webrtc = WebRTCServer(
            self.muxer,
            stun_server=stun_server,
            turn_server=turn_server,
            turn_username=turn_username,
            turn_password=turn_password,
        )
        self.style_filter = StyleFilter()

        self._inference_task: Optional[asyncio.Task] = None
        self._transport_task: Optional[asyncio.Task] = None
        self._running = False
        self._gpu_available = False

    async def start(self) -> bool:
        """
        Load models and start pipeline workers.
        Returns True if GPU pipeline is active, False if degraded (audio-only).
        """
        print("[Pipeline] Starting avatar pipeline...")

        # Check GPU availability
        try:
            import torch
            if not torch.cuda.is_available():
                print("[Pipeline] WARNING: No CUDA GPU detected. Avatar pipeline disabled.")
                return False
            gpu_name = torch.cuda.get_device_name(0)
            vram = torch.cuda.get_device_properties(0).total_memory / (1024**3)
            print(f"[Pipeline] GPU: {gpu_name} ({vram:.1f} GB VRAM)")
        except Exception as e:
            print(f"[Pipeline] GPU check failed: {e}. Avatar pipeline disabled.")
            return False

        # Load models
        try:
            self.afe.load()
            self.vfg.load()
            self.encoder.init()
        except Exception as e:
            print(f"[Pipeline] Model loading failed: {e}. Avatar pipeline disabled.")
            return False

        # Set reference image if provided
        if self._avatar_image:
            try:
                self.vfg.set_reference_image(self._avatar_image)
                # Pass reference image to WebRTC for idle animation
                if self.vfg._ref_image is not None:
                    self.webrtc.set_reference_image(self.vfg._ref_image)
            except FileNotFoundError as e:
                print(f"[Pipeline] Avatar image not found: {e}")
                print("[Pipeline] Pipeline will wait for avatar image to be set.")

        # Set driving video for idle animation if provided
        if self._idle_video:
            import os
            if os.path.exists(self._idle_video):
                self.webrtc.set_driving_video(self._idle_video)
            else:
                print(f"[Pipeline] Idle video not found: {self._idle_video}")

        # Start worker tasks
        self._running = True
        self._inference_task = asyncio.create_task(self._inference_worker())
        self._transport_task = asyncio.create_task(self._transport_worker())
        self._gpu_available = True

        print("[Pipeline] Avatar pipeline started successfully.")
        return True

    async def feed_audio(self, pcm_24k: bytes) -> None:
        """
        Feed raw 24kHz PCM audio from Gemini into the pipeline.
        Audio goes two ways:
        1. Direct to WebRTC audio track (for playback)
        2. To splitter for inference (for lip-sync video)
        """
        if not self._running or not pcm_24k:
            return

        # 1. Push audio directly to WebRTC for immediate playback
        pcm_48k = self._resample_24k_to_48k(pcm_24k)
        await self.webrtc.push_audio(pcm_48k)

        # 2. Feed to splitter for video inference
        await self.splitter.feed(pcm_24k)

    def _resample_24k_to_48k(self, pcm_24k: bytes) -> bytes:
        """Resample 24kHz to 48kHz (1:2 ratio — duplicate each sample)."""
        import struct
        sample_count = len(pcm_24k) // 2
        if sample_count == 0:
            return pcm_24k
        samples = struct.unpack(f'<{sample_count}h', pcm_24k)
        resampled = []
        for s in samples:
            resampled.extend([s, s])  # 24k * 2 = 48k
        return struct.pack(f'<{len(resampled)}h', *resampled)

    async def _inference_worker(self) -> None:
        """
        Continuously pulls 200ms chunks from the inference queue,
        extracts audio features, generates video frames, and submits
        them to the muxer.

        GPU inference runs in a thread executor to avoid blocking
        the asyncio event loop (which would starve WebRTC tracks).
        """
        print("[Pipeline] Inference worker started.")
        loop = asyncio.get_running_loop()

        while self._running:
            try:
                chunk = await asyncio.wait_for(
                    self.splitter.inference_queue.get(), timeout=0.5
                )
            except asyncio.TimeoutError:
                continue
            except asyncio.CancelledError:
                break

            if not self.vfg.has_reference:
                continue

            try:
                t_start = time.monotonic()

                # Run GPU-heavy work in thread to not block event loop
                audio_prompt = await loop.run_in_executor(
                    None, self.afe.extract, chunk['pcm']
                )
                frame = await loop.run_in_executor(
                    None, self.vfg.generate_frame, audio_prompt
                )

                # Apply cyberpunk style filter
                if frame is not None:
                    frame = await loop.run_in_executor(
                        None, self.style_filter.apply, frame
                    )

                if frame is not None:
                    t_elapsed = (time.monotonic() - t_start) * 1000
                    q_depth = self.splitter.inference_queue.qsize()
                    if q_depth > 5 or t_elapsed > 200:
                        print(f"[Pipeline] Inference: {t_elapsed:.0f}ms, queue: {q_depth}")
                    # Submit to muxer
                    await self.muxer.submit_frame(frame, chunk['pts'])

            except Exception as e:
                print(f"[Pipeline] Inference error: {e}")

    async def _transport_worker(self) -> None:
        """
        Continuously moves audio from the splitter's transport queue
        to the delay buffer.
        """
        print("[Pipeline] Transport worker started.")
        while self._running:
            try:
                segment = await asyncio.wait_for(
                    self.splitter.transport_queue.get(), timeout=0.5
                )
            except asyncio.TimeoutError:
                continue
            except asyncio.CancelledError:
                break

            await self.delay_buffer.write(segment)

    async def stop(self) -> None:
        """Stop pipeline and clean up."""
        print("[Pipeline] Stopping avatar pipeline...")
        self._running = False

        if self._inference_task:
            self._inference_task.cancel()
            try:
                await self._inference_task
            except asyncio.CancelledError:
                pass

        if self._transport_task:
            self._transport_task.cancel()
            try:
                await self._transport_task
            except asyncio.CancelledError:
                pass

        self.splitter.reset()
        await self.delay_buffer.reset()
        await self.muxer.reset()
        print("[Pipeline] Avatar pipeline stopped.")

    def reset_for_new_turn(self) -> None:
        """Reset buffers for a new conversation turn (keep models loaded)."""
        print("[Pipeline] Resetting for new turn...")
        self.splitter.reset()
        # Reset delay buffer synchronously-safe
        self.delay_buffer._buffer.clear()
        self.delay_buffer._first_write_time = None
        # Drain muxer queues
        for q in (self.muxer.video_queue, self.muxer.audio_queue):
            while not q.empty():
                try:
                    q.get_nowait()
                except asyncio.QueueEmpty:
                    break
        # Reset all connected WebRTC audio tracks
        for track in self.webrtc._audio_tracks:
            track.reset()
        # Reset all connected WebRTC video tracks
        for track in self.webrtc._video_tracks:
            track.reset()

    @property
    def is_active(self) -> bool:
        return self._running and self._gpu_available

    def set_reference_image(self, image_path: str) -> None:
        """Set or change the avatar reference image at runtime."""
        self._avatar_image = image_path
        if self.vfg.is_loaded:
            self.vfg.set_reference_image(image_path)
            # Also update WebRTC idle animation
            if self.vfg._ref_image is not None:
                self.webrtc.set_reference_image(self.vfg._ref_image)
            print(f"[Pipeline] Reference image updated: {image_path}")
