
import asyncio
import struct
import numpy as np
from typing import Optional, Tuple


class AVMuxer:
    """
    Synchronization node: joins encoded video frames with delayed audio by PTS.

    When VFG completes a frame:
    1. Looks up matching audio from the delay buffer
    2. Resamples audio to 48kHz for WebRTC/Opus
    3. Packages matched A/V pair into output queue

    Consumers: WebRTC VideoStreamTrack and AudioStreamTrack pull from here.
    """

    def __init__(self, audio_delay_buffer, target_audio_rate: int = 48000, source_audio_rate: int = 16000):
        self._delay_buffer = audio_delay_buffer
        self._target_audio_rate = target_audio_rate
        self._source_audio_rate = source_audio_rate

        # Output queues for WebRTC tracks
        self.video_queue: asyncio.Queue = asyncio.Queue(maxsize=30)
        self.audio_queue: asyncio.Queue = asyncio.Queue(maxsize=60)

    async def submit_frame(self, frame_bgr: np.ndarray, pts: float) -> None:
        """
        Submit a generated video frame with its PTS.
        Retrieves matching audio from delay buffer and pushes both to output queues.

        Args:
            frame_bgr: BGR numpy array (full-resolution composited frame)
            pts: presentation timestamp matching the audio chunk
        """
        # Push video frame
        video_entry = {'frame': frame_bgr, 'pts': pts}
        try:
            self.video_queue.put_nowait(video_entry)
        except asyncio.QueueFull:
            try:
                self.video_queue.get_nowait()
            except asyncio.QueueEmpty:
                pass
            self.video_queue.put_nowait(video_entry)

        # Retrieve matching audio from delay buffer
        audio_pcm = await self._delay_buffer.read(pts, tolerance=0.05)

        if audio_pcm is None:
            # No matching audio — generate silence for this frame duration
            frame_duration_ms = 40  # ~25fps
            samples_16k = int(self._source_audio_rate * frame_duration_ms / 1000)
            audio_pcm = b'\x00' * (samples_16k * 2)

        # Resample 16kHz → 48kHz for WebRTC Opus
        audio_48k = self._resample_16k_to_48k(audio_pcm)

        audio_entry = {'pcm_48k': audio_48k, 'pts': pts}
        try:
            self.audio_queue.put_nowait(audio_entry)
        except asyncio.QueueFull:
            try:
                self.audio_queue.get_nowait()
            except asyncio.QueueEmpty:
                pass
            self.audio_queue.put_nowait(audio_entry)

    def _resample_16k_to_48k(self, pcm_16k: bytes) -> bytes:
        """
        Resample 16kHz PCM to 48kHz using simple interpolation (1:3 ratio).
        48000 / 16000 = 3, so each sample is tripled.
        """
        sample_count = len(pcm_16k) // 2
        if sample_count == 0:
            return pcm_16k

        samples = struct.unpack(f'<{sample_count}h', pcm_16k)

        # Simple triplication (nearest-neighbor upsampling)
        resampled = []
        for s in samples:
            resampled.extend([s, s, s])

        return struct.pack(f'<{len(resampled)}h', *resampled)

    async def reset(self) -> None:
        """Clear output queues."""
        for q in (self.video_queue, self.audio_queue):
            while not q.empty():
                try:
                    q.get_nowait()
                except asyncio.QueueEmpty:
                    break
