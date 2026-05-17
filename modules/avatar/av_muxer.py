
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
        self.video_queue: asyncio.Queue = asyncio.Queue(maxsize=250)
        self.audio_queue: asyncio.Queue = asyncio.Queue(maxsize=500)

    async def submit_frame(self, frame_bgr: np.ndarray, pts: float,
                           chunk_duration_ms: int = 40) -> None:
        """
        Submit a generated video frame with its PTS.
        Splits matching audio into synchronized segments.

        Args:
            frame_bgr: BGR numpy array (lip-synced frame)
            pts: presentation timestamp matching the audio chunk
            chunk_duration_ms: duration of the source audio chunk in ms (now 40ms)
        """
        fps = 25
        frame_duration_ms = 1000 / fps  # 40ms
        num_video_frames = max(1, int(chunk_duration_ms / frame_duration_ms))  # 1

        # Retrieve matching audio from delay buffer
        audio_pcm = await self._delay_buffer.read(pts, tolerance=0.5)

        if audio_pcm is None:
            # No matching audio — generate silence for the chunk
            samples_16k = int(self._source_audio_rate * chunk_duration_ms / 1000)
            audio_pcm = b'\x00' * (samples_16k * 2)

        # Resample full chunk 16kHz → 48kHz
        audio_48k = self._resample_16k_to_48k(audio_pcm)

        # Split audio evenly across the video frames
        audio_bytes_per_frame = len(audio_48k) // num_video_frames

        for i in range(num_video_frames):
            frame_pts = pts + (i * frame_duration_ms / 1000.0)

            # Push duplicated video frame
            video_entry = {'frame': frame_bgr, 'pts': frame_pts}
            try:
                self.video_queue.put_nowait(video_entry)
            except asyncio.QueueFull:
                try:
                    self.video_queue.get_nowait()
                except asyncio.QueueEmpty:
                    pass
                self.video_queue.put_nowait(video_entry)

            # Push matching audio slice
            audio_start = i * audio_bytes_per_frame
            audio_end = audio_start + audio_bytes_per_frame
            audio_slice = audio_48k[audio_start:audio_end]

            audio_entry = {'pcm_48k': audio_slice, 'pts': frame_pts}
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
