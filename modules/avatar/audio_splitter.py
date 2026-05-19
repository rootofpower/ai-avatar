
import asyncio
import time
import struct
from typing import Optional


class AudioSplitter:
    """
    Receives raw PCM audio from Gemini and forks it into two async queues:
    1. Inference queue: 200ms chunks (3200 samples at 16kHz) for Whisper/MuseTalk
    2. Transport queue: raw audio segments for the Audio Delay Buffer

    Input: 24kHz 16-bit mono PCM from Gemini.
    Internally resamples to 16kHz for inference.
    """

    INPUT_SAMPLE_RATE = 24000
    TARGET_SAMPLE_RATE = 16000
    SAMPLE_WIDTH = 2  # 16-bit = 2 bytes
    CHUNK_DURATION_MS = 40  # 40ms exactly matches 25 FPS video (1 frame per chunk)
    CHUNK_SAMPLES = int(TARGET_SAMPLE_RATE * CHUNK_DURATION_MS / 1000)  # 640
    STALE_THRESHOLD_MS = 1000

    def __init__(self):
        self.inference_queue: asyncio.Queue = asyncio.Queue(maxsize=500)
        self.transport_queue: asyncio.Queue = asyncio.Queue(maxsize=500)

        self._accumulator = bytearray()
        self._samples_processed = 0
        self._inference_samples_emitted = 0
        self._base_time: Optional[float] = None
        self._running = False

    def _resample_24k_to_16k(self, pcm_24k: bytes) -> bytes:
        """Resample 24kHz PCM to 16kHz using simple linear decimation (2:3 ratio)."""
        # 24000 / 16000 = 3/2, so we pick 2 out of every 3 samples
        sample_count = len(pcm_24k) // self.SAMPLE_WIDTH
        samples = struct.unpack(f'<{sample_count}h', pcm_24k)

        resampled = []
        # Simple decimation: for every 3 input samples, output 2
        i = 0
        while i + 2 < len(samples):
            resampled.append(samples[i])
            resampled.append(samples[i + 1])
            i += 3
        # Handle remainder
        if i < len(samples):
            resampled.append(samples[i])

        return struct.pack(f'<{len(resampled)}h', *resampled)

    def _assign_pts(self, num_samples: int) -> float:
        """Assign PTS in seconds based on sample count."""
        if self._base_time is None:
            self._base_time = time.monotonic()
        pts = self._samples_processed / self.TARGET_SAMPLE_RATE
        self._samples_processed += num_samples
        return pts

    async def feed(self, pcm_24k: bytes) -> None:
        """
        Feed raw 24kHz PCM audio from Gemini.
        Resamples to 16kHz, duplicates to both queues.
        """
        if not pcm_24k:
            return

        pcm_16k = self._resample_24k_to_16k(pcm_24k)
        segment_samples = len(pcm_16k) // self.SAMPLE_WIDTH
        pts = self._assign_pts(segment_samples)

        # Transport queue: send raw 16kHz segment with PTS
        transport_segment = {
            'pcm': bytes(pcm_16k),
            'pts': pts,
            'samples': segment_samples,
        }
        try:
            self.transport_queue.put_nowait(transport_segment)
        except asyncio.QueueFull:
            # Drop oldest if buffer is full (back-pressure)
            try:
                self.transport_queue.get_nowait()
            except asyncio.QueueEmpty:
                pass
            self.transport_queue.put_nowait(transport_segment)

        # Inference queue: sliding window of 200ms (3200 samples) at 40ms steps (640 samples)
        self._accumulator.extend(pcm_16k)

        WINDOW_SIZE_SAMPLES = 3200
        PAST_CONTEXT_SAMPLES = 1280
        FUTURE_CONTEXT_SAMPLES = 1920
        STEP_SAMPLES = 640

        while (len(self._accumulator) // self.SAMPLE_WIDTH) >= (self._inference_samples_emitted + FUTURE_CONTEXT_SAMPLES):
            S = self._inference_samples_emitted
            start_sample = S - PAST_CONTEXT_SAMPLES
            end_sample = S + FUTURE_CONTEXT_SAMPLES

            if start_sample < 0:
                silence_samples = -start_sample
                actual_samples = WINDOW_SIZE_SAMPLES - silence_samples
                chunk_data = b'\x00' * (silence_samples * self.SAMPLE_WIDTH) + \
                             self._accumulator[0 : actual_samples * self.SAMPLE_WIDTH]
            else:
                chunk_data = self._accumulator[start_sample * self.SAMPLE_WIDTH : end_sample * self.SAMPLE_WIDTH]

            chunk_pts = S / self.TARGET_SAMPLE_RATE
            inference_chunk = {
                'pcm': bytes(chunk_data),
                'pts': chunk_pts,
                'samples': STEP_SAMPLES,
            }

            try:
                self.inference_queue.put_nowait(inference_chunk)
            except asyncio.QueueFull:
                try:
                    self.inference_queue.get_nowait()
                except asyncio.QueueEmpty:
                    pass
                self.inference_queue.put_nowait(inference_chunk)

            self._inference_samples_emitted += STEP_SAMPLES

    def reset(self) -> None:
        """Reset state for a new conversation turn."""
        self._accumulator.clear()
        self._samples_processed = 0
        self._inference_samples_emitted = 0
        self._base_time = None
        # Drain queues
        for q in (self.inference_queue, self.transport_queue):
            while not q.empty():
                try:
                    q.get_nowait()
                except asyncio.QueueEmpty:
                    break
