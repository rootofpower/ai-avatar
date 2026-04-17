
import asyncio
import time
from collections import OrderedDict
from typing import Optional


class AudioDelayBuffer:
    """
    Ring buffer holding transport audio until matching video frames are ready.
    PTS-indexed FIFO with garbage collection for stale segments.

    Write path: receives PCM segments with PTS from AudioSplitter's transport queue.
    Read path: get(pts) returns audio segment matching a specific PTS.
    """

    STALE_THRESHOLD_S = 30.0  # Keep segments longer — inference may lag behind
    COLD_START_SILENCE_MS = 500  # Silence padding during cold start

    def __init__(self, sample_rate: int = 16000, sample_width: int = 2):
        self._buffer: OrderedDict[float, dict] = OrderedDict()
        self._sample_rate = sample_rate
        self._sample_width = sample_width
        self._lock = asyncio.Lock()
        self._first_write_time: Optional[float] = None

    async def write(self, segment: dict) -> None:
        """
        Write a transport audio segment into the buffer.
        segment: {'pcm': bytes, 'pts': float, 'samples': int}
        """
        async with self._lock:
            if self._first_write_time is None:
                self._first_write_time = time.monotonic()
            self._buffer[segment['pts']] = segment
            self._gc()

    async def read(self, pts: float, tolerance: float = 0.02) -> Optional[bytes]:
        """
        Read audio matching the given PTS (within tolerance).
        Returns PCM bytes or None if not found.
        """
        async with self._lock:
            # Find closest PTS
            best_key = None
            best_diff = float('inf')
            for key in self._buffer:
                diff = abs(key - pts)
                if diff < best_diff:
                    best_diff = diff
                    best_key = key

            if best_key is not None and best_diff <= tolerance:
                segment = self._buffer.pop(best_key)
                return segment['pcm']

            return None

    async def read_range(self, pts_start: float, pts_end: float) -> bytes:
        """
        Read all audio segments within a PTS range, concatenated.
        """
        async with self._lock:
            result = bytearray()
            keys_to_remove = []
            for key in self._buffer:
                if pts_start <= key <= pts_end:
                    result.extend(self._buffer[key]['pcm'])
                    keys_to_remove.append(key)
            for key in keys_to_remove:
                del self._buffer[key]
            return bytes(result)

    def _gc(self) -> None:
        """Remove stale segments older than threshold."""
        if not self._buffer:
            return
        now_pts = max(self._buffer.keys())
        stale_keys = [k for k in self._buffer if now_pts - k > self.STALE_THRESHOLD_S]
        for k in stale_keys:
            del self._buffer[k]

    def generate_silence(self, duration_ms: int) -> bytes:
        """Generate silence PCM for cold-start padding."""
        num_samples = int(self._sample_rate * duration_ms / 1000)
        return b'\x00' * (num_samples * self._sample_width)

    async def reset(self) -> None:
        """Clear buffer for new conversation."""
        async with self._lock:
            self._buffer.clear()
            self._first_write_time = None

    @property
    def size(self) -> int:
        return len(self._buffer)
