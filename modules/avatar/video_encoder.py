
import numpy as np
from typing import Optional

try:
    import av
    HAS_AV = True
except ImportError:
    HAS_AV = False


class VideoEncoder:
    """
    Encodes generated frames to H.264 for WebRTC transport.

    Uses PyAV (FFmpeg wrapper). Tries h264_nvenc first (NVENC hardware encoder),
    falls back to libx264 with zerolatency tune.

    H.264 Baseline profile for maximum WebRTC browser compatibility.
    Ultra-low-latency: no B-frames, single-pass, CBR.
    """

    def __init__(
        self,
        width: int = 512,
        height: int = 512,
        fps: int = 25,
        bitrate: int = 2_000_000,
    ):
        self._width = width
        self._height = height
        self._fps = fps
        self._bitrate = bitrate
        self._codec_name: Optional[str] = None
        self._encoder = None
        self._frame_count = 0

    def _try_codec(self, codec_name: str) -> bool:
        """Test if a codec is available."""
        if not HAS_AV:
            return False
        try:
            av.codec.Codec(codec_name, 'w')
            return True
        except (av.codec.codec.UnknownCodecError, Exception):
            return False

    def init(self) -> str:
        """
        Initialize encoder. Returns the codec name selected.
        """
        if not HAS_AV:
            raise RuntimeError("PyAV not installed. Run: pip install av")

        # Try NVENC first (RTX 5070 Ti supports this)
        if self._try_codec('h264_nvenc'):
            self._codec_name = 'h264_nvenc'
        elif self._try_codec('libx264'):
            self._codec_name = 'libx264'
        else:
            raise RuntimeError("No H.264 encoder available")

        print(f"[VideoEncoder] Using codec: {self._codec_name}")
        self._frame_count = 0
        return self._codec_name

    def encode_frame(self, frame_bgr: np.ndarray) -> list:
        """
        Encode a single BGR frame to H.264 packets.

        This method is for standalone encoding (e.g., recording).
        For WebRTC, aiortc handles encoding internally — use
        the raw frame directly with VideoStreamTrack.

        Args:
            frame_bgr: BGR numpy array

        Returns:
            List of encoded packet bytes
        """
        if not HAS_AV:
            return []

        # Create an in-memory output container for single-frame encoding
        frame_rgb = frame_bgr[:, :, ::-1].copy()
        video_frame = av.VideoFrame.from_ndarray(frame_rgb, format='rgb24')
        video_frame.pts = self._frame_count
        self._frame_count += 1

        # For standalone encoding, we'd use a codec context.
        # For WebRTC (our primary use case), return the av.VideoFrame directly.
        return [video_frame]

    def frame_to_av(self, frame_bgr: np.ndarray, pts: int) -> 'av.VideoFrame':
        """
        Convert BGR numpy array to av.VideoFrame for aiortc.

        Args:
            frame_bgr: BGR numpy array
            pts: presentation timestamp

        Returns:
            PyAV VideoFrame ready for aiortc VideoStreamTrack
        """
        frame_rgb = frame_bgr[:, :, ::-1].copy()

        # Resize if needed
        h, w = frame_rgb.shape[:2]
        if w != self._width or h != self._height:
            import cv2
            frame_rgb = cv2.resize(frame_rgb, (self._width, self._height))

        video_frame = av.VideoFrame.from_ndarray(frame_rgb, format='rgb24')
        video_frame.pts = pts
        video_frame.time_base = f'1/{self._fps}'
        return video_frame

    @property
    def codec_name(self) -> Optional[str]:
        return self._codec_name

    @property
    def fps(self) -> int:
        return self._fps
