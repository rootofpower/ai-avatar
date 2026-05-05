
import cv2
import numpy as np
from typing import Optional


class StyleFilter:
    """
    Post-processing filter that transforms realistic MuseTalk output
    into a stylized cyberpunk/neon avatar look.

    Applied per-frame after MuseTalk inference, before WebRTC delivery.
    All operations are pure OpenCV — no GPU required (~1ms per frame).
    """

    def __init__(
        self,
        tint_color: tuple = (255, 50, 255),  # BGR: magenta/pink
        edge_strength: float = 0.6,
        glow_intensity: float = 0.3,
        scanline_opacity: float = 0.08,
        brightness_boost: float = 1.2,
    ):
        self._tint = np.array(tint_color, dtype=np.float32) / 255.0
        self._edge_strength = edge_strength
        self._glow_intensity = glow_intensity
        self._scanline_opacity = scanline_opacity
        self._brightness_boost = brightness_boost
        self._scanline_cache: Optional[np.ndarray] = None

    def apply(self, frame_bgr: np.ndarray) -> np.ndarray:
        """
        Apply cyberpunk style filter to a BGR frame.

        Pipeline:
        1. Convert to luminance (grayscale)
        2. Boost contrast
        3. Tint with target color
        4. Add edge glow (Canny edges with blur)
        5. Add subtle scanline effect
        """
        h, w = frame_bgr.shape[:2]

        # 1. Grayscale luminance
        gray = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2GRAY).astype(np.float32) / 255.0

        # 2. Contrast boost via CLAHE
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        gray_u8 = (gray * 255).astype(np.uint8)
        gray_u8 = clahe.apply(gray_u8)
        gray = gray_u8.astype(np.float32) / 255.0

        # Brightness boost
        gray = np.clip(gray * self._brightness_boost, 0, 1)

        # 3. Colorize with tint
        colored = np.zeros((h, w, 3), dtype=np.float32)
        for c in range(3):
            colored[:, :, c] = gray * self._tint[c]

        # 4. Edge glow
        if self._edge_strength > 0:
            edges = cv2.Canny(gray_u8, 50, 150)
            # Dilate edges slightly for glow
            kernel = np.ones((2, 2), np.uint8)
            edges = cv2.dilate(edges, kernel, iterations=1)
            # Blur for glow effect
            glow = cv2.GaussianBlur(edges.astype(np.float32) / 255.0, (7, 7), 0)
            glow_colored = np.zeros_like(colored)
            for c in range(3):
                glow_colored[:, :, c] = glow * self._tint[c]
            colored = colored + glow_colored * self._edge_strength
            colored = np.clip(colored, 0, 1)

        # 5. Scanline effect
        if self._scanline_opacity > 0:
            if self._scanline_cache is None or self._scanline_cache.shape[:2] != (h, w):
                self._scanline_cache = self._make_scanlines(h, w)
            colored = colored * (1.0 - self._scanline_cache[:, :, np.newaxis] * self._scanline_opacity)

        # Convert back to uint8 BGR
        result = (colored * 255).clip(0, 255).astype(np.uint8)
        return result

    def _make_scanlines(self, h: int, w: int) -> np.ndarray:
        """Generate horizontal scanline pattern."""
        pattern = np.zeros((h, w), dtype=np.float32)
        pattern[::2, :] = 1.0  # Every other line
        # Slight blur for subtlety
        pattern = cv2.GaussianBlur(pattern, (1, 3), 0)
        return pattern
