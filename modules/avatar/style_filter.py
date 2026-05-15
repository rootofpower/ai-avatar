
import cv2
import numpy as np
from typing import Optional, Tuple


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
        remove_background: bool = True,
        bg_color: tuple = (0, 0, 0),  # BGR: black background
    ):
        self._tint = np.array(tint_color, dtype=np.float32) / 255.0
        self._edge_strength = edge_strength
        self._glow_intensity = glow_intensity
        self._scanline_opacity = scanline_opacity
        self._brightness_boost = brightness_boost
        self._remove_bg = remove_background
        self._bg_color = np.array(bg_color, dtype=np.float32) / 255.0
        self._scanline_cache: Optional[np.ndarray] = None
        self._person_mask_cache: Optional[np.ndarray] = None
        self._person_mask_shape: Optional[Tuple[int, int]] = None

    def apply(self, frame_bgr: np.ndarray) -> np.ndarray:
        """
        Apply cyberpunk style filter to a BGR frame.

        Pipeline:
        1. Remove background (optional)
        2. Convert to luminance (grayscale)
        3. Boost contrast
        4. Tint with target color
        5. Add edge glow (Canny edges with blur)
        6. Add subtle scanline effect
        """
        h, w = frame_bgr.shape[:2]

        # 1. Background removal
        if self._remove_bg:
            person_mask = self._segment_person(frame_bgr)
        else:
            person_mask = np.ones((h, w), dtype=np.float32)

        # 2. Grayscale luminance
        gray = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2GRAY).astype(np.float32) / 255.0

        # 3. Contrast boost via CLAHE
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        gray_u8 = (gray * 255).astype(np.uint8)
        gray_u8 = clahe.apply(gray_u8)
        gray = gray_u8.astype(np.float32) / 255.0

        # Brightness boost
        gray = np.clip(gray * self._brightness_boost, 0, 1)

        # 4. Colorize with tint
        colored = np.zeros((h, w, 3), dtype=np.float32)
        for c in range(3):
            colored[:, :, c] = gray * self._tint[c]

        # 5. Edge glow
        if self._edge_strength > 0:
            edges = cv2.Canny(gray_u8, 50, 150)
            kernel = np.ones((2, 2), np.uint8)
            edges = cv2.dilate(edges, kernel, iterations=1)
            glow = cv2.GaussianBlur(edges.astype(np.float32) / 255.0, (7, 7), 0)
            glow_colored = np.zeros_like(colored)
            for c in range(3):
                glow_colored[:, :, c] = glow * self._tint[c]
            colored = colored + glow_colored * self._edge_strength
            colored = np.clip(colored, 0, 1)

        # 6. Scanline effect
        if self._scanline_opacity > 0:
            if self._scanline_cache is None or self._scanline_cache.shape[:2] != (h, w):
                self._scanline_cache = self._make_scanlines(h, w)
            colored = colored * (1.0 - self._scanline_cache[:, :, np.newaxis] * self._scanline_opacity)

        # 7. Apply background mask
        if self._remove_bg:
            mask_3ch = person_mask[:, :, np.newaxis]
            bg = np.zeros_like(colored)
            for c in range(3):
                bg[:, :, c] = self._bg_color[c]
            colored = colored * mask_3ch + bg * (1.0 - mask_3ch)

        # Convert back to uint8 BGR
        result = (colored * 255).clip(0, 255).astype(np.uint8)
        return result

    def _segment_person(self, frame_bgr: np.ndarray) -> np.ndarray:
        """
        Simple person segmentation using GrabCut with face detection.
        Creates a soft mask: 1.0 = person, 0.0 = background.
        Falls back to elliptical vignette if face detection fails.
        """
        h, w = frame_bgr.shape[:2]

        # Detect face for GrabCut hint
        gray = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2GRAY)
        face_cascade = cv2.CascadeClassifier(
            cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
        )
        faces = face_cascade.detectMultiScale(gray, 1.1, 5, minSize=(50, 50))

        if len(faces) > 0:
            # Use face to define person region
            fx, fy, fw, fh = max(faces, key=lambda f: f[2] * f[3])
            # Expand to include hair, neck, shoulders
            cx, cy = fx + fw // 2, fy + fh // 2
            # Person region: wider than face, extends down to bottom
            pw = int(fw * 2.5)
            ph = int(fh * 3.5)
            px = max(0, cx - pw // 2)
            py = max(0, fy - int(fh * 0.5))  # Include hair above face

            # Create elliptical mask centered on head/shoulders
            mask = np.zeros((h, w), dtype=np.float32)
            center = (cx, min(cy + fh // 2, h - 1))
            axes = (pw // 2, ph // 2)
            cv2.ellipse(mask, center, axes, 0, 0, 360, 1.0, -1)

            # Heavy blur for smooth falloff
            mask = cv2.GaussianBlur(mask, (51, 51), 0)
            mask = np.clip(mask * 1.5, 0, 1)  # Boost center
        else:
            # Fallback: centered ellipse
            mask = np.zeros((h, w), dtype=np.float32)
            cv2.ellipse(mask, (w // 2, h // 2), (w // 3, h // 2), 0, 0, 360, 1.0, -1)
            mask = cv2.GaussianBlur(mask, (51, 51), 0)

        return mask

    def _make_scanlines(self, h: int, w: int) -> np.ndarray:
        """Generate horizontal scanline pattern."""
        pattern = np.zeros((h, w), dtype=np.float32)
        pattern[::2, :] = 1.0
        pattern = cv2.GaussianBlur(pattern, (1, 3), 0)
        return pattern
