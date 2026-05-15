
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

        # Setup Mediapipe Selfie Segmentation if available
        self._segmenter = None
        if self._remove_bg:
            try:
                import mediapipe as mp
                # Using general model (0) for better quality, landscape (1) is faster
                self._segmenter = mp.solutions.selfie_segmentation.SelfieSegmentation(model_selection=0)
                print("[StyleFilter] Successfully loaded Mediapipe Selfie Segmentation.")
            except ImportError:
                print("[StyleFilter] WARNING: Mediapipe not installed. Background removal will use Haar Cascade fallback. Run `pip install mediapipe` for perfect background removal.")
            except AttributeError:
                print("[StyleFilter] WARNING: Incompatible Mediapipe version. Background removal will use Haar Cascade fallback.")


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
        Person segmentation for MuseTalk 512x512 frames.
        Uses face detection to build a generous mask covering head, hair,
        neck, and upper body. Background fades to black with a soft edge.

        For MuseTalk output the face is always roughly centered, so
        we can be aggressive with the mask coverage.
        """
        h, w = frame_bgr.shape[:2]

        if self._segmenter is not None:
            # Mediapipe is incredibly fast and gives pixel-perfect segmentation
            # Convert BGR to RGB for mediapipe
            frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
            results = self._segmenter.process(frame_rgb)
            if results.segmentation_mask is not None:
                mask = results.segmentation_mask.astype(np.float32)
                # Apply a slight blur for softer edges
                mask = cv2.GaussianBlur(mask, (5, 5), 0)
                return mask

        # Fallback: Use cached mask if available and same size (face pos doesn't change much)
        if (self._person_mask_cache is not None and
                self._person_mask_shape == (h, w)):
            return self._person_mask_cache

        # Detect face
        gray = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2GRAY)
        face_cascade = cv2.CascadeClassifier(
            cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
        )
        faces = face_cascade.detectMultiScale(gray, 1.1, 5, minSize=(50, 50))

        mask = np.zeros((h, w), dtype=np.float32)

        if len(faces) > 0:
            fx, fy, fw, fh = max(faces, key=lambda f: f[2] * f[3])
            cx = fx + fw // 2
            cy = fy + fh // 2

            # Build a generous person region:
            # - Width: 3x face width (covers hair + ears + shoulders)
            # - Top: start well above face (hair)
            # - Bottom: extend to frame bottom (body)
            region_w = min(int(fw * 3.0), w)
            region_top = max(0, fy - int(fh * 0.8))   # hair above forehead
            region_left = max(0, cx - region_w // 2)
            region_right = min(w, cx + region_w // 2)
            region_bottom = h  # extend to bottom

            # Fill the person rectangle
            mask[region_top:region_bottom, region_left:region_right] = 1.0

            # Round the top corners with an ellipse for natural head shape
            head_ellipse_cy = region_top + int(fh * 0.3)
            head_ellipse_rx = (region_right - region_left) // 2
            head_ellipse_ry = int(fh * 0.8)
            cv2.ellipse(mask,
                        (cx, head_ellipse_cy),
                        (head_ellipse_rx, head_ellipse_ry),
                        0, 180, 360, 1.0, -1)

            # Heavy Gaussian blur for very soft falloff at edges
            mask = cv2.GaussianBlur(mask, (71, 71), 25)
            # Boost so the center is fully opaque
            mask = np.clip(mask * 1.8, 0, 1)
        else:
            # Fallback: generous centered ellipse covering most of the frame
            cv2.ellipse(mask, (w // 2, int(h * 0.45)),
                        (int(w * 0.42), int(h * 0.50)),
                        0, 0, 360, 1.0, -1)
            mask = cv2.GaussianBlur(mask, (71, 71), 25)
            mask = np.clip(mask * 1.5, 0, 1)

        # Cache for next frame
        self._person_mask_cache = mask
        self._person_mask_shape = (h, w)

        return mask

    def _make_scanlines(self, h: int, w: int) -> np.ndarray:
        """Generate horizontal scanline pattern."""
        pattern = np.zeros((h, w), dtype=np.float32)
        pattern[::2, :] = 1.0
        pattern = cv2.GaussianBlur(pattern, (1, 3), 0)
        return pattern
