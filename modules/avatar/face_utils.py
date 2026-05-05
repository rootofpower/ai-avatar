
import cv2
import numpy as np
from PIL import Image
from typing import Tuple, List, Optional


def detect_face_bbox(image: np.ndarray, bbox_shift: int = 0) -> Optional[Tuple[int, int, int, int]]:
    """
    Detect face bounding box using OpenCV's Haar cascade.
    Falls back to simple heuristic if no face found.

    Args:
        image: BGR numpy array
        bbox_shift: vertical shift for bbox adjustment

    Returns:
        (x1, y1, x2, y2) bounding box or None
    """
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(64, 64))

    if len(faces) == 0:
        # Fallback: assume face is centered in the image
        h, w = image.shape[:2]
        size = min(h, w) // 2
        cx, cy = w // 2, h // 2
        return (cx - size // 2, cy - size // 2, cx + size // 2, cy + size // 2)

    # Take the largest face
    x, y, w, h = max(faces, key=lambda f: f[2] * f[3])
    return (x, y, x + w, y + h + bbox_shift)


def crop_face(image: np.ndarray, bbox: Tuple[int, int, int, int],
              target_size: int = 256, extra_margin: int = 10) -> Tuple[np.ndarray, Tuple[int, int, int, int]]:
    """
    Crop face region from image and resize to target_size.

    Args:
        image: BGR numpy array
        bbox: (x1, y1, x2, y2)
        target_size: output size (square)
        extra_margin: extra pixels below chin (V1.5 feature)

    Returns:
        (cropped_resized_image, adjusted_bbox)
    """
    x1, y1, x2, y2 = bbox
    h, w = image.shape[:2]

    # Add extra margin for V1.5
    y2 = min(y2 + extra_margin, h)
    adjusted_bbox = (x1, y1, x2, y2)

    crop = image[y1:y2, x1:x2]
    resized = cv2.resize(crop, (target_size, target_size), interpolation=cv2.INTER_LANCZOS4)
    return resized, adjusted_bbox


def create_lower_face_mask(height: int = 256, width: int = 256) -> np.ndarray:
    """
    Create a mask that zeros out the lower half of the face (mouth region).
    Upper half = 1 (keep), Lower half = 0 (inpaint).
    """
    mask = np.zeros((height, width), dtype=np.float32)
    mask[:height // 2, :] = 1.0
    return mask


def get_crop_box(face_box: Tuple[int, int, int, int],
                 expand: float = 1.5) -> Tuple[Tuple[int, int, int, int], int]:
    """Compute expanded crop box for blending context."""
    x, y, x1, y1 = face_box
    x_c, y_c = (x + x1) // 2, (y + y1) // 2
    w, h = x1 - x, y1 - y
    s = int(max(w, h) // 2 * expand)
    crop_box = (x_c - s, y_c - s, x_c + s, y_c + s)
    return crop_box, s


def stitch_face(
    original_frame: np.ndarray,
    generated_face: np.ndarray,
    bbox: Tuple[int, int, int, int],
    mask_array: Optional[np.ndarray] = None,
    crop_box: Optional[Tuple[int, int, int, int]] = None,
) -> np.ndarray:
    """
    Stitch generated face back into the original frame with feathered blending.

    Uses MuseTalk's blending approach: paste face into expanded crop region,
    then composite onto body using a gaussian-blurred mask.

    Args:
        original_frame: full-resolution BGR frame
        generated_face: 256x256 BGR face from VFG
        bbox: (x1, y1, x2, y2) face bounding box
        mask_array: pre-computed blending mask (optional)
        crop_box: expanded crop region (optional)

    Returns:
        Composited full-resolution BGR frame
    """
    x1, y1, x2, y2 = bbox

    # Resize generated face to match bbox dimensions
    face_w, face_h = x2 - x1, y2 - y1
    if face_w <= 0 or face_h <= 0:
        return original_frame

    resized_face = cv2.resize(generated_face, (face_w, face_h), interpolation=cv2.INTER_LANCZOS4)

    if mask_array is not None and crop_box is not None:
        ih, iw = original_frame.shape[:2]

        # Clamp crop_box to image bounds
        x_s, y_s, x_e, y_e = crop_box
        x_s_c = max(0, x_s)
        y_s_c = max(0, y_s)
        x_e_c = min(iw, x_e)
        y_e_c = min(ih, y_e)

        if x_e_c <= x_s_c or y_e_c <= y_s_c:
            # Fallback: just paste directly
            result = original_frame.copy()
            result[max(0,y1):min(ih,y2), max(0,x1):min(iw,x2)] = resized_face[
                max(0,-y1):face_h-max(0,y2-ih),
                max(0,-x1):face_w-max(0,x2-iw)
            ]
            return result

        body = Image.fromarray(original_frame[:, :, ::-1])  # BGR to RGB
        face_img = Image.fromarray(resized_face[:, :, ::-1])

        # Crop body region (clamped)
        face_large = body.crop((x_s_c, y_s_c, x_e_c, y_e_c))

        # Paste position of face inside the clamped crop
        paste_x1 = x1 - x_s_c
        paste_y1 = y1 - y_s_c
        paste_x2 = paste_x1 + face_w
        paste_y2 = paste_y1 + face_h

        # Resize mask to match clamped crop size
        crop_w = x_e_c - x_s_c
        crop_h = y_e_c - y_s_c
        mask_resized = cv2.resize(mask_array, (crop_w, crop_h), interpolation=cv2.INTER_LINEAR)
        mask_image = Image.fromarray(mask_resized).convert("L")

        # Clamp paste box to crop region
        paste_x1 = max(0, paste_x1)
        paste_y1 = max(0, paste_y1)
        paste_x2 = min(crop_w, paste_x2)
        paste_y2 = min(crop_h, paste_y2)
        face_clipped = face_img.crop((0, 0, paste_x2 - paste_x1, paste_y2 - paste_y1))

        face_large.paste(face_clipped, (paste_x1, paste_y1))
        body.paste(face_large, (x_s_c, y_s_c), mask_image)

        return np.array(body)[:, :, ::-1]  # RGB back to BGR
    else:
        # Simple feathered blending fallback
        result = original_frame.copy()

        # Create a gradient mask for smooth blending
        blend_mask = np.ones((face_h, face_w), dtype=np.float32)
        feather = min(face_w, face_h) // 8
        if feather > 0:
            for i in range(feather):
                alpha = i / feather
                blend_mask[i, :] *= alpha
                blend_mask[-(i + 1), :] *= alpha
                blend_mask[:, i] *= alpha
                blend_mask[:, -(i + 1)] *= alpha

        blend_mask_3ch = blend_mask[:, :, np.newaxis]

        # Clamp bbox to image bounds
        ih, iw = result.shape[:2]
        dx1 = max(0, -x1)
        dy1 = max(0, -y1)
        dx2 = max(0, x2 - iw)
        dy2 = max(0, y2 - ih)
        sx1, sy1 = x1 + dx1, y1 + dy1
        sx2, sy2 = x2 - dx2, y2 - dy2

        roi = result[sy1:sy2, sx1:sx2].astype(np.float32)
        face_roi = resized_face[dy1:face_h - dy2, dx1:face_w - dx2].astype(np.float32)
        mask_roi = blend_mask_3ch[dy1:face_h - dy2, dx1:face_w - dx2]

        blended = roi * (1 - mask_roi) + face_roi * mask_roi
        result[sy1:sy2, sx1:sx2] = blended.astype(np.uint8)

        return result


def prepare_blending_mask(
    frame: np.ndarray,
    bbox: Tuple[int, int, int, int],
    upper_boundary_ratio: float = 0.5,
    expand: float = 1.5,
) -> Tuple[np.ndarray, Tuple[int, int, int, int]]:
    """
    Prepare face-parsing blending mask for seamless stitching.

    Simplified version that doesn't require the BiSeNet face parser.
    Creates a soft mask covering the lower face region.

    Returns:
        (mask_array, crop_box)
    """
    crop_box, s = get_crop_box(bbox, expand)
    x_s, y_s, x_e, y_e = crop_box
    x1, y1, x2, y2 = bbox

    ori_w = x_e - x_s
    ori_h = y_e - y_s
    if ori_w <= 0 or ori_h <= 0:
        return np.zeros((1, 1), dtype=np.uint8), crop_box

    # Create mask for the face region within the expanded crop
    mask = np.zeros((ori_h, ori_w), dtype=np.uint8)

    # Fill the face bbox area
    fx1 = max(0, x1 - x_s)
    fy1 = max(0, y1 - y_s)
    fx2 = min(ori_w, x2 - x_s)
    fy2 = min(ori_h, y2 - y_s)
    mask[fy1:fy2, fx1:fx2] = 255

    # Keep only lower portion (mouth/chin area)
    face_h = fy2 - fy1
    top_boundary = fy1 + int(face_h * upper_boundary_ratio)
    mask[:top_boundary, :] = 0

    # Gaussian blur for smooth edges
    blur_size = int(0.1 * ori_w // 2 * 2) + 1
    mask = cv2.GaussianBlur(mask, (blur_size, blur_size), 0)

    return mask, crop_box
