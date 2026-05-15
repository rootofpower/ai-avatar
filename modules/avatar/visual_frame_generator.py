
import os
import cv2
import json
import numpy as np
import torch
from typing import Optional, Tuple

from modules.avatar.face_utils import (
    detect_face_bbox, crop_face, stitch_face, prepare_blending_mask,
)


class VisualFrameGenerator:
    """
    Core GPU module: takes reference image + audio embeddings → generates
    lip-synced 256×256 face frames using MuseTalk V1.5 inference.

    Pipeline per frame:
    1. VAE encode masked face → latent
    2. UNet inpaint with audio cross-attention (single step, NOT diffusion)
    3. VAE decode → 256×256 face
    4. Stitch back into full frame
    """

    def __init__(
        self,
        models_dir: str = "./models",
        device: str = "cuda",
        unet_config: str = None,
        unet_model_path: str = None,
        vae_type: str = "sd-vae",
    ):
        self._models_dir = models_dir
        self._device = torch.device(device if torch.cuda.is_available() else "cpu")
        self._unet_config = unet_config or os.path.join(models_dir, "musetalkV15", "musetalk.json")
        self._unet_model_path = unet_model_path or os.path.join(models_dir, "musetalkV15", "unet.pth")
        self._vae_path = os.path.join(models_dir, vae_type)

        self._vae = None
        self._unet = None
        self._timesteps = None

        # Pre-computed reference data
        self._ref_image: Optional[np.ndarray] = None
        self._ref_bbox: Optional[Tuple[int, int, int, int]] = None
        self._ref_latent: Optional[torch.Tensor] = None
        self._ref_mask_array: Optional[np.ndarray] = None
        self._ref_crop_box: Optional[Tuple[int, int, int, int]] = None

        # Sequence data for dynamic driving video
        self._ref_frames = []
        self._frame_idx = 0

    def load(self) -> None:
        """Load VAE and UNet models. Call once at startup."""
        print("[VFG] Loading models...")
        self._load_vae()
        self._load_unet()
        self._timesteps = torch.tensor([0], device=self._device)
        print(f"[VFG] Models loaded on {self._device}")

    def _load_vae(self) -> None:
        """Load the Stable Diffusion VAE (sd-vae-ft-mse)."""
        from diffusers import AutoencoderKL

        self._vae = AutoencoderKL.from_pretrained(self._vae_path)
        self._vae = self._vae.half().to(self._device).eval()
        self._vae.requires_grad_(False)
        self._scaling_factor = self._vae.config.scaling_factor
        print(f"[VFG] VAE loaded (scaling_factor={self._scaling_factor})")

    def _load_unet(self) -> None:
        """Load the MuseTalk UNet."""
        from diffusers import UNet2DConditionModel

        with open(self._unet_config, 'r') as f:
            config = json.load(f)

        self._unet = UNet2DConditionModel(**config)
        weights = torch.load(self._unet_model_path, map_location=self._device)
        self._unet.load_state_dict(weights)
        self._unet = self._unet.half().to(self._device).eval()
        self._unet.requires_grad_(False)
        print(f"[VFG] UNet loaded from {self._unet_model_path}")

    def set_reference_image(self, image_path: str, bbox_shift: int = 0, extra_margin: int = 10) -> None:
        """
        Pre-process a reference image: detect face, crop, compute latent,
        prepare blending mask. Call once per avatar.

        Args:
            image_path: path to avatar reference image
            bbox_shift: vertical bbox adjustment
            extra_margin: extra margin below chin (V1.5)
        """
        image = cv2.imread(image_path)
        if image is None:
            raise FileNotFoundError(f"Cannot load image: {image_path}")

        self._ref_image = image
        self._ref_bbox = detect_face_bbox(image, bbox_shift)

        # Crop and get reference latent
        cropped, adjusted_bbox = crop_face(image, self._ref_bbox, extra_margin=extra_margin)
        self._ref_bbox = adjusted_bbox
        self._ref_latent = self._encode_face(cropped)

        # Prepare blending mask
        self._ref_mask_array, self._ref_crop_box = prepare_blending_mask(
            image, self._ref_bbox
        )

        self._ref_frames = [] # clear any driving video sequence
        print(f"[VFG] Reference image set: {image_path}, bbox={self._ref_bbox}")

    def set_driving_video(self, video_path: str, bbox_shift: int = 0, extra_margin: int = 10, max_frames: int = 100) -> None:
        """
        Pre-process a driving video for dynamic head movements.
        Extracts up to max_frames and precomputes latents for each to avoid inference freezing.
        """
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            raise FileNotFoundError(f"Cannot open driving video: {video_path}")

        self._ref_frames = []
        self._frame_idx = 0

        print(f"[VFG] Pre-processing driving video for generation: {video_path}")
        while len(self._ref_frames) < max_frames:
            ret, frame = cap.read()
            if not ret:
                break
            
            try:
                # Need to detect bbox per frame to follow head movements
                bbox = detect_face_bbox(frame, bbox_shift)
                cropped, adjusted_bbox = crop_face(frame, bbox, extra_margin=extra_margin)
                latent = self._encode_face(cropped)
                mask_array, crop_box = prepare_blending_mask(frame, adjusted_bbox)
                
                self._ref_frames.append({
                    'image': frame,
                    'bbox': adjusted_bbox,
                    'latent': latent,
                    'mask_array': mask_array,
                    'crop_box': crop_box
                })
            except Exception as e:
                print(f"[VFG] Skipping frame due to error: {e}")
                continue
                
        cap.release()
        
        if not self._ref_frames:
            raise ValueError(f"No valid frames could be read from {video_path}")
            
        print(f"[VFG] Driving video loaded: {len(self._ref_frames)} frames pre-processed.")
        
        # Set the first frame as the fallback reference
        self._ref_image = self._ref_frames[0]['image']
        self._ref_latent = self._ref_frames[0]['latent']
        self._ref_bbox = self._ref_frames[0]['bbox']
        self._ref_mask_array = self._ref_frames[0]['mask_array']
        self._ref_crop_box = self._ref_frames[0]['crop_box']

    def _preprocess_face(self, face_bgr: np.ndarray, half_mask: bool = False) -> torch.Tensor:
        """
        Preprocess a face crop for VAE encoding.
        Matches MuseTalk VAE.preprocess_img().
        """
        face_rgb = cv2.cvtColor(face_bgr, cv2.COLOR_BGR2RGB)
        x = np.asarray(face_rgb, dtype=np.float32) / 255.0
        x = np.transpose(x, (2, 0, 1))  # HWC -> CHW
        x = torch.FloatTensor(x)

        if half_mask:
            h = x.shape[1]
            x[:, h // 2:, :] = 0  # Zero out lower half

        # Normalize to [-1, 1]
        x = (x - 0.5) / 0.5
        x = x.unsqueeze(0).to(device=self._device, dtype=torch.float16)
        return x

    @torch.no_grad()
    def _encode_face(self, face_bgr: np.ndarray) -> torch.Tensor:
        """
        Encode a 256×256 face crop into the UNet input latent.
        Returns concatenated [masked_latent, ref_latent] matching MuseTalk's format.
        """
        # Masked version (lower face zeroed)
        masked_input = self._preprocess_face(face_bgr, half_mask=True)
        masked_latent = self._vae.encode(masked_input).latent_dist.mode()
        masked_latent = masked_latent * self._scaling_factor

        # Full reference
        ref_input = self._preprocess_face(face_bgr, half_mask=False)
        ref_latent = self._vae.encode(ref_input).latent_dist.mode()
        ref_latent = ref_latent * self._scaling_factor

        # Concatenate along channel dim: [1, 8, 32, 32]
        return torch.cat([masked_latent, ref_latent], dim=1)

    @torch.no_grad()
    def generate_frame(self, audio_prompt: torch.Tensor) -> Optional[np.ndarray]:
        """
        Generate a single lip-synced frame.

        Args:
            audio_prompt: whisper embedding from AudioFeatureExtractor.extract()

        Returns:
            Full-resolution composited BGR frame, or None if not ready.
        """
        if not self.has_reference:
            return None

        # Determine which reference data to use (static image or next video frame)
        if hasattr(self, '_ref_frames') and self._ref_frames:
            frame_data = self._ref_frames[self._frame_idx]
            self._frame_idx = (self._frame_idx + 1) % len(self._ref_frames)
        else:
            frame_data = {
                'image': self._ref_image,
                'bbox': self._ref_bbox,
                'latent': self._ref_latent,
                'mask_array': self._ref_mask_array,
                'crop_box': self._ref_crop_box
            }

        # UNet inference (single step — NOT diffusion)
        latent_input = frame_data['latent'].to(dtype=self._unet.dtype)
        pred_latents = self._unet(
            latent_input,
            self._timesteps,
            encoder_hidden_states=audio_prompt,
        ).sample

        # VAE decode
        pred_latents = pred_latents.to(dtype=self._vae.dtype)
        pred_latents = (1 / self._scaling_factor) * pred_latents
        decoded = self._vae.decode(pred_latents).sample

        # Post-process: [-1, 1] -> [0, 255] BGR
        image = (decoded / 2 + 0.5).clamp(0, 1)
        image = image.detach().cpu().permute(0, 2, 3, 1).float().numpy()
        image = (image[0] * 255).round().astype(np.uint8)
        face_bgr = image[:, :, ::-1]  # RGB to BGR

        # Stitch into full frame
        result = stitch_face(
            frame_data['image'].copy(),
            face_bgr,
            frame_data['bbox'],
            mask_array=frame_data['mask_array'],
            crop_box=frame_data['crop_box'],
        )

        return result

    @property
    def is_loaded(self) -> bool:
        return self._vae is not None and self._unet is not None

    @property
    def has_reference(self) -> bool:
        return self._ref_latent is not None
