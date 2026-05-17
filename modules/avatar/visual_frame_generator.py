
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
        self._onnx_dir = os.path.join(models_dir, "onnx")

        self._vae = None
        self._unet = None
        self._timesteps = None

        # ONNX Runtime sessions (None = use PyTorch fallback)
        self._ort_unet = None
        self._ort_vae_decoder = None
        self._use_onnx = False

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
        
        # Try to load ONNX accelerated models
        self._try_load_onnx()
        
        backend = "ONNX Runtime" if self._use_onnx else "PyTorch"
        print(f"[VFG] Models loaded on {self._device} (backend: {backend})")

    def _try_load_onnx(self) -> None:
        """Try to load ONNX models for accelerated inference."""
        unet_onnx = os.path.join(self._onnx_dir, "unet.onnx")
        vae_onnx = os.path.join(self._onnx_dir, "vae_decoder.onnx")
        
        if not os.path.exists(unet_onnx) or not os.path.exists(vae_onnx):
            print(f"[VFG] ONNX models not found in {self._onnx_dir}, using PyTorch")
            print(f"[VFG] Run 'python scripts/export_onnx.py' to create ONNX models")
            return
        
        try:
            import onnxruntime as ort
            
            # Build provider list: TensorRT > CUDA > CPU
            providers = []
            available = ort.get_available_providers()
            
            if 'TensorrtExecutionProvider' in available:
                providers.append(('TensorrtExecutionProvider', {
                    'trt_max_workspace_size': 2 * 1024 * 1024 * 1024,
                    'trt_fp16_enable': True,
                    'trt_engine_cache_enable': True,
                    'trt_engine_cache_path': os.path.join(self._onnx_dir, 'trt_cache'),
                }))
            if 'CUDAExecutionProvider' in available:
                providers.append(('CUDAExecutionProvider', {
                    'device_id': 0,
                }))
            providers.append('CPUExecutionProvider')
            
            # Load sessions
            sess_opts = ort.SessionOptions()
            sess_opts.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
            
            self._ort_unet = ort.InferenceSession(unet_onnx, sess_options=sess_opts, providers=providers)
            self._ort_vae_decoder = ort.InferenceSession(vae_onnx, sess_options=sess_opts, providers=providers)
            
            active_unet = self._ort_unet.get_providers()[0]
            active_vae = self._ort_vae_decoder.get_providers()[0]
            print(f"[VFG] ONNX UNet loaded with {active_unet}")
            print(f"[VFG] ONNX VAE decoder loaded with {active_vae}")
            self._use_onnx = True
            
        except ImportError:
            print("[VFG] onnxruntime not installed, using PyTorch")
            print("[VFG] Install with: pip install onnxruntime-gpu")
        except Exception as e:
            print(f"[VFG] Failed to load ONNX models: {e}")
            print("[VFG] Falling back to PyTorch")

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
                # Resize and center crop to maintain aspect ratio
                if frame.shape[0] != 512 or frame.shape[1] != 512:
                    h, w = frame.shape[:2]
                    scale = 512 / min(h, w)
                    new_w = int(w * scale)
                    new_h = int(h * scale)
                    frame = cv2.resize(frame, (new_w, new_h), interpolation=cv2.INTER_LANCZOS4)
                    y_off = (new_h - 512) // 2
                    x_off = (new_w - 512) // 2
                    frame = frame[y_off:y_off + 512, x_off:x_off + 512]
                    
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

        # UNet + VAE inference (single step — NOT diffusion)
        face_bgr = self._run_inference_single(
            frame_data['latent'], audio_prompt
        )

        # Stitch into full frame
        result = stitch_face(
            frame_data['image'].copy(),
            face_bgr,
            frame_data['bbox'],
            mask_array=frame_data['mask_array'],
            crop_box=frame_data['crop_box'],
        )

        return result
    @torch.no_grad()
    def _run_inference_single(self, latent: torch.Tensor, audio_prompt: torch.Tensor) -> np.ndarray:
        """Run UNet + VAE inference for a single frame (ONNX or PyTorch)."""
        if self._use_onnx:
            return self._run_inference_batch_onnx([latent], [audio_prompt])[0]
        
        # PyTorch path
        latent_input = latent.to(dtype=self._unet.dtype)
        pred_latents = self._unet(
            latent_input,
            self._timesteps,
            encoder_hidden_states=audio_prompt,
        ).sample

        pred_latents = pred_latents.to(dtype=self._vae.dtype)
        pred_latents = (1 / self._scaling_factor) * pred_latents
        decoded = self._vae.decode(pred_latents).sample
        
        return self._postprocess_decoded(decoded)[0]

    def _run_inference_batch_onnx(self, latents: list, audio_prompts: list) -> list:
        """Run batched UNet + VAE inference via ONNX Runtime."""
        batch_size = len(latents)
        
        # Stack into batches
        batched_latents = torch.cat(latents, dim=0)
        batched_audio = torch.cat(audio_prompts, dim=0)
        
        # Convert to numpy for ONNX Runtime
        sample_np = batched_latents.cpu().numpy().astype(np.float16)
        timestep_np = np.zeros(batch_size, dtype=np.int64)
        hidden_np = batched_audio.cpu().numpy().astype(np.float16)
        
        # UNet inference via ONNX
        unet_output = self._ort_unet.run(None, {
            'sample': sample_np,
            'timestep': timestep_np,
            'encoder_hidden_states': hidden_np,
        })[0]
        
        # Scale latents for VAE
        pred_latents_np = (unet_output / self._scaling_factor).astype(np.float16)
        
        # VAE decode via ONNX
        decoded_np = self._ort_vae_decoder.run(None, {
            'latent': pred_latents_np,
        })[0]
        
        # Post-process: [-1, 1] -> [0, 255] BGR
        decoded_np = np.clip(decoded_np / 2 + 0.5, 0, 1)
        decoded_np = (decoded_np * 255).round().astype(np.uint8)
        # Shape: [B, 3, 256, 256] -> list of [256, 256, 3] BGR
        results = []
        for i in range(batch_size):
            face_rgb = np.transpose(decoded_np[i], (1, 2, 0))  # CHW -> HWC
            face_bgr = face_rgb[:, :, ::-1].copy()  # RGB -> BGR
            results.append(face_bgr)
        return results

    @staticmethod
    def _postprocess_decoded(decoded: torch.Tensor) -> list:
        """Post-process VAE decoded output to list of BGR face images."""
        images = (decoded / 2 + 0.5).clamp(0, 1)
        images = images.detach().cpu().permute(0, 2, 3, 1).float().numpy()
        results = []
        for i in range(images.shape[0]):
            face_rgb = (images[i] * 255).round().astype(np.uint8)
            face_bgr = face_rgb[:, :, ::-1]  # RGB to BGR
            results.append(face_bgr)
        return results

    @torch.no_grad()
    def generate_frames_batch(self, audio_prompts: list) -> list:
        """
        Generate multiple lip-synced frames in a single batched GPU pass.
        
        Args:
            audio_prompts: list of whisper embedding tensors from AFE.extract()
            
        Returns:
            List of full-resolution composited BGR frames.
        """
        if not self.has_reference or not audio_prompts:
            return []

        batch_size = len(audio_prompts)
        
        # Collect frame data for each item in the batch
        frame_datas = []
        latent_inputs = []
        for _ in range(batch_size):
            if hasattr(self, '_ref_frames') and self._ref_frames:
                fd = self._ref_frames[self._frame_idx]
                self._frame_idx = (self._frame_idx + 1) % len(self._ref_frames)
            else:
                fd = {
                    'image': self._ref_image,
                    'bbox': self._ref_bbox,
                    'latent': self._ref_latent,
                    'mask_array': self._ref_mask_array,
                    'crop_box': self._ref_crop_box
                }
            frame_datas.append(fd)
            latent_inputs.append(fd['latent'])

        # Run batched inference (ONNX or PyTorch)
        if self._use_onnx:
            face_bgrs = self._run_inference_batch_onnx(latent_inputs, audio_prompts)
        else:
            # PyTorch batch path
            batched_latents = torch.cat(latent_inputs, dim=0).to(dtype=self._unet.dtype)
            batched_audio = torch.cat(audio_prompts, dim=0).to(dtype=self._unet.dtype)
            batched_timesteps = self._timesteps.expand(batch_size)

            pred_latents = self._unet(
                batched_latents,
                batched_timesteps,
                encoder_hidden_states=batched_audio,
            ).sample

            pred_latents = pred_latents.to(dtype=self._vae.dtype)
            pred_latents = (1 / self._scaling_factor) * pred_latents
            decoded = self._vae.decode(pred_latents).sample
            face_bgrs = self._postprocess_decoded(decoded)

        # Stitch each face into its full frame
        results = []
        for i in range(batch_size):
            result = stitch_face(
                frame_datas[i]['image'].copy(),
                face_bgrs[i],
                frame_datas[i]['bbox'],
                mask_array=frame_datas[i]['mask_array'],
                crop_box=frame_datas[i]['crop_box'],
            )
            results.append(result)

        return results

    @property
    def is_loaded(self) -> bool:
        return self._vae is not None and self._unet is not None

    @property
    def has_reference(self) -> bool:
        return self._ref_latent is not None
