
import struct
import numpy as np
import torch
from typing import Optional


class AudioFeatureExtractor:
    """
    Converts raw PCM audio chunks into articulation embeddings for MuseTalk.
    Uses Whisper-Tiny encoder (MuseTalk's native audio encoder).

    Input: 200ms PCM chunks (16kHz, 16-bit, mono) with PTS.
    Output: whisper embedding tensor for MuseTalk UNet cross-attention.
    """

    SAMPLE_RATE = 16000
    SAMPLE_WIDTH = 2

    def __init__(
        self,
        whisper_dir: str = "./models/whisper",
        device: str = "cuda",
        audio_padding_length_left: int = 2,
        audio_padding_length_right: int = 2,
        fps: int = 25,
    ):
        self._device = torch.device(device if torch.cuda.is_available() else "cpu")
        self._fps = fps
        self._audio_padding_left = audio_padding_length_left
        self._audio_padding_right = audio_padding_length_right
        self._audio_feature_length = 2 * (audio_padding_length_left + audio_padding_length_right + 1)

        self._whisper = None
        self._feature_extractor = None
        self._pe = None
        self._weight_dtype = torch.float16
        self._whisper_dir = whisper_dir

    def load(self) -> None:
        """Load Whisper model and feature extractor. Call once at startup."""
        from transformers import WhisperModel, AutoFeatureExtractor

        print(f"[AFE] Loading Whisper-Tiny from {self._whisper_dir}...")
        self._feature_extractor = AutoFeatureExtractor.from_pretrained(self._whisper_dir)
        self._whisper = WhisperModel.from_pretrained(self._whisper_dir)
        self._whisper = self._whisper.to(device=self._device, dtype=self._weight_dtype).eval()
        self._whisper.requires_grad_(False)

        # Positional encoding (same as MuseTalk's)
        self._pe = self._build_positional_encoding(d_model=384)
        self._pe = self._pe.to(device=self._device, dtype=self._weight_dtype)
        print(f"[AFE] Whisper-Tiny loaded on {self._device}")

    def _build_positional_encoding(self, d_model: int = 384, max_len: int = 5000) -> torch.nn.Module:
        """Build sinusoidal positional encoding matching MuseTalk's PositionalEncoding."""
        import math

        class PE(torch.nn.Module):
            def __init__(self):
                super().__init__()
                pe = torch.zeros(max_len, d_model)
                position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
                div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
                pe[:, 0::2] = torch.sin(position * div_term)
                pe[:, 1::2] = torch.cos(position * div_term)
                pe = pe.unsqueeze(0)
                self.register_buffer('pe', pe)

            def forward(self, x):
                b, seq_len, d_model = x.size()
                return x + self.pe[:, :seq_len, :].to(x.device)

        return PE()

    def pcm_to_mel_features(self, pcm_bytes: bytes) -> torch.Tensor:
        """
        Convert raw PCM bytes to Whisper mel-spectrogram input features.
        """
        # Decode PCM bytes to float32 numpy array
        sample_count = len(pcm_bytes) // self.SAMPLE_WIDTH
        samples = struct.unpack(f'<{sample_count}h', pcm_bytes)
        audio_np = np.array(samples, dtype=np.float32) / 32768.0

        # Use Whisper's feature extractor (computes log-Mel spectrogram)
        features = self._feature_extractor(
            audio_np,
            return_tensors="pt",
            sampling_rate=self.SAMPLE_RATE,
        ).input_features
        return features.to(dtype=self._weight_dtype)

    @torch.no_grad()
    def extract(self, pcm_bytes: bytes) -> torch.Tensor:
        """
        Extract audio features from a PCM chunk for MuseTalk.

        Args:
            pcm_bytes: 16kHz 16-bit mono PCM

        Returns:
            Audio prompt tensor ready for MuseTalk UNet cross-attention.
            Shape: [1, audio_feature_length * hidden_layers, 384]
        """
        # Get mel features
        mel_features = self.pcm_to_mel_features(pcm_bytes)
        mel_features = mel_features.to(self._device)

        # Run through Whisper encoder, extract hidden states
        encoder_output = self._whisper.encoder(mel_features, output_hidden_states=True)
        hidden_states = encoder_output.hidden_states
        audio_feats = torch.stack(hidden_states, dim=2)  # [B, T, layers, dim]

        # Calculate the frame index for the audio
        audio_fps = 50  # Whisper's internal frame rate
        whisper_idx_multiplier = audio_fps / self._fps

        # Add padding (matching MuseTalk's audio_processor.py logic)
        padding_nums = int(np.ceil(whisper_idx_multiplier))
        audio_feats = torch.cat([
            torch.zeros_like(audio_feats[:, :padding_nums * self._audio_padding_left]),
            audio_feats,
            torch.zeros_like(audio_feats[:, :padding_nums * 3 * self._audio_padding_right]),
        ], dim=1)

        # Extract clip for a single frame
        audio_index = 0
        audio_clip = audio_feats[:, audio_index:audio_index + self._audio_feature_length]

        # Reshape: from [B, time, layers, dim] to [B, time*layers, dim]
        from einops import rearrange
        audio_clip = rearrange(audio_clip, 'b c h w -> b (c h) w')

        # Apply positional encoding
        audio_prompt = self._pe(audio_clip)

        return audio_prompt

    @torch.no_grad()
    def extract_batch(self, pcm_list: list) -> list:
        """
        Extract audio features from multiple PCM chunks in a single batched pass.
        
        Instead of N sequential Whisper encoder calls, computes mel features
        for each chunk (CPU), stacks them, and runs ONE encoder forward pass (GPU).
        
        Args:
            pcm_list: list of PCM byte chunks (16kHz 16-bit mono)
            
        Returns:
            List of audio prompt tensors, each [1, audio_feature_length * hidden_layers, 384]
        """
        if not pcm_list:
            return []
        
        from einops import rearrange
        
        # Step 1: Compute mel features for each chunk (CPU-bound, fast)
        mel_batch = []
        for pcm_bytes in pcm_list:
            mel = self.pcm_to_mel_features(pcm_bytes)
            mel_batch.append(mel)
        
        # Stack into batch: [N, 80, 3000]
        batched_mel = torch.cat(mel_batch, dim=0).to(self._device)
        
        # Step 2: Single batched Whisper encoder call (GPU)
        encoder_output = self._whisper.encoder(batched_mel, output_hidden_states=True)
        hidden_states = encoder_output.hidden_states
        audio_feats = torch.stack(hidden_states, dim=2)  # [N, T, layers, dim]
        
        # Step 3: Post-process each item in the batch
        audio_fps = 50
        whisper_idx_multiplier = audio_fps / self._fps
        padding_nums = int(np.ceil(whisper_idx_multiplier))
        
        audio_feats = torch.cat([
            torch.zeros_like(audio_feats[:, :padding_nums * self._audio_padding_left]),
            audio_feats,
            torch.zeros_like(audio_feats[:, :padding_nums * 3 * self._audio_padding_right]),
        ], dim=1)
        
        audio_index = 0
        audio_clip = audio_feats[:, audio_index:audio_index + self._audio_feature_length]
        audio_clip = rearrange(audio_clip, 'b c h w -> b (c h) w')
        audio_prompts_batched = self._pe(audio_clip)
        
        # Split back into individual tensors
        return [audio_prompts_batched[i:i+1] for i in range(len(pcm_list))]

    @property
    def is_loaded(self) -> bool:
        return self._whisper is not None
