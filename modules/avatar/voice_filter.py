import numpy as np
import scipy.signal as signal

class PitchShifter:
    """
    Clean, single-tap circular delay-line pitch shifter.
    Operates in the time-domain with smooth crossfading and zero added noise/jitter.
    """
    def __init__(self, sample_rate=24000, pitch_ratio=1.1892, d_min=40.0, d_max=800.0):
        self.sr = sample_rate
        self.R = pitch_ratio
        self.d_min = d_min
        self.d_max = d_max
        self.P = d_max - d_min
        
        self.buf_size = 65536
        self.buffer = np.zeros(self.buf_size, dtype=np.float32)
        self.write_ptr = 0
        self.delay_ptr = d_min + self.P / 2.0  # start in the middle

    def reset(self):
        self.buffer.fill(0)
        self.write_ptr = 0
        self.delay_ptr = self.d_min + self.P / 2.0

    def process(self, x: np.ndarray) -> np.ndarray:
        n_samples = len(x)
        if n_samples == 0:
            return x
            
        # 1. Write current chunk to circular buffer
        write_indices = (self.write_ptr + np.arange(n_samples)) % self.buf_size
        self.buffer[write_indices] = x
        
        # 2. Compute delay pointers
        steps = np.arange(1, n_samples + 1)
        delay_pts = self.delay_ptr + steps * (1.0 - self.R)
        delay_pts = self.d_min + (delay_pts - self.d_min) % self.P
        
        # 3. Two taps out-of-phase by half a period
        delay1 = delay_pts
        delay2 = self.d_min + (delay_pts - self.d_min + self.P / 2.0) % self.P
        
        read_idx1 = (write_indices - delay1) % self.buf_size
        read_idx2 = (write_indices - delay2) % self.buf_size
        
        # Linear interpolation
        idx1_floor = np.floor(read_idx1).astype(np.int32)
        idx1_ceil = (idx1_floor + 1) % self.buf_size
        frac1 = read_idx1 - idx1_floor
        val1 = (1.0 - frac1) * self.buffer[idx1_floor] + frac1 * self.buffer[idx1_ceil]
        
        idx2_floor = np.floor(read_idx2).astype(np.int32)
        idx2_ceil = (idx2_floor + 1) % self.buf_size
        frac2 = read_idx2 - idx2_floor
        val2 = (1.0 - frac2) * self.buffer[idx2_floor] + frac2 * self.buffer[idx2_ceil]
        
        # 4. Crossfade mix
        mid = self.d_min + self.P / 2.0
        w = np.where(delay1 < mid, 
                     (delay1 - self.d_min) / (self.P / 2.0), 
                     (self.d_max - delay1) / (self.P / 2.0))
        
        out = w * val1 + (1.0 - w) * val2
        
        # 5. Update states
        self.write_ptr = (self.write_ptr + n_samples) % self.buf_size
        self.delay_ptr = delay_pts[-1]
        
        return out


class BiquadShelvingFilter:
    """
    Standard audio biquad shelving filter (Low-Shelf or High-Shelf).
    Maintains filter history (zi) for click-free chunk-by-chunk real-time streaming.
    """
    def __init__(self, sample_rate, cutoff, gain_db, filter_type='low'):
        self.sr = sample_rate
        self.cutoff = cutoff
        self.gain_db = gain_db
        self.filter_type = filter_type
        
        # Design filter coefficients
        A = 10.0 ** (gain_db / 40.0)
        omega = 2.0 * np.pi * cutoff / sample_rate
        # Slope S = 1.0, simplifies alpha formula:
        alpha = np.sin(omega) / 2.0 * (A + 1.0) / np.sqrt(A)
        cos_w = np.cos(omega)
        two_sqrt_A_alpha = 2.0 * np.sqrt(A) * alpha
        
        if filter_type == 'low':
            b0 = A * ((A + 1.0) - (A - 1.0) * cos_w + two_sqrt_A_alpha)
            b1 = 2.0 * A * ((A - 1.0) - (A + 1.0) * cos_w)
            b2 = A * ((A + 1.0) - (A - 1.0) * cos_w - two_sqrt_A_alpha)
            a0 = (A + 1.0) + (A - 1.0) * cos_w + two_sqrt_A_alpha
            a1 = -2.0 * ((A - 1.0) + (A + 1.0) * cos_w)
            a2 = (A + 1.0) + (A - 1.0) * cos_w - two_sqrt_A_alpha
        else: # high-shelf
            b0 = A * ((A + 1.0) + (A - 1.0) * cos_w + two_sqrt_A_alpha)
            b1 = -2.0 * A * ((A - 1.0) + (A + 1.0) * cos_w)
            b2 = A * ((A + 1.0) + (A - 1.0) * cos_w - two_sqrt_A_alpha)
            a0 = (A + 1.0) - (A - 1.0) * cos_w + two_sqrt_A_alpha
            a1 = 2.0 * ((A - 1.0) - (A + 1.0) * cos_w)
            a2 = (A + 1.0) - (A - 1.0) * cos_w - two_sqrt_A_alpha
            
        self.b = np.array([b0, b1, b2]) / a0
        self.a = np.array([a0, a1, a2]) / a0
        self.zi = signal.lfilter_zi(self.b, self.a)
        self.initial_zi = self.zi.copy()

    def reset(self):
        self.zi = self.initial_zi.copy()

    def process(self, x: np.ndarray) -> np.ndarray:
        if len(x) == 0:
            return x
        y, self.zi = signal.lfilter(self.b, self.a, x, zi=self.zi)
        return y


class VoiceFilter:
    """
    Unified minimal voice filter wrapper.
    Processes incoming 16-bit PCM bytes in-place.
    Supports backwards-compatible signatures.
    """
    def __init__(self, sample_rate=24000, enabled=True, pitch_ratio=1.1892, hpf_cutoff=None, eq_gain=None):
        self.enabled = enabled
        self.sample_rate = sample_rate
        
        if self.enabled:
            # PitchShift at semitones=3.0 corresponds exactly to pitch_ratio = 2**(3/12) ≈ 1.1892
            # pitch_ratio passed from .env can override this default
            self.shifter = PitchShifter(sample_rate=sample_rate, pitch_ratio=pitch_ratio)
            
            # Conservative shelf filters (gain within ±2dB)
            # Low-shelf at 200Hz to gently attenuate deep low-end (-2.0dB)
            self.low_shelf = BiquadShelvingFilter(sample_rate, cutoff=200.0, gain_db=-2.0, filter_type='low')
            # High-shelf at 3000Hz to gently boost high clarity (+2.0dB)
            self.high_shelf = BiquadShelvingFilter(sample_rate, cutoff=3000.0, gain_db=2.0, filter_type='high')
            
            print(f"[VoiceFilter] Minimal voice filter initialized (pitch_ratio={pitch_ratio})")
        else:
            print("[VoiceFilter] Voice filter disabled.")

    def reset(self):
        if self.enabled:
            self.shifter.reset()
            self.low_shelf.reset()
            self.high_shelf.reset()

    def process(self, pcm_bytes: bytes, reset=False) -> bytes:
        if not self.enabled or not pcm_bytes:
            return pcm_bytes
            
        if reset:
            self.reset()
            
        # Convert bytes to float32 in range [-1.0, 1.0]
        x = np.frombuffer(pcm_bytes, dtype=np.int16).astype(np.float32) / 32768.0
        
        # Input clipping check: reduce peak to 0.85 if it exceeds 0.95
        peak = np.max(np.abs(x))
        if peak > 0.95:
            x = x * (0.85 / peak)
            
        # Apply clean pitch shift
        x_shifted = self.shifter.process(x)
        
        # Apply low-shelf and high-shelf EQ
        x_low = self.low_shelf.process(x_shifted)
        x_filtered = self.high_shelf.process(x_low)
        
        # Convert back to 16-bit PCM bytes
        y_pcm = (x_filtered * 32767.0).clip(-32768, 32767).astype(np.int16)
        return y_pcm.tobytes()
