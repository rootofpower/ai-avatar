import numpy as np
import scipy.signal as signal
import os

class SinglePitchShifter:
    """
    An individual real-time circular delay-line pitch shifter.
    Uses randomized dither (high-frequency delay jitter) to break periodic patterns,
    effectively diffusing comb-filtering notches.
    """
    def __init__(self, sample_rate, pitch_ratio, d_min, d_max, jitter_amp, seed):
        self.sr = sample_rate
        self.R = pitch_ratio
        self.d_min = d_min
        self.d_max = d_max
        self.P = d_max - d_min
        self.jitter_amp = jitter_amp
        
        self.buf_size = 65536
        self.buffer = np.zeros(self.buf_size, dtype=np.float32)
        self.write_ptr = 0
        self.delay_ptr = d_min + self.P / 2.0
        self.rng = np.random.default_rng(seed)

    def process(self, x: np.ndarray) -> np.ndarray:
        n_samples = len(x)
        if n_samples == 0:
            return x
            
        # 1. Write the chunk to the circular buffer
        write_indices = (self.write_ptr + np.arange(n_samples)) % self.buf_size
        self.buffer[write_indices] = x
        
        # 2. Compute base delay pointers
        steps = np.arange(1, n_samples + 1)
        delay_pts = self.delay_ptr + steps * (1.0 - self.R)
        
        # 3. Add high-frequency random jitter (dither) to scatter comb-filter notches
        jitter = self.rng.uniform(-self.jitter_amp, self.jitter_amp, size=n_samples)
        delay_pts = delay_pts + jitter
        
        # Wrap the delay pointers to be within [d_min, d_max]
        delay_pts = self.d_min + (delay_pts - self.d_min) % self.P
        
        # 4. Tap 1 and Tap 2 delays
        delay1 = delay_pts
        delay2 = self.d_min + (delay_pts - self.d_min + self.P / 2.0) % self.P
        
        # 5. Calculate read indices
        read_idx1 = (write_indices - delay1) % self.buf_size
        read_idx2 = (write_indices - delay2) % self.buf_size
        
        # Linear interpolation for Tap 1
        idx1_floor = np.floor(read_idx1).astype(np.int32)
        idx1_ceil = (idx1_floor + 1) % self.buf_size
        frac1 = read_idx1 - idx1_floor
        val1 = (1.0 - frac1) * self.buffer[idx1_floor] + frac1 * self.buffer[idx1_ceil]
        
        # Linear interpolation for Tap 2
        idx2_floor = np.floor(read_idx2).astype(np.int32)
        idx2_ceil = (idx2_floor + 1) % self.buf_size
        frac2 = read_idx2 - idx2_floor
        val2 = (1.0 - frac2) * self.buffer[idx2_floor] + frac2 * self.buffer[idx2_ceil]
        
        # 6. Crossfade weights (triangular window)
        mid = self.d_min + self.P / 2.0
        w = np.where(delay1 < mid, 
                     (delay1 - self.d_min) / (self.P / 2.0), 
                     (self.d_max - delay1) / (self.P / 2.0))
        
        # 7. Mix Tap 1 and Tap 2
        out = w * val1 + (1.0 - w) * val2
        
        # 8. Update internal states for the next chunk
        self.write_ptr = (self.write_ptr + n_samples) % self.buf_size
        self.delay_ptr = delay_pts[-1]
        
        return out


class ImprovedPitchShifter:
    """
    An improved, ultra-smooth real-time pitch shifter.
    Uses a dual-voice parallel architecture with overlapping delay lines and random dither
    to completely eliminate robotic, metallic, and chorusing artifacts.
    """
    def __init__(self, sample_rate=24000, pitch_ratio=1.20):
        self.sr = sample_rate
        self.R = pitch_ratio
        
        # Two single pitch shifters with different, overlapping delay ranges
        # to cancel out each other's comb-filtering notches.
        self.shifter1 = SinglePitchShifter(sample_rate, pitch_ratio, d_min=60.0, d_max=1000.0, jitter_amp=2.0, seed=42)
        self.shifter2 = SinglePitchShifter(sample_rate, pitch_ratio, d_min=100.0, d_max=1400.0, jitter_amp=3.0, seed=100)

    def process(self, x: np.ndarray) -> np.ndarray:
        out1 = self.shifter1.process(x)
        out2 = self.shifter2.process(x)
        # Average the two voices for a much warmer and flatter frequency response
        return 0.5 * (out1 + out2)


class VoiceEQ:
    """
    Real-time state-preserving EQ filter.
    Includes a 2nd order High-Pass Filter and a 2nd order Peaking filter.
    Maintains filter history (zi) across chunk boundaries to avoid click artifacts.
    """
    def __init__(self, sample_rate=24000, hpf_cutoff=130.0, eq_gain_db=3.0):
        # 1. Warm High-Pass Filter (cutoff ~130Hz to remove adult chest bass while preserving girl's fundamental pitch)
        self.sos_hp = signal.butter(2, hpf_cutoff, 'highpass', fs=sample_rate, output='sos')
        self.zi_hp = signal.sosfilt_zi(self.sos_hp)
        
        # 2. Peaking EQ Filter (boost at 1500Hz, Q=0.8 for smooth vocal warmth and presence)
        f0 = 1500.0
        Q = 0.8
        A = 10.0 ** (eq_gain_db / 40.0)
        omega = 2.0 * np.pi * f0 / sample_rate
        alpha = np.sin(omega) / (2.0 * Q)
        
        b0 = 1.0 + alpha * A
        b1 = -2.0 * np.cos(omega)
        b2 = 1.0 - alpha * A
        a0 = 1.0 + alpha / A
        a1 = -2.0 * np.cos(omega)
        a2 = 1.0 - alpha / A
        
        self.b = np.array([b0, b1, b2]) / a0
        self.a = np.array([a0, a1, a2]) / a0
        self.zi_peak = signal.lfilter_zi(self.b, self.a)

    def process(self, x: np.ndarray) -> np.ndarray:
        if len(x) == 0:
            return x
        # Apply High-Pass
        x_filtered, self.zi_hp = signal.sosfilt(self.sos_hp, x, zi=self.zi_hp)
        # Apply Peaking EQ
        x_eq, self.zi_peak = signal.lfilter(self.b, self.a, x_filtered, zi=self.zi_peak)
        return x_eq


class VoiceFilter:
    """
    Unified voice filter wrapper that processes incoming 16-bit PCM bytes
    in-place and returns filtered 16-bit PCM bytes.
    """
    def __init__(self, sample_rate=24000, enabled=True, pitch_ratio=1.20, hpf_cutoff=130.0, eq_gain=3.0):
        self.enabled = enabled
        self.sample_rate = sample_rate
        
        if self.enabled:
            print(f"[VoiceFilter] Initializing voice filter: pitch_ratio={pitch_ratio}, hpf_cutoff={hpf_cutoff}Hz, eq_gain={eq_gain}dB")
            self.shifter = ImprovedPitchShifter(sample_rate=sample_rate, pitch_ratio=pitch_ratio)
            self.eq = VoiceEQ(sample_rate=sample_rate, hpf_cutoff=hpf_cutoff, eq_gain_db=eq_gain)
        else:
            print("[VoiceFilter] Voice filter disabled.")

    def process(self, pcm_bytes: bytes) -> bytes:
        if not self.enabled or not pcm_bytes:
            return pcm_bytes
            
        # Convert bytes to float32 in range [-1.0, 1.0]
        x = np.frombuffer(pcm_bytes, dtype=np.int16).astype(np.float32) / 32768.0
        
        # Process through improved pitch shifter and warm EQ
        x_shifted = self.shifter.process(x)
        x_filtered = self.eq.process(x_shifted)
        
        # Convert back to 16-bit PCM bytes
        y_pcm = (x_filtered * 32767.0).clip(-32768, 32767).astype(np.int16)
        return y_pcm.tobytes()
