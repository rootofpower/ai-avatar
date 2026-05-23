import numpy as np
import scipy.signal as signal
import soundfile as sf
import time
import os
from dotenv import load_dotenv

# Load env
load_dotenv()

from modules.avatar.voice_filter import VoiceFilter

def test_voice_filter_integration():
    # 1. Load the original wave file
    data, sr = sf.read('response_audio.wav')
    print(f"Loaded response_audio.wav: shape={data.shape}, sr={sr}")
    
    # Convert float32 array [-1.0, 1.0] back to 16-bit PCM bytes to simulate Gemini input
    pcm_bytes_int16 = (data * 32767.0).clip(-32768, 32767).astype(np.int16)
    pcm_bytes = pcm_bytes_int16.tobytes()
    print(f"PCM bytes length: {len(pcm_bytes)} bytes (~{len(pcm_bytes)/2/sr:.2f} seconds)")

    # 2. Initialize VoiceFilter using the configurations loaded from .env
    voice_filter = VoiceFilter(
        sample_rate=sr,
        enabled=os.getenv("AVATAR_VOICE_FILTER_ENABLED", "true").lower() == "true",
        pitch_ratio=float(os.getenv("AVATAR_VOICE_FILTER_PITCH", "1.20")),
        hpf_cutoff=float(os.getenv("AVATAR_VOICE_FILTER_HPF", "220.0")),
        eq_gain=float(os.getenv("AVATAR_VOICE_FILTER_EQ_GAIN", "4.0")),
    )

    # Chunk size of 20ms at 24kHz = 480 samples = 960 bytes
    chunk_size_bytes = int(sr * 0.02) * 2 # 960 bytes
    num_chunks = len(pcm_bytes) // chunk_size_bytes
    
    out_chunks = []
    
    # 3. Process chunk by chunk
    t_start = time.time()
    for i in range(num_chunks):
        chunk = pcm_bytes[i*chunk_size_bytes : (i+1)*chunk_size_bytes]
        
        t0 = time.time()
        filtered_chunk = voice_filter.process(chunk)
        t_elapsed = (time.time() - t0) * 1000
        
        out_chunks.append(filtered_chunk)
        if i < 5:
            print(f"Chunk {i} (960 bytes) processed in {t_elapsed:.4f}ms")
            
    total_time = (time.time() - t_start) * 1000
    print(f"\nVerification Results:")
    print(f"Processed {num_chunks} chunks ({num_chunks*20}ms of audio) in {total_time:.2f}ms")
    print(f"Average time per 20ms chunk: {total_time/num_chunks:.4f}ms")
    
    # Reassemble and save to test_out.wav
    all_filtered_bytes = b"".join(out_chunks)
    all_filtered_int16 = np.frombuffer(all_filtered_bytes, dtype=np.int16)
    all_filtered_float32 = all_filtered_int16.astype(np.float32) / 32768.0
    
    sf.write('test_out.wav', all_filtered_float32, sr)
    print("Saved reassembled output to test_out.wav")

if __name__ == "__main__":
    test_voice_filter_integration()
