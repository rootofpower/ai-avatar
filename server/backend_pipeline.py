import numpy as np
import cv2
import torch
import os
import sys

# We will need to clone the Wav2Lip repository on the server into the `server/Wav2Lip` directory.
# This is a wrapper that will interface with the Wav2Lip model.

# Add Wav2Lip to path
sys.path.append(os.path.join(os.path.dirname(__file__), 'Wav2Lip'))

try:
    from models import Wav2Lip
    import audio
except ImportError:
    print("Warning: Wav2Lip modules not found. Ensure you clone the repository and download the weights on the server.")
    WAV2LIP_AVAILABLE = False
else:
    WAV2LIP_AVAILABLE = True


def get_smoothened_boxes(boxes, T):
    for i in range(len(boxes)):
        if i + T > len(boxes):
            window = boxes[len(boxes) - T:]
        else:
            window = boxes[i : i + T]
        boxes[i] = np.mean(window, axis=0)
    return boxes

def _load(checkpoint_path):
    if torch.cuda.is_available():
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")
    checkpoint = torch.load(checkpoint_path, map_location=device)
    return checkpoint

def load_model(path):
    model = Wav2Lip()
    print(f"Load checkpoint from: {path}")
    checkpoint = _load(path)
    s = checkpoint["state_dict"]
    new_s = {}
    for k, v in s.items():
        new_s[k.replace('module.', '')] = v
    model.load_state_dict(new_s)

    model = model.to(torch.device("cuda" if torch.cuda.is_available() else "cpu"))
    return model.eval()


class StreamingWav2LipPipeline:
    def __init__(self, image_path: str, checkpoint_path: str):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"Using device: {self.device}")
        
        self.image_path = image_path
        
        if not WAV2LIP_AVAILABLE:
            print("Wav2Lip not available, running in DUMMY mode.")
            self.model = None
        else:
            if not os.path.exists(checkpoint_path):
                print(f"Checkpoint not found at {checkpoint_path}")
                self.model = None
            else:
                self.model = load_model(checkpoint_path)
                print("Model loaded successfully.")
                
        # Load and preprocess the base frame
        self.base_frame = cv2.imread(image_path)
        if self.base_frame is None:
            print(f"Could not read image: {image_path}, using dummy")
            self.base_frame = np.zeros((512, 512, 3), dtype=np.uint8)
            
        # 1. Detect face and get bounding box (Do this ONCE for the static image)
        # For simplicity in this script, we'll assume the generated image has the face somewhat centered.
        # In a full implementation, you'd use face_alignment or cv2.dnn for robust face detection.
        # Here we provide a fixed crop for the "Gemini_Generated_Image" since it's a static image
        
        # Determine a generic bounding box (adjust for your specific image)
        h, w = self.base_frame.shape[:2]
        # Rough guess for a generic portrait
        self.face_box = [int(w*0.2), int(h*0.2), int(w*0.8), int(h*0.8)] 
        
        # Prepare the face crop for the model (Wav2Lip expects 96x96 rgb for the face window)
        y1, y2, x1, x2 = self.face_box[1], self.face_box[3], self.face_box[0], self.face_box[2]
        self.face_crop = cv2.resize(self.base_frame[y1:y2, x1:x2], (96, 96))
        
        self.img_batch = []
        self.mel_step_size = 16 # Magic number for Wav2Lip (16 mel frames per chunk)
        
        # Buffer for incoming audio stream
        self.audio_buffer = np.array([], dtype=np.float32)

    def process_audio_chunk(self, audio_data: bytes) -> bytes:
        """
        Receives raw audio bytes (PCM 16k or 24k), processes it through the model,
        and returns a JPEG frame.
        """
        if self.model is None:
            # DUMMY MODE
            frame = self.base_frame.copy()
            cv2.putText(frame, "WAV2LIP DUMMY", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
            _, buffer = cv2.imencode('.jpg', frame, [int(cv2.IMWRITE_JPEG_QUALITY), 80])
            return buffer.tobytes()

        # 1. Convert byte array to numpy float32 audio
        # Gemini native audio is usually 24kHz PCM 16-bit
        audio_chunk_np = np.frombuffer(audio_data, dtype=np.int16).astype(np.float32) / 32768.0
        
        # 2. Wav2Lip expects 16kHz audio. We need to resample it dynamically 
        # For a production application running on a server, we'd use scipy.signal.resample or librosa
        import librosa
        if len(audio_chunk_np) > 0:
             # Assuming input is 24k based on your save_audio settings
             audio_chunk_np = librosa.resample(audio_chunk_np, orig_sr=24000, target_sr=16000)

        # Append to buffer
        self.audio_buffer = np.concatenate((self.audio_buffer, audio_chunk_np))
        
        # Process if we have enough audio for one chunk
        # Wav2Lip syncs 1 video frame (25fps) to 16 mel-spectrogram steps (~0.2s of audio)
        # We need a certain amount of audio to generate a frame.
        
        # For streaming, we need to extract the mel spectrogram
        # This implementation requires custom streaming logic over the base Wav2Lip audio.py
        
        # IMPORTANT: Since this is meant to be run on the university server, the final logic 
        # depends on having the Wav2Lip repo pulled there.
        # This is a structural skeleton for the worker.
        
        # DUMMY return for now until server is setup
        frame = self.base_frame.copy()
        cv2.putText(frame, "WAV2LIP SYNCING...", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        _, buffer = cv2.imencode('.jpg', frame, [int(cv2.IMWRITE_JPEG_QUALITY), 80])
        return buffer.tobytes()
