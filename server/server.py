import asyncio
import base64
import json
import numpy as np
import cv2
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
import uvicorn
import logging

from backend_pipeline import StreamingWav2LipPipeline

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI()

# Configuration
CHECKPOINT_PATH = "Wav2Lip/checkpoints/wav2lip_gan.pth"
IMAGE_PATH = "../Gemini_Generated_Image_gx6wbwgx6wbwgx6w.png" 
# Change this path to match where the server mounts the files

lipsync_pipeline = None

@app.on_event("startup")
async def startup_event():
    global lipsync_pipeline
    try:
        lipsync_pipeline = StreamingWav2LipPipeline(IMAGE_PATH, CHECKPOINT_PATH)
    except Exception as e:
        logger.error(f"Failed to load pipeline: {e}")

@app.websocket("/ws/lipsync")
async def websocket_endpoint(websocket: WebSocket):
    await websocket.accept()
    logger.info("Client connected for lip-sync stream")
    
    try:
        while True:
            # Receive audio chunk from client (expecting binary PCM data)
            data = await websocket.receive_bytes()
            
            if lipsync_pipeline:
                # Process audio and get new frame
                jpg_bytes = lipsync_pipeline.process_audio_chunk(data)
                
                # Send frame back to client
                await websocket.send_bytes(jpg_bytes)
            else:
                await asyncio.sleep(0.01)
                
    except WebSocketDisconnect:
        logger.info("Client disconnected")
    except Exception as e:
        logger.error(f"WebSocket error: {e}")

if __name__ == "__main__":
    uvicorn.run("server:app", host="0.0.0.0", port=8000, reload=True)
