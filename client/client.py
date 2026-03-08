import asyncio
import json
import logging
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
import uvicorn
import websockets
import os
import sys

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from modules.llm import GeminiHandler

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI()

os.makedirs("client/static", exist_ok=True)
app.mount("/static", StaticFiles(directory="client/static"), name="static")

html_content = """
<!DOCTYPE html>
<html>
<head>
    <title>OLA - AI Assistant</title>
    <style>
        body { margin: 0; padding: 0; background-color: #0d1117; display: flex; flex-direction: column; align-items: center; justify-content: center; height: 100vh; color: white; font-family: 'Inter', sans-serif; }
        #video-container { position: relative; width: 512px; height: 512px; border-radius: 20px; overflow: hidden; box-shadow: 0 10px 30px rgba(0,0,0,0.5); }
        #idle-video { position: absolute; top: 0; left: 0; width: 100%; height: 100%; object-fit: cover; z-index: 1; transition: opacity 0.3s; }
        #live-canvas { position: absolute; top: 0; left: 0; width: 100%; height: 100%; z-index: 2; opacity: 0; transition: opacity 0.3s; }
        .controls { margin-top: 20px; display: flex; gap: 10px; }
        input { padding: 10px; width: 300px; border-radius: 5px; border: none; outline: none; }
        button { padding: 10px 20px; border-radius: 5px; border: none; background-color: #2ea043; color: white; cursor: pointer; font-weight: bold; }
        button:hover { background-color: #3fb950; }
        #status { margin-top: 10px; font-size: 0.9em; color: #8b949e; }
    </style>
</head>
<body>
    <div id="video-container">
        <!-- Placeholder for future 3D VRM model -->
        <h2 style="color: #8b949e; margin-top: 50%;">Avatar Interface (Audio Only)</h2>
    </div>

    <div class="controls">
        <input type="text" id="chat-input" placeholder="Type to talk to OLA..." />
        <button id="send-btn">Send</button>
    </div>
    <div id="status">Status: Disconnected</div>

    <script>
        const ws = new WebSocket(`ws://${location.host}/ws/chat`);
        const statusEl = document.getElementById('status');
        const inputEl = document.getElementById('chat-input');
        const sendBtn = document.getElementById('send-btn');
        
        const idleVideo = null;
        const liveCanvas = null;

        // Audio playback setup
        // We will queue audio chunks and play them seamlessly
        const audioQueue = [];
        let isPlaying = false;
        let audioContext;

        function initAudio() {
            if (!audioContext) {
                audioContext = new (window.AudioContext || window.webkitAudioContext)();
            }
        }

        ws.onopen = () => {
            statusEl.innerText = "Status: Connected";
        };

        let nextPlayTime = 0;

        async function playAudioChunk(blob) {
            const arrayBuffer = await blob.arrayBuffer();
            const view = new Int16Array(arrayBuffer);
            const floatArray = new Float32Array(view.length);
            for (let i = 0; i < view.length; i++) {
                floatArray[i] = view[i] / 32768.0; // Convert 16-bit PCM to float
            }
            const buffer = audioContext.createBuffer(1, floatArray.length, 24000);
            buffer.getChannelData(0).set(floatArray);
            
            const source = audioContext.createBufferSource();
            source.buffer = buffer;
            source.connect(audioContext.destination);
            
            if (nextPlayTime < audioContext.currentTime) {
                nextPlayTime = audioContext.currentTime;
            }
            source.start(nextPlayTime);
            nextPlayTime += buffer.duration;
        }

        ws.onmessage = async (event) => {
            // Received binary audio chunk
            if (event.data instanceof Blob) {
                await playAudioChunk(event.data);
            } else {
                const msg = JSON.parse(event.data);
                if (msg.type === "state") {
                    if (msg.value === "speaking") {
                        statusEl.innerText = "Status: Speaking...";
                    } else if (msg.value === "idle") {
                        statusEl.innerText = "Status: Connected (Idle)";
                    }
                }
            }
        };

        sendBtn.onclick = () => {
            initAudio();
            const text = inputEl.value;
            if (text) {
                ws.send(JSON.stringify({type: 'prompt', text: text}));
                inputEl.value = '';
                statusEl.innerText = "Status: Generating...";
            }
        };
        
        // Handle enter key
        inputEl.addEventListener("keypress", function(event) {
          if (event.key === "Enter") {
            event.preventDefault();
            sendBtn.click();
          }
        });
    </script>
</body>
</html>
"""

@app.get("/")
async def get():
    return HTMLResponse(html_content)

@app.websocket("/ws/chat")
async def websocket_endpoint(websocket: WebSocket):
    await websocket.accept()
    
    gemini = GeminiHandler()
    logger.info("Local UI Client connected")

    try:
        while True:
            data_str = await websocket.receive_text()
            data = json.loads(data_str)
            
            if data['type'] == 'prompt':
                prompt = data['text']
                logger.info(f"Received prompt: {prompt}")
                
                await websocket.send_text(json.dumps({"type": "state", "value": "speaking"}))
                
                async for chunk, is_final in gemini.generate_response_streaming(prompt):
                    if chunk:
                        await websocket.send_bytes(chunk)
                    
                    if is_final:
                        break
                
                await websocket.send_text(json.dumps({"type": "state", "value": "idle"}))

    except WebSocketDisconnect:
        logger.info("Client disconnected")

if __name__ == "__main__":
    import os
    os.makedirs("client/static", exist_ok=True)
    uvicorn.run("client:app", host="0.0.0.0", port=8001, reload=True)
