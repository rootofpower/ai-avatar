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

from fastapi.responses import FileResponse

@app.get("/")
async def get():
    return FileResponse("client/static/index.html")

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
                
                await websocket.send_text(json.dumps({"type": "state", "value": "thinking"}))
                
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
