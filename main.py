
import asyncio
import logging
import os
import threading
from typing import Optional
from modules.mqtt_handler import MQTTHandler
from modules.tuna_chat import TunaChatAPI
from modules.llm import GeminiHandler
from dotenv import load_dotenv


load_dotenv()

# Suppress noisy third-party debug logs
for _logger_name in [
    "aiortc", "aioice", "websockets", "av",
    "aiortc.rtcrtpsender", "aiortc.rtcrtpreceiver",
    "aiortc.rtcpeerconnection", "aioice.ice",
    "websockets.client",
]:
    logging.getLogger(_logger_name).setLevel(logging.WARNING)

AVATAR_ENABLED = os.getenv("AVATAR_ENABLED", "false").lower() == "true"
AVATAR_IMAGE = os.getenv("AVATAR_IMAGE", "")
AVATAR_IDLE_VIDEO = os.getenv("AVATAR_IDLE_VIDEO", "")
WEBRTC_PORT = int(os.getenv("WEBRTC_PORT", "8080"))

print(f"[Config] AVATAR_ENABLED={AVATAR_ENABLED}, AVATAR_IMAGE='{AVATAR_IMAGE}', AVATAR_IDLE_VIDEO='{AVATAR_IDLE_VIDEO}'")


class VoiceAssistant:
    """
    Voice Assistant Prototype.
    
    Integrates TUNA Chat API for knowledge retrieval and Gemini Live API for voice synthesis.
    When AVATAR_ENABLED=true, also runs the avatar generation pipeline with WebRTC delivery.
    """
    
    def __init__(self):
        print("Initializing Voice Assistant...")
        
        # Initialize TUNA Chat API
        try:
            self.tuna_chat = TunaChatAPI()
            print(f"[OK] TUNA Chat API Initialized ({self.tuna_chat.chat_endpoint})")
        except Exception as e:
            print(f"[ERROR] TUNA Chat API Init Failed: {e}")
            self.tuna_chat = None

        # Initialize LLM
        try:
            self.llm = GeminiHandler()
            print("[OK] Gemini Handler Initialized")
        except Exception as e:
            print(f"[ERROR] Gemini Init Failed: {e}")
            self.llm = None

        # Initialize MQTT
        self.mqtt = MQTTHandler(self.handle_voice_input_wrapper)
        print("[OK] MQTT Handler Initialized")
        
        # Avatar pipeline (lazy init)
        self.avatar_pipeline = None
        
        # Async coordination
        self.input_queue: asyncio.Queue = asyncio.Queue()
        self.loop: Optional[asyncio.AbstractEventLoop] = None

    def handle_voice_input_wrapper(self, text: str) -> None:
        """Thread-safe wrapper to put input into the async queue."""
        if self.loop:
            self.loop.call_soon_threadsafe(self.input_queue.put_nowait, text)

    async def _init_avatar_pipeline(self) -> None:
        """Initialize the avatar pipeline if enabled."""
        if not AVATAR_ENABLED:
            print("[Avatar] Avatar pipeline disabled (AVATAR_ENABLED=false)")
            return

        try:
            from modules.avatar.pipeline import AvatarPipeline
            self.avatar_pipeline = AvatarPipeline(
                models_dir="./models",
                avatar_image=AVATAR_IMAGE if AVATAR_IMAGE else None,
                idle_video=AVATAR_IDLE_VIDEO if AVATAR_IDLE_VIDEO else None,
                webrtc_port=WEBRTC_PORT,
            )
            active = await self.avatar_pipeline.start()
            if active:
                print(f"[OK] Avatar Pipeline Active (WebRTC on port {WEBRTC_PORT})")
            else:
                print("[WARN] Avatar Pipeline degraded (no GPU or model load failure)")
                self.avatar_pipeline = None
        except ImportError as e:
            print(f"[WARN] Avatar pipeline dependencies not installed: {e}")
            self.avatar_pipeline = None
        except Exception as e:
            print(f"[ERROR] Avatar Pipeline Init Failed: {e}")
            self.avatar_pipeline = None

    async def process_voice_input(self) -> None:
        """Process voice input and generate audio response."""
        print("Worker started. Waiting for input...")
        
        while True:
            text = await self.input_queue.get()
            print(f"\n>>> Processing: {text}")

            # Get answer from TUNA Chat API
            answer_text = None
            if self.tuna_chat:
                print("Querying TUNA Chat API...")
                response = await self.tuna_chat.send_message(text)
                if response:
                    answer_text = self.tuna_chat.get_answer(response)
                    print(f"TUNA answer: {answer_text[:100]}..." if len(answer_text) > 100 else f"TUNA answer: {answer_text}")
                else:
                    print("TUNA API returned no response, Gemini will answer directly.")

            if not self.llm:
                print("LLM not available.")
                self.input_queue.task_done()
                continue

            # Reset avatar pipeline buffers for new turn
            if self.avatar_pipeline and self.avatar_pipeline.is_active:
                self.avatar_pipeline.reset_for_new_turn()

            # Stream audio from Gemini and feed to avatar pipeline
            print("Generating streaming response...")
            audio_chunks = []
            
            async for audio_chunk, is_final in self.llm.generate_response_streaming(text, answer_text):
                if audio_chunk:
                    audio_chunks.append(audio_chunk)
                    
                    # Feed audio to avatar pipeline for lip-sync generation
                    if self.avatar_pipeline and self.avatar_pipeline.is_active:
                        await self.avatar_pipeline.feed_audio(audio_chunk)
                
                if is_final:
                    break
            
            # Save and play complete audio
            if audio_chunks:
                complete_audio = b"".join(audio_chunks)
                audio_size_kb = len(complete_audio) / 1024
                duration_s = len(complete_audio) / (24000 * 2)  # 24kHz, 16-bit
                print(f"[Gemini] Response: {len(audio_chunks)} chunks, {audio_size_kb:.1f} KB, ~{duration_s:.1f}s audio")
                audio_filename = "response_audio.wav"
                self._save_audio(complete_audio, audio_filename)
                print(f"Playing audio...")
                os.system(f"aplay {audio_filename} > /dev/null 2>&1 &")
            else:
                print("[Gemini] No audio received!")
            
            self.input_queue.task_done()

    def _save_audio(self, audio_data: bytes, filename: str) -> None:
        """Save raw PCM audio to WAV file."""
        import wave
        with wave.open(filename, "wb") as wav_file:
            wav_file.setnchannels(1)
            wav_file.setsampwidth(2)
            wav_file.setframerate(24000)
            wav_file.writeframes(audio_data)

    async def run(self) -> None:
        """Main application entry point."""
        print("\n--- Voice Assistant Started ---")
        
        self.loop = asyncio.get_running_loop()
        
        # Initialize avatar pipeline
        await self._init_avatar_pipeline()
        
        # Start MQTT
        self.mqtt.start()
        print("Waiting for voice commands on MQTT...")
        
        # Start WebRTC server if avatar pipeline is active
        if self.avatar_pipeline and self.avatar_pipeline.is_active:
            import uvicorn
            config = uvicorn.Config(
                self.avatar_pipeline.webrtc.app,
                host="0.0.0.0",
                port=WEBRTC_PORT,
                log_level="warning",
            )
            server = uvicorn.Server(config)
            # Run WebRTC server in background
            webrtc_task = asyncio.create_task(server.serve())
            print(f"[WebRTC] Server running on http://0.0.0.0:{WEBRTC_PORT}")
        
        try:
            await self.process_voice_input()
        except asyncio.CancelledError:
            print("\nStopping...")
        finally:
            if self.avatar_pipeline:
                await self.avatar_pipeline.stop()
            self.mqtt.stop()
            print("Goodbye.")


if __name__ == "__main__":
    app = VoiceAssistant()
    try:
        asyncio.run(app.run())
    except KeyboardInterrupt:
        pass
