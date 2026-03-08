
import asyncio
import os
from typing import Optional
from modules.mqtt_handler import MQTTHandler
from modules.tuna_chat import TunaChatAPI
from modules.llm import GeminiHandler
from dotenv import load_dotenv


load_dotenv()


class VoiceAssistant:
    """
    Voice Assistant Prototype.
    
    Integrates TUNA Chat API for knowledge retrieval and Gemini Live API for voice synthesis.
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
        
        # Async coordination
        self.input_queue: asyncio.Queue = asyncio.Queue()
        self.loop: Optional[asyncio.AbstractEventLoop] = None

    def handle_voice_input_wrapper(self, text: str) -> None:
        """Thread-safe wrapper to put input into the async queue."""
        if self.loop:
            self.loop.call_soon_threadsafe(self.input_queue.put_nowait, text)

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

            # Stream audio from Gemini and sync with avatar
            print("Generating streaming response...")
            audio_chunks = []
            
            async for audio_chunk, is_final in self.llm.generate_response_streaming(text, answer_text):
                if audio_chunk:
                    audio_chunks.append(audio_chunk)
                
                if is_final:
                    break
            
            # Save and play complete audio
            if audio_chunks:
                complete_audio = b"".join(audio_chunks)
                audio_filename = "response_audio.wav"
                self._save_audio(complete_audio, audio_filename)
                print(f"Playing audio...")
                os.system(f"aplay {audio_filename} > /dev/null 2>&1 &")
            
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
        
        # Start MQTT
        self.mqtt.start()
        print("Waiting for voice commands on MQTT...")
        
        try:
            await self.process_voice_input()
        except asyncio.CancelledError:
            print("\nStopping...")
        finally:
            self.mqtt.stop()
            print("Goodbye.")


if __name__ == "__main__":
    app = VoiceAssistant()
    try:
        asyncio.run(app.run())
    except KeyboardInterrupt:
        pass
