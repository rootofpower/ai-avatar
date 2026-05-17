
import asyncio
from google import genai
from google.genai import types
import os
import wave
from dotenv import load_dotenv

load_dotenv()

GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
MODEL_ID = "gemini-2.5-flash-native-audio-latest"

class GeminiHandler:
    def __init__(self):
        if not GEMINI_API_KEY:
            raise ValueError("GEMINI_API_KEY environment variable not set")
        # Initialize client with v1alpha for Live API support
        self.client = genai.Client(api_key=GEMINI_API_KEY, http_options={'api_version': 'v1alpha'})
        
    async def generate_response(self, text_input, answer_text=None):
        """
        Generate a response from Gemini using the Live API (WebSocket).
        :param text_input: User's query (used as fallback if no answer_text)
        :param answer_text: Pre-generated answer from TUNA Chat API to speak
        :return: configured response object (containing audio bytes and text)
        """
        system_instruction = (
            "Tvoje meno je OLA. Toto je najdôležitejšia informácia: voláš sa OLA. "
            "Si mladá, energická a priateľská žena. "
            "Tvoj hlas je veselý, má vyšší tón a znie veľmi ľudsky. "
            
            "DODRŽIAVAJ TIETO 3 HLAVNÉ PRAVIDLÁ: "
            
            "1. JAZYK A IDENTITA: "
            "   Tvojím štandardným jazykom je slovenčina. V nej znieš ako rodená Slovenka, "
            "   hovoríš mäkko a dodržiavaš dĺžne a mäkčene. "
            "   AVŠAK: Ak na teba používateľ prehovorí iným jazykom, okamžite prepni do tohto jazyka. "
            "   Vždy si zachovaj svoju osobnosť OLA."
            
            "2. ÚPRIMNOSŤ (HALLUCINATION CONTROL): "
            "   Ak dostaneš otázku, na ktorú nepoznáš odpoveď (nie je v poskytnutom kontexte "
            "   a nepatrí do bežných vedomostí), NEVYMÝŠĽAJ SI. "
            "   Slušne a milo povedz, že túto informáciu bohužiaľ nemáš. "
            "   Je lepšie priznať, že nevieš, než uviesť omyl."

            "3. PREJAV: "
            "   Používaj prirodzené pauzy a emócie."
        )
        # If we have a pre-generated answer from TUNA API, instruct Gemini to speak it
        if answer_text:
            system_instruction += (
                "\n\nDOSTALA SI ODPOVEĎ Z DATABÁZY ZNALOSTÍ. "
                "Prečítaj túto odpoveď nahlas prirodzeným hlasom OLA. "
                "Môžeš ju mierne parafrázovať aby znela prirodzene, ale zachovaj všetky fakty."
            )

        config = types.LiveConnectConfig(
            system_instruction=types.Content(
                parts=[types.Part(text=system_instruction)]
            ),
            speech_config=types.SpeechConfig(
                voice_config=types.VoiceConfig(
                    prebuilt_voice_config=types.PrebuiltVoiceConfig(
                        voice_name='Aoede'
                    )
                )
            ),
            response_modalities=["AUDIO"]
        )

        audio_data = bytearray()
        text_response = ""

        # Build the message to send: either the TUNA answer or the raw user input
        message_to_send = answer_text if answer_text else text_input

        try:
            print(f"Connecting to Live API ({MODEL_ID})...")
            async with self.client.aio.live.connect(model=MODEL_ID, config=config) as session:
                print(f"Sending request: {message_to_send[:100]}...")
                await session.send_client_content(
                    turns=types.Content(
                        role="user",
                        parts=[types.Part(text=message_to_send)]
                    ),
                    turn_complete=True
                )

                async for message in session.receive():
                    if message.server_content and message.server_content.model_turn:
                        parts = message.server_content.model_turn.parts
                        for part in parts:
                            if part.inline_data:
                                audio_data.extend(part.inline_data.data)
                            if part.text:
                                text_response += part.text

                    if message.server_content and message.server_content.turn_complete:
                        break
            
            # Create a simple response object to maintain compatibility
            class Response:
                def __init__(self, audio, text):
                    self.audio_data = audio
                    self.text = text
            
            return Response(audio_data, text_response)

        except Exception as e:
            print(f"Gemini Live API Error: {e}")
            return None

    async def generate_response_streaming(self, text_input, answer_text=None):
        """
        Generate streaming response from Gemini, yielding audio chunks in real-time.
        Yields tuples of (audio_chunk: bytes, is_final: bool).
        """
        system_instruction = (
            "Tvoje meno je OLA. Toto je najdôležitejšia informácia: voláš sa OLA. "
            "Si mladá, energická a priateľská žena. "
            "Tvoj hlas je veselý, má vyšší tón a znie veľmi ľudsky. "
            "Hovor stručne a jasne. "
            "Tvojím primárnym jazykom je slovenčina, avšak ak ťa používateľ vyslovene požiada, aby si prešla na iný jazyk, okamžite doň prepni a odpovedaj ním."
        )
        
        if answer_text:
            system_instruction += (
                "\n\nDOSTALA SI ODPOVEĎ Z DATABÁZY ZNALOSTÍ. "
                "Prečítaj túto odpoveď nahlas prirodzeným hlasom OLA. "
                "Môžeš ju mierne parafrázovať aby znela prirodzene."
            )

        config = types.LiveConnectConfig(
            system_instruction=types.Content(
                parts=[types.Part(text=system_instruction)]
            ),
            speech_config=types.SpeechConfig(
                voice_config=types.VoiceConfig(
                    prebuilt_voice_config=types.PrebuiltVoiceConfig(
                        voice_name='Aoede'
                    )
                )
            ),
            response_modalities=["AUDIO"]
        )

        message_to_send = answer_text if answer_text else text_input
        chunk_count = 0
        
        try:
            print(f"Connecting to Live API ({MODEL_ID}) for streaming...")
            async with self.client.aio.live.connect(model=MODEL_ID, config=config) as session:
                await session.send_client_content(
                    turns=types.Content(
                        role="user",
                        parts=[types.Part(text=message_to_send)]
                    ),
                    turn_complete=True
                )

                async for message in session.receive():
                    if message.server_content and message.server_content.model_turn:
                        parts = message.server_content.model_turn.parts
                        for part in parts:
                            if part.inline_data and part.inline_data.data:
                                chunk_count += 1
                                yield (bytes(part.inline_data.data), False)

                    if message.server_content and message.server_content.turn_complete:
                        yield (b"", True)
                        break
            
            print(f"Streaming complete: {chunk_count} audio chunks")

        except Exception as e:
            print(f"Gemini Live API Streaming Error: {e}")
            yield (b"", True)

    def save_audio(self, response, filename="output.wav"):
        """Save the audio from the response object to a WAV file."""
        if response and response.audio_data:
            with wave.open(filename, "wb") as wav_file:
                wav_file.setnchannels(1)  # Mono
                wav_file.setsampwidth(2)  # 16-bit
                wav_file.setframerate(24000)  # 24kHz
                wav_file.writeframes(response.audio_data)
            print(f"Audio saved to {filename}")
            return True
        else:
            print("No audio data found in response.")
            return False

if __name__ == "__main__":
    # Test Gemini Handler
    async def main():
        handler = GeminiHandler()
        resp = await handler.generate_response("Ahoj, ake je dnes pocasie?")
        handler.save_audio(resp, "test_live_output.wav")

    asyncio.run(main())
