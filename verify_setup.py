
import sys
import os

# Add current directory to path
sys.path.append(os.getcwd())

print("Verifying imports...")
try:
    from modules.mqtt_handler import MQTTHandler
    print("[OK] MQTTHandler imported")
except ImportError as e:
    print(f"[FAIL] MQTTHandler import failed: {e}")

try:
    from modules.tuna_chat import TunaChatAPI
    print("[OK] TunaChatAPI imported")
except ImportError as e:
    print(f"[FAIL] TunaChatAPI import failed: {e}")

try:
    from modules.llm import GeminiHandler
    print("[OK] GeminiHandler imported")
except ImportError as e:
    print(f"[FAIL] GeminiHandler import failed: {e}")

print("\nVerifying Environment Variables...")
from dotenv import load_dotenv
load_dotenv()
if os.getenv("GEMINI_API_KEY"):
    print("[OK] GEMINI_API_KEY found")
else:
    print("[FAIL] GEMINI_API_KEY not set")

if os.getenv("MQTT_BROKER"):
    print(f"[OK] MQTT_BROKER set to {os.getenv('MQTT_BROKER')}")
else:
    print("[WARN] MQTT_BROKER not set, using default")

print("\nVerification Complete.")
