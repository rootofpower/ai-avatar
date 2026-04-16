
import json
import os
import threading
import paho.mqtt.client as mqtt
from paho.mqtt.enums import CallbackAPIVersion
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

MQTT_BROKER = os.getenv("MQTT_BROKER", "localhost")
MQTT_PORT = int(os.getenv("MQTT_PORT", 1883))
MQTT_TOPIC = "experiments/voice/recognition/avatar"

class MQTTHandler:
    def __init__(self, callback):
        """
        Initialize the MQTT Handler.
        :param callback: Function to call when recognized text is received. 
                         Signature: callback(text: str)
        """
        self.client_id = f"bc_pract_client_{os.getpid()}"
        self.client = mqtt.Client(CallbackAPIVersion.VERSION2, client_id=self.client_id)
        self.client.on_connect = self.on_connect
        self.client.on_message = self.on_message
        self.client.on_disconnect = self.on_disconnect
        self.callback = callback
        self.running = False

    def on_connect(self, client, userdata, flags, reason_code, properties):
        print(f"Connected to MQTT Broker: {MQTT_BROKER}")
        client.subscribe(MQTT_TOPIC)
        print(f"Subscribed to topic: {MQTT_TOPIC}")

    def on_disconnect(self, client, userdata, flags, reason_code, properties):
        print("Disconnected from MQTT Broker")

    def on_message(self, client, userdata, message):
        try:
            payload = message.payload.decode("utf-8").strip()
            
            # Try JSON format first (from voice recognition system)
            try:
                data = json.loads(payload)
                if data.get("status") == "recognized":
                    text = data.get("recognized", "")
                    if text:
                        print(f"Recognized Speech: {text}")
                        self.callback(text)
                        return
            except json.JSONDecodeError:
                pass
            
            # Fallback: treat as plain text (for manual testing)
            if payload:
                print(f"Plain text input: {payload}")
                self.callback(payload)
                    
        except Exception as e:
            print(f"Error processing MQTT message: {e}")

    def start(self):
        """Start the MQTT client loop in a background thread."""
        if not self.running:
            print(f"Connecting to {MQTT_BROKER}:{MQTT_PORT}...")
            try:
                self.client.connect(MQTT_BROKER, MQTT_PORT, 60)
                self.client.loop_start()
                self.running = True
            except Exception as e:
                print(f"Failed to connect to MQTT broker: {e}")

    def stop(self):
        """Stop the MQTT client."""
        if self.running:
            self.client.loop_stop()
            self.client.disconnect()
            self.running = False
            print("MQTT Handler stopped.")

if __name__ == "__main__":
    # Test the MQTT Handler
    def test_callback(text):
        print(f"TEST CALLBACK RECEIVED: {text}")

    handler = MQTTHandler(test_callback)
    try:
        handler.start()
        print("Press Ctrl+C to stop...")
        # Keep main thread alive
        threading.Event().wait()
    except KeyboardInterrupt:
        handler.stop()
