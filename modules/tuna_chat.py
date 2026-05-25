
import asyncio
import aiohttp
import os
import logging
import traceback
from dotenv import load_dotenv

load_dotenv()

# --- Logging Setup ---
logger = logging.getLogger("tuna_chat")
logger.setLevel(logging.INFO)

TUNA_API_BASE_URL = os.getenv("TUNA_API_URL", "http://localhost:11136")


class TunaChatAPI:
    """Client for the TUNA Chat API (POST /api/chat)."""

    def __init__(self):
        self.base_url = TUNA_API_BASE_URL.rstrip("/")
        self.chat_endpoint = f"{self.base_url}/api/chat"
        self.session_id = None  # Track session across messages
        logger.info("TunaChatAPI initialized | endpoint=%s", self.chat_endpoint)

    async def send_message(self, message: str) -> dict | None:
        """
        Send a message to the TUNA Chat API and get a RAG-powered answer.
        :param message: User's message/question
        :return: dict with 'answer', 'sources', 'sessionId', 'messageId' or None on error
        """
        payload = {"message": message}
        if self.session_id:
            payload["sessionId"] = self.session_id

        logger.debug("Sending request to %s | payload=%s", self.chat_endpoint, payload)

        try:
            async with aiohttp.ClientSession() as session:
                logger.debug("ClientSession created, sending POST...")
                async with session.post(
                    self.chat_endpoint,
                    json=payload,
                    timeout=aiohttp.ClientTimeout(total=20),
                ) as resp:
                    logger.info("Response received | status=%s, content_type=%s", resp.status, resp.content_type)
                    logger.debug("Response headers: %s", dict(resp.headers))

                    if resp.status == 200:
                        raw_text = await resp.text()
                        logger.debug("Raw response body (first 500 chars): %s", raw_text[:500])
                        data = await resp.json(content_type=None)
                        # Track session for follow-up messages
                        self.session_id = data.get("sessionId")
                        logger.info("Success | sessionId=%s, keys=%s", self.session_id, list(data.keys()))
                        return data
                    else:
                        error_text = await resp.text()
                        logger.error(
                            "TUNA API HTTP Error | status=%s, body=%s",
                            resp.status, error_text[:500],
                        )
                        return None

        except aiohttp.ClientConnectorError as e:
            logger.error("Connection failed (cannot reach host) | %s", e)
            logger.debug("Full traceback:\n%s", traceback.format_exc())
            return None
        except aiohttp.ClientResponseError as e:
            logger.error("Bad response | status=%s, message=%s", e.status, e.message)
            logger.debug("Full traceback:\n%s", traceback.format_exc())
            return None
        except asyncio.TimeoutError:
            logger.error("Request timed out after 20s to %s", self.chat_endpoint)
            return None
        except aiohttp.ClientError as e:
            logger.error("TUNA API ClientError | type=%s, detail=%s", type(e).__name__, e)
            logger.debug("Full traceback:\n%s", traceback.format_exc())
            return None
        except Exception as e:
            logger.error(
                "TUNA API Unexpected Error | type=%s, detail=%s",
                type(e).__name__, e,
            )
            logger.debug("Full traceback:\n%s", traceback.format_exc())
            return None

    def get_answer(self, response: dict | None) -> str:
        """Extract the answer text from a TUNA API response."""
        if response and "answer" in response:
            return response["answer"]
        return ""

    def reset_session(self):
        """Reset the session ID to start a new conversation."""
        self.session_id = None


if __name__ == "__main__":
    import asyncio

    async def test():
        api = TunaChatAPI()
        logger.info("Endpoint: %s", api.chat_endpoint)
        result = await api.send_message("Jaroslav Poruban")
        if result:
            logger.info("Answer: %s", result.get("answer", "N/A"))
            logger.info("Session ID: %s", result.get("sessionId", "N/A"))
            if result.get("sources"):
                logger.info("Sources: %s", result["sources"])
        else:
            logger.warning("No response received.")

    asyncio.run(test())
