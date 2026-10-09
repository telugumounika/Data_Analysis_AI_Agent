"""Stage 6: talk to an LLM through an OpenAI-compatible HTTP API.

LM Studio (local Qwen), OpenAI, Groq, OpenRouter, Ollama... all accept the same
request format, so changing provider = changing 3 values in .env.
"""
import httpx

from .config import LLM_API_KEY, LLM_BASE_URL, LLM_MODEL, LLM_TEMPERATURE, LLM_TIMEOUT


class LLMError(Exception):
    """The LLM could not be reached or returned something unusable."""


def chat(messages: list[dict], tools: list[dict] | None = None) -> dict:
    """Send the conversation to the LLM and return the assistant message (a dict)."""
    payload = {
        "model": LLM_MODEL,
        "messages": messages,
        "temperature": LLM_TEMPERATURE,  # low = more deterministic, good for analysis
    }
    if tools:
        payload["tools"] = tools
        payload["tool_choice"] = "auto"
    try:
        resp = httpx.post(
            f"{LLM_BASE_URL.rstrip('/')}/chat/completions",
            json=payload,
            headers={"Authorization": f"Bearer {LLM_API_KEY}"},
            timeout=LLM_TIMEOUT,
        )
        resp.raise_for_status()
        return resp.json()["choices"][0]["message"]
    except httpx.ConnectError as exc:
        raise LLMError(f"Cannot connect to the LLM at {LLM_BASE_URL}. Is LM Studio's server running?") from exc
    except httpx.TimeoutException as exc:
        raise LLMError("The LLM took too long to respond.") from exc
    except httpx.HTTPStatusError as exc:
        raise LLMError(f"LLM API error {exc.response.status_code}: {exc.response.text[:200]}") from exc
    except (KeyError, IndexError, ValueError) as exc:
        raise LLMError("The LLM returned an unexpected response format.") from exc


def check_status() -> dict:
    """Quick health check used by the UI badge: is the LLM server up, and is our model loaded?"""
    try:
        resp = httpx.get(f"{LLM_BASE_URL.rstrip('/')}/models",
                         headers={"Authorization": f"Bearer {LLM_API_KEY}"}, timeout=2.5)
        resp.raise_for_status()
        models = [m.get("id") for m in resp.json().get("data", [])]
        return {"reachable": True, "models": models[:20], "model_ok": LLM_MODEL in models if models else True}
    except Exception as exc:  # any failure just means "not available"
        return {"reachable": False, "models": [], "model_ok": False, "error": type(exc).__name__}
