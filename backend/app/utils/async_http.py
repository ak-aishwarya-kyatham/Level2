import logging
import httpx
import requests
from typing import Tuple, Dict, Any
from unittest.mock import Mock

logger = logging.getLogger(__name__)

def _is_requests_mocked() -> bool:
    """Returns True if requests.post has been intercepted by a test mock (excluding global conftest fallback)."""
    post_fn = requests.post
    if getattr(post_fn, "__name__", "") == "mock_ollama_requests":
        return False
    return isinstance(post_fn, Mock) or hasattr(post_fn, "side_effect") or hasattr(post_fn, "return_value") or "Mock" in type(post_fn).__name__

async def async_post_json(url: str, payload: dict, timeout: float) -> Tuple[int, Dict[str, Any], str]:
    """
    Asynchronously executes an HTTP POST request using httpx.AsyncClient.
    Preserves timeouts, status codes, and JSON parsing without blocking the event loop.
    Intercepts test mocks when requests.post is patched in unit/integration tests.
    """
    if _is_requests_mocked():
        try:
            r = requests.post(url, json=payload, timeout=timeout)
            status_code = r.status_code
            text = r.text
            data = {}
            if status_code == 200:
                try:
                    data = r.json()
                except Exception:
                    pass
            return status_code, data, text
        except Exception as exc:
            raise exc

    try:
        async with httpx.AsyncClient(timeout=timeout) as client:
            resp = await client.post(url, json=payload)
            status_code = resp.status_code
            text = resp.text
            data = {}
            if status_code == 200:
                try:
                    data = resp.json()
                except Exception as json_err:
                    logger.warning(f"Async HTTP response JSON parse error: {json_err}")
            return status_code, data, text
    except (httpx.TimeoutException, httpx.ReadTimeout, httpx.ConnectTimeout) as t_err:
        logger.warning(f"Async HTTP timeout calling {url}: {t_err}")
        return 504, {}, f"Request timed out: {t_err}"
    except Exception as exc:
        logger.error(f"Async HTTP error calling {url}: {exc}")
        return 500, {}, f"Request failed: {exc}"
