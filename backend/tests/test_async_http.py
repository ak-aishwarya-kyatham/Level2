import pytest
import httpx
from unittest.mock import patch, MagicMock
from app.utils.async_http import async_post_json

pytestmark = pytest.mark.unit

async def test_async_post_json_success():
    """1. Test successful HTTP 200 response with valid JSON payload using httpx.AsyncClient."""
    fake_response = MagicMock(spec=httpx.Response)
    fake_response.status_code = 200
    fake_response.json.return_value = {"response": "Success result", "status": "ok"}
    fake_response.text = '{"response": "Success result", "status": "ok"}'

    with patch("httpx.AsyncClient.post", return_value=fake_response):
        status, data, text = await async_post_json("http://localhost:11434/api/generate", {"prompt": "hi"}, timeout=5.0)

    assert status == 200
    assert data == {"response": "Success result", "status": "ok"}
    assert "Success result" in text


async def test_async_post_json_timeout():
    """2. Test HTTP timeout handling with httpx.TimeoutException."""
    with patch("httpx.AsyncClient.post", side_effect=httpx.TimeoutException("Request timed out")):
        with pytest.raises((httpx.TimeoutException, Exception)):
            await async_post_json("http://localhost:11434/api/generate", {"prompt": "hi"}, timeout=0.1)


async def test_async_post_json_http_error():
    """3. Test HTTP non-200 error response (e.g. 500 Internal Server Error)."""
    fake_response = MagicMock(spec=httpx.Response)
    fake_response.status_code = 500
    fake_response.json.return_value = {"error": "Internal Server Error"}
    fake_response.text = '{"error": "Internal Server Error"}'

    with patch("httpx.AsyncClient.post", return_value=fake_response):
        status, data, text = await async_post_json("http://localhost:11434/api/generate", {"prompt": "hi"}, timeout=5.0)

    assert status == 500
    assert data == {}
    assert "Internal Server Error" in text


async def test_async_post_json_malformed_response():
    """4. Test malformed non-JSON response body handling."""
    fake_response = MagicMock(spec=httpx.Response)
    fake_response.status_code = 200
    fake_response.json.side_effect = Exception("JSON Decode Error")
    fake_response.text = "<html>502 Bad Gateway</html>"

    with patch("httpx.AsyncClient.post", return_value=fake_response):
        status, data, text = await async_post_json("http://localhost:11434/api/generate", {"prompt": "hi"}, timeout=5.0)

    assert status == 200
    assert data == {}
    assert "502 Bad Gateway" in text
