import os
import requests
import pytest
from unittest.mock import patch, MagicMock, AsyncMock, PropertyMock

orig_post = requests.post

@pytest.fixture(autouse=True)
def mock_external_boundaries(tmp_path, request):
    """
    Autouse fixture that isolates environment data files using pytest's default tmp_path
    and mocks external service boundaries (Qdrant, Redis, Ollama, HuggingFace embeddings)
    for all offline tests (unless explicitly marked with @pytest.mark.live).
    """
    eval_file = str(tmp_path / "test_eval_runs.json")
    logs_file = str(tmp_path / "test_agent_logs.json")
    articles_file = str(tmp_path / "test_articles.json")

    env_overrides = {
        "EVAL_DATA_FILE": eval_file,
        "AGENT_LOGS_DATA_FILE": logs_file,
        "ARTICLES_DATA_FILE": articles_file,
    }

    # Skip mocking for explicit @pytest.mark.live tests
    if "live" in request.keywords:
        with patch.dict(os.environ, env_overrides):
            yield
        return

    from app.agents.duplicate import generate_fallback_embedding

    def mock_ollama_requests(url, *args, **kwargs):
        if "11434" in str(url) or "ollama" in str(url).lower():
            mock_resp = MagicMock()
            mock_resp.status_code = 500
            mock_resp.text = "Ollama offline mock in test mode"
            return mock_resp
        return orig_post(url, *args, **kwargs)

    def mock_get_embedding_sync(text: str) -> list:
        return generate_fallback_embedding(text)

    async def mock_get_embedding_async(text: str) -> list:
        return generate_fallback_embedding(text)

    with patch.dict(os.environ, env_overrides), \
         patch("app.mcp_client.mcp_client.session", None), \
         patch("qdrant_client.QdrantClient", side_effect=Exception("Qdrant offline in test mode")), \
         patch("redis.Redis", side_effect=Exception("Redis offline in test mode")), \
         patch("requests.post", side_effect=mock_ollama_requests), \
         patch("app.agents.embedding.EmbeddingAgent.run", side_effect=mock_get_embedding_async), \
         patch("app.agents.embedding.EmbeddingAgent.get_embedding_sync", side_effect=mock_get_embedding_sync):
        yield
