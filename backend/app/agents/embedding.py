import logging
import requests
from typing import List

logger = logging.getLogger(__name__)

OLLAMA_URL = "http://localhost:11434"

class EmbeddingAgent:
    """
    Generates embeddings using BAAI/bge-m3 via local Ollama or lazy transformer fallback.
    """
    def __init__(self):
        self.model_name = "BAAI/bge-m3"
        self.tokenizer = None
        self.model = None
        self.attempted_load = False

    def _lazy_load(self):
        if self.attempted_load:
            return
        self.attempted_load = True
        logger.info(f"Lazily loading embedding model {self.model_name}...")
        try:
            from transformers import AutoModel, AutoTokenizer
            self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
            self.model = AutoModel.from_pretrained(self.model_name)
            self.model.eval()
        except Exception as e:
            logger.error(f"Failed to load embedding model: {e}")
            self.tokenizer = None
            self.model = None

    async def run(self, text: str) -> List[float]:
        # Try Ollama bge-m3 first
        try:
            from app.utils.async_http import async_post_json
            status_code, data, res_text = await async_post_json(
                f"{OLLAMA_URL}/api/embeddings",
                payload={"model": "bge-m3:latest", "prompt": text},
                timeout=5.0
            )
            if status_code == 200:
                emb = data.get("embedding", [])
                if emb:
                    return emb
        except Exception as e:
            logger.debug(f"Ollama embedding lookup skipped/failed: {e}")

        # Local fallback
        self._lazy_load()
        if not self.model or not self.tokenizer:
            logger.warning("Embedding model not loaded. Returning deterministic fallback embedding vector.")
            from app.agents.duplicate import generate_fallback_embedding
            return generate_fallback_embedding(text)
            
        try:
            import torch
            encoded_input = self.tokenizer([text], padding=True, truncation=True, return_tensors='pt', max_length=512)
            with torch.no_grad():
                model_output = self.model(**encoded_input)
            sentence_embeddings = model_output[0][:, 0]
            sentence_embeddings = torch.nn.functional.normalize(sentence_embeddings, p=2, dim=1)
            return sentence_embeddings[0].tolist()
        except Exception as e:
            logger.error(f"Embedding error: {e}")
            from app.agents.duplicate import generate_fallback_embedding
            return generate_fallback_embedding(text)

    def get_embedding_sync(self, text: str) -> List[float]:
        """Synchronous version of embedding generation for fallback purposes."""
        # Try Ollama bge-m3 first
        try:
            r = requests.post(
                f"{OLLAMA_URL}/api/embeddings",
                json={"model": "bge-m3:latest", "prompt": text},
                timeout=5
            )
            if r.status_code == 200:
                emb = r.json().get("embedding", [])
                if emb:
                    return emb
        except Exception as e:
            logger.debug(f"Ollama sync embedding lookup failed: {e}")

        self._lazy_load()
        if not self.model or not self.tokenizer:
            from app.agents.duplicate import generate_fallback_embedding
            return generate_fallback_embedding(text)
        try:
            import torch
            encoded_input = self.tokenizer([text], padding=True, truncation=True, return_tensors='pt', max_length=512)
            with torch.no_grad():
                model_output = self.model(**encoded_input)
            sentence_embeddings = model_output[0][:, 0]
            sentence_embeddings = torch.nn.functional.normalize(sentence_embeddings, p=2, dim=1)
            return sentence_embeddings[0].tolist()
        except Exception as e:
            logger.error(f"Sync embedding error: {e}")
            from app.agents.duplicate import generate_fallback_embedding
            return generate_fallback_embedding(text)


