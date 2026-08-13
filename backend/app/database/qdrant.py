import os
import hashlib
import logging
from typing import List, Dict, Any, Optional
from qdrant_client import QdrantClient
from qdrant_client.models import Distance, VectorParams, PointStruct

logger = logging.getLogger(__name__)

class QdrantManager:
    """
    Manages Qdrant vector database connections, collection initialization,
    vector searching, and article embedding upserts.
    Configurable via QDRANT_URL environment variable.
    """
    def __init__(self):
        qdrant_url = os.getenv("QDRANT_URL", "http://localhost:6333")
        try:
            self.client = QdrantClient(url=qdrant_url, timeout=0.5)
            self.collection_name = "news_articles"
            self._ensure_collection()
            logger.info(f"[Qdrant] Connected to Qdrant at {qdrant_url}.")
        except Exception as e:
            logger.info(f"[Qdrant] Server offline at {qdrant_url} ({e}). Vector search will use in-memory cosine fallback.")
            self.client = None

    def _ensure_collection(self):
        if self.client and not self.client.collection_exists(self.collection_name):
            self.client.create_collection(
                collection_name=self.collection_name,
                vectors_config=VectorParams(size=1024, distance=Distance.COSINE),
            )
            logger.info(f"[Qdrant] Created collection: {self.collection_name}")

    def upsert_article(self, article_id: str, vector: list, payload: dict) -> bool:
        """Upserts a single vector embedding point into Qdrant."""
        if not self.client or not vector:
            return False
        try:
            # Deterministic integer point ID derived from article ID
            point_id = int(hashlib.md5(article_id.encode('utf-8')).hexdigest()[:15], 16)
            point = PointStruct(
                id=point_id,
                vector=vector,
                payload=payload
            )
            self.client.upsert(
                collection_name=self.collection_name,
                points=[point]
            )
            logger.info(f"[Qdrant] Upserted vector point for article: {article_id}")
            return True
        except Exception as e:
            logger.error(f"[Qdrant Error] Upsert failed for article {article_id}: {e}")
            return False

    def search(self, query_vector: list, top_k: int = 5):
        """Searches nearest neighbor vectors in Qdrant."""
        if not self.client or not query_vector:
            return []
        try:
            results = self.client.search(
                collection_name=self.collection_name,
                query_vector=query_vector,
                limit=top_k
            )
            return results
        except Exception as e:
            logger.error(f"[Qdrant Error] Vector search failed: {e}")
            return []

qdrant_manager = QdrantManager()
