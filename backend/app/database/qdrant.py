import os
from qdrant_client import QdrantClient
from qdrant_client.models import Distance, VectorParams
import logging

logger = logging.getLogger(__name__)

class QdrantManager:
    def __init__(self):
        qdrant_url = os.getenv("QDRANT_URL", "http://localhost:6333")
        try:
            self.client = QdrantClient(url=qdrant_url)
            self.collection_name = "news_articles"
            self._ensure_collection()
            logger.info("Connected to Qdrant.")
        except Exception as e:
            logger.error(f"Failed to connect to Qdrant: {e}")
            self.client = None

    def _ensure_collection(self):
        if not self.client.collection_exists(self.collection_name):
            self.client.create_collection(
                collection_name=self.collection_name,
                vectors_config=VectorParams(size=1024, distance=Distance.COSINE), # BAAI/bge-m3 output size
            )
            logger.info(f"Created Qdrant collection: {self.collection_name}")

    def search(self, query_vector: list, top_k: int = 5):
        if not self.client:
            return []
        
        try:
            results = self.client.search(
                collection_name=self.collection_name,
                query_vector=query_vector,
                limit=top_k
            )
            return results
        except Exception as e:
            logger.error(f"Qdrant search error: {e}")
            return []
