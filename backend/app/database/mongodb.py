import logging
import os
from typing import Any, Dict, List

logger = logging.getLogger(__name__)

class MongoDBManager:
    """
    Manages MongoDB persistence for news article metadata.
    Configurable via MONGO_URI and MONGO_DB_NAME environment variables.
    Falls back gracefully to JSON store if MongoDB is unavailable.
    """
    def __init__(self):
        self.mongo_uri = os.getenv("MONGO_URI", "mongodb://localhost:27017")
        self.db_name = os.getenv("MONGO_DB_NAME", "newsintel")
        self.client = None
        self.db = None
        self.collection = None
        self.is_connected = False
        self._connect()

    def _connect(self):
        try:
            import pymongo
            client = pymongo.MongoClient(self.mongo_uri, serverSelectionTimeoutMS=100)
            client.server_info()  # Trigger fast connection check
            self.client = client
            self.db = client[self.db_name]
            self.collection = self.db["articles"]
            self.is_connected = True
            logger.info(f"[MongoDB] Connected to MongoDB at {self.mongo_uri} (db: {self.db_name}).")
        except Exception as e:
            logger.info(f"[MongoDB] Server offline at {self.mongo_uri} ({e}). Falling back to JSON store.")
            self.client = None
            self.is_connected = False

    async def insert_article(self, article_data: Dict[str, Any]) -> bool:
        """Inserts a single article dictionary into MongoDB."""
        if not self.is_connected or self.collection is None:
            return False
        try:
            doc = dict(article_data)
            if "id" in doc and "_id" not in doc:
                doc["_id"] = doc["id"]
            self.collection.update_one({"_id": doc["_id"]}, {"$set": doc}, upsert=True)
            return True
        except Exception as e:
            logger.error(f"[MongoDB Error] Failed to insert article: {e}")
            return False

    async def get_all_articles(self, limit: int = 100) -> List[Dict[str, Any]]:
        """Retrieves articles from MongoDB."""
        if not self.is_connected or self.collection is None:
            return []
        try:
            cursor = self.collection.find({}, {"_id": 0}).limit(limit)
            return list(cursor)
        except Exception as e:
            logger.error(f"[MongoDB Error] Failed to query articles: {e}")
            return []

mongodb_manager = MongoDBManager()
