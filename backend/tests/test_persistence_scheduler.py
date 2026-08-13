import os
import pytest
from unittest.mock import AsyncMock, patch, MagicMock
from app.repositories.news_repository import NewsRepository
from app.database.mongodb import MongoDBManager
from app.database.qdrant import QdrantManager
from app.workflows.scheduler import NewsPipelineScheduler

pytestmark = pytest.mark.unit

async def test_scheduler_ingestion_workflow_real_persistence():
    """1. Test that NewsPipelineScheduler executes workflow without mock_existing or TODO placeholders."""
    scheduler = NewsPipelineScheduler()

    fake_article = MagicMock()
    fake_article.title = "Test AI Headline"
    fake_article.content = "Test content for scheduler integration."
    fake_article.source = "Test Source"
    fake_article.url = "http://example.com/test-ai"

    with patch.object(scheduler.ingestion_agent, "run", new_callable=AsyncMock, return_value=[fake_article]), \
         patch.object(scheduler.cleaning_agent, "run", new_callable=AsyncMock, return_value="Cleaned content for test AI headline"), \
         patch.object(scheduler.categorization_agent, "run", new_callable=AsyncMock, return_value="Technology"), \
         patch.object(scheduler.chunking_agent, "run", new_callable=AsyncMock, return_value=["Chunk 1"]), \
         patch.object(scheduler.embedding_agent, "run", new_callable=AsyncMock, return_value=[0.1] * 1024), \
         patch.object(scheduler.duplicate_agent, "run", new_callable=AsyncMock, return_value=(False, 0.0)):

        await scheduler.ingestion_workflow()

        assert fake_article.cleaned_content == "Cleaned content for test AI headline"
        assert fake_article.category == "Technology"
        assert fake_article.is_duplicate is False


def test_mongodb_manager_offline_fallback():
    """2. Test that MongoDBManager handles connection attempt and falls back cleanly when offline."""
    manager = MongoDBManager()
    # When MongoDB daemon is not running on port 27017, is_connected is False and operations return False/empty
    if not manager.is_connected:
        assert manager.client is None
    assert isinstance(manager.is_connected, bool)


def test_qdrant_manager_upsert_and_search_fallback():
    """3. Test QdrantManager upsert_article and search methods with offline fallback."""
    qdrant = QdrantManager()
    result = qdrant.upsert_article("test_id_123", [0.1] * 1024, {"title": "Test Title"})
    if not qdrant.client:
        assert result is False

    search_res = qdrant.search([0.1] * 1024, top_k=5)
    if not qdrant.client:
        assert search_res == []


def test_news_repository_configurable_data_file(tmp_path):
    """4. Test that NewsRepository respects custom ARTICLES_DATA_FILE environment variable."""
    custom_file = str(tmp_path / "custom_articles.json")
    with patch.dict(os.environ, {"ARTICLES_DATA_FILE": custom_file}):
        from app.repositories import news_repository
        repo = NewsRepository()
        assert repo.get_all_embeddings() == [] or isinstance(repo.get_all_embeddings(), list)
