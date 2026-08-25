import logging

from apscheduler.schedulers.asyncio import AsyncIOScheduler
from apscheduler.triggers.interval import IntervalTrigger

from app.agents.categorization import CategorizationAgent
from app.agents.chunking import ChunkingAgent
from app.agents.cleaning import CleaningAgent
from app.agents.duplicate import DuplicateDetectionAgent
from app.agents.embedding import EmbeddingAgent

# Import our agents
from app.agents.ingestion import NewsIngestionAgent

logger = logging.getLogger(__name__)

class NewsPipelineScheduler:
    """
    APScheduler to trigger the ingestion workflow every 15-30 minutes.
    """
    def __init__(self):
        self.scheduler = AsyncIOScheduler()
        self.ingestion_agent = NewsIngestionAgent()
        self.cleaning_agent = CleaningAgent()
        self.categorization_agent = CategorizationAgent()
        self.chunking_agent = ChunkingAgent()
        self.embedding_agent = EmbeddingAgent()
        self.duplicate_agent = DuplicateDetectionAgent()

    async def ingestion_workflow(self):
        logger.info("Starting scheduled ingestion workflow...")

        # 1. Ingestion
        articles = await self.ingestion_agent.run()

        for article in articles:
            try:
                # 2. Cleaning
                cleaned_text = await self.cleaning_agent.run(article)
                article.cleaned_content = cleaned_text

                # 3. Categorization
                category = await self.categorization_agent.run(cleaned_text)
                article.category = category

                # 4. Chunking
                chunks = await self.chunking_agent.run(cleaned_text)
                article.chunks = chunks

                # 5. Embedding
                # We can embed the whole document or just chunks.
                # According to the flow: Article -> Chunks -> Embeddings
                # Here we just embed the cleaned text for duplicate detection.
                embedding = await self.embedding_agent.run(cleaned_text)
                article.embedding = embedding

                # 6. Duplicate Detection using real stored embeddings from repository
                from app.database.mongodb import mongodb_manager
                from app.database.qdrant import qdrant_manager
                from app.repositories.news_repository import news_repository

                existing_embeddings = news_repository.get_all_embeddings()
                is_duplicate, _ = await self.duplicate_agent.run(embedding, existing_embeddings)
                article.is_duplicate = is_duplicate

                if not is_duplicate:
                    logger.info(f"New unique article: {article.title}. Persisting to repositories...")
                    import hashlib
                    raw_id = getattr(article, "id", None)
                    art_id = raw_id if isinstance(raw_id, str) and raw_id else f"art_{hashlib.md5(str(article.title).encode('utf-8')).hexdigest()[:12]}"
                    art_source = str(getattr(article, "source", "Scheduled RSS Feed"))
                    art_url = str(getattr(article, "url", "#"))
                    art_content = str(getattr(article, "content", cleaned_text))

                    article_dict = {
                        "id": art_id,
                        "title": str(article.title),
                        "content": art_content,
                        "cleaned_content": str(cleaned_text),
                        "source": art_source,
                        "url": art_url,
                        "category": str(category),
                        "chunks": chunks if isinstance(chunks, list) else [str(chunks)],
                        "embedding": embedding if isinstance(embedding, list) else []
                    }
                    # Persist to Primary JSON Store
                    news_repository.articles.append(article_dict)
                    news_repository._save_to_disk()

                    # Persist to MongoDB (if connected)
                    await mongodb_manager.insert_article(article_dict)

                    # Persist to Qdrant Vector Store (if connected)
                    if embedding:
                        qdrant_manager.upsert_article(
                            article_id=article_dict["id"],
                            vector=embedding,
                            payload={"title": article_dict["title"], "url": article_dict["url"], "category": category}
                        )
                else:
                    logger.info(f"Article is duplicate: {article.title}. Skipping.")

            except Exception as e:
                logger.error(f"Error processing article {article.title}: {e}")

        logger.info("Scheduled ingestion workflow completed.")

    def start(self):
        # Run every 30 minutes
        self.scheduler.add_job(
            self.ingestion_workflow,
            trigger=IntervalTrigger(minutes=30),
            id='news_ingestion_job',
            name='News Ingestion Pipeline',
            replace_existing=True
        )
        self.scheduler.start()
        logger.info("News Pipeline Scheduler started. Job will run every 30 minutes.")

    def shutdown(self):
        self.scheduler.shutdown()
        logger.info("News Pipeline Scheduler shut down.")
