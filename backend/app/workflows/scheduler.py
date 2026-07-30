import logging
from apscheduler.schedulers.asyncio import AsyncIOScheduler
from apscheduler.triggers.interval import IntervalTrigger

# Import our agents
from app.agents.ingestion import NewsIngestionAgent
from app.agents.cleaning import CleaningAgent
from app.agents.categorization import CategorizationAgent
from app.agents.chunking import ChunkingAgent
from app.agents.embedding import EmbeddingAgent
from app.agents.duplicate import DuplicateDetectionAgent

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
                
                # 6. Duplicate Detection
                # Mock existing embeddings from DB
                mock_existing = [] 
                is_duplicate, _ = await self.duplicate_agent.run(embedding, mock_existing)
                article.is_duplicate = is_duplicate
                
                if not is_duplicate:
                    logger.info(f"New unique article: {article.title}. Ready for DB insertion.")
                    # TODO: Insert to MongoDB and Qdrant
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
