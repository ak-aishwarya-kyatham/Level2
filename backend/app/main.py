import asyncio
import logging
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from app.routers import chat, dashboard, analytics, compare, sources
from app.repositories.news_repository import news_repository
from app.mcp_client import mcp_client

from contextlib import asynccontextmanager

logger = logging.getLogger("uvicorn")

async def periodic_news_fetcher():
    """Background task to continuously poll RSS feeds every 5 minutes via MCP Tool."""
    while True:
        try:
            logger.info("[MCP Background Worker] Triggering periodic live RSS news fetch via MCP...")
            await mcp_client.call_tool("fetch_latest_rss_feeds")
        except Exception as e:
            logger.error(f"[MCP Background Worker Error]: {e}")
        await asyncio.sleep(300)

@asynccontextmanager
async def lifespan(app: FastAPI):
    logger.info("Initializing NewsIntel AI MCP Server & Client architecture...")
    await mcp_client.start()
    tools = await mcp_client.list_available_tools()
    logger.info(f"Registered MCP Tools: {tools}")
    
    # Fetch immediately on startup via MCP tool
    asyncio.create_task(mcp_client.call_tool("fetch_latest_rss_feeds"))
    # Start continuous background fetcher
    asyncio.create_task(periodic_news_fetcher())
    yield
    await mcp_client.stop()


app = FastAPI(
    title="NewsIntel AI - Multi-Agent Enterprise Platform (MCP Architecture)",
    description="Backend API powered by Model Context Protocol (MCP) Server & Client Tools",
    version="2.0.0",
    lifespan=lifespan
)

# CORS configuration
app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:5173",
        "http://127.0.0.1:5173",
        "http://localhost:3000",
        "http://localhost:8000"
    ],
    allow_origin_regex=r"http://(localhost|127\.0\.0\.1):\d+",
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


app.include_router(chat.router)
app.include_router(dashboard.router)
app.include_router(analytics.router)
app.include_router(compare.router)
app.include_router(sources.router)

@app.get("/")
async def read_root():
    return {"status": "ok", "architecture": "Model Context Protocol (MCP)", "message": "NewsIntel AI MCP Live Intelligence Backend is active", "live_articles_count": len(news_repository.articles), "available_mcp_tools": ["search_live_news", "fetch_latest_rss_feeds", "get_dashboard_analytics", "compare_news_sources"], "available_mcp_resources": ["news://store/articles", "news://analytics/metrics"]}

@app.get("/api/health")
async def health():
    """Simple health check endpoint"""
    return {"status": "healthy"}

@app.get("/ping")
async def ping():
    """Simple ping endpoint for debugging"""
    return {"ping": "pong"}
