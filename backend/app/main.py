import asyncio
import logging
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from app.routers import chat, dashboard, analytics, compare, sources
from app.repositories.news_repository import news_repository
from app.mcp_client import mcp_client

from contextlib import asynccontextmanager

from app.utils.task_lifecycle import task_manager

logger = logging.getLogger("uvicorn")

async def periodic_news_fetcher():
    """Background task to poll RSS feeds every 10 minutes via MCP Tool."""
    try:
        while True:
            await asyncio.sleep(600)
            try:
                logger.info("[MCP Background Worker] Triggering periodic live RSS news fetch via MCP...")
                await mcp_client.call_tool("fetch_latest_rss_feeds")
            except asyncio.CancelledError:
                logger.info("[MCP Background Worker] Cancellation requested.")
                raise
            except Exception as e:
                logger.error(f"[MCP Background Worker Error]: {e}")
    except asyncio.CancelledError:
        logger.info("[MCP Background Worker] Task cleanly shut down.")
        raise

@asynccontextmanager
async def lifespan(app: FastAPI):
    logger.info("Initializing NewsIntel AI architecture...")
    tools = await mcp_client.list_available_tools()
    tool_names = [t if isinstance(t, str) else t.get("name", "") for t in tools if t]
    logger.info(f"Registered MCP Tools ({len(tools)}): {tool_names}")
    
    # Pre-flight check for Ollama local LLM server
    try:
        from app.utils.async_http import async_get_json
        import os
        ollama_url = os.getenv("OLLAMA_URL", "http://localhost:11434")
        ollama_model = os.getenv("OLLAMA_MODEL", "qwen2.5:3b")
        status, data, _ = await async_get_json(f"{ollama_url}/api/tags", timeout=1.5)
        if status == 200 and isinstance(data, dict):
            models = [m.get("name", "") for m in data.get("models", [])]
            installed = any(ollama_model.lower() in m.lower() or m.lower() in ollama_model.lower() for m in models)
            if installed:
                logger.info(f"[LLM Health] Ollama online with model '{ollama_model}' verified.")
            else:
                logger.warning(f"[LLM Health Warning] Ollama online, but model '{ollama_model}' not found in {models}. Run: ollama pull {ollama_model}")
        else:
            logger.warning(f"[LLM Health Warning] Ollama not reachable at {ollama_url}. Operating in local synthesis fallback mode.")
    except Exception as e:
        logger.warning(f"[LLM Health] Ollama pre-flight check skipped: {e}")

    # Track background tasks via task_manager
    task_manager.create_task(periodic_news_fetcher(), name="periodic_news_fetcher")
    yield
    
    logger.info("Application shutdown initiated. Cancelling background tasks...")
    await task_manager.cancel_all()
    await mcp_client.stop()
    logger.info("Application shutdown complete.")


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
    fallback_tools = [
        "search_live_news", "fetch_latest_rss_feeds", "get_dashboard_analytics", "compare_news_sources", "get_articles_by_category"
    ]
    if mcp_client.is_connected:
        try:
            tools = await mcp_client.list_available_tools()
            tool_names = [t if isinstance(t, str) else t.get("name", "") for t in tools if t] or fallback_tools
        except Exception:
            tool_names = fallback_tools
    else:
        tool_names = fallback_tools

    return {
        "status": "ok",
        "architecture": "Model Context Protocol (MCP)",
        "transport": "stdio JSON-RPC",
        "mcp_connected": mcp_client.is_connected,
        "message": "NewsIntel AI MCP Live Intelligence Backend is active",
        "live_articles_count": len(news_repository.articles),
        "available_mcp_tools": tool_names,
        "available_mcp_resources": ["news://store/articles", "news://analytics/metrics"]
    }

@app.get("/api/health")
async def health():
    """Simple health check endpoint"""
    return {"status": "healthy"}

@app.get("/ping")
async def ping():
    """Simple ping endpoint for debugging"""
    return {"ping": "pong"}
