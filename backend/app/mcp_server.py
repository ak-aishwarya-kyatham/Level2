import logging
from typing import Dict, Any, List
try:
    from mcp.server.fastmcp import FastMCP
except ImportError:
    try:
        from mcp.server import FastMCP
    except ImportError:
        class FastMCP:
            def __init__(self, name: str):
                self.name = name
            def tool(self, *args, **kwargs):
                def decorator(fn):
                    return fn
                return decorator
            def resource(self, *args, **kwargs):
                def decorator(fn):
                    return fn
                return decorator
            def prompt(self, *args, **kwargs):
                def decorator(fn):
                    return fn
                return decorator

from app.repositories.news_repository import news_repository

logger = logging.getLogger(__name__)

# Initialize FastMCP server for NewsIntel AI
mcp_server = FastMCP(
    name="NewsIntel-AI-MCP-Server"
)

@mcp_server.tool()
async def search_live_news(query: str = "", category: str = "", source: str = "", limit: int = 20) -> List[Dict[str, Any]]:
    """
    MCP Tool: Search indexed live news articles, location trends, or topic news (e.g. Telangana, India, Artificial Intelligence). MUST be used for all topic, state, or location queries.
    """
    logger.info(f"[MCP Tool: search_live_news] Query: '{query}', Category: '{category}', Source: '{source}'")
    return await news_repository.search_articles(query=query, category=category, source=source, limit=limit)

@mcp_server.tool()
async def fetch_latest_rss_feeds() -> Dict[str, Any]:
    """
    MCP Tool: Trigger live RSS news ingestion from configured sources.
    """
    logger.info("[MCP Tool: fetch_latest_rss_feeds] Executing ingestion...")
    articles = await news_repository.fetch_and_index_live_news()
    return {
        "status": "success",
        "total_articles": len(articles),
        "duplicates_prevented": news_repository.duplicates_prevented
    }

@mcp_server.tool()
async def get_dashboard_analytics() -> Dict[str, Any]:
    """
    MCP Tool: Retrieve overall global system statistics and high-level platform analytics. DO NOT use for location, state, or specific topic queries.
    """
    logger.info("[MCP Tool: get_dashboard_analytics] Retrieving stats...")
    return news_repository.get_dashboard_stats()

@mcp_server.tool()
async def compare_news_sources(source1: str, source2: str) -> Dict[str, Any]:
    """
    MCP Tool: Compare coverage, exclusive topics, and common news between two media outlets.
    """
    logger.info(f"[MCP Tool: compare_news_sources] Comparing '{source1}' vs '{source2}'")
    return news_repository.compare_sources(source1, source2)

@mcp_server.tool()
async def get_articles_by_category(category: str = "", limit: int = 20) -> List[Dict[str, Any]]:
    """
    MCP Tool: Retrieve live news articles matching exact category strictly.
    """
    logger.info(f"[MCP Tool: get_articles_by_category] Category: '{category}'")
    return await news_repository.get_articles_by_category(category=category, limit=limit)

@mcp_server.resource("news://store/articles")
def get_articles_resource() -> List[Dict[str, Any]]:
    """
    MCP Resource: Access complete stored articles dataset.
    """
    return news_repository.get_all_articles()

@mcp_server.resource("news://analytics/metrics")
def get_analytics_resource() -> Dict[str, Any]:
    """
    MCP Resource: Access platform analytics and source distribution metrics.
    """
    return news_repository.get_analytics_metrics()


if __name__ == "__main__":
    mcp_server.run(transport="stdio")


