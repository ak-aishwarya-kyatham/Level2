from typing import Optional
from fastapi import APIRouter
from app.mcp_client import mcp_client

router = APIRouter(prefix="/api/dashboard", tags=["dashboard"])

@router.get("/stats")
async def get_dashboard_stats():
    """Route dashboard stats request via MCP tool 'get_dashboard_analytics'"""
    return await mcp_client.call_tool("get_dashboard_analytics")

@router.get("/latest")
async def get_latest_news(
    query: Optional[str] = "",
    category: Optional[str] = "All",
    source: Optional[str] = "All",
    limit: int = 50
):
    """Route latest news search via MCP tool 'search_live_news'"""
    try:
        articles = await mcp_client.call_tool(
            "search_live_news",
            {"query": query or "", "category": category or "All", "source": source or "All", "limit": limit}
        )
        if not isinstance(articles, list):
            articles = []
        return {
            "articles": articles,
            "total_count": len(articles),
            "query": query,
            "category": category,
            "source": source
        }
    except Exception as e:
        return {
            "articles": [],
            "total_count": 0,
            "query": query,
            "category": category,
            "source": source,
            "error": str(e)
        }

@router.get("/by-category")
async def get_articles_by_category(
    category: str = "All",
    limit: int = 50
):
    """Route exact category news request via MCP tool 'get_articles_by_category'"""
    try:
        articles = await mcp_client.call_tool(
            "get_articles_by_category",
            {"category": category or "All", "limit": limit}
        )
        if not isinstance(articles, list):
            articles = []
        return {
            "articles": articles,
            "total_count": len(articles),
            "category": category
        }
    except Exception as e:
        return {
            "articles": [],
            "total_count": 0,
            "category": category,
            "error": str(e)
        }

@router.post("/refresh")
async def refresh_live_news():
    """Route refresh request via MCP tool 'fetch_latest_rss_feeds'"""
    res = await mcp_client.call_tool("fetch_latest_rss_feeds")
    return {
        "status": "success",
        "message": f"Successfully refreshed live news feeds via MCP server!",
        "total_articles": res.get("total_articles", 0) if isinstance(res, dict) else 0
    }

