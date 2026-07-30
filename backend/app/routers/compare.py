from fastapi import APIRouter
from pydantic import BaseModel
from app.mcp_client import mcp_client

router = APIRouter(prefix="/api/compare", tags=["compare"])

class CompareRequest(BaseModel):
    source1: str
    source2: str

@router.post("/")
async def compare_sources_post(req: CompareRequest):
    """Route source comparison via MCP tool 'compare_news_sources' (POST)"""
    return await mcp_client.call_tool("compare_news_sources", {"source1": req.source1, "source2": req.source2})

@router.get("/")
async def compare_sources_get(source1: str = "Times of India", source2: str = "The Hindu"):
    """Route source comparison via MCP tool 'compare_news_sources' (GET)"""
    return await mcp_client.call_tool("compare_news_sources", {"source1": source1, "source2": source2})

