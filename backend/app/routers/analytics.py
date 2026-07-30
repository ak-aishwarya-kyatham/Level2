from fastapi import APIRouter
from app.mcp_client import mcp_client

router = APIRouter(prefix="/api/analytics", tags=["analytics"])

@router.get("/metrics")
async def get_analytics_metrics():
    """Route analytics metrics request via MCP resource 'news://analytics/metrics'"""
    return await mcp_client.read_resource("news://analytics/metrics")
