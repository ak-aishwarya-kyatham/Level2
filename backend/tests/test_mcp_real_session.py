import os
import sys

import pytest

# Ensure backend root is on sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from app.mcp_client import NewsIntelMCPClient


@pytest.mark.asyncio
async def test_mcp_real_stdio_subprocess_session():
    """
    Asserts that NewsIntelMCPClient establishes a genuine stdio ClientSession
    with the FastMCP subprocess, discovers all 5 registered tools dynamically,
    and executes a tool call over the protocol.
    """
    client = NewsIntelMCPClient()
    try:
        await client.start()
        assert client.is_connected is True, "MCP ClientSession failed to connect via stdio transport."
        assert client.session is not None, "ClientSession is None after start()."

        # 1. Discover registered tools dynamically from FastMCP server
        tools = await client.list_available_tools()
        tool_names = [t.get("name") for t in tools if t.get("name")]

        expected_tools = {
            "search_live_news",
            "fetch_latest_rss_feeds",
            "get_dashboard_analytics",
            "compare_news_sources",
            "get_articles_by_category"
        }

        for exp in expected_tools:
            assert exp in tool_names, f"Expected tool '{exp}' not discovered from MCP server. Found: {tool_names}"
        assert len(tools) >= 5, f"Expected at least 5 tools, got {len(tools)}: {tool_names}"

        # 2. Execute tool call over MCP protocol
        analytics = await client.call_tool("get_dashboard_analytics", {})
        assert isinstance(analytics, dict), f"Tool execution did not return dict: {analytics}"
        assert "total_articles" in analytics or "articles_by_category" in analytics or "category_distribution" in analytics

        # 3. Read MCP resource
        res = await client.read_resource("news://analytics/metrics")
        assert isinstance(res, dict), f"Resource read did not return dict: {res}"

    finally:
        await client.stop()
        assert client.is_connected is False
        assert client.session is None
