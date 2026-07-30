import json
import logging
from typing import Dict, Any, List, Optional
from app.mcp_server import mcp_server

logger = logging.getLogger(__name__)

class NewsIntelMCPClient:
    """
    MCP Client interface for NewsIntel AI.
    Routes agent tool calls through the Model Context Protocol (MCP) server primitives.
    """
    def __init__(self):
        self.server = mcp_server

    async def list_available_tools(self) -> List[str]:
        """
        Query the MCP server for registered tools.
        """
        try:
            tools = [t.name for t in self.server._tool_manager.list_tools()]
            logger.info(f"[MCP Client] Available server tools: {tools}")
            return tools
        except Exception:
            return ["search_live_news", "fetch_latest_rss_feeds", "get_dashboard_analytics", "compare_news_sources"]


    async def call_tool(self, tool_name: str, arguments: Optional[Dict[str, Any]] = None) -> Any:
        """
        Invoke an MCP tool by name via standard MCP JSON-RPC payload interface.
        Handles FastMCP single-item and multi-item Content object list parsing cleanly.
        """
        arguments = arguments or {}
        logger.info(f"[MCP Client -> MCP Server] Invoking tool '{tool_name}' with args {arguments}")
        
        try:
            result = await self.server.call_tool(tool_name, arguments)
            
            if isinstance(result, tuple) and len(result) > 0:
                content_list = result[0]
                if isinstance(content_list, list) and len(content_list) > 0:

                    # Multi-item list returned by FastMCP (e.g. multiple news articles)
                    if len(content_list) > 1:
                        parsed_list = []
                        for item in content_list:
                            if hasattr(item, "text"):
                                try:
                                    parsed_list.append(json.loads(item.text))
                                except Exception:
                                    parsed_list.append(item.text)
                            else:
                                parsed_list.append(item)
                        return parsed_list
                    # Single-item returned by FastMCP
                    else:
                        single_item = content_list[0]
                        if hasattr(single_item, "text"):
                            try:
                                parsed = json.loads(single_item.text)
                                return parsed
                            except Exception:
                                return single_item.text
                        return single_item
            return result
        except Exception as e:
            logger.error(f"[MCP Client Error] Failed to execute MCP tool '{tool_name}': {e}")
            raise e

    async def read_resource(self, uri: str) -> Any:
        """
        Read an MCP resource by URI. Parses JSON payload automatically.
        """
        logger.info(f"[MCP Client -> MCP Server] Reading resource '{uri}'")
        res = await self.server.read_resource(uri)
        try:
            if isinstance(res, list) and len(res) > 0:
                item = res[0]
                text_content = getattr(item, "text", None) or getattr(item, "content", None)
                if text_content and isinstance(text_content, str):
                    return json.loads(text_content)
        except Exception as e:
            logger.error(f"[MCP Client Error] Failed to parse resource '{uri}': {e}")
        return res


# Singleton MCP Client instance
mcp_client = NewsIntelMCPClient()
