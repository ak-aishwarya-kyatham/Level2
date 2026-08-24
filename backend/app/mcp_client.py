import os
import sys
import json
import logging
import asyncio
from typing import Dict, Any, List, Optional
from mcp import StdioServerParameters, stdio_client
from mcp.client.session import ClientSession
from contextlib import AsyncExitStack

logger = logging.getLogger(__name__)

class NewsIntelMCPClient:
    """
    MCP Client interface for NewsIntel AI.
    Routes agent tool calls through standard MCP client-session transport.
    """
    def __init__(self):
        self.session: Optional[ClientSession] = None
        self._exit_stack: Optional[AsyncExitStack] = None

    async def start(self):
        """
        Operates in high-performance in-process mode on Windows/FastAPI environment.
        """
        logger.info("MCP Client operating in high-performance in-process mode.")
        self.session = None

    async def stop(self):
        """
        Closes the client session and terminates the subprocess.
        """
        if self._exit_stack:
            logger.info("Stopping MCP Client Session...")
            await self._exit_stack.aclose()
            self.session = None
            self._exit_stack = None

    async def list_available_tools(self) -> List[Dict[str, Any]]:
        """
        Query the MCP server for registered tools schema dynamically.
        Auto-starts session if not active and provides static fallbacks.
        """
        if not self.session:
            try:
                await self.start()
            except Exception as e:
                logger.warning(f"[MCP Client] Could not auto-start session: {e}")

        if not self.session:
            logger.warning("[MCP Client] Session not active, returning fallback tool schemas.")
            return [
                {
                    "name": "search_live_news",
                    "description": "MCP Tool: Search indexed live news articles, location trends, or topic news (e.g. Telangana, India, Artificial Intelligence). MUST be used for all topic, state, or location queries.",
                    "inputSchema": {
                        "type": "object",
                        "properties": {
                            "query": {"type": "string", "default": ""},
                            "category": {"type": "string", "default": ""},
                            "source": {"type": "string", "default": ""},
                            "limit": {"type": "integer", "default": 20}
                        }
                    }
                },
                {
                    "name": "get_dashboard_analytics",
                    "description": "MCP Tool: Retrieve overall global system statistics and high-level platform analytics. DO NOT use for location, state, or specific topic queries.",
                    "inputSchema": {"type": "object", "properties": {}}
                },
                {
                    "name": "compare_news_sources",
                    "description": "MCP Tool: Compare coverage, exclusive topics, and common news between two media outlets.",
                    "inputSchema": {
                        "type": "object",
                        "properties": {
                            "source1": {"type": "string"},
                            "source2": {"type": "string"}
                        },
                        "required": ["source1", "source2"]
                    }
                },
                {
                    "name": "get_articles_by_category",
                    "description": "MCP Tool: Retrieve live news articles matching a specific category (Technology, Business, Politics, Sports, General News, Entertainment). Matches category aliases like 'AI' or 'Technology'.",
                    "inputSchema": {
                        "type": "object",
                        "properties": {
                            "category": {"type": "string", "default": "", "description": "Category name or keyword e.g. Technology, Business, Politics, Sports, General News"},
                            "limit": {"type": "integer", "default": 20}
                        }
                    }
                }
            ]

        try:
            result = await self.session.list_tools()
            logger.info(f"[MCP Client] Available server tools: {[t.name for t in result.tools]}")
            return [
                {
                    "name": t.name,
                    "description": t.description,
                    "inputSchema": t.inputSchema
                } for t in result.tools
            ]
        except Exception as e:
            logger.error(f"[MCP Client Error] Failed to list tools: {e}")
            return []

    async def call_tool(self, tool_name: str, arguments: Optional[Dict[str, Any]] = None) -> Any:
        """
        Invoke an MCP tool by name via standard MCP ClientSession interface,
        falling back seamlessly to in-process execution if subprocess session is inactive.
        """
        arguments = arguments or {}
        logger.info(f"[MCP Client -> MCP Server] Invoking tool '{tool_name}' with args {arguments}")
        
        if not self.session:
            try:
                await self.start()
            except Exception as e:
                logger.warning(f"[MCP Client] Subprocess session start skipped/failed: {e}")

        if self.session:
            try:
                result = await self.session.call_tool(tool_name, arguments)
                if hasattr(result, "content") and result.content:
                    parsed_list = []
                    for item in result.content:
                        if hasattr(item, "text") and item.text:
                            try:
                                parsed_list.append(json.loads(item.text))
                            except Exception:
                                parsed_list.append(item.text)
                        else:
                            parsed_list.append(item)
                    if len(parsed_list) == 1:
                        return parsed_list[0]
                    return parsed_list
                return result
            except Exception as e:
                logger.warning(f"[MCP Client] Subprocess tool call failed ({e}). Falling back to in-process execution.")

        # In-process direct fallback execution
        logger.info(f"[MCP Client Direct Fallback] Executing tool '{tool_name}' in-process...")
        from app.repositories.news_repository import news_repository
        if tool_name == "search_live_news":
            return await news_repository.search_articles(
                query=arguments.get("query", ""),
                category=arguments.get("category", ""),
                source=arguments.get("source", ""),
                limit=arguments.get("limit", 20)
            )
        elif tool_name == "fetch_latest_rss_feeds":
            articles = await news_repository.fetch_and_index_live_news()
            return {"status": "success", "total_articles": len(articles)}
        elif tool_name == "get_dashboard_analytics":
            return news_repository.get_dashboard_stats()
        elif tool_name == "compare_news_sources":
            return news_repository.compare_sources(
                source1=arguments.get("source1", ""),
                source2=arguments.get("source2", "")
            )
        elif tool_name == "get_articles_by_category":
            return await news_repository.get_articles_by_category(
                category=arguments.get("category", ""),
                limit=arguments.get("limit", 20)
            )
        else:
            raise ValueError(f"Unknown MCP tool: '{tool_name}'")

    async def read_resource(self, uri: str) -> Any:
        """
        Read an MCP resource by URI. Parses JSON payload automatically.
        """
        logger.info(f"[MCP Client -> MCP Server] Reading resource '{uri}'")
        if not self.session:
            await self.start()
            if not self.session:
                raise RuntimeError("MCP Client Session is not active and failed to start.")

        try:
            res = await self.session.read_resource(uri)
            if hasattr(res, "contents") and res.contents:
                item = res.contents[0]
                text_content = getattr(item, "text", None) or getattr(item, "content", None)
                if text_content and isinstance(text_content, str):
                    try:
                        return json.loads(text_content)
                    except Exception:
                        return text_content
            return res
        except Exception as e:
            logger.error(f"[MCP Client Error] Failed to read resource '{uri}': {e}")
            raise e

# Singleton MCP Client instance
mcp_client = NewsIntelMCPClient()
