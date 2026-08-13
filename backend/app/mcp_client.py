import os
import sys
import json
import logging
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
        Spawns the MCP Server as a subprocess and establishes a client session.
        """
        if self.session is not None:
            return
        
        logger.info("Starting MCP Server subprocess and establishing session...")
        backend_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        server_params = StdioServerParameters(
            command=sys.executable,
            args=["-u", "-m", "app.mcp_server"],
            env={**os.environ, "PYTHONPATH": backend_dir}
        )
        
        self._exit_stack = AsyncExitStack()
        try:
            read_stream, write_stream = await self._exit_stack.enter_async_context(
                stdio_client(server_params)
            )
            self.session = await self._exit_stack.enter_async_context(
                ClientSession(read_stream, write_stream)
            )
            await self.session.initialize()
            logger.info("MCP Client Session successfully initialized!")
        except Exception as e:
            logger.error(f"Failed to initialize MCP Client session: {e}", exc_info=True)
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
        """
        if not self.session:
            logger.warning("[MCP Client] Session not active, returning empty tool list.")
            return []
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
        Invoke an MCP tool by name via standard MCP ClientSession interface.
        """
        arguments = arguments or {}
        logger.info(f"[MCP Client -> MCP Server] Invoking tool '{tool_name}' with args {arguments}")
        
        if not self.session:
            # Fallback/Lazy initialization if session is missing
            await self.start()
            if not self.session:
                raise RuntimeError("MCP Client Session is not active and failed to start.")

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
            logger.error(f"[MCP Client Error] Failed to execute MCP tool '{tool_name}': {e}")
            raise e

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
