import pytest
from app.main import read_root

@pytest.mark.unit
@pytest.mark.asyncio
async def test_read_main():
    res_json = await read_root()
    assert isinstance(res_json, dict)
    assert res_json["status"] == "ok"
    assert "NewsIntel AI MCP Live Intelligence Backend" in res_json["message"]
    assert "available_mcp_tools" in res_json
    assert len(res_json["available_mcp_tools"]) >= 4
