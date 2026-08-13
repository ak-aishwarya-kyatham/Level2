import pytest
from fastapi.testclient import TestClient
from app.main import app

client = TestClient(app)

@pytest.mark.unit
def test_read_main():
    response = client.get("/")
    assert response.status_code == 200
    res_json = response.json()
    assert res_json["status"] == "ok"
    assert "NewsIntel AI MCP Live Intelligence Backend" in res_json["message"]
