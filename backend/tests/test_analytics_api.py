import pytest
import os
from fastapi.testclient import TestClient
from app.main import app
from app.repositories.news_repository import news_repository

from unittest.mock import patch, AsyncMock

client = TestClient(app)

@pytest.fixture(autouse=True)
def isolate_eval_store(tmp_path):
    news_repository.evaluation_runs = []
    yield
    news_repository.evaluation_runs = []

@pytest.mark.unit
def test_analytics_metrics_endpoint_empty_state():
    """Verify GET /api/analytics/metrics returns 10 metrics with N/A / null when no runs exist."""
    response = client.get("/api/analytics/metrics")
    assert response.status_code == 200
    res = response.json()
    assert "evaluation_metrics" in res
    
    eval_metrics = res["evaluation_metrics"]
    expected_keys = {
        "precision_at_5",
        "mrr_at_10",
        "faithfulness",
        "groundedness",
        "answer_relevance",
        "routing_accuracy",
        "latency_seconds",
        "cache_hit_rate",
        "deduplication_recall",
        "categorization_f1"
    }
    
    returned_keys = {m["metric_key"] for m in eval_metrics}
    assert expected_keys.issubset(returned_keys), f"Missing keys: {expected_keys - returned_keys}"
    
    for metric in eval_metrics:
        assert metric["score"] is None
        assert metric["percentage"] is None
        assert metric["value_text"] == "N/A"
        assert metric["status"] == "UNAVAILABLE"
        assert "source" in metric
        assert "calculation" in metric
        assert metric["runs_count"] == 0

@pytest.mark.unit
def test_analytics_evaluate_and_metrics_flow():
    """Verify POST /api/analytics/evaluate triggers benchmark and populates metrics with timestamp & run_id."""
    dummy_article = {
        "id": "art_test_1",
        "title": "Artificial Intelligence and Tech Developments",
        "content": "Latest breakthroughs in AI, machine learning, and semiconductor chips.",
        "cleaned_content": "Latest breakthroughs in AI, machine learning, and semiconductor chips.",
        "source": "Tech Source",
        "category": "Technology"
    }
    news_repository.articles = [dummy_article]
    mock_benchmark = AsyncMock(return_value={"routing_accuracy": 0.8, "coverage": 1.0, "total_queries": 15})
    with patch("app.utils.evaluator.run_routing_benchmark", new=mock_benchmark):
        eval_resp = client.post("/api/analytics/evaluate")
        assert eval_resp.status_code == 200
        eval_data = eval_resp.json()
        assert eval_data["status"] == "success"
        assert "metrics" in eval_data
        
        metrics_resp = client.get("/api/analytics/metrics")
        assert metrics_resp.status_code == 200
        res = metrics_resp.json()
        
        eval_metrics = res["evaluation_metrics"]
        assert len(eval_metrics) >= 10
        
        for metric in eval_metrics:
            assert metric["score"] is not None
            assert metric["percentage"] is not None
            assert metric["status"] in ("PASSED", "FAILED")
            assert metric["runs_count"] >= 1
            assert "latest_run_id" in metric
            assert "timestamp" in metric
            assert "source" in metric

