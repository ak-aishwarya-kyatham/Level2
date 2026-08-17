import pytest
import os
import json
import time
from app.repositories.news_repository import NewsRepository

@pytest.fixture
def clean_repo(tmp_path):
    """Provides a fresh NewsRepository instance isolated from previous evaluation data."""
    old_eval = os.environ.get("EVAL_DATA_FILE")
    old_logs = os.environ.get("AGENT_LOGS_DATA_FILE")
    eval_file = str(tmp_path / "test_eval_runs.json")
    logs_file = str(tmp_path / "test_agent_logs.json")
    os.environ["EVAL_DATA_FILE"] = eval_file
    os.environ["AGENT_LOGS_DATA_FILE"] = logs_file
    repo = NewsRepository()
    repo.evaluation_runs = []
    repo.agent_execution_logs = []
    yield repo
    if old_eval:
        os.environ["EVAL_DATA_FILE"] = old_eval
    else:
        os.environ.pop("EVAL_DATA_FILE", None)
    if old_logs:
        os.environ["AGENT_LOGS_DATA_FILE"] = old_logs
    else:
        os.environ.pop("AGENT_LOGS_DATA_FILE", None)


def test_metrics_return_null_when_no_evaluation_runs(clean_repo):
    """
    Requirements Check:
    If a metric cannot be calculated because there is no evaluation run yet, return:
    null / unavailable / N/A rather than inventing a number.
    """
    analytics = clean_repo.get_analytics_metrics()
    eval_metrics = analytics.get("evaluation_metrics", [])
    
    assert len(eval_metrics) > 0
    for metric in eval_metrics:
        assert metric["score"] is None, f"Expected None for score in {metric['name']}, got {metric['score']}"
        assert metric["percentage"] is None, f"Expected None for percentage in {metric['name']}, got {metric['percentage']}"
        assert metric["status"] == "UNAVAILABLE"
        assert metric["value_text"] == "N/A"
        assert "source" in metric
        assert "calculation" in metric
        assert metric["runs_count"] == 0

    agent_perf = analytics.get("agent_performance", [])
    assert len(agent_perf) > 0
    for agent in agent_perf:
        assert agent["avg_latency_ms"] is None
        assert agent["success_rate"] is None
        assert agent["status"] == "UNAVAILABLE"
        assert agent["execution_count"] == 0


def test_metrics_calculated_from_actual_evaluation_data(clean_repo):
    """
    Requirements Check:
    When evaluation data is recorded, dashboard metrics must be computed strictly
    from the actual evaluation data.
    """
    # 1. Record an actual evaluation run
    sample_run = {
        "run_id": "run_test_1001",
        "query": "Test evaluation query",
        "metrics": {
            "precision_at_5": 0.80,
            "faithfulness": 0.90,
            "routing_accuracy": 1.0,
            "latency_seconds": 1.5,
            "deduplication_recall": 0.95,
            "categorization_f1": 0.88
        }
    }
    clean_repo.record_evaluation_run(sample_run)
    
    # 2. Record actual agent execution log
    clean_repo.record_agent_execution("RAG Search & Retrieval Agent", latency_ms=150.0, success=True)
    clean_repo.record_agent_execution("RAG Search & Retrieval Agent", latency_ms=250.0, success=True)

    analytics = clean_repo.get_analytics_metrics()
    eval_metrics = {m["metric_key"]: m for m in analytics["evaluation_metrics"]}

    # Verify precision_at_5 comes directly from actual evaluation run (0.80 -> 80.0%)
    assert eval_metrics["precision_at_5"]["score"] == 0.80
    assert eval_metrics["precision_at_5"]["percentage"] == 80.0
    assert eval_metrics["precision_at_5"]["status"] == "PASSED"
    assert eval_metrics["precision_at_5"]["runs_count"] == 1
    assert eval_metrics["precision_at_5"]["latest_run_id"] == "run_test_1001"

    # Verify faithfulness comes directly from actual evaluation run (0.90 -> 90.0%)
    assert eval_metrics["faithfulness"]["score"] == 0.90
    assert eval_metrics["faithfulness"]["percentage"] == 90.0
    assert eval_metrics["faithfulness"]["status"] == "PASSED"

    # Verify agent performance calculated dynamically (avg(150, 250) = 200ms)
    agent_map = {a["agent"]: a for a in analytics["agent_performance"]}
    rag_agent = agent_map["RAG Search & Retrieval Agent"]
    assert rag_agent["avg_latency_ms"] == 200.0
    assert rag_agent["success_rate"] == 100.0
    assert rag_agent["execution_count"] == 2
    assert rag_agent["status"] == "OPERATIONAL"
