import json
import os
import time
from datetime import datetime, timezone

from fastapi import APIRouter

from app.mcp_client import mcp_client

router = APIRouter(prefix="/api/analytics", tags=["analytics"])

@router.get("/metrics")
async def get_analytics_metrics():
    """Route analytics metrics request via MCP resource 'news://analytics/metrics' with fallback to repository."""
    try:
        if mcp_client.is_connected:
            return await mcp_client.read_resource("news://analytics/metrics")
    except Exception:
        pass
    from app.repositories.news_repository import news_repository
    return news_repository.get_analytics_metrics()

@router.post("/evaluate")
async def run_evaluation_benchmark():
    """Triggers an actual dynamic evaluation run against live indexed news articles and records execution metrics."""
    from app.repositories.news_repository import news_repository
    from app.utils.evaluator import evaluate_execution, run_routing_benchmark
    from app.workflows.main_workflow import _synthesize_from_observations

    # Load eval dataset if available
    eval_dataset_path = os.path.join(os.path.dirname(__file__), "..", "data", "eval_dataset.json")
    eval_dataset = None
    if os.path.exists(eval_dataset_path):
        try:
            with open(eval_dataset_path, "r", encoding="utf-8") as f:
                eval_dataset = json.load(f)
        except Exception:
            eval_dataset = None

    t_start = time.perf_counter()

    # 1. Run actual routing benchmark against policy agent and eval dataset
    routing_res = await run_routing_benchmark(dataset=eval_dataset)

    # 2. Get real articles from repository
    real_articles = news_repository.get_all_articles()[:5]

    if real_articles:
        sample_title = real_articles[0].get("title", "Artificial Intelligence and Tech Developments")
        query = f"Latest news on {sample_title[:30]}"
        obs_text = "\n".join([f"{a.get('title')}: {a.get('cleaned_content') or a.get('content')}" for a in real_articles])
        retrieved_docs = real_articles
        observations = [{"tool": "search_live_news", "result": obs_text}]
    else:
        query = "Artificial Intelligence tech developments"
        retrieved_docs = []
        observations = []

    # Call real synthesis step to generate actual response text
    response = _synthesize_from_observations(query, observations, intent="search")

    t_elapsed = time.perf_counter() - t_start

    # 3. Calculate metrics using real measured latency and generated response
    eval_res = evaluate_execution(
        query=query,
        response=response,
        retrieved_docs=retrieved_docs,
        observations=observations,
        intent="search",
        latency=t_elapsed,
        cache_hit=False
    )

    # 4. Use honest routing accuracy from policy agent benchmark (no flooring/clamping)
    eval_res["routing_accuracy"] = routing_res.get("routing_accuracy", 0.0)

    news_repository.record_evaluation_run({
        "query": query,
        "model_name": routing_res.get("model_name", "qwen2.5:3b"),
        "tool_calls": routing_res.get("tool_calls", []),
        "test_cases": routing_res.get("test_cases", []),
        "metrics": eval_res,
        "timestamp": datetime.now(timezone.utc).isoformat()
    })

    return {
        "status": "success",
        "message": "Dynamic evaluation benchmark executed and recorded.",
        "metrics": eval_res
    }

