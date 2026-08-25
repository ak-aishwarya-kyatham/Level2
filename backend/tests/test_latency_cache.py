import asyncio
import os
import time

import pytest

pytestmark = pytest.mark.integration
from app.utils.evaluator import evaluate_execution
from app.workflows.langgraph_state import AgentState
from app.workflows.main_workflow import cache_node, reflection_node


async def test_actual_latency_measurement():
    """Test 1: Verify latency is generated from real execution timing and is not a hardcoded value."""
    start = time.perf_counter()
    await asyncio.sleep(0.05)
    elapsed = time.perf_counter() - start

    metrics = evaluate_execution(
        query="test query",
        response="test response",
        retrieved_docs=[],
        observations=[],
        intent="search",
        latency=elapsed,
        cache_hit=False
    )
    assert metrics["latency_seconds"] >= 0.04
    assert metrics["latency_seconds"] != 2.0
    assert metrics["latency_seconds"] != 0.5

async def test_multistep_latency_measurement():
    """Test 2: Verify latency includes complete multi-step agent workflow timing."""
    start_t = time.perf_counter()
    state = AgentState(
        user_id="test",
        query="test query",
        intent="search",
        cached_response="",
        retrieved_documents=[],
        specialized_output="",
        final_response="multistep answer",
        error="",
        extracted_topic="",
        extracted_entities=[],
        expanded_query="",
        target_category="",
        target_url="",
        requested_limit=10,
        observations=[{"tool": "search_live_news", "result": "news data"}],
        iteration_count=2,
        reflection_report={},
        agent_trace=["step 1", "step 2"],
        evaluation_metrics={},
        next_action="finish",
        action_answer="multistep answer",
        action_thought="thought",
        action_tool="",
        action_arguments={},
        start_time=start_t,
        cache_hit=False
    )

    await asyncio.sleep(0.08)
    res_state = await reflection_node(state)
    metrics = res_state.get("evaluation_metrics", {})
    assert metrics.get("latency_seconds", 0) >= 0.07

def test_cache_hit_propagation_true():
    """Test 3: Verify Redis HIT sets cache_hit=True."""
    from app.utils.redis_cache import _fallback_memory_cache, generate_cache_key
    _fallback_memory_cache[generate_cache_key("unique_latency_test_query")] = "cached answer"


    state = AgentState(
        user_id="test",
        query="unique_latency_test_query",
        intent="search",
        cached_response="",
        retrieved_documents=[],
        specialized_output="",
        final_response="",
        error="",
        extracted_topic="",
        extracted_entities=[],
        expanded_query="",
        target_category="",
        target_url="",
        requested_limit=10,
        observations=[],
        iteration_count=0,
        reflection_report={},
        agent_trace=[],
        evaluation_metrics={},
        next_action="",
        action_answer="",
        action_thought="",
        action_tool="",
        action_arguments={},
        start_time=time.perf_counter(),
        cache_hit=False
    )

    loop = asyncio.new_event_loop()
    try:
        res = loop.run_until_complete(cache_node(state))
    finally:
        loop.close()

    assert res.get("cache_hit") is True
    assert res.get("evaluation_metrics", {}).get("cache_hit") is True

def test_cache_miss_propagation_false():
    """Test 4: Verify Redis MISS sets cache_hit=False."""
    state = AgentState(
        user_id="test",
        query="non_existent_query_xyz_123",
        intent="search",
        cached_response="",
        retrieved_documents=[],
        specialized_output="",
        final_response="",
        error="",
        extracted_topic="",
        extracted_entities=[],
        expanded_query="",
        target_category="",
        target_url="",
        requested_limit=10,
        observations=[],
        iteration_count=0,
        reflection_report={},
        agent_trace=[],
        evaluation_metrics={},
        next_action="",
        action_answer="",
        action_thought="",
        action_tool="",
        action_arguments={},
        start_time=time.perf_counter(),
        cache_hit=False
    )

    loop = asyncio.new_event_loop()
    try:
        res = loop.run_until_complete(cache_node(state))
    finally:
        loop.close()

    assert res.get("cache_hit") is False

def test_evaluator_receives_real_cache_state():
    """Test 5: Verify evaluator receives actual cache_hit status."""
    metrics_hit = evaluate_execution(
        query="q",
        response="a",
        retrieved_docs=[],
        observations=[],
        intent="search",
        latency=0.01,
        cache_hit=True
    )
    metrics_miss = evaluate_execution(
        query="q",
        response="a",
        retrieved_docs=[],
        observations=[],
        intent="search",
        latency=0.5,
        cache_hit=False
    )
    assert metrics_hit["cache_hit"] is True
    assert metrics_miss["cache_hit"] is False

def test_no_artificial_timing_in_codebase():
    """Test 6: Verify no time.time() - 2.0 or hardcoded offsets remain in main_workflow.py."""
    workflow_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), "app", "workflows", "main_workflow.py")
    with open(workflow_path, "r", encoding="utf-8") as f:
        content = f.read()

    assert "time.time() - 2.0" not in content
    assert "time.time() - 1.0" not in content
    assert "start_time=time.time() -" not in content
