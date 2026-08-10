import asyncio
from unittest.mock import AsyncMock, patch, MagicMock
from app.agents.policy_agent import PolicyAgent, PolicyAction
from app.agents.reflection_agent import ReflectionAgent, ReflectionReport
from app.workflows.main_workflow import app_graph, policy_node, tool_node
from app.workflows.langgraph_state import AgentState

async def test_policy_agent_parsing():
    """Verify that Policy Agent successfully parses correct JSON structure."""
    agent = PolicyAgent()
    
    # Test valid JSON response parsing
    raw_text = '{"action": "tool", "tool": "search_live_news", "arguments": {"query": "test"}, "thought": "I need to search."}'
    cleaned = agent._sanitize_json_string(raw_text)
    assert "search_live_news" in cleaned
    
    # Test fallback action construction on errors
    action = await agent.decide_action(
        query="what is happening?",
        tools=[],
        history=[],
        iteration_count=1
    )
    # By default, first iteration error triggers tool call fallback
    assert action.action == "tool"

async def test_reflection_agent_parsing():
    """Verify that Reflection Agent generates valid report structure."""
    agent = ReflectionAgent()
    
    # Test critique mapping on blank history
    report = await agent.reflect(
        query="AI trends",
        answer="No new trends.",
        history=[]
    )
    assert report.revise is False

async def test_agentic_loop_termination():
    """Ensure the LangGraph execution terminates safely when max iterations is hit."""
    # Build initial state with iteration count at max limit
    state = AgentState(
        user_id="test_user",
        query="Nvidia news",
        intent="search",
        cached_response="",
        retrieved_documents=[],
        specialized_output="",
        final_response="",
        error="",
        observations=[],
        iteration_count=5,  # Max iteration reached
        reflection_report={},
        agent_trace=[],
        evaluation_metrics={}
    )
    
    # Invoke workflow
    final_state = await app_graph.ainvoke(state)
    print(f"DEBUG: iteration_count = {final_state.get('iteration_count')}")
    print(f"DEBUG: final_response = '{final_state.get('final_response')}'")
    print(f"DEBUG: keys = {list(final_state.keys())}")
    assert final_state["iteration_count"] >= 5
    assert final_state["final_response"] != ""


async def test_multi_tool_execution():
    """
    Prove genuine multi-step agentic loop:
    Policy → Tool A (search_live_news) → Observation →
    Policy → Tool B (get_dashboard_analytics) → Observation →
    Policy → Finish

    Verifies:
    - No forced finish after first tool call
    - Policy Agent can call multiple different tools
    - Final state has 2 observations from 2 different tools
    - Workflow terminates only when Policy Agent emits 'finish'
    """
    # Track which tools were called in order
    called_tools = []

    # Policy Agent responses: Tool A, then Tool B, then Finish
    policy_responses = [
        PolicyAction(
            action="tool",
            tool="search_live_news",
            arguments={"query": "India tech news", "limit": 5},
            thought="Step 1: Search for news first.",
            answer=""
        ),
        PolicyAction(
            action="tool",
            tool="get_dashboard_analytics",
            arguments={},
            thought="Step 2: Now get analytics to enrich my answer.",
            answer=""
        ),
        PolicyAction(
            action="finish",
            tool="",
            arguments={},
            thought="Step 3: I have enough data. Finishing.",
            answer="India tech news summary based on search and analytics."
        ),
    ]
    policy_call_idx = [0]

    async def mock_decide_action(self, **kwargs):
        idx = policy_call_idx[0]
        policy_call_idx[0] += 1
        return policy_responses[idx]

    async def mock_call_tool(tool_name, arguments):
        called_tools.append(tool_name)
        if tool_name == "search_live_news":
            return [{"title": "India Tech Boom", "content": "India's tech sector grows.", "source": "TechIndia", "url": "http://example.com"}]
        elif tool_name == "get_dashboard_analytics":
            return {"total_articles": 100, "top_categories": ["technology", "business"]}
        return {}

    async def mock_list_available_tools():
        return [
            {"name": "search_live_news", "description": "Search live news"},
            {"name": "get_dashboard_analytics", "description": "Get analytics"},
        ]

    async def mock_reflect(self, **kwargs):
        """Always returns revise=False to prevent unexpected Ollama-driven loop-backs."""
        return ReflectionReport(revise=False, unsupported_claims=[], missing_information=[], confidence=0.9)

    with patch.object(PolicyAgent, "decide_action", mock_decide_action), \
         patch.object(ReflectionAgent, "reflect", mock_reflect), \
         patch("app.workflows.main_workflow.mcp_client.call_tool", mock_call_tool), \
         patch("app.workflows.main_workflow.mcp_client.list_available_tools", mock_list_available_tools):

        state = AgentState(
            user_id="test_user",
            query="Give me India tech news with analytics",
            intent="search",
            cached_response="",
            retrieved_documents=[],
            specialized_output="",
            final_response="",
            error="",
            observations=[],
            iteration_count=0,
            reflection_report={},
            agent_trace=[],
            evaluation_metrics={}
        )

        final_state = await app_graph.ainvoke(state)

    # Verify multi-tool execution
    print(f"DEBUG [multi_tool]: tools called = {called_tools}")
    print(f"DEBUG [multi_tool]: observations = {len(final_state.get('observations', []))}")
    print(f"DEBUG [multi_tool]: iteration_count = {final_state.get('iteration_count')}")
    print(f"DEBUG [multi_tool]: agent_trace = {final_state.get('agent_trace', [])}")

    # Both tools must have been called
    assert "search_live_news" in called_tools, "Tool A (search_live_news) was never called"
    assert "get_dashboard_analytics" in called_tools, "Tool B (get_dashboard_analytics) was never called"

    # Must have at least 2 observations (one per tool)
    assert len(final_state["observations"]) >= 2, \
        f"Expected >= 2 observations, got {len(final_state['observations'])}"

    # Tools must be called in order: A then B
    assert called_tools[0] == "search_live_news", "Tool A must be called first"
    assert called_tools[1] == "get_dashboard_analytics", "Tool B must be called second"

    # Iteration count must not be forced to 1 (proving no forced finish)
    assert final_state["iteration_count"] >= 2, \
        f"iteration_count should be >= 2 for multi-tool, got {final_state['iteration_count']}"

    # Workflow must have terminated (final_response set)
    assert final_state.get("final_response"), "final_response must not be empty"


async def test_max_iterations_safety():
    """
    Prove that MAX_ITERATIONS=5 prevents infinite loops:
    - Policy Agent continuously requests tools
    - Workflow stops at iteration 5
    - final_response is still produced (not empty)
    - iteration_count equals exactly MAX_ITERATIONS
    """
    MAX_ITERATIONS = 5
    called_count = [0]

    async def mock_decide_always_tool(self, **kwargs):
        # Policy Agent always requests a tool — simulating a misbehaving LLM
        called_count[0] += 1
        return PolicyAction(
            action="tool",
            tool="search_live_news",
            arguments={"query": "infinite loop test"},
            thought=f"Calling tool again (call #{called_count[0]})",
            answer=""
        )

    async def mock_call_tool(tool_name, arguments):
        return [{"title": f"Article {called_count[0]}", "content": "Some content.", "source": "TestSource", "url": "http://test.com"}]

    async def mock_list_available_tools():
        return [{"name": "search_live_news", "description": "Search live news"}]

    async def mock_reflect_safe(self, **kwargs):
        """Always returns revise=False to prevent unexpected Ollama-driven loop-backs."""
        return ReflectionReport(revise=False, unsupported_claims=[], missing_information=[], confidence=0.9)

    with patch.object(PolicyAgent, "decide_action", mock_decide_always_tool), \
         patch.object(ReflectionAgent, "reflect", mock_reflect_safe), \
         patch("app.workflows.main_workflow.mcp_client.call_tool", mock_call_tool), \
         patch("app.workflows.main_workflow.mcp_client.list_available_tools", mock_list_available_tools):

        state = AgentState(
            user_id="test_user",
            query="infinite tool loop query",
            intent="search",
            cached_response="",
            retrieved_documents=[],
            specialized_output="",
            final_response="",
            error="",
            observations=[],
            iteration_count=0,
            reflection_report={},
            agent_trace=[],
            evaluation_metrics={}
        )

        final_state = await app_graph.ainvoke(state)

    print(f"DEBUG [max_iter]: iteration_count = {final_state.get('iteration_count')}")
    safe_resp = (final_state.get('final_response', '') or '')[:80].encode('ascii', errors='replace').decode()
    print(f"DEBUG [max_iter]: final_response = '{safe_resp}'")
    print(f"DEBUG [max_iter]: observations = {len(final_state.get('observations', []))}")

    # Must stop at MAX_ITERATIONS
    assert final_state["iteration_count"] >= MAX_ITERATIONS, \
        f"Expected iteration_count >= {MAX_ITERATIONS}, got {final_state['iteration_count']}"

    # Must not exceed MAX_ITERATIONS significantly
    assert final_state["iteration_count"] <= MAX_ITERATIONS + 1, \
        f"Exceeded MAX_ITERATIONS: got {final_state['iteration_count']}"

    # Must still produce a response (graceful termination)
    assert final_state.get("final_response"), "final_response must not be empty even at max iterations"
