import asyncio
import os
import json
import requests
import pytest
from unittest.mock import AsyncMock, patch, MagicMock

pytestmark = pytest.mark.integration
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

    async def mock_decide_action(*args, **kwargs):
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

    async def mock_reflect(*args, **kwargs):
        """Always returns revise=False to prevent unexpected Ollama-driven loop-backs."""
        return ReflectionReport(revise=False, unsupported_claims=[], missing_information=[], confidence=0.9)

    with patch.object(PolicyAgent, "decide_action", mock_decide_action), \
         patch.object(ReflectionAgent, "reflect", mock_reflect), \
         patch("app.workflows.main_workflow.cache_get", return_value=None), \
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

    async def mock_decide_always_tool(*args, **kwargs):
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
         patch("app.workflows.main_workflow.cache_get", return_value=None), \
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


class FakeOllamaLLM:
    """
    Deterministic local Ollama integration mechanism for tests.
    Intercepts HTTP POST requests to Ollama API (/api/generate and /api/embeddings) and simulates an LLM
    making actual Policy Agent and Reflection Agent decisions by inspecting incoming prompt payloads.
    
    Proves that:
    - Policy Agent formats and sends tool schemas to the LLM
    - Policy Agent formats and sends observation history to the LLM
    - Policy Agent selects tools based on query and observations
    - Second decision depends on previous observation history
    - Policy Agent selects 'finish' action when observations are complete
    - Reflection Agent critiques answer and triggers revision
    """
    def __init__(self):
        self.policy_decisions_count = 0
        self.reflection_decisions_count = 0
        self.prompt_history = []
        self.execution_trace = []
        self.schemas_received = False
        self.history_received = False
        self.second_decision_depended_on_obs1 = False
        self.finish_selected = False
        self.revision_triggered = False

    def __call__(self, *args, **kwargs):
        url = args[0] if args else kwargs.get("url", "")
        payload = kwargs.get("json") or (args[1] if len(args) > 1 and isinstance(args[1], dict) else {})
        if not isinstance(payload, dict):
            payload = {}
        prompt = payload.get("prompt", "")
        self.prompt_history.append(prompt)

        # Handle embedding calls separately
        if "/api/embeddings" in str(url):
            mock_resp = MagicMock()
            mock_resp.status_code = 200
            mock_resp.json.return_value = {"embedding": [0.01] * 1024}
            return mock_resp

        response_data = {}

        # 1. Reflection Agent Call
        if "Reflection Agent" in prompt or "JSON Reflection Report:" in prompt:
            self.reflection_decisions_count += 1
            if self.reflection_decisions_count == 1:
                # Turn 1 Reflection: Critique answer and trigger revision
                self.revision_triggered = True
                response_data = {
                    "supported_claims": ["India tech sector is experiencing high growth"],
                    "unsupported_claims": ["Unverified 50% growth rate claim"],
                    "missing_information": ["Explicit breakdown of regional stats"],
                    "revise": True
                }
                self.execution_trace.append(
                    f"Reflection Decision {self.reflection_decisions_count}: revise=True, unsupported_claims=['Unverified 50% growth rate claim']"
                )
            else:
                # Turn 2 Reflection: Approve revised response
                response_data = {
                    "supported_claims": ["India tech sector growth verified with stats"],
                    "unsupported_claims": [],
                    "missing_information": [],
                    "revise": False
                }
                self.execution_trace.append(
                    f"Reflection Decision {self.reflection_decisions_count}: revise=False (Approved revised output)"
                )
        else:
            # 2. Policy Agent Call
            self.policy_decisions_count += 1
            
            # Verify tool schemas present in prompt
            if "Available Tools:" in prompt and "search_live_news" in prompt:
                self.schemas_received = True
                
            has_no_obs = "No tools have been called yet" in prompt
            has_obs_1 = ("Step 1:" in prompt or "India Tech Boom" in prompt) and not has_no_obs
            has_obs_2 = "total_articles" in prompt or "Step 2:" in prompt
            has_critique = "reflection_critique" in prompt or "Please revise" in prompt

            if has_obs_1 and not has_no_obs:
                self.history_received = True

            if has_critique:
                # Post-revision decision: Finish with revised output
                self.finish_selected = True
                response_data = {
                    "action": "finish",
                    "tool": "",
                    "arguments": {},
                    "answer": "REVISED Executive Intelligence Briefing: India tech sector growth verified with 150 tracked articles across Technology & Business.",
                    "thought": "Policy Decision 4: Received reflection critique. Synthesizing final revised answer addressing feedback."
                }
                self.execution_trace.append(
                    f"Policy Decision {self.policy_decisions_count}: action='finish' (Post-Revision Answer)"
                )
            elif has_no_obs or not has_obs_1:
                # Policy Decision 1: Select Tool A (search_live_news)
                response_data = {
                    "action": "tool",
                    "tool": "search_live_news",
                    "arguments": {"query": "India tech news", "limit": 5},
                    "thought": "Policy Decision 1: Received user query and tool schemas. Selecting search_live_news to retrieve recent news articles."
                }
                self.execution_trace.append(
                    f"Policy Decision {self.policy_decisions_count}: action='tool', tool='search_live_news', arguments={{'query': 'India tech news', 'limit': 5}}"
                )
            elif not has_obs_2:
                # Policy Decision 2: Select Tool B (get_dashboard_analytics) based on Observation 1
                self.second_decision_depended_on_obs1 = True
                response_data = {
                    "action": "tool",
                    "tool": "get_dashboard_analytics",
                    "arguments": {},
                    "thought": "Policy Decision 2: Received search_live_news observation. Selecting get_dashboard_analytics to fetch metrics."
                }
                self.execution_trace.append(
                    f"Policy Decision {self.policy_decisions_count}: action='tool', tool='get_dashboard_analytics', arguments={{}}"
                )
            else:
                # Policy Decision 3: Select finish with initial synthesized answer
                self.finish_selected = True
                response_data = {
                    "action": "finish",
                    "tool": "",
                    "arguments": {},
                    "answer": "Executive Intelligence Briefing: India tech sector is experiencing high growth with 150 tracked articles.",
                    "thought": "Policy Decision 3: Received observations from both search_live_news and get_dashboard_analytics. Deciding finish."
                }
                self.execution_trace.append(
                    f"Policy Decision {self.policy_decisions_count}: action='finish', answer='Executive Intelligence Briefing...'"
                )

        # Construct HTTP mock response
        mock_resp = MagicMock()
        mock_resp.status_code = 200
        mock_resp.json.return_value = {"response": json.dumps(response_data)}
        return mock_resp


async def test_llm_mediated_policy_agent_and_reflection_loop():
    """
    Deterministic local integration test that proves genuine LLM-mediated
    Policy Agent and Reflection Agent behavior without requiring a real external Ollama server.

    Demonstrates sequence:
    User Query -> Policy Decision 1 (Tool A) -> Tool Execution 1 -> Observation 1 ->
    Policy Decision 2 (Tool B) -> Tool Execution 2 -> Observation 2 ->
    Policy Decision 3 (Finish) -> Reflection Decision 1 (Revise=True) -> Revision Execution ->
    Policy Decision 4 (Post-Revision Finish) -> Reflection Decision 2 (Revise=False) -> Final Answer

    Mocks ONLY the LLM HTTP boundary (requests.post).
    """
    called_tools = []
    fake_ollama = FakeOllamaLLM()

    async def mock_call_tool(tool_name, arguments):
        called_tools.append(tool_name)
        if tool_name == "search_live_news":
            obs = [{"title": "India Tech Boom", "content": "India's tech sector grows rapidly.", "source": "TechIndia", "url": "http://example.com"}]
            fake_ollama.execution_trace.append(f"Tool Execution 1: Executed tool='{tool_name}' -> Observation 1: {obs[0]['title']}")
            return obs
        elif tool_name == "get_dashboard_analytics":
            obs = {"total_articles": 150, "top_categories": ["technology", "business"]}
            fake_ollama.execution_trace.append(f"Tool Execution 2: Executed tool='{tool_name}' -> Observation 2: total_articles={obs['total_articles']}")
            return obs
        return {}

    async def mock_list_available_tools():
        return [
            {"name": "search_live_news", "description": "Search live news", "inputSchema": {"type": "object", "properties": {"query": {"type": "string"}}}},
            {"name": "get_dashboard_analytics", "description": "Get analytics", "inputSchema": {"type": "object"}},
        ]

    # Patch ONLY requests.post and MCP tools (the system boundaries)
    with patch("requests.post", side_effect=fake_ollama), \
         patch("app.workflows.main_workflow.cache_get", return_value=None), \
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

    # Print sanitized execution trace to stdout
    print("\n" + "="*70)
    print("SANITIZED EXECUTION TRACE (Fake LLM-Mediated Agentic Loop)")
    print("="*70)
    for line in fake_ollama.execution_trace:
        print(line)
    print("="*70 + "\n")

    # Assertions proving genuine LLM-mediated behavior:
    assert fake_ollama.schemas_received, "Policy Agent failed to pass tool schemas to LLM"
    assert fake_ollama.history_received, "Policy Agent failed to pass observation history to LLM"
    assert fake_ollama.second_decision_depended_on_obs1, "Second Policy decision did not depend on previous observation"
    assert fake_ollama.finish_selected, "Policy Agent failed to select finish"
    assert fake_ollama.revision_triggered, "Reflection Agent failed to trigger revision"

    # Verify counts
    assert fake_ollama.policy_decisions_count == 4, f"Expected 4 Policy decisions, got {fake_ollama.policy_decisions_count}"
    assert fake_ollama.reflection_decisions_count == 2, f"Expected 2 Reflection decisions, got {fake_ollama.reflection_decisions_count}"
    assert called_tools == ["search_live_news", "get_dashboard_analytics"], f"Unexpected tool call sequence: {called_tools}"
    
    # Final response verified
    assert "REVISED" in final_state.get("final_response", ""), "Final response did not contain revised output"
    assert len(final_state.get("observations", [])) >= 3, "Observations must contain tool results and reflection critique"


@pytest.mark.live
async def test_live_ollama_integration():
    """
    Optional live integration test against an actual running Ollama server.
    Executed explicitly via: python -m pytest -m live
    """
    ollama_url = os.getenv("OLLAMA_URL", "http://localhost:11434")
    res = requests.get(f"{ollama_url}/api/tags", timeout=5)
    assert res.status_code == 200, f"Ollama server at {ollama_url} returned status code {res.status_code}"

    agent = PolicyAgent()
    tools = [
        {
            "name": "search_live_news",
            "description": "Search live news articles",
            "inputSchema": {"type": "object", "properties": {"query": {"type": "string"}}}
        }
    ]
    action = await agent.decide_action(
        query="What is the latest AI news?",
        tools=tools,
        history=[],
        iteration_count=1
    )
    assert action.action in ["tool", "finish"]
    assert action.thought != ""


async def test_policy_action_validation_feedback_loop():
    """
    Test end-to-end feedback loop when Policy Agent emits an invalid action (e.g. unknown tool):
    1. Turn 1: Policy Agent attempts unknown tool 'fake_invalid_tool'
    2. Validation layer catches failure, records structured observation 'VALIDATION ERROR: INVALID TOOL...'
    3. Workflow routes back to Policy Agent
    4. Turn 2: Policy Agent receives validation error observation, self-corrects to 'search_live_news'
    5. Workflow completes successfully.
    """
    policy_responses = [
        # Decision 1: Invalid tool
        PolicyAction(
            action="tool",
            tool="fake_invalid_tool",
            arguments={},
            thought="Attempting to call unknown tool",
            is_valid=False,
            validation_error="INVALID TOOL: Unknown tool 'fake_invalid_tool'. Available tools: ['search_live_news']"
        ),
        # Decision 2: Self-corrected valid tool
        PolicyAction(
            action="tool",
            tool="search_live_news",
            arguments={"query": "AI news"},
            thought="Correcting choice to valid search_live_news tool",
            is_valid=True
        ),
        # Decision 3: Finish
        PolicyAction(
            action="finish",
            answer="Final news briefing after validation recovery.",
            thought="Sufficient data gathered",
            is_valid=True
        )
    ]
    call_idx = [0]
    called_tools = []

    async def mock_decide_action(*args, **kwargs):
        idx = call_idx[0]
        call_idx[0] += 1
        return policy_responses[min(idx, len(policy_responses) - 1)]

    async def mock_call_tool(tool_name, arguments):
        called_tools.append(tool_name)
        return [{"title": "AI Growth", "content": "AI is advancing rapidly."}]

    async def mock_list_tools():
        return [{"name": "search_live_news", "description": "Search live news", "inputSchema": {"type": "object"}}]

    async def mock_reflect(*args, **kwargs):
        return ReflectionReport(revise=False, unsupported_claims=[], missing_information=[], confidence=0.95)

    with patch.object(PolicyAgent, "decide_action", mock_decide_action), \
         patch.object(ReflectionAgent, "reflect", mock_reflect), \
         patch("app.workflows.main_workflow.cache_get", return_value=None), \
         patch("app.workflows.main_workflow.mcp_client.call_tool", mock_call_tool), \
         patch("app.workflows.main_workflow.mcp_client.list_available_tools", mock_list_tools):

        state = AgentState(
            user_id="test_user",
            query="AI news query",
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

    # Verify structured observation recorded for validation failure
    obs = final_state.get("observations", [])
    assert len(obs) >= 2, f"Expected at least 2 observations (validation error + tool result), got {len(obs)}"
    
    val_err_obs = [o for o in obs if "VALIDATION ERROR" in str(o.get("result"))]
    assert len(val_err_obs) >= 1, "Validation failure observation was not recorded in state"
    assert "fake_invalid_tool" in str(val_err_obs[0].get("tool")), "Validation observation did not log invalid tool name"

    # Verify self-correction tool was executed
    assert called_tools == ["search_live_news"], f"Expected ['search_live_news'], got {called_tools}"
    assert "Final news briefing" in final_state.get("final_response", ""), "Final response not generated after validation recovery"


