import asyncio
from app.agents.policy_agent import PolicyAgent, PolicyAction
from app.agents.reflection_agent import ReflectionAgent, ReflectionReport
from app.workflows.main_workflow import app_graph
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
