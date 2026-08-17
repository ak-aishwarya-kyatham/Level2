import asyncio
import json
import time
import pytest
from unittest.mock import AsyncMock, patch, MagicMock

pytestmark = pytest.mark.integration

from app.workflows.main_workflow import app_graph
from app.workflows.langgraph_state import AgentState
from app.agents.policy_agent import PolicyAction
from app.agents.reflection_agent import ReflectionReport


class DeterministicWorkflowSimulator:
    """
    Deterministic simulator for Policy Agent & Reflection Agent boundaries.
    Proves the full agentic loop without requiring internet, Ollama, HuggingFace, or external MCP servers.
    """
    def __init__(self):
        self.policy_calls = 0
        self.reflection_calls = 0
        self.trace = []
        
        self.schemas_received = []
        self.history_received = []
        self.second_decision_depended_on_obs1 = False
        self.finish_selected = False
        self.revision_triggered = False
        self.final_grounded_answer_generated = False

    async def fake_policy_decide(self, query: str, tools: list, history: list, iteration_count: int):
        self.policy_calls += 1
        self.schemas_received.append(tools)
        self.history_received.append(list(history))

        if self.policy_calls == 1:
            # Turn 1: Policy receives schemas, chooses search_live_news tool
            self.trace.append("USER QUERY\n->")
            self.trace.append(f"POLICY: tool=search_live_news, arguments={{'query': '{query}'}}")
            return PolicyAction(
                action="tool",
                tool="search_live_news",
                arguments={"query": query, "limit": 10},
                thought="Searching live news feeds for initial articles.",
                is_valid=True
            )
        elif self.policy_calls == 2:
            # Turn 2: Policy receives Observation 1 in history, chooses compare_news_sources tool
            if len(history) >= 1 and history[0].get("tool") == "search_live_news":
                self.second_decision_depended_on_obs1 = True
            
            self.trace.append("->\nPOLICY: tool=compare_news_sources, arguments={'source1': 'The Hindu', 'source2': 'Indian Express'}")
            return PolicyAction(
                action="tool",
                tool="compare_news_sources",
                arguments={"source1": "The Hindu", "source2": "Indian Express"},
                thought="Comparing news coverage across sources based on initial observation.",
                is_valid=True
            )
        elif self.policy_calls == 3:
            # Turn 3: Policy receives Observation 1 & 2 in history, chooses finish
            self.finish_selected = True
            self.trace.append("->\nPOLICY: finish, answer='Preliminary comparative summary of Telangana housing...'")
            return PolicyAction(
                action="finish",
                answer="Preliminary comparative summary of Telangana housing allocation.",
                thought="Sufficient observations gathered to synthesize briefing.",
                is_valid=True
            )
        else:
            # Turn 4: Policy receives Reflection critique (revise=True) in history, provides revised grounded answer
            critique_obs = [h for h in history if h.get("tool") == "reflection_critique"]
            assert len(critique_obs) > 0, "Policy Agent did not receive reflection critique in history"
            
            revised_answer = (
                "## 📰 Executive Intelligence Briefing: Telangana Housing Allocation\n\n"
                "**Overview:** Telangana Housing Minister urges Centre to allocate 11.56 lakh houses under PMAY-G 2.0.\n\n"
                "**Key Details & Implications:** As reported by The Hindu and Indian Express, Telangana Housing Minister "
                "submitted a representation to Union Minister Shivraj Singh Chouhan in New Delhi requesting 11.56 lakh houses under PMAY-G 2.0. "
                "This key development reflects ongoing regional housing policy transitions and inter-governmental coordination."
            )
            self.final_grounded_answer_generated = True
            self.trace.append("->\nREVISION\n->\nPOLICY: finish (Revised), answer='REVISED: Complete grounded comparative briefing...'")
            return PolicyAction(
                action="finish",
                answer=revised_answer,
                thought="Synthesizing revised answer addressing reflection critique.",
                is_valid=True
            )

    async def fake_reflection_reflect(self, query: str, answer: str, history: list):
        self.reflection_calls += 1
        if self.reflection_calls == 1:
            # Turn 1 Reflection: Output critique with revise=True
            self.revision_triggered = True
            self.trace.append("->\nRESPONSE SYNTHESIS\n->\nREFLECTION: revise=True, status=REVISED, unsupported_claims=['Initial claim missing numerical house count details']")
            return ReflectionReport(
                revise=True,
                reflection_status="REVISED",
                fallback_used=False,
                unsupported_claims=["Initial claim missing numerical house count details"],
                missing_information=["Include exact 11.56 lakh house count figures from observations."],
                suggested_revision="Add explicit 11.56 lakh house allocation details."
            )
        else:
            # Turn 2 Reflection: Validated revised answer with revise=False
            self.trace.append("->\nREFLECTION: revise=False, status=PASSED\n->\nFINAL ANSWER")
            return ReflectionReport(
                revise=False,
                reflection_status="PASSED",
                fallback_used=False,
                unsupported_claims=[],
                missing_information=[],
                suggested_revision=""
            )


@pytest.mark.asyncio
async def test_full_agent_workflow_end_to_end():
    """
    Integration test proving the complete 11-stage agentic workflow:
    User Query -> Policy Agent (Schemas & History) -> MCP Tool -> Observation ->
    Policy Decision 2 (Observation Dependent) -> MCP Tool 2 -> Policy Finish ->
    Response Synthesis -> Reflection Agent -> Revision (revise=True) -> Final Grounded Response.
    """
    simulator = DeterministicWorkflowSimulator()

    # Fake tool definitions
    fake_tools = [
        {
            "name": "search_live_news",
            "description": "Search live media news articles",
            "parameters": {"type": "object", "properties": {"query": {"type": "string"}}}
        },
        {
            "name": "compare_news_sources",
            "description": "Compare coverage across news outlets",
            "parameters": {"type": "object", "properties": {"source1": {"type": "string"}, "source2": {"type": "string"}}}
        }
    ]

    async def fake_call_tool(name: str, args: dict):
        if name == "search_live_news":
            simulator.trace.append("->\nOBSERVATION 1: Retrieved 1 live article (Telangana PMAY-G 2.0 Housing)")
            return [
                {
                    "title": "Allocate 11.56 lakh houses to Telangana under PMAY-G 2.0",
                    "content": "Telangana Housing Minister urges Centre to allocate 11.56 lakh houses under PMAY-G 2.0 scheme.",
                    "source": "The Hindu",
                    "url": "https://www.thehindu.com/news/national/telangana/article71337811.ece",
                    "category": "Government & Policy",
                    "published_date": "2026-08-13T17:00:00Z"
                }
            ]
        elif name == "compare_news_sources":
            simulator.trace.append("->\nOBSERVATION 2: Retrieved comparative analysis between The Hindu & Indian Express")
            return {
                "source1": "The Hindu",
                "source2": "Indian Express",
                "common_news": [
                    {
                        "source1_title": "Allocate 11.56 lakh houses to Telangana under PMAY-G 2.0",
                        "source2_title": "Telangana seeks legal opinion on Family Register Certificate",
                        "summary": "Both outlets cover major Telangana governance and policy developments."
                    }
                ]
            }
        return {"error": "Unknown tool"}

    with patch("app.workflows.main_workflow.policy_agent.decide_action", side_effect=simulator.fake_policy_decide), \
         patch("app.workflows.main_workflow.reflection_agent.reflect", side_effect=simulator.fake_reflection_reflect), \
         patch("app.agents.response.synthesize_executive_summary", side_effect=lambda q, docs, llm_sum, **kw: llm_sum or "Preliminary comparative summary of Telangana housing allocation."), \
         patch("app.workflows.main_workflow.mcp_client.list_available_tools", new_callable=AsyncMock, return_value=fake_tools), \
         patch("app.workflows.main_workflow.mcp_client.call_tool", side_effect=fake_call_tool), \
         patch("app.workflows.main_workflow.cache_get", return_value=None):

        initial_state = AgentState(
            user_id="test_user_flow",
            query="Compare Telangana news coverage across Indian Express and The Hindu",
            intent="",
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

        final_state = await app_graph.ainvoke(initial_state)

    # Print and Save Sanitized Execution Trace
    import os
    trace_file = os.path.join(os.path.dirname(__file__), "sanitized_agent_trace.txt")
    with open(trace_file, "w", encoding="utf-8") as f:
        f.write("========== AGENT TRACE ==========\n\n")
        for step in simulator.trace:
            f.write(step + "\n")
        f.write("\n==================================\n")

    print("\n" + "=" * 70)
    print("SANITIZED AGENTIC WORKFLOW EXECUTION TRACE")
    print("=" * 70)
    for step in simulator.trace:
        print(step)
    print("=" * 70 + "\n")

    # STAGE ASSERTIONS: Proving all 11 requirements
    # 1. Policy Agent receives tool schemas
    assert len(simulator.schemas_received[0]) == 2, "Stage 1 Failed: Policy Agent did not receive tool schemas"

    # 2. Policy Agent receives observation history
    assert len(simulator.history_received[1]) >= 1, "Stage 2 Failed: Policy Agent did not receive observation history"

    # 3. First action selects a tool
    assert simulator.history_received[1][0]["tool"] == "search_live_news", "Stage 3 Failed: First action did not select search_live_news"

    # 4. MCP/tool execution produces an observation
    assert len(final_state["observations"]) >= 2, "Stage 4 Failed: MCP execution did not produce observations"

    # 5. Second Policy decision depended on observation 1
    assert simulator.second_decision_depended_on_obs1, "Stage 5 Failed: Second decision did not depend on observation 1"

    # 6. Policy Agent eventually chooses finish
    assert simulator.finish_selected, "Stage 6 Failed: Policy Agent did not select finish"

    # 7. Response synthesis receives observations
    assert len(final_state.get("retrieved_documents", [])) > 0 or len(final_state.get("observations", [])) > 0, "Stage 7 Failed: Response synthesis missing observations"

    # 8. Reflection Agent runs
    assert simulator.reflection_calls == 2, f"Stage 8 Failed: Expected 2 Reflection runs, got {simulator.reflection_calls}"

    # 9. Reflection produces revise=True
    assert simulator.revision_triggered, "Stage 9 Failed: Reflection failed to produce revise=True"

    # 10. Revision occurs (critique added to observations and policy called again)
    critique_found = any(obs.get("tool") == "reflection_critique" for obs in final_state["observations"])
    assert critique_found, "Stage 10 Failed: Reflection critique observation not found in history"

    # 11. Final response generated from observations
    assert simulator.final_grounded_answer_generated, "Stage 11 Failed: Final response not generated"
    assert "11.56 lakh" in final_state.get("final_response", ""), "Final response missing grounded details"
    assert final_state.get("evaluation_metrics", {}).get("faithfulness") is not None, "Evaluation metrics not populated"
