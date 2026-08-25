from unittest.mock import AsyncMock, patch

import pytest

pytestmark = pytest.mark.integration

from app.agents.reflection_agent import REFLECTION_STATUS_UNVERIFIED, ReflectionReport
from app.agents.response import validate_faithfulness
from app.workflows.langgraph_state import AgentState
from app.workflows.main_workflow import _synthesize_from_observations, reflection_node


def test_response_synthesis_grounded_in_observations():
    """
    Test 1: Proves that when retrieved observations contain 'Company X announced Product Y',
    the response synthesis generates a briefing grounded in that observation, preserving source details.
    """
    observations = [
        {
            "iteration": 1,
            "tool": "search_live_news",
            "result": [
                {
                    "title": "Company X announced Product Y at annual summit",
                    "content": "Company X announced Product Y featuring advanced processing capabilities and next-gen integration.",
                    "source": "TechCrunch",
                    "url": "https://techcrunch.com/company-x-product-y"
                }
            ]
        }
    ]

    query = "Tell me more about Company X announced Product Y"
    response = _synthesize_from_observations(query, observations)

    # Assertions
    assert "Company X" in response, "Final response must include 'Company X' from observation"
    assert "Product Y" in response, "Final response must include 'Product Y' from observation"
    assert "TechCrunch" in response, "Source name TechCrunch must be preserved"
    assert "https://techcrunch.com/company-x-product-y" in response, "Source URL citation must be preserved"


def test_faithfulness_prevents_unsupported_hallucinations():
    """
    Test 2: Proves that when a summary attempts to include ungrounded claimed facts
    (e.g., forbidden buzzwords or ungrounded claims not in retrieved docs),
    the faithfulness validator strips or filters out ungrounded content.
    """
    docs = [
        {
            "title": "Company X announced Product Y",
            "content": "Company X unveiled Product Y at the tech conference in San Francisco.",
            "source": "Reuters",
            "url": "https://reuters.com/company-x"
        }
    ]

    # Summary containing ungrounded forbidden buzzword claims not present in source text
    hallucinated_summary = (
        "**Overview:** Company X announced Product Y\n\n"
        "**Key Details & Implications:** Company X introduced Product Y with neural processor architecture and zero-trust edge AI."
    )

    validated = validate_faithfulness(hallucinated_summary, docs)

    # Assertions
    assert "neural processor" not in validated.lower(), "Faithfulness validator must strip ungrounded forbidden buzzwords"
    assert "zero-trust" not in validated.lower(), "Faithfulness validator must strip ungrounded claims"
    assert "Company X" in validated, "Grounded factual content must be preserved"


def test_empty_observations_returns_safe_response():
    """
    Test 3: Proves that when no observations are available ([]),
    the response synthesis returns a safe non-hallucinated response instead of erroring or hallucinating.
    """
    query = "What is the latest quantum computing breakthrough?"
    empty_obs = []

    response = _synthesize_from_observations(query, empty_obs)

    # Assertions
    assert response != "", "Empty observations must produce a non-empty response"
    assert "No relevant observations were retrieved" in response or "No relevant live news" in response, \
        "Empty observations must produce a safe disclaimer response"


@pytest.mark.asyncio
async def test_reflection_detects_unsupported_content_and_triggers_disclaimer():
    """
    Test 4: Proves that when Reflection detects unsupported content,
    the response is marked with an explicit UNVERIFIED warning disclaimer.
    """
    initial_state = AgentState(
        user_id="test_user_reflection",
        query="Unverified claim query",
        intent="search",
        cached_response="",
        retrieved_documents=[],
        specialized_output="",
        final_response="",
        error="",
        observations=[
            {
                "iteration": 1,
                "tool": "search_live_news",
                "result": [{"title": "Unverified claim news article", "content": "Article body content", "source": "Media Source"}]
            }
        ],
        iteration_count=4,  # Max iteration hit so reflection completes without further revision
        action_answer="Prelim answer with unverified claim X.",
        reflection_report={},
        agent_trace=[],
        evaluation_metrics={}
    )

    unverified_report = ReflectionReport(
        revise=False,
        reflection_status=REFLECTION_STATUS_UNVERIFIED,
        fallback_used=True,
        unsupported_claims=["Claim X is unverified"],
        missing_information=["Missing primary source verification"],
        suggested_revision="Add unverified warning"
    )

    with patch("app.workflows.main_workflow.reflection_agent.reflect", new_callable=AsyncMock, return_value=unverified_report), \
         patch("app.workflows.main_workflow.cache_set", return_value=None):
        final_state = await reflection_node(initial_state)

    # Assertions
    assert "UNVERIFIED" in final_state["final_response"], "Final response must contain UNVERIFIED warning when reflection flags fallback/unverified status"
    assert "Potentially unverified claims" in final_state["final_response"], "Final response must list flagged unsupported claims"
    assert final_state["reflection_report"]["reflection_status"] == REFLECTION_STATUS_UNVERIFIED
