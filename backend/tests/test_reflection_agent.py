"""
Tests for Reflection Agent fail-safe behavior.

Tests:
1. Successful reflection → valid answer → VERIFIED, revise=False
2. Reflection detects hallucination → revise=True, REVISED
3. Reflection detects unsupported claim → revise=True, REVISED
4. Reflection LLM failure → deterministic fallback runs (never revise=False by default)
5. Fallback verifier fails to ground claims → UNVERIFIED, revise=True
6. Reflection failure MUST NEVER automatically become revise=False / pass=True
"""
import pytest
from unittest.mock import patch, MagicMock

pytestmark = pytest.mark.integration
from app.agents.reflection_agent import (
    ReflectionAgent,
    ReflectionReport,
    REFLECTION_STATUS_VERIFIED,
    REFLECTION_STATUS_UNVERIFIED,
    REFLECTION_STATUS_REVISED,
)

# ---------------------------------------------------------------------------
# Shared fixtures
# ---------------------------------------------------------------------------

SAMPLE_HISTORY_WITH_DATA = [
    {
        "iteration": 1,
        "tool": "search_live_news",
        "thought": "Searching for India tech news",
        "arguments": {"query": "India tech"},
        "result": [
            {
                "title": "India launches national AI mission",
                "description": "India announced a national artificial intelligence research mission worth 10000 crore rupees.",
                "url": "https://example.com/india-ai-mission",
            },
            {
                "title": "Bengaluru startup ecosystem growing rapidly",
                "description": "Bengaluru startup companies received record venture capital investment this quarter.",
                "url": "https://example.com/bengaluru-startups",
            },
        ],
        "timestamp": 1234567890,
        "execution_time": 0.5,
    }
]

GROUNDED_ANSWER = (
    "India announced a national artificial intelligence mission with significant investment. "
    "The Bengaluru startup ecosystem received record venture capital investment this quarter. "
    "These developments highlight India's growing technology sector."
)

HALLUCINATED_ANSWER = (
    "India deployed quantum teleportation infrastructure across 500 cities last week. "
    "Martian real estate prices surged 300% following the lunar currency collapse. "
    "Extraterrestrial investors dominate the Mumbai stock exchange."
)


async def _make_llm_response(data: dict):
    """Helper to build a mock requests.post response returning JSON."""
    mock_resp = MagicMock()
    mock_resp.status_code = 200
    mock_resp.json.return_value = {"response": str(data).replace("'", '"').replace("True", "true").replace("False", "false")}
    return mock_resp


# ---------------------------------------------------------------------------
# Test 1: Successful reflection → valid answer → VERIFIED
# ---------------------------------------------------------------------------
async def test_1_successful_reflection_valid_answer():
    """LLM returns clean reflection → revise=False → status=VERIFIED."""
    llm_output = {
        "supported_claims": ["India announced a national AI mission.", "Bengaluru startups received record investment."],
        "unsupported_claims": [],
        "missing_information": [],
        "revise": False,
    }
    import json
    mock_resp = MagicMock()
    mock_resp.status_code = 200
    mock_resp.json.return_value = {"response": json.dumps(llm_output)}

    agent = ReflectionAgent()

    with patch("requests.post", return_value=mock_resp):
        report = await agent.reflect(
            query="Tell me about India tech news",
            answer=GROUNDED_ANSWER,
            history=SAMPLE_HISTORY_WITH_DATA,
        )

    assert report.revise is False
    assert report.reflection_status == REFLECTION_STATUS_VERIFIED
    assert report.fallback_used is False
    assert len(report.supported_claims) >= 1
    assert report.unsupported_claims == []
    assert report.missing_information == []


# ---------------------------------------------------------------------------
# Test 2: Reflection detects hallucination → revise=True, status=REVISED
# ---------------------------------------------------------------------------
async def test_2_reflection_detects_hallucination():
    """LLM flags hallucinated claims → revise=True → status=REVISED."""
    import json
    llm_output = {
        "supported_claims": [],
        "unsupported_claims": [
            "India deployed quantum teleportation infrastructure.",
            "Martian real estate prices surged 300%.",
        ],
        "missing_information": [],
        "revise": True,
    }
    mock_resp = MagicMock()
    mock_resp.status_code = 200
    mock_resp.json.return_value = {"response": json.dumps(llm_output)}

    agent = ReflectionAgent()

    with patch("requests.post", return_value=mock_resp):
        report = await agent.reflect(
            query="Tell me about India tech news",
            answer=HALLUCINATED_ANSWER,
            history=SAMPLE_HISTORY_WITH_DATA,
        )

    assert report.revise is True
    assert report.reflection_status == REFLECTION_STATUS_REVISED
    assert report.fallback_used is False
    assert len(report.unsupported_claims) >= 1
    assert "quantum teleportation" in " ".join(report.unsupported_claims).lower() or len(report.unsupported_claims) >= 1


# ---------------------------------------------------------------------------
# Test 3: Reflection detects unsupported claim → revise=True
# ---------------------------------------------------------------------------
async def test_3_reflection_detects_unsupported_claim():
    """LLM flags a specific unsupported claim → revise=True → status=REVISED."""
    import json
    llm_output = {
        "supported_claims": ["India announced a national AI mission."],
        "unsupported_claims": ["The article claimed 500 billion dollars in investment, which is not in the observations."],
        "missing_information": ["Exact investment figure is missing from observations."],
        "revise": True,
    }
    mock_resp = MagicMock()
    mock_resp.status_code = 200
    mock_resp.json.return_value = {"response": json.dumps(llm_output)}

    agent = ReflectionAgent()

    with patch("requests.post", return_value=mock_resp):
        report = await agent.reflect(
            query="What was the investment amount for India's AI mission?",
            answer="India's AI mission received 500 billion dollars in investment.",
            history=SAMPLE_HISTORY_WITH_DATA,
            skip_llm_if_grounded=False,
        )

    assert report.revise is True
    assert report.reflection_status == REFLECTION_STATUS_REVISED
    assert len(report.unsupported_claims) >= 1
    assert report.fallback_used is False


# ---------------------------------------------------------------------------
# Test 4: Reflection LLM failure → deterministic fallback runs
# ---------------------------------------------------------------------------
async def test_4_llm_failure_triggers_deterministic_fallback():
    """When Ollama is down, deterministic fallback verifier must run (not silent pass)."""
    agent = ReflectionAgent()

    with patch("requests.post", side_effect=ConnectionError("Ollama offline")):
        report = await agent.reflect(
            query="India tech news",
            answer=GROUNDED_ANSWER,
            history=SAMPLE_HISTORY_WITH_DATA,
            skip_llm_if_grounded=False,
        )

    # Fallback must have been invoked
    assert report.fallback_used is True
    # Status must NOT be VERIFIED without actual evidence from LLM
    # (may be VERIFIED if deterministic verifier confirms grounding, but must NOT be silent pass)
    assert report.reflection_status in (REFLECTION_STATUS_VERIFIED, REFLECTION_STATUS_UNVERIFIED)
    # The key invariant: if fallback ran, it must have actually checked claims
    if report.reflection_status == REFLECTION_STATUS_VERIFIED:
        # Grounded — all sentences matched observations
        assert len(report.supported_claims) >= 1
    else:
        # Could not ground — UNVERIFIED
        assert report.reflection_status == REFLECTION_STATUS_UNVERIFIED


# ---------------------------------------------------------------------------
# Test 5: Fallback verifier cannot ground claims → UNVERIFIED, revise=True
# ---------------------------------------------------------------------------
async def test_5_fallback_verification_fails_gives_unverified():
    """When fallback verifier finds ungroundable claims → UNVERIFIED + revise=True."""
    agent = ReflectionAgent()

    with patch("requests.post", side_effect=ConnectionError("Ollama offline")):
        report = await agent.reflect(
            query="Tell me about space colonization",
            answer=HALLUCINATED_ANSWER,  # Completely unrelated to observations
            history=SAMPLE_HISTORY_WITH_DATA,
        )

    assert report.fallback_used is True
    assert report.reflection_status == REFLECTION_STATUS_UNVERIFIED
    assert report.revise is True
    assert len(report.unsupported_claims) >= 1


# ---------------------------------------------------------------------------
# Test 6: Reflection failure must NEVER automatically become revise=False/pass=True
# ---------------------------------------------------------------------------
async def test_6_reflection_failure_never_auto_passes():
    """
    Critical invariant: When the LLM fails, the result must NEVER be:
      revise=False AND reflection_status=VERIFIED AND fallback_used=False

    That combination is the old fail-open bug we are fixing.
    """
    agent = ReflectionAgent()

    failure_scenarios = [
        ConnectionError("Connection refused"),
        TimeoutError("Read timed out"),
        Exception("Unknown error"),
    ]

    for exc in failure_scenarios:
        with patch("requests.post", side_effect=exc):
            report = await agent.reflect(
                query="AI news",
                answer="Some AI-related answer with claims about the future.",
                history=SAMPLE_HISTORY_WITH_DATA,
            )

        # The forbidden combination: silent pass after LLM failure
        is_silent_pass = (
            report.revise is False
            and report.reflection_status == REFLECTION_STATUS_VERIFIED
            and report.fallback_used is False
        )
        assert not is_silent_pass, (
            f"FAIL-OPEN BUG: After LLM failure ({exc.__class__.__name__}), "
            f"got revise=False + VERIFIED + fallback_used=False. "
            f"This is the forbidden silent-pass behavior."
        )

        # Fallback MUST have run
        assert report.fallback_used is True, (
            f"Expected fallback_used=True after LLM error ({exc.__class__.__name__}), "
            f"got fallback_used={report.fallback_used}"
        )
