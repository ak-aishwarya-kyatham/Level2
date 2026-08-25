import pytest

from app.agents.policy_agent import (
    parse_and_validate_policy_action,
)

pytestmark = pytest.mark.unit

SAMPLE_TOOLS = [
    {
        "name": "search_live_news",
        "description": "Search live news",
        "inputSchema": {
            "type": "object",
            "properties": {
                "query": {"type": "string"},
                "category": {"type": "string"},
                "limit": {"type": "integer", "minimum": 1, "maximum": 100}
            },
            "required": ["query"]
        }
    },
    {
        "name": "compare_news_sources",
        "description": "Compare two sources",
        "inputSchema": {
            "type": "object",
            "properties": {
                "source1": {"type": "string"},
                "source2": {"type": "string"}
            },
            "required": ["source1", "source2"]
        }
    },
    {
        "name": "get_dashboard_analytics",
        "description": "Get analytics",
        "inputSchema": {
            "type": "object",
            "properties": {}
        }
    }
]

def test_1_valid_tool_action():
    """Test 1: Valid tool action passes validation cleanly."""
    raw = '{"action": "tool", "tool": "search_live_news", "arguments": {"query": "AI news", "limit": 10}, "thought": "Searching for AI news"}'
    action = parse_and_validate_policy_action(raw, SAMPLE_TOOLS)
    assert action.is_valid is True
    assert action.action == "tool"
    assert action.tool == "search_live_news"
    assert action.arguments == {"query": "AI news", "limit": 10}
    assert action.validation_error is None

def test_2_valid_finish_action():
    """Test 2: Valid finish action passes validation cleanly."""
    raw = '{"action": "finish", "answer": "Here is the final news summary.", "thought": "Synthesized complete answer"}'
    action = parse_and_validate_policy_action(raw, SAMPLE_TOOLS)
    assert action.is_valid is True
    assert action.action == "finish"
    assert action.answer == "Here is the final news summary."
    assert action.validation_error is None

def test_3_invalid_action():
    """Test 3: Invalid action literal (e.g. 'search_news') is rejected."""
    raw = '{"action": "search_news", "tool": "search_live_news", "arguments": {"query": "test"}, "thought": "Invalid action string"}'
    action = parse_and_validate_policy_action(raw, SAMPLE_TOOLS)
    assert action.is_valid is False
    assert "INVALID OUTPUT" in action.validation_error or "Input should be 'tool' or 'finish'" in action.validation_error

def test_4_unknown_tool():
    """Test 4: Tool name not present in MCP tool schemas is rejected."""
    raw = '{"action": "tool", "tool": "fake_nonexistent_tool", "arguments": {}, "thought": "Calling non-existent tool"}'
    action = parse_and_validate_policy_action(raw, SAMPLE_TOOLS)
    assert action.is_valid is False
    assert "INVALID TOOL: Unknown tool 'fake_nonexistent_tool'" in action.validation_error

def test_5_missing_required_argument():
    """Test 5: Tool call missing a required schema argument is rejected."""
    # compare_news_sources requires both source1 and source2
    raw = '{"action": "tool", "tool": "compare_news_sources", "arguments": {"source1": "BBC"}, "thought": "Missing source2"}'
    action = parse_and_validate_policy_action(raw, SAMPLE_TOOLS)
    assert action.is_valid is False
    assert "MISSING REQUIRED ARGUMENT: Tool 'compare_news_sources' requires argument 'source2'" in action.validation_error

def test_6_invalid_argument_type():
    """Test 6: Tool argument with wrong data type (string instead of int) is rejected."""
    raw = '{"action": "tool", "tool": "search_live_news", "arguments": {"query": "tech", "limit": "twenty"}, "thought": "Invalid limit type"}'
    action = parse_and_validate_policy_action(raw, SAMPLE_TOOLS)
    assert action.is_valid is False
    assert "INVALID ARGUMENT TYPE: Argument 'limit' for tool 'search_live_news' must be of type 'integer'" in action.validation_error

def test_7_invalid_argument_value():
    """Test 7: Tool argument violating schema limits (minimum limit < 1) is rejected."""
    raw = '{"action": "tool", "tool": "search_live_news", "arguments": {"query": "tech", "limit": -5}, "thought": "Limit below minimum"}'
    action = parse_and_validate_policy_action(raw, SAMPLE_TOOLS)
    assert action.is_valid is False
    assert "INVALID ARGUMENT VALUE: Argument 'limit' (-5) is below minimum of 1" in action.validation_error

def test_8_malformed_json_output():
    """Test 8: Malformed JSON output from LLM is caught cleanly with structured error."""
    raw = '{"action": "tool", "tool": "search_live_news", "arguments": {query: unquoted_val}'
    action = parse_and_validate_policy_action(raw, SAMPLE_TOOLS)
    assert action.is_valid is False
    assert "MALFORMED OUTPUT: Response is not valid JSON" in action.validation_error
