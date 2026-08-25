"""
DEPRECATED MODULE:
These specialized routing agent stubs were part of the initial fixed-graph architecture.
The active Level 2 agent architecture executes via LangGraph dynamic ReAcT loop:
  cache_node -> policy_node (PolicyAgent) <-> tool_node (MCP Client) -> reflection_node (ReflectionAgent) -> END
Retained for backwards compatibility.
"""
import logging
from app.workflows.langgraph_state import AgentState

logger = logging.getLogger(__name__)

def summary_agent(state: AgentState) -> AgentState:
    logger.debug("[Deprecated] Summary Agent called.")
    state["specialized_output"] = "Requesting summary from LLM."
    return state

def compare_agent(state: AgentState) -> AgentState:
    logger.debug("[Deprecated] Compare Agent called.")
    state["specialized_output"] = "Requesting comparison from LLM."
    return state

def trend_agent(state: AgentState) -> AgentState:
    logger.debug("[Deprecated] Trend Agent called.")
    state["specialized_output"] = "Requesting trend analysis from LLM."
    return state

def translation_agent(state: AgentState) -> AgentState:
    logger.debug("[Deprecated] Translation Agent called.")
    return state
