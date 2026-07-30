import logging
from app.workflows.langgraph_state import AgentState

logger = logging.getLogger(__name__)

def summary_agent(state: AgentState) -> AgentState:
    logger.info("Summary Agent structuring summary request...")
    state["specialized_output"] = "Requesting summary from LLM."
    return state

def compare_agent(state: AgentState) -> AgentState:
    logger.info("Compare Agent structuring comparison request...")
    state["specialized_output"] = "Requesting comparison from LLM."
    return state

def trend_agent(state: AgentState) -> AgentState:
    logger.info("Trend Agent analyzing topics...")
    state["specialized_output"] = "Requesting trend analysis from LLM."
    return state

def translation_agent(state: AgentState) -> AgentState:
    logger.info("Translation Agent active...")
    # Optional translation logic if required by intent or user prefs
    return state
