import logging
from app.workflows.langgraph_state import AgentState

logger = logging.getLogger(__name__)

def triage_agent(state: AgentState) -> AgentState:
    """
    Analyzes user intent and routes to the appropriate agent.
    """
    logger.info("Triage Agent analyzing intent...")
    query = state.get("query", "").lower()
    
    # Basic keyword-based intent extraction (Can be replaced with LLM call)
    if "compare" in query or "vs" in query:
        state["intent"] = "compare"
    elif "trend" in query or "trending" in query:
        state["intent"] = "trend"
    elif "summarize" in query or "summary" in query:
        state["intent"] = "summarize"
    else:
        state["intent"] = "search"
        
    logger.info(f"Intent determined as: {state['intent']}")
    return state
