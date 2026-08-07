import logging
from app.workflows.langgraph_state import AgentState

logger = logging.getLogger(__name__)

import re

def triage_agent(state: AgentState) -> AgentState:
    """
    Analyzes user intent and routes to the appropriate agent.
    """
    logger.info("Triage Agent analyzing intent...")
    query = state.get("query", "").lower().strip()
    
    # Extract requested number of items if specified (e.g. "top 1", "top 3", "first 5", "1 feed")
    limit_match = re.search(r'\b(?:top|first|only|limit)\s*(\d+)\b', query) or re.search(r'\b(\d+)\s*(?:items?|feeds?|articles?|topics?)\b', query)
    if limit_match:
        try:
            state["requested_limit"] = int(limit_match.group(1))
        except Exception:
            state["requested_limit"] = 10
    else:
        state["requested_limit"] = 10

    # Keyword-based intent extraction
    if "summarize" in query or "summary" in query or "briefing" in query:
        state["intent"] = "summarize"
    elif "compare" in query or " vs " in query:
        state["intent"] = "compare"
    elif re.search(r'\b(?:tr[ee]nd(?:ing|nign)?|trnd|top topics?|topics? of)\b', query):
        state["intent"] = "trend"
    elif any(k in query for k in ["live feed", "live feeds", "news feed", "real-time feed", "latest feed"]) or \
         query in ["latest news", "latest live news", "what is the latest news", "what is the live feed", "show live feed", "get live feed", "live feed", "what is the live feed latest one"] or \
         (re.search(r'\b(?:top|first)\s*\d+\b', query) and not any(k in query for k in ["about", "on ", "for "])) or \
         (("latest" in query or "live" in query or "top" in query) and ("feed" in query or "dashboard" in query or "news" in query) and not any(k in query for k in ["about", "on ", "for "])):
        state["intent"] = "live_feed"
    else:
        state["intent"] = "search"
        
    logger.info(f"Intent determined as: {state['intent']} (requested_limit={state.get('requested_limit')})")
    return state

