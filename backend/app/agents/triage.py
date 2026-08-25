import logging
import re

from app.workflows.langgraph_state import AgentState

logger = logging.getLogger(__name__)


def triage_agent(state: AgentState) -> AgentState:
    """
    Analyzes user intent and routes to the appropriate agent.
    """
    logger.info("Triage Agent analyzing intent...")
    query = state.get("query", "").lower().strip()

    # Strip UI greeting/prefill artifacts (e.g. ", or compare news sources! what is...")
    query = re.sub(r"^(?:,\s*or\s+compare\s+news\s+sources[!.]*|ask\s+me\s+to\s+summarize\s+recent\s+news[,\s]*|search\s+specific\s+topics[,\s]*|analyze\s+trends[,\s]*)+", "", query, flags=re.IGNORECASE).strip()

    # Extract requested number of items if specified (e.g. "top 1", "top 3", "give me 1", "give me one", "1 topic", "only 2")
    number_words = {"one": 1, "two": 2, "three": 3, "four": 4, "five": 5, "six": 6, "seven": 7, "eight": 8, "nine": 9, "ten": 10}

    limit_match = re.search(r'\b(?:top|first|only|limit|give me|show me|get|just)\s*(\d+)\b', query) or \
                  re.search(r'\b(\d+)\s*(?:items?|feeds?|articles?|topics?|news)?\b', query)

    if limit_match and limit_match.group(1):
        try:
            state["requested_limit"] = int(limit_match.group(1))
        except Exception:
            state["requested_limit"] = 10
    else:
        found_limit = None
        for word_str, word_val in number_words.items():
            if re.search(rf'\b(?:give me|show me|top|first|only|just|get)?\s*{word_str}\b', query):
                found_limit = word_val
                break
        if found_limit is not None:
            state["requested_limit"] = found_limit
        elif re.search(r'\b(?:the\s+)?(?:latest\s+)?trending\s+topic\b', query) and not re.search(r'\b(?:topics|articles|feeds)\b', query):
            state["requested_limit"] = 1
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

