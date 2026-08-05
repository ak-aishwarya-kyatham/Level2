from langgraph.graph import StateGraph, END
import logging
from app.workflows.langgraph_state import AgentState

# Agent nodes
from app.agents.triage import triage_agent
from app.agents.query_understanding import query_understanding_agent
from app.agents.search_retrieval import search_agent, retrieval_agent
from app.agents.specialized import summary_agent, compare_agent, trend_agent, translation_agent
from app.agents.response import response_generation_agent

logger = logging.getLogger(__name__)

# Authentication and Cache would typically happen before hitting the LangGraph, 
# or can be the first nodes. We will implement them as middleware/routers in FastAPI, 
# but for completeness, we can have a dummy cache node here.

def cache_agent(state: AgentState) -> AgentState:
    logger.info("Checking cache...")
    # Mock cache check
    state["cached_response"] = ""
    return state

def should_use_cache(state: AgentState) -> str:
    if state.get("cached_response"):
        return "end"
    return "triage"

def route_triage(state: AgentState) -> str:
    intent = state.get("intent", "search").lower()
    if intent == "compare":
        return "compare"
    elif intent == "summarize":
        return "summary"
    elif intent == "trend":
        return "trend"
    else:
        return "search"

# Build Graph
graph = StateGraph(AgentState)

# Add nodes
graph.add_node("cache", cache_agent)
graph.add_node("triage", triage_agent)
graph.add_node("search", search_agent)
graph.add_node("query_understanding", query_understanding_agent)
graph.add_node("retrieval", retrieval_agent)
graph.add_node("summary", summary_agent)
graph.add_node("compare", compare_agent)
graph.add_node("trend", trend_agent)
graph.add_node("translation", translation_agent)
graph.add_node("response", response_generation_agent)

# Add edges
graph.set_entry_point("cache")
graph.add_conditional_edges("cache", should_use_cache, {"end": END, "triage": "triage"})
graph.add_conditional_edges("triage", route_triage, {
    "compare": "compare",
    "summary": "summary",
    "trend": "trend",
    "search": "search"
})

# All specialized nodes route to query_understanding, which then routes to retrieval.
graph.add_edge("search", "query_understanding")
graph.add_edge("compare", "query_understanding")
graph.add_edge("summary", "query_understanding")
graph.add_edge("trend", "query_understanding")

graph.add_edge("query_understanding", "retrieval")
graph.add_edge("retrieval", "response")
graph.add_edge("translation", "response") # Optional translation step
graph.add_edge("response", END)

app_graph = graph.compile()
