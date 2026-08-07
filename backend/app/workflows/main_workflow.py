from langgraph.graph import StateGraph, END
import logging
import time
import re
from typing import Dict, Any

from app.workflows.langgraph_state import AgentState
from app.agents.policy_agent import PolicyAgent
from app.agents.reflection_agent import ReflectionAgent
from app.mcp_client import mcp_client
from app.utils.evaluator import evaluate_execution

logger = logging.getLogger(__name__)

policy_agent = PolicyAgent()
reflection_agent = ReflectionAgent()

# Cache checking node
async def cache_node(state: AgentState) -> AgentState:
    logger.info("[Workflow] Checking cache and running intent analysis...")
    # Initialize fields if not present
    if "observations" not in state or state["observations"] is None:
        state["observations"] = []
    if "iteration_count" not in state or state["iteration_count"] is None:
        state["iteration_count"] = 0
    if "agent_trace" not in state or state["agent_trace"] is None:
        state["agent_trace"] = []
    if "evaluation_metrics" not in state or state["evaluation_metrics"] is None:
        state["evaluation_metrics"] = {}
    if "retrieved_documents" not in state or state["retrieved_documents"] is None:
        state["retrieved_documents"] = []
    
    state["next_action"] = state.get("next_action", "")
    state["action_answer"] = state.get("action_answer", "")
    state["action_thought"] = state.get("action_thought", "")
    state["action_tool"] = state.get("action_tool", "")
    state["action_arguments"] = state.get("action_arguments", {})
    
    state["cached_response"] = ""  # Mock cache miss for now

    # Run triage and query understanding
    from app.agents.triage import triage_agent
    from app.agents.query_understanding import query_understanding_agent
    try:
        state = triage_agent(state)
        state = query_understanding_agent(state)
    except Exception as ex:
        logger.error(f"[Workflow] Query understanding error: {ex}")

    return state

def should_use_cache(state: AgentState) -> str:
    if state.get("cached_response"):
        return "end"
    return "policy"

# Policy decision node
async def policy_node(state: AgentState) -> AgentState:
    # Ensure fields are initialized
    if not state.get("observations"):
        state["observations"] = []
    if not state.get("agent_trace"):
        state["agent_trace"] = []
    if state.get("iteration_count") is None:
        state["iteration_count"] = 0

    # Max iteration check — force finish after 2 tool calls
    if state["iteration_count"] >= 2:
        logger.warning("[Workflow] Max iterations (2) reached. Synthesizing from observations.")
        state["next_action"] = "finish"
        # Synthesize answer from collected observations
        synth = _synthesize_from_observations(state["query"], state["observations"])
        state["action_answer"] = synth or "Could not retrieve sufficient information."
        state["action_thought"] = "Reached iteration limit. Synthesizing from collected data."
        return state

    query = state["query"]
    tools = await mcp_client.list_available_tools()
    
    # Run Policy Agent
    decision = await policy_agent.decide_action(
        query=query,
        tools=tools,
        history=state["observations"],
        iteration_count=state["iteration_count"] + 1
    )
    
    # Safeguard 1: If LLM says "finish" but we have zero observations,
    # force a tool call so we always retrieve real data first.
    if decision.action == "finish" and len(state["observations"]) == 0:
        logger.warning("[Workflow] LLM tried to finish with no observations. Forcing search_live_news tool call.")
        decision.action = "tool"
        decision.tool = "search_live_news"
        decision.arguments = {"query": query, "limit": 10}
        decision.thought = "Overridden: Must search for data before finishing."
    
    # Safeguard 2: If we already have observations and LLM says "tool" again,
    # force finish and synthesize from data (prevents infinite loops).
    if decision.action == "tool" and len(state["observations"]) >= 1:
        logger.info("[Workflow] Already have observations. Forcing finish with synthesis.")
        decision.action = "finish"
        synth = _synthesize_from_observations(query, state["observations"], intent=state.get("intent", ""))
        decision.answer = synth or decision.answer or "No detailed information was found."
        decision.thought = "Have sufficient data from tools. Synthesizing answer."
    
    # Store decisions in temporary state variables (or agent_trace)
    state["next_action"] = decision.action
    state["action_thought"] = decision.thought
    
    if decision.action == "tool":
        state["action_tool"] = decision.tool
        state["action_arguments"] = decision.arguments
    else:
        state["action_answer"] = decision.answer
        
    state["agent_trace"].append(f"Thought: {decision.thought}")
    return state

# Conditional routing edge
def route_policy(state: AgentState) -> str:
    action = state.get("next_action", "finish")
    if action == "tool":
        return "tool"
    return "reflection"

# Sanitization and prompt injection protection
def sanitize_observation(data: Any) -> Any:
    """Recursively sanitize strings to remove potential prompt injections and HTML tags."""
    if isinstance(data, list):
        return [sanitize_observation(x) for x in data]
    elif isinstance(data, dict):
        return {k: sanitize_observation(v) for k, v in data.items()}
    elif isinstance(data, str):
        # Redact common prompt injection keywords/sentences
        patterns = [
            r"(?i)ignore\s+(?:all\s+)?prior\s+instructions",
            r"(?i)ignore\s+(?:all\s+)?previous\s+instructions",
            r"(?i)you\s+must\s+now\s+act\s+as",
            r"(?i)override\s+system\s+prompt",
            r"(?i)jailbreak",
            r"(?i)forget\s+everything\s+you\s+were\s+told"
        ]
        sanitized = data
        for pattern in patterns:
            sanitized = re.sub(pattern, "[POTENTIAL INJECTION BLOCKED]", sanitized)
        # Strip simple HTML/XML tags
        sanitized = re.sub(r"<[^>]+>", "", sanitized)
        return sanitized
    return data

# Tool execution node
async def tool_node(state: AgentState) -> AgentState:
    tool_name = state.get("action_tool")
    arguments = state.get("action_arguments", {})
    thought = state.get("action_thought", "")
    iteration = state.get("iteration_count", 0) + 1
    
    # 1. Validate tool arguments
    if not isinstance(arguments, dict):
        logger.warning(f"[Workflow] Arguments for tool '{tool_name}' is not a dict. Forcing to empty dict.")
        arguments = {}
        
    # Escape query/params string args
    for k, v in list(arguments.items()):
        if isinstance(v, str):
            # Basic validation/cleanup
            arguments[k] = v.strip()
            
    t_start = time.time()
    try:
        # Call tool via standard MCP ClientSession
        result = await mcp_client.call_tool(tool_name, arguments)
    except Exception as e:
        logger.error(f"[Workflow] Tool execution failed: {e}")
        result = {"error": f"Failed to execute tool: {str(e)}"}
        
    t_end = time.time()
    
    # 2. Sanitize result before recording
    sanitized_result = sanitize_observation(result)
    
    # Record observation
    observation = {
        "iteration": iteration,
        "thought": thought,
        "tool": tool_name,
        "arguments": arguments,
        "result": sanitized_result,
        "timestamp": t_end,
        "execution_time": round(t_end - t_start, 3)
    }
    state["observations"].append(observation)
    state["iteration_count"] = iteration
    
    # Store retrieved documents if search tool is used
    if tool_name == "search_live_news" and isinstance(sanitized_result, list):
        state["retrieved_documents"].extend(sanitized_result)
    elif tool_name == "get_articles_by_category" and isinstance(sanitized_result, list):
        state["retrieved_documents"].extend(sanitized_result)
        
    state["agent_trace"].append(f"Action: Called tool '{tool_name}' -> Observation length: {len(str(sanitized_result))}")
    return state


def _synthesize_from_observations(query: str, observations: list, intent: str = "") -> str:
    """Build a readable response from raw tool observations using grounded executive summarization."""
    from app.agents.response import synthesize_executive_summary, synthesize_comparison_briefing
    articles = []
    source1_val = None
    source2_val = None

    for obs in observations:
        result = obs.get("result")
        if obs.get("tool") == "reflection_critique":
            continue
        if isinstance(result, list):
            for item in result:
                if isinstance(item, dict):
                    title = item.get("title", "")
                    content = item.get("cleaned_content") or item.get("content") or item.get("description") or item.get("summary") or ""
                    source = item.get("source", "")
                    url = item.get("url", "#")
                    category = item.get("category", "")
                    published_date = item.get("published_date", "")
                    if title:
                        articles.append({
                            "title": title,
                            "content": content,
                            "cleaned_content": content,
                            "source": source,
                            "url": url,
                            "category": category,
                            "published_date": published_date
                        })
        elif isinstance(result, dict):
            s1 = result.get("source1")
            s2 = result.get("source2")
            if s1: source1_val = s1
            if s2: source2_val = s2

            # Handle compare_news_sources dict
            for item in result.get("common_news", []):
                if isinstance(item, dict):
                    t1 = item.get("source1_title") or item.get("title") or ""
                    if t1:
                        articles.append({
                            "title": t1,
                            "content": item.get("summary") or t1,
                            "cleaned_content": item.get("summary") or t1,
                            "source": s1 or "Source 1",
                            "comparison_source": s1 or "Source 1",
                            "url": item.get("source1_url", "#")
                        })
                    t2 = item.get("source2_title") or ""
                    if t2:
                        articles.append({
                            "title": t2,
                            "content": item.get("summary") or t2,
                            "cleaned_content": item.get("summary") or t2,
                            "source": s2 or "Source 2",
                            "comparison_source": s2 or "Source 2",
                            "url": item.get("source2_url", "#")
                        })
            for item in result.get("exclusive_source1", []):
                if isinstance(item, dict) and item.get("title"):
                    articles.append({
                        "title": item["title"],
                        "content": item.get("title"),
                        "cleaned_content": item.get("title"),
                        "source": s1 or "Source 1",
                        "comparison_source": s1 or "Source 1",
                        "url": item.get("url", "#")
                    })
            for item in result.get("exclusive_source2", []):
                if isinstance(item, dict) and item.get("title"):
                    articles.append({
                        "title": item["title"],
                        "content": item.get("title"),
                        "cleaned_content": item.get("title"),
                        "source": s2 or "Source 2",
                        "comparison_source": s2 or "Source 2",
                        "url": item.get("url", "#")
                    })

            for key in ["articles", "results", "data", "recent_articles"]:
                if isinstance(result.get(key), list):
                    for item in result[key]:
                        if isinstance(item, dict):
                            title = item.get("title", "")
                            content = item.get("cleaned_content") or item.get("content") or item.get("description") or item.get("summary") or ""
                            source = item.get("source", "")
                            url = item.get("url", "#")
                            category = item.get("category", "")
                            published_date = item.get("published_date", "")
                            if title:
                                articles.append({
                                    "title": title,
                                    "content": content,
                                    "cleaned_content": content,
                                    "source": source,
                                    "url": url,
                                    "category": category,
                                    "published_date": published_date
                                })
    
    if not articles:
        return ""
    
    # Deduplicate by title
    seen = set()
    unique = []
    for a in articles:
        if a["title"] not in seen:
            seen.add(a["title"])
            unique.append(a)
    
    if intent == "compare" or source1_val or source2_val:
        return synthesize_comparison_briefing(query, unique, source1=source1_val, source2=source2_val)

    return synthesize_executive_summary(query, unique[:5], llm_summary=None, intent=intent or "summarize")


# Reflection node
async def reflection_node(state: AgentState) -> AgentState:
    query = state["query"]
    answer = state.get("action_answer", "")
    history = state.get("observations", [])
    
    # Run reflection
    report = await reflection_agent.reflect(query=query, answer=answer, history=history)
    state["reflection_report"] = report.dict()
    
    if report.revise and state.get("iteration_count", 0) < 4:
        # Loop back to Policy Agent by setting action to tool (or keep looping)
        logger.info("[Workflow] Reflection agent requested a revision.")
        state["agent_trace"].append(f"Critique: Revision requested. Unsupported: {report.unsupported_claims}")
        state["iteration_count"] = state.get("iteration_count", 0) + 1
        state["next_action"] = "tool"  # Forces workflow to loop back
        # Append critique to history so Policy Agent is aware
        state["observations"].append({
            "iteration": state["iteration_count"],
            "thought": "Reflection criticism",
            "tool": "reflection_critique",
            "arguments": {},
            "result": f"Please revise the response. Critique: Unsupported claims: {report.unsupported_claims}. Missing: {report.missing_information}",
            "timestamp": time.time(),
            "execution_time": 0.0
        })
    else:
        resp_str = answer
        retrieved = state.get("retrieved_documents", [])
        
        # Check if LLM output is generic, missing, or just a raw single-line headline string
        is_raw_list = bool(resp_str and resp_str.startswith("Summary:") and len(resp_str.split("\n")) <= 2)
        is_generic = bool(not resp_str or resp_str in ["No response generated.", "No detailed information was found."])
        
        if (is_generic or is_raw_list) and (history or retrieved):
            synth_docs = retrieved if retrieved else []
            if synth_docs:
                from app.agents.response import synthesize_executive_summary
                resp_str = synthesize_executive_summary(query, synth_docs[:5], llm_summary=None, intent=state.get("intent", ""))
            else:
                resp_str = _synthesize_from_observations(query, history, intent=state.get("intent", ""))

        if not resp_str:
            resp_str = "No response generated."

        state["final_response"] = resp_str
        # Evaluate actual metrics
        metrics = evaluate_execution(
            query=query,
            response=resp_str,
            retrieved_docs=state.get("retrieved_documents", []),
            observations=history,
            intent=state.get("intent", "search"),
            start_time=time.time() - 2.0,  # approximate duration
            cache_hit=False
        )
        state["evaluation_metrics"] = metrics
        state["agent_trace"].append("Reflection complete. Finalizing response.")
        
    return state

# Build Graph
graph = StateGraph(AgentState)

# Add nodes
graph.add_node("cache", cache_node)
graph.add_node("policy", policy_node)
graph.add_node("tool", tool_node)
graph.add_node("reflection", reflection_node)

# Add edges
graph.set_entry_point("cache")
graph.add_conditional_edges("cache", should_use_cache, {"end": END, "policy": "policy"})
graph.add_conditional_edges("policy", route_policy, {"tool": "tool", "reflection": "reflection"})
graph.add_edge("tool", "policy")
graph.add_conditional_edges("reflection", lambda state: "policy" if state.get("next_action") == "tool" else "end", {
    "policy": "policy",
    "end": END
})

app_graph = graph.compile()
