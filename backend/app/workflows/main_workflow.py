from langgraph.graph import StateGraph, END
import logging
import time
import re
from typing import Dict, Any
from datetime import datetime

from app.workflows.langgraph_state import AgentState
from app.agents.policy_agent import PolicyAgent
from app.agents.reflection_agent import ReflectionAgent
from app.mcp_client import mcp_client
from app.utils.evaluator import evaluate_execution
from app.utils.redis_cache import cache_get, cache_set, get_cache_hit_rate

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

    if "start_time" not in state or state["start_time"] is None or state["start_time"] == 0:
        state["start_time"] = time.perf_counter()

    # Real Redis cache lookup — replaces mock cache miss
    query = state.get("query", "")
    cached = cache_get(query)
    if cached:
        logger.info("[Workflow] Cache HIT — returning cached response, skipping agentic workflow.")
        state["cached_response"] = cached
        state["final_response"] = cached
        state["cache_hit"] = True
        latency = time.perf_counter() - state["start_time"]
        metrics = evaluate_execution(
            query=query,
            response=cached,
            retrieved_docs=[],
            observations=[],
            intent=state.get("intent", "search"),
            latency=latency,
            cache_hit=True
        )
        metrics["cache_hit_rate"] = get_cache_hit_rate()
        state["evaluation_metrics"] = metrics
        return state
    else:
        state["cached_response"] = ""  # Cache miss — proceed to agentic workflow
        state["cache_hit"] = False


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

    MAX_ITERATIONS = 5

    # Max iteration safety check — only fires to prevent infinite loops.
    # Normal termination must come from the Policy Agent via {"action": "finish"}.
    if state["iteration_count"] >= MAX_ITERATIONS:
        logger.warning(f"[Workflow] Max iterations ({MAX_ITERATIONS}) reached. Synthesizing from observations.")
        state["next_action"] = "finish"
        synth = _synthesize_from_observations(state["query"], state["observations"])
        state["action_answer"] = synth or "Could not retrieve sufficient information."
        state["action_thought"] = f"Reached MAX_ITERATIONS={MAX_ITERATIONS}. Synthesizing from collected data."
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
    
    # 1. Validation Check: If LLM output failed Pydantic or MCP schema validation,
    # record a structured observation detailing the validation failure and return to Policy Agent for re-decision.
    if not decision.is_valid:
        err_msg = decision.validation_error or "Unknown validation error."
        logger.warning(f"[Workflow] Policy Agent produced invalid action/arguments: {err_msg}")
        validation_obs = {
            "iteration": state["iteration_count"] + 1,
            "thought": decision.thought or "Policy Agent action failed schema validation.",
            "tool": decision.tool or "invalid_action",
            "arguments": decision.arguments or {},
            "result": f"VALIDATION ERROR: {err_msg}",
            "timestamp": time.time(),
            "execution_time": 0.0
        }
        state["observations"].append(validation_obs)
        state["iteration_count"] += 1
        state["next_action"] = "policy"
        state["agent_trace"].append(f"Validation Failure: {err_msg}")
        return state

    # Safeguard 1: If LLM says "finish" but we have zero observations,
    # force a tool call so we always retrieve real data first.
    if decision.action == "finish" and len(state["observations"]) == 0:
        logger.warning("[Workflow] LLM tried to finish with no observations. Forcing search_live_news tool call.")
        decision.action = "tool"
        decision.tool = "search_live_news"
        decision.arguments = {"query": query, "limit": 10}
        decision.thought = "Overridden: Must search for data before finishing."

    # Multi-tool support: if Policy Agent requests another tool after observations,
    # allow it — the agent decides when it has enough data.
    if decision.action == "tool":
        obs_count = len(state["observations"])
        logger.info(f"[Workflow] Policy Agent selected tool '{decision.tool}' (observation #{obs_count + 1}).")
    
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
    if action == "policy":
        return "policy"
    elif action == "tool":
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
    
    # Record agent execution metrics in news_repository
    try:
        from app.repositories.news_repository import news_repository
        agent_label = "RAG Search & Retrieval Agent" if tool_name == "search_live_news" else "Ingestion Agent (RSS & Web)"
        is_success = not (isinstance(result, dict) and "error" in result)
        news_repository.record_agent_execution(agent_label, (t_end - t_start) * 1000, is_success)
    except Exception as ex_log:
        logger.warning(f"Could not record agent execution log: {ex_log}")

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
        if isinstance(result, str):
            try:
                import json
                parsed = json.loads(result)
                if isinstance(parsed, (list, dict)):
                    result = parsed
            except Exception:
                pass
        if isinstance(result, list):
            for item in result:
                if isinstance(item, dict):
                    title = item.get("title", "")
                    content = item.get("cleaned_content") or item.get("content") or item.get("description") or item.get("summary") or ""
                    source = item.get("source") or "Live Media"
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
                            source = item.get("source") or "Live Media"
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
    
    try:
        if not articles:
            # Extract title from query quotes if available
            quoted_title = re.findall(r'["\u201c\u201d]([^"\u201c\u201d]+)["\u201c\u201d]', query)
            if not quoted_title:
                quoted_title = re.findall(r"['\u2018\u2019]([^'\u2018\u2019]+)['\u2018\u2019]", query)
            if quoted_title:
                t = quoted_title[0]
                articles = [{
                    "title": t,
                    "content": t,
                    "cleaned_content": t,
                    "source": "Live Media",
                    "url": "#",
                    "category": "General News",
                    "published_date": ""
                }]
            else:
                return "No relevant observations were retrieved for your query. Please verify search parameters or refresh the news feeds."
        
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
    except Exception as ex:
        logger.error(f"[Workflow] Exception during response synthesis: {ex}")
        return "Unable to synthesize response from observations due to processing format errors."


# Reflection node
async def reflection_node(state: AgentState) -> AgentState:
    from app.agents.reflection_agent import REFLECTION_STATUS_UNVERIFIED, REFLECTION_STATUS_REVISED
    query = state["query"]
    answer = state.get("action_answer", "")
    history = state.get("observations", [])

    # Run reflection
    report = await reflection_agent.reflect(query=query, answer=answer, history=history)
    state["reflection_report"] = report.model_dump() if hasattr(report, "model_dump") else report.dict()

    reflection_status = report.reflection_status
    fallback_used     = report.fallback_used

    if report.revise and state.get("iteration_count", 0) < 4:
        # Loop back to Policy Agent for revision
        logger.info(
            f"[Workflow] Reflection agent requested revision. "
            f"Status={reflection_status}, Fallback={fallback_used}, "
            f"Unsupported={report.unsupported_claims}"
        )
        state["agent_trace"].append(
            f"Critique: Revision requested (status={reflection_status}). "
            f"Unsupported: {report.unsupported_claims}"
        )
        state["iteration_count"] = state.get("iteration_count", 0) + 1
        state["next_action"] = "tool"  # Forces workflow to loop back
        # Append critique to history so Policy Agent is aware
        state["observations"].append({
            "iteration": state["iteration_count"],
            "thought": "Reflection criticism",
            "tool": "reflection_critique",
            "arguments": {},
            "result": (
                f"Please revise the response. "
                f"Reflection status: {reflection_status}. "
                f"Unsupported claims: {report.unsupported_claims}. "
                f"Missing: {report.missing_information}"
            ),
            "timestamp": time.time(),
            "execution_time": 0.0
        })
    else:
        resp_str  = answer
        retrieved = state.get("retrieved_documents", [])

        # Check if LLM output is generic, missing, single-line, or contains an empty Key Summary template
        is_raw_list    = bool(resp_str and resp_str.startswith("Summary:") and len(resp_str.split("\n")) <= 2)
        is_generic     = bool(not resp_str or resp_str in ["No response generated.", "No detailed information was found."])
        is_blank_summary = bool(resp_str and re.search(r'\*\*Key Summary:\*\*\s*(?:---|#|\*\*Primary Source Links:\*\*|\s*$)', resp_str))

        if (is_generic or is_raw_list or is_blank_summary) and (history or retrieved):
            synth_docs = retrieved if retrieved else []
            if synth_docs:
                from app.agents.response import synthesize_executive_summary
                resp_str = synthesize_executive_summary(query, synth_docs[:5], llm_summary=None, intent=state.get("intent", ""))
            else:
                resp_str = _synthesize_from_observations(query, history, intent=state.get("intent", ""))

        if not resp_str:
            resp_str = "No response generated."

        # ----------------------------------------------------------------
        # FAIL-SAFE: If reflection could not verify the answer, attach a
        # clear UNVERIFIED disclaimer. NEVER suppress this silently.
        # ----------------------------------------------------------------
        if reflection_status == REFLECTION_STATUS_UNVERIFIED or fallback_used:
            unverified_note = (
                "\n\n---\n"
                "> ⚠️ **UNVERIFIED**: The Reflection Agent could not fully verify this response "
                "against retrieved sources (LLM verification was unavailable). "
                "Claims have been checked deterministically against observations, "
                "but independent verification is recommended."
            )
            if report.unsupported_claims:
                unverified_note += (
                    f"\n> **Potentially unverified claims**: {'; '.join(report.unsupported_claims[:3])}"
                )
            resp_str += unverified_note
            logger.warning(
                f"[Workflow] Reflection UNVERIFIED — disclaimer appended to final response. "
                f"Fallback={fallback_used}, Unsupported={len(report.unsupported_claims)}"
            )
            state["agent_trace"].append(
                f"Reflection UNVERIFIED (fallback={fallback_used}). Disclaimer added to response."
            )

        state["final_response"] = resp_str

        # Store final response in Redis for future cache hits
        query = state.get("query", "")
        if resp_str and query:
            cache_set(query, resp_str)

        start_t = state.get("start_time")
        if start_t:
            if start_t > 1000000000:
                calc_latency = time.time() - start_t
            else:
                calc_latency = time.perf_counter() - start_t
        else:
            calc_latency = 0.0

        # Evaluate actual metrics — pass real latency and cache_hit state
        metrics = evaluate_execution(
            query=query,
            response=resp_str,
            retrieved_docs=state.get("retrieved_documents", []),
            observations=history,
            intent=state.get("intent", "search"),
            latency=calc_latency,
            cache_hit=state.get("cache_hit", False)
        )
        # Override cache_hit_rate with actual dynamic value from Redis counters
        metrics["cache_hit_rate"] = get_cache_hit_rate()
        state["evaluation_metrics"] = metrics
        
        # Persist dynamic evaluation run to news_repository store
        try:
            from app.repositories.news_repository import news_repository
            news_repository.record_evaluation_run({
                "query": query,
                "metrics": metrics,
                "timestamp": datetime.utcnow().isoformat()
            })
        except Exception as ex_eval:
            logger.warning(f"Could not record evaluation run: {ex_eval}")

        state["agent_trace"].append(
            f"Reflection complete. Status={reflection_status}. Finalizing response."
        )

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
graph.add_conditional_edges("policy", route_policy, {"tool": "tool", "reflection": "reflection", "policy": "policy"})
graph.add_edge("tool", "policy")
graph.add_conditional_edges("reflection", lambda state: "policy" if state.get("next_action") == "tool" else "end", {
    "policy": "policy",
    "end": END
})

app_graph = graph.compile()
