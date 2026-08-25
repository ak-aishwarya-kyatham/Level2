import os
import sys
import asyncio
import requests

# Ensure app module is importable
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from app.mcp_client import mcp_client
from app.agents.policy_agent import PolicyAgent
from app.agents.reflection_agent import ReflectionAgent
from app.workflows.main_workflow import _synthesize_from_observations

OLLAMA_URL = os.getenv("OLLAMA_URL", "http://localhost:11434")
OLLAMA_MODEL = os.getenv("OLLAMA_MODEL", "qwen2.5:3b")


from typing import Tuple, List

def check_ollama_status() -> Tuple[bool, bool, List[str]]:
    """
    Verifies whether the Ollama server is reachable and whether the target OLLAMA_MODEL is installed.
    Returns: (server_online, model_installed, list_of_available_models)
    """
    try:
        resp = requests.get(f"{OLLAMA_URL}/api/tags", timeout=3.0)
        if resp.status_code == 200:
            data = resp.json()
            models = [m.get("name", "") for m in data.get("models", [])]
            # Match exact name or base name (e.g., qwen2.5:3b or qwen2.5:3b-instruct)
            model_installed = any(
                OLLAMA_MODEL.lower() in m.lower() or m.lower() in OLLAMA_MODEL.lower()
                for m in models
            )
            return True, model_installed, models
    except Exception:
        pass
    return False, False, []


async def run_live_trace():
    """
    Executes an unmocked, live query flow through PolicyAgent, MCP Client server session,
    Response Synthesis, and ReflectionAgent, logging every step to tests/live_agent_trace.txt.
    """
    print(f"Checking Ollama status at {OLLAMA_URL}...")
    server_online, model_installed, available_models = check_ollama_status()

    if not server_online:
        print(f"\n[ERROR] Ollama server is unreachable at {OLLAMA_URL}.")
        print("Please start Ollama service on http://localhost:11434 before running a live trace.")
        print(f"   ollama run {OLLAMA_MODEL}")
        sys.exit(1)

    if not model_installed:
        print(f"\n[ERROR] Ollama server is online at {OLLAMA_URL}, but model '{OLLAMA_MODEL}' is NOT pulled/installed.")
        print(f"Available installed models in Ollama: {available_models or 'None'}")
        print(f"To pull the required model, run:")
        print(f"   ollama pull {OLLAMA_MODEL}")
        print("Or set OLLAMA_MODEL environment variable to use one of the installed models.")
        sys.exit(1)

    print(f"[OK] Ollama server is online and model '{OLLAMA_MODEL}' is verified installed. Starting live unmocked trace...\n")

    trace_lines = []

    def log_step(text: str):
        try:
            sys.stdout.buffer.write((text + "\n").encode("utf-8", errors="replace"))
            sys.stdout.buffer.flush()
        except Exception:
            print(text.encode("ascii", errors="replace").decode("ascii"))
        trace_lines.append(text)

    log_step("========== LIVE UNMOCKED AGENT TRACE ==========")
    log_step(f"Ollama Endpoint: {OLLAMA_URL} | Model: {OLLAMA_MODEL}\n")

    query = "Compare news coverage of AI developments across top media sources"
    log_step(f"USER QUERY:\n-> {query}\n")

    # 1. Connect real MCP Client Session (spawns real FastMCP server via stdio transport)
    await mcp_client.start()
    if not mcp_client.is_connected:
        print("[ERROR] MCP Client could not establish stdio ClientSession with FastMCP server.")
        sys.exit(1)

    tools = await mcp_client.list_available_tools()
    discovered_names = [t.get("name") for t in tools if t.get("name")]
    log_step(f"MCP TRANSPORT: stdio ClientSession (Active: {mcp_client.is_connected})")
    log_step(f"DISCOVERED TOOLS FROM SERVER: {discovered_names}\n")

    # 2. Multi-iteration Policy & Reflection Loop (Max 3 iterations total: 1 initial + 2 revisions)
    policy_agent = PolicyAgent()
    reflection_agent = ReflectionAgent()
    
    history = []
    observations = []
    max_iterations = 3
    iteration = 1
    synthesized_answer = "No answer generated."

    while iteration <= max_iterations:
        log_step(f"\n--- [ITERATION {iteration}/{max_iterations}] ---")

        # Step A: Policy Agent Decision
        log_step(f"[{iteration}.1] Executing PolicyAgent decision (no mocks)...")
        action = await policy_agent.decide_action(
            query=query, tools=tools, history=history, iteration_count=iteration
        )

        log_step(f"POLICY DECISION:\n-> Action: {action.action} | Tool: {action.tool} | Thought: {action.thought}")
        if action.arguments:
            log_step(f"-> Arguments: {action.arguments}")

        # Step B: Real MCP Tool Execution
        if action.action == "tool" and action.tool:
            log_step(f"\n[{iteration}.2] Calling real MCP tool '{action.tool}' via MCP client session...")
            obs_result = await mcp_client.call_tool(action.tool, action.arguments)
            log_step(f"OBSERVATION RETURNED:\n-> Tool '{action.tool}' returned {len(str(obs_result))} bytes of data.")

            obs_entry = {
                "iteration": iteration,
                "thought": action.thought,
                "tool": action.tool,
                "arguments": action.arguments,
                "result": obs_result
            }
            observations.append(obs_entry)
            history.append(obs_entry)

        # Step C: Grounded Response Synthesis
        log_step(f"\n[{iteration}.3] Running grounded response synthesis...")
        synthesized_answer = _synthesize_from_observations(query, observations, intent="search")
        log_step(f"SYNTHESIZED RESPONSE:\n-> {synthesized_answer[:300]}...")

        # Step D: Reflection Agent Validation
        log_step(f"\n[{iteration}.4] Running ReflectionAgent validation...")
        report = await reflection_agent.reflect(query=query, answer=synthesized_answer, history=observations, skip_llm_if_grounded=False)
        log_step(f"REFLECTION VERDICT:\n-> Status: {report.reflection_status} | Revise: {report.revise} | Fallback Used: {report.fallback_used}")
        if report.unsupported_claims:
            log_step(f"-> Unsupported Claims: {report.unsupported_claims}")

        # Step E: Handle Revision Request
        if not report.revise or report.reflection_status == "VERIFIED":
            log_step(f"\n[OK] Response verified by ReflectionAgent on iteration {iteration}.")
            break
        else:
            if iteration < max_iterations:
                feedback_entry = {
                    "tool": "reflection_feedback",
                    "result": f"Unsupported claims to address: {report.unsupported_claims}"
                }
                history.append(feedback_entry)
                log_step(
                    f"\n[REVISION REQUESTED] ReflectionAgent flagged unsupported claims. "
                    f"Proceeding to revision iteration {iteration + 1}..."
                )
                iteration += 1
            else:
                log_step(
                    f"\n[CAP REACHED] Reached maximum iteration cap ({max_iterations}) "
                    "without achieving VERIFIED reflection status."
                )
                break

    log_step("\n========== FINAL RESPONSE OUTPUT ==========")
    log_step(synthesized_answer)
    log_step("============================================\n")

    # Clean disconnect
    await mcp_client.stop()

    # Save to live_agent_trace.txt
    output_path = os.path.join(os.path.dirname(__file__), "..", "tests", "live_agent_trace.txt")
    with open(output_path, "w", encoding="utf-8") as f:
        f.write("\n".join(trace_lines))
    print(f"Live agent trace saved to {output_path}")


if __name__ == "__main__":
    asyncio.run(run_live_trace())
