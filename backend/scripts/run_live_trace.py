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


def check_ollama_status() -> bool:
    """Verifies whether the Ollama server endpoint is reachable."""
    try:
        resp = requests.get(f"{OLLAMA_URL}/api/tags", timeout=3.0)
        if resp.status_code == 200:
            return True
    except Exception:
        pass
    return False


async def run_live_trace():
    """
    Executes an unmocked, live query flow through PolicyAgent, MCP Client server session,
    Response Synthesis, and ReflectionAgent, logging every step to tests/live_agent_trace.txt.
    """
    print(f"Checking Ollama status at {OLLAMA_URL}...")
    if not check_ollama_status():
        print(f"\n❌ ERROR: Ollama server is unreachable at {OLLAMA_URL}.")
        print("Please start Ollama with your desired model before running a live trace:")
        print(f"   ollama run {OLLAMA_MODEL}")
        print("Or set OLLAMA_URL environment variable if running on a custom host/port.")
        sys.exit(1)

    print(f"Ollama server is reachable. Using model '{OLLAMA_MODEL}'. Starting live unmocked trace...\n")

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

    # 1. Connect real MCP Client Session (spawns real MCP server)
    await mcp_client.start()
    tools = await mcp_client.list_available_tools()

    # 2. Real Policy Agent Decision
    policy_agent = PolicyAgent()
    log_step("[1] Executing real PolicyAgent decision (no mocks)...")
    action = await policy_agent.decide_action(query=query, tools=tools, history=[], iteration_count=1)

    log_step(f"POLICY DECISION:\n-> Action: {action.action} | Tool: {action.tool} | Thought: {action.thought}")
    if action.arguments:
        log_step(f"-> Arguments: {action.arguments}")

    observations = []

    # 3. Real MCP Tool Execution
    if action.action == "tool" and action.tool:
        log_step(f"\n[2] Calling real MCP tool '{action.tool}' via stdio JSON-RPC...")
        obs_result = await mcp_client.call_tool(action.tool, action.arguments)
        log_step(f"OBSERVATION RETURNED:\n-> Tool '{action.tool}' returned {len(str(obs_result))} bytes of data.")

        obs_entry = {
            "iteration": 1,
            "thought": action.thought,
            "tool": action.tool,
            "arguments": action.arguments,
            "result": obs_result
        }
        observations.append(obs_entry)

    # 4. Real Response Synthesis
    log_step("\n[3] Running real grounded response synthesis...")
    synthesized_answer = _synthesize_from_observations(query, observations, intent="search")
    log_step(f"SYNTHESIZED RESPONSE:\n-> {synthesized_answer[:300]}...")

    # 5. Real Reflection Agent Validation
    log_step("\n[4] Running real ReflectionAgent validation...")
    reflection_agent = ReflectionAgent()
    report = await reflection_agent.reflect(query=query, answer=synthesized_answer, history=observations)
    log_step(f"REFLECTION VERDICT:\n-> Status: {report.reflection_status} | Revise: {report.revise} | Fallback Used: {report.fallback_used}")
    if report.unsupported_claims:
        log_step(f"-> Unsupported Claims: {report.unsupported_claims}")

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
