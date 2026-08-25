import os
import sys

def verify_live_trace():
    base_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
    candidate_paths = [
        os.path.join(base_dir, "tests", "live_agent_trace.txt"),
        os.path.join(os.path.dirname(base_dir), "tests", "live_agent_trace.txt")
    ]

    trace_path = None
    for p in candidate_paths:
        if os.path.exists(p):
            trace_path = p
            break

    if not trace_path:
        print("❌ Error: Could not locate tests/live_agent_trace.txt artifact.")
        sys.exit(1)

    with open(trace_path, "r", encoding="utf-8", errors="replace") as f:
        content = f.read()

    errors = []

    # 1. Assert stdio transport
    if "MCP TRANSPORT: stdio ClientSession (Active: True)" not in content and "MCP client session" not in content:
        errors.append("Trace lacks verified MCP stdio ClientSession connection proof.")

    # 2. Assert discovered tools from server
    if "DISCOVERED TOOLS FROM SERVER:" not in content:
        errors.append("Trace lacks dynamic tool discovery proof from MCP server.")

    # 3. Assert no offline fallback
    if "Fallback Used: True" in content:
        errors.append("Trace indicates reflection fallback was used.")

    if "Ollama circuit breaker active" in content or "Ollama fallback triggered" in content or "Ollama failed or is offline" in content:
        errors.append("Trace was generated with LLM fallback active — does not demonstrate live model reasoning.")

    # 4. Assert policy decisions and reflection verdicts exist
    if "POLICY DECISION:" not in content:
        errors.append("Trace lacks PolicyAgent decision entries.")

    if "REFLECTION VERDICT:" not in content:
        errors.append("Trace lacks ReflectionAgent validation entries.")

    if errors:
        print("⚠️ [TRACE VERIFICATION FAILED]:")
        for err in errors:
            print(f"  - {err}")
        print("\nRe-run `python scripts/run_live_trace.py` with active MCP session & Ollama qwen2.5:3b.")
        sys.exit(1)
    else:
        print("✅ [TRACE VERIFIED]: Trace confirms genuine MCP stdio session, dynamic tool discovery, LLM policy execution, and reflection validation.")
        sys.exit(0)

if __name__ == "__main__":
    verify_live_trace()
