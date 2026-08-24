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

    fallback_detected = False

    if "Fallback Used: True" in content:
        fallback_detected = True

    if "POLICY DECISION:" in content:
        policy_section = content.split("POLICY DECISION:")[1].split("[2]")[0]
        if "Ollama failed or is offline" in policy_section:
            fallback_detected = True

    if fallback_detected:
        msg = (
            "[WARNING] live_agent_trace.txt was generated with LLM fallback active — "
            "this does NOT demonstrate live model reasoning. Re-run scripts/run_live_trace.py "
            "with Ollama serving qwen2.5:3b before treating this as evidence."
        )
        try:
            print("⚠️ " + msg)
        except Exception:
            print("[WARNING] " + msg)
        sys.exit(1)
    else:
        try:
            print("✅ Trace shows genuine live LLM invocation.")
        except Exception:
            print("[OK] Trace shows genuine live LLM invocation.")
        sys.exit(0)

if __name__ == "__main__":
    verify_live_trace()
