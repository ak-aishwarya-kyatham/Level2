import os
import json
import logging
import time
import requests
from typing import List, Dict, Any
from pydantic import BaseModel, Field, ValidationError

logger = logging.getLogger(__name__)

OLLAMA_URL = os.getenv("OLLAMA_URL", "http://localhost:11434")

class PolicyAction(BaseModel):
    action: str = Field(..., description="Must be 'tool' or 'finish'")
    tool: str = Field(None, description="Name of tool if action is 'tool'")
    arguments: Dict[str, Any] = Field(default_factory=dict, description="Arguments for tool if action is 'tool'")
    answer: str = Field(None, description="Final response to user if action is 'finish'")
    thought: str = Field(default="Analyzing query and deciding action.", description="Detailed thoughts/reasoning behind the decision")

class PolicyAgent:
    """
    Policy Agent that decides whether to invoke an MCP tool or finish the request.
    Uses local Ollama (Qwen) LLM and outputs validated JSON.
    """
    def __init__(self, model_name: str = "qwen2.5:3b"):
        self.model_name = model_name

    def _sanitize_json_string(self, text: str) -> str:
        """Clean code block formatting or leading/trailing text from LLM JSON response."""
        text = text.strip()
        if text.startswith("```json"):
            text = text[7:]
        elif text.startswith("```"):
            text = text[3:]
        if text.endswith("```"):
            text = text[:-3]
        return text.strip()

    async def decide_action(self, query: str, tools: List[Dict[str, Any]], history: List[Dict[str, Any]], iteration_count: int) -> PolicyAction:
        """
        Queries Ollama to decide the next action.
        """
        # Format tools description
        tools_str = ""
        for tool in tools:
            tools_str += f"- Name: {tool.get('name')}\n"
            tools_str += f"  Description: {tool.get('description')}\n"
            tools_str += f"  Input Schema: {json.dumps(tool.get('inputSchema', {}))}\n\n"

        # Format history description
        history_str = ""
        if history:
            for idx, step in enumerate(history, 1):
                history_str += f"Step {idx}:\n"
                history_str += f"  Thought: {step.get('thought')}\n"
                history_str += f"  Tool Called: {step.get('tool')}\n"
                history_str += f"  Arguments: {json.dumps(step.get('arguments'))}\n"
                res = step.get('result')
                if isinstance(res, list):
                    items_str = []
                    for item in res:
                        if isinstance(item, dict):
                            t = item.get("title", "")
                            s = item.get("source", "")
                            c = item.get("cleaned_content") or item.get("content") or item.get("description") or ""
                            items_str.append(f"  - Title: {t}\n    Source: {s}\n    Content snippet: {c[:250]}")
                        else:
                            items_str.append(f"  - {str(item)[:250]}")
                    history_str += f"  Observation/Result ({len(res)} items):\n" + "\n".join(items_str[:10]) + "\n\n"
                else:
                    history_str += f"  Observation/Result: {str(res)[:1000]}\n\n"
        else:
            history_str = "No tools have been called yet.\n"

        system_prompt = (
            "You are a Policy Agent in an L2 Agentic loop. Your job is to decide whether to call a tool or finalize the answer.\n"
            "You must output a single valid JSON object strictly conforming to the following structure:\n"
            "{\n"
            "  \"action\": \"tool\" or \"finish\",\n"
            "  \"tool\": \"tool_name_here\" (required if action is 'tool'),\n"
            "  \"arguments\": { ... } (required if action is 'tool'),\n"
            "  \"answer\": \"your final synthesized answer\" (required if action is 'finish'),\n"
            "  \"thought\": \"reasoning explaining why you are choosing this action\"\n"
            "}\n"
            "Rules:\n"
            "1. IMPORTANT: If no tools have been called yet (Observation History is empty), you MUST choose action='tool' and call search_live_news. You are NOT allowed to choose 'finish' without any observations.\n"
            "2. Analyze the user query and the observation history.\n"
            "3. If you need more information, call an available tool. Do not hallucinate.\n"
            "4. Only choose action='finish' AFTER you have at least one observation with real data. Provide a complete, detailed answer in the 'answer' field.\n"
            "5. DO NOT generate markdown wrapping. Return ONLY the JSON object.\n"
            f"6. Current Iteration: {iteration_count}/5. If you reach iteration 5, you MUST choose 'finish'.\n"
        )

        prompt = (
            f"Available Tools:\n{tools_str}\n"
            f"Observation History:\n{history_str}\n"
            f"User Query: {query}\n\n"
            "JSON Action:"
        )

        logger.info(f"[Policy Agent] Querying LLM on iteration {iteration_count}...")
        
        try:
            response = requests.post(
                f"{OLLAMA_URL}/api/generate",
                json={
                    "model": self.model_name,
                    "prompt": system_prompt + "\n" + prompt,
                    "stream": False,
                    "format": "json",
                    "options": {
                        "num_predict": 300,
                        "temperature": 0.0
                    }
                },
                timeout=12
            )

            if response.status_code == 200:
                raw_text = response.json().get("response", "").strip()
                cleaned_text = self._sanitize_json_string(raw_text)
                try:
                    data = json.loads(cleaned_text)
                    # Normalize actions / check inputs
                    if data.get("action") == "finish" and not data.get("answer"):
                        data["answer"] = "No detailed information was found."
                    
                    action = PolicyAction(**data)
                    logger.info(f"[Policy Agent] Decided action: {action.action} (Thought: {action.thought})")
                    return action
                except (json.JSONDecodeError, ValidationError) as parse_err:
                    logger.warning(f"[Policy Agent] Malformed LLM response: {raw_text}. Error: {parse_err}")
            else:
                logger.warning(f"[Policy Agent] Ollama returned status code: {response.status_code}")
        except Exception as e:
            logger.error(f"[Policy Agent Exception] Ollama query failed: {e}")

        # Safe Fallback: Finish with local retrieval if possible, or simple search
        logger.info("[Policy Agent] Using fallback default action")
        if iteration_count == 1:
            return PolicyAction(
                action="tool",
                tool="search_live_news",
                arguments={"query": query, "limit": 10},
                thought="Ollama failed or is offline. Falling back to search_live_news tool."
            )
        else:
            return PolicyAction(
                action="finish",
                answer="",
                thought="Encountered an exception. Synthesizing answer from retrieved observations."
            )

