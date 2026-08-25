import json
import logging
import os
import time
from typing import Any, Dict, List, Literal, Optional, Tuple

from pydantic import BaseModel, Field, ValidationError, model_validator

logger = logging.getLogger(__name__)

OLLAMA_URL = os.getenv("OLLAMA_URL", "http://localhost:11434")

class PolicyAction(BaseModel):
    action: Literal["tool", "finish"] = Field(..., description="Action type must strictly be 'tool' or 'finish'")
    tool: Optional[str] = Field(None, description="Name of tool if action is 'tool'")
    arguments: Dict[str, Any] = Field(default_factory=dict, description="Arguments for tool if action is 'tool'")
    answer: Optional[str] = Field(None, description="Final response to user if action is 'finish'")
    thought: str = Field(default="Analyzing query and deciding action.", description="Detailed thoughts/reasoning behind the decision")
    is_valid: bool = Field(default=True, description="Whether action passed strict schema validation")
    validation_error: Optional[str] = Field(default=None, description="Structured error message if validation failed")

    @model_validator(mode="after")
    def validate_action_fields(self) -> "PolicyAction":
        if self.action == "tool":
            if not self.tool or not str(self.tool).strip():
                raise ValueError("Field 'tool' is required and cannot be empty when action is 'tool'.")
            if not isinstance(self.arguments, dict):
                raise ValueError("Field 'arguments' must be a dictionary/object when action is 'tool'.")
        elif self.action == "finish":
            if self.answer is None or not str(self.answer).strip():
                raise ValueError("Field 'answer' is required and cannot be empty when action is 'finish'.")
        return self

def validate_policy_action_against_mcp_schemas(
    action: PolicyAction,
    tools: List[Dict[str, Any]]
) -> Tuple[bool, str]:
    """
    Validates a PolicyAction against dynamically discovered MCP tool schemas.
    Checks:
    1. Action validity ('tool' or 'finish')
    2. Tool existence in MCP tool schemas
    3. Required arguments
    4. Argument types (string, integer, number, boolean, array, object)
    5. Value limits/ranges (minimum, maximum, minLength, maxLength, enum)
    """
    if not action.is_valid and action.validation_error:
        return False, action.validation_error

    if action.action == "finish":
        if action.answer is None or not str(action.answer).strip():
            return False, "INVALID FINISH ACTION: 'answer' field is required when action is 'finish'."
        return True, "Valid finish action."

    if action.action == "tool":
        tool_name = action.tool
        if not tool_name or not str(tool_name).strip():
            return False, "INVALID TOOL: 'tool' field is required when action is 'tool'."

        # Look up tool in MCP schemas
        matching_tool = None
        for t in tools:
            if t.get("name") == tool_name:
                matching_tool = t
                break

        if not matching_tool:
            available_names = [t.get("name") for t in tools if t.get("name")]
            return False, f"INVALID TOOL: Unknown tool '{tool_name}'. Available tools: {available_names}"

        # Validate arguments against inputSchema
        input_schema = matching_tool.get("inputSchema") or matching_tool.get("input_schema") or {}
        properties = input_schema.get("properties", {})
        required_fields = input_schema.get("required", [])

        args = action.arguments if isinstance(action.arguments, dict) else {}

        # 1. Required arguments check
        for req_field in required_fields:
            if req_field not in args or args[req_field] is None:
                return False, f"MISSING REQUIRED ARGUMENT: Tool '{tool_name}' requires argument '{req_field}'."

        # 2. Argument types & range checks
        for arg_key, arg_val in args.items():
            if arg_key in properties:
                prop_schema = properties[arg_key]
                expected_type = prop_schema.get("type")

                if expected_type:
                    valid_type = True
                    if expected_type == "string" and not isinstance(arg_val, str):
                        valid_type = False
                    elif expected_type == "integer" and (not isinstance(arg_val, int) or isinstance(arg_val, bool)):
                        valid_type = False
                    elif expected_type == "number" and (not isinstance(arg_val, (int, float)) or isinstance(arg_val, bool)):
                        valid_type = False
                    elif expected_type == "boolean" and not isinstance(arg_val, bool):
                        valid_type = False
                    elif expected_type == "array" and not isinstance(arg_val, list):
                        valid_type = False
                    elif expected_type == "object" and not isinstance(arg_val, dict):
                        valid_type = False

                    if not valid_type:
                        return False, f"INVALID ARGUMENT TYPE: Argument '{arg_key}' for tool '{tool_name}' must be of type '{expected_type}', got '{type(arg_val).__name__}'."

                # Range / value checks
                if "minimum" in prop_schema and isinstance(arg_val, (int, float)):
                    min_val = prop_schema["minimum"]
                    if arg_val < min_val:
                        return False, f"INVALID ARGUMENT VALUE: Argument '{arg_key}' ({arg_val}) is below minimum of {min_val}."

                if "maximum" in prop_schema and isinstance(arg_val, (int, float)):
                    max_val = prop_schema["maximum"]
                    if arg_val > max_val:
                        return False, f"INVALID ARGUMENT VALUE: Argument '{arg_key}' ({arg_val}) exceeds maximum of {max_val}."

                if "minLength" in prop_schema and isinstance(arg_val, str):
                    min_len = prop_schema["minLength"]
                    if len(arg_val) < min_len:
                        return False, f"INVALID ARGUMENT VALUE: Argument '{arg_key}' length ({len(arg_val)}) is below minLength of {min_len}."

                if "maxLength" in prop_schema and isinstance(arg_val, str):
                    max_len = prop_schema["maxLength"]
                    if len(arg_val) > max_len:
                        return False, f"INVALID ARGUMENT VALUE: Argument '{arg_key}' length ({len(arg_val)}) exceeds maxLength of {max_len}."

                if "enum" in prop_schema:
                    allowed_enum = prop_schema["enum"]
                    if arg_val not in allowed_enum:
                        return False, f"INVALID ARGUMENT VALUE: Argument '{arg_key}' ('{arg_val}') must be one of {allowed_enum}."

        return True, "Valid tool action."

    return False, f"INVALID ACTION: Unknown action type '{action.action}'."

def parse_and_validate_policy_action(
    raw_input: Any,
    tools: List[Dict[str, Any]]
) -> PolicyAction:
    """
    Parses raw LLM text, dict, or object into a PolicyAction and performs strict Pydantic + MCP schema validation.
    If parsing or validation fails, returns an invalid PolicyAction containing the structured validation error message.
    """
    if isinstance(raw_input, str):
        cleaned = raw_input.strip()
        if cleaned.startswith("```json"):
            cleaned = cleaned[7:]
        elif cleaned.startswith("```"):
            cleaned = cleaned[3:]
        if cleaned.endswith("```"):
            cleaned = cleaned[:-3]
        cleaned = cleaned.strip()

        try:
            data = json.loads(cleaned)
        except json.JSONDecodeError as err:
            return PolicyAction(
                action="tool",
                tool="invalid_json",
                arguments={},
                thought="Failed to parse LLM response as JSON.",
                is_valid=False,
                validation_error=f"MALFORMED OUTPUT: Response is not valid JSON. Error: {str(err)}"
            )
    elif isinstance(raw_input, dict):
        data = raw_input
    elif isinstance(raw_input, PolicyAction):
        # Validate existing PolicyAction against schemas
        is_valid, err_msg = validate_policy_action_against_mcp_schemas(raw_input, tools)
        raw_input.is_valid = is_valid
        raw_input.validation_error = err_msg if not is_valid else None
        return raw_input
    else:
        return PolicyAction(
            action="tool",
            tool="invalid_output",
            arguments={},
            thought="LLM response was neither string nor dict.",
            is_valid=False,
            validation_error="MALFORMED OUTPUT: LLM response must be a JSON string or dict."
        )

    if not isinstance(data, dict):
        return PolicyAction(
            action="tool",
            tool="invalid_json_structure",
            arguments={},
            thought="JSON payload is not an object/dict.",
            is_valid=False,
            validation_error="MALFORMED OUTPUT: JSON response must be a dictionary object."
        )

    # Attempt Pydantic model validation
    try:
        action_obj = PolicyAction.model_validate(data)
    except ValidationError as val_err:
        err_msg = val_err.errors()[0].get("msg", str(val_err))
        tool_name = str(data.get("tool") or "invalid_action")
        return PolicyAction(
            action="tool",
            tool=tool_name,
            arguments=data.get("arguments") if isinstance(data.get("arguments"), dict) else {},
            thought=str(data.get("thought", "Pydantic validation failed.")),
            is_valid=False,
            validation_error=f"INVALID OUTPUT: {err_msg}"
        )
    except Exception as ex:
        return PolicyAction(
            action="tool",
            tool="invalid_action",
            arguments={},
            thought="Parsing failed.",
            is_valid=False,
            validation_error=f"INVALID OUTPUT: {str(ex)}"
        )

    # Perform MCP schema & parameter validation
    is_valid, error_msg = validate_policy_action_against_mcp_schemas(action_obj, tools)
    action_obj.is_valid = is_valid
    action_obj.validation_error = error_msg if not is_valid else None
    return action_obj

_OLLAMA_OFFLINE_UNTIL = 0.0

def mark_ollama_offline(seconds: float = 60.0):
    global _OLLAMA_OFFLINE_UNTIL
    _OLLAMA_OFFLINE_UNTIL = time.time() + seconds

def is_ollama_circuit_open() -> bool:
    return time.time() < _OLLAMA_OFFLINE_UNTIL

class PolicyAgent:
    """
    Policy Agent that decides whether to invoke an MCP tool or finish the request.
    Uses local Ollama (Qwen) LLM and outputs strictly validated JSON.
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
        Queries Ollama to decide the next action and validates the result.
        """
        if is_ollama_circuit_open():
            fallback_reason = "Ollama circuit breaker active from previous timeout"
            logger.info(f"[Policy Agent] Circuit breaker active — skipping Ollama wait on iteration {iteration_count}.")
            if iteration_count == 1:
                fallback = PolicyAction(
                    action="tool",
                    tool="search_live_news",
                    arguments={"query": query, "limit": 10},
                    thought=f"Ollama circuit breaker active ({fallback_reason}). Falling back to search_live_news tool.",
                    is_valid=True
                )
            else:
                fallback = PolicyAction(
                    action="finish",
                    answer="No detailed information was found.",
                    thought=f"Ollama circuit breaker active ({fallback_reason}). Routing to local synthesis from retrieved observations.",
                    is_valid=True
                )
            return parse_and_validate_policy_action(fallback, tools)

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
                tool_name = step.get('tool', '')
                if tool_name in ["reflection_feedback", "reflection_critique"]:
                    history_str += f"Step {idx} [REFLECTION AGENT CRITIQUE - REVISION REQUIRED]:\n"
                    history_str += f"  Reflection Feedback: {step.get('result')}\n"
                    history_str += "  REVISION DIRECTIVE: The Reflection Agent flagged unsupported/unverified claims in the previous synthesis. You MUST choose a DIFFERENT tool or a MORE SPECIFIC search query/category (e.g. search_live_news with specific topic keywords) to retrieve grounded evidence to address these claims. Do NOT repeat identical tool calls with identical arguments!\n\n"
                    continue
                history_str += f"Step {idx}:\n"
                history_str += f"  Thought: {step.get('thought')}\n"
                history_str += f"  Tool Called: {tool_name}\n"
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
            "  \"answer\": \"concise summary of findings\" (required if action is 'finish'),\n"
            "  \"thought\": \"reasoning explaining why you are choosing this action\"\n"
            "}\n"
            "Rules:\n"
            "1. IMPORTANT: If no tools have been called yet (Observation History is empty), you MUST choose action='tool' and call search_live_news. You are NOT allowed to choose 'finish' without any observations.\n"
            "2. Analyze the user query and the observation history.\n"
            "3. If you need more information, call an available tool. Do not hallucinate.\n"
            "4. Only choose action='finish' AFTER you have at least one observation with real data. Provide a concise summary of findings in the 'answer' field (under 150 words).\n"
            "5. DO NOT generate markdown wrapping. Return ONLY the JSON object.\n"
            f"6. Current Iteration: {iteration_count}/5. If you reach iteration 5, you MUST choose 'finish'.\n"
            "7. CRITICAL: Do NOT call the exact same tool with the exact same arguments if observations are already present in the Observation History. Choose action='finish' once you have retrieved observations.\n"
            "8. REVISION DIRECTIVE: If a [REFLECTION AGENT CRITIQUE] step is present in the Observation History flagging unsupported claims, you MUST address those claims. If calling a tool, choose a DIFFERENT tool or use a MORE SPECIFIC search query (e.g., search_live_news with specific search keywords) to find supporting evidence. Do NOT repeat the exact same tool and arguments.\n"
        )

        prompt = (
            f"Available Tools:\n{tools_str}\n"
            f"Observation History:\n{history_str}\n"
            f"User Query: {query}\n\n"
            "JSON Action:"
        )

        ollama_timeout = float(os.getenv("OLLAMA_TIMEOUT", "15.0"))
        logger.info(f"[Policy Agent] Querying LLM on iteration {iteration_count} (Model: '{self.model_name}', Timeout: {ollama_timeout}s)...")
        fallback_reason = "Unknown Ollama error"
        try:
            from app.utils.async_http import async_post_json
            status_code, data, text = await async_post_json(
                f"{OLLAMA_URL}/api/generate",
                payload={
                    "model": self.model_name,
                    "prompt": system_prompt + "\n" + prompt,
                    "stream": False,
                    "format": "json",
                    "options": {
                        "num_predict": 250,
                        "temperature": 0.0
                    }
                },
                timeout=ollama_timeout
            )

            if status_code == 200:
                raw_text = data.get("response", "").strip()
                action = parse_and_validate_policy_action(raw_text, tools)
                if action.is_valid:
                    logger.info(f"[Policy Agent] Decided action: {action.action} (Valid: {action.is_valid}, Thought: {action.thought})")
                    return action
                else:
                    fallback_reason = f"LLM output validation failed: {action.validation_error}"
                    logger.warning(
                        f"[Policy Agent Warning] LLM response schema validation failed: {action.validation_error}\n"
                        f"Raw LLM output: {raw_text[:500]}"
                    )
            elif status_code == 404:
                fallback_reason = f"MODEL NOT PULLED: Ollama returned HTTP 404 for model '{self.model_name}'"
                logger.error(
                    f"[Policy Agent Error] MODEL NOT PULLED: Ollama returned HTTP 404 for model '{self.model_name}'. "
                    f"Please pull the model first by running: ollama pull {self.model_name}\nResponse text: {text}"
                )
            else:
                mark_ollama_offline(60.0)
                fallback_reason = f"Ollama HTTP POST /api/generate failed with status {status_code}: {text[:200]}"
                logger.error(
                    f"[Policy Agent Error] Ollama HTTP POST /api/generate failed with status {status_code}.\n"
                    f"Response text: {text[:500]}"
                )
        except Exception as e:
            exc_name = type(e).__name__
            fallback_reason = f"Ollama call threw {exc_name}: {str(e)}"
            logger.error(
                f"[Policy Agent Exception] Ollama /api/generate call threw {exc_name}: {str(e)}\n"
                f"URL: {OLLAMA_URL}/api/generate | Model: '{self.model_name}'"
            )
            import traceback
            logger.error(f"[Policy Agent Exception Traceback]:\n{traceback.format_exc()}")

        # Safe Fallback: Finish with local retrieval if possible, or simple search
        logger.warning(f"[Policy Agent] Triggering fallback branch due to: {fallback_reason}")
        if iteration_count == 1:
            fallback = PolicyAction(
                action="tool",
                tool="search_live_news",
                arguments={"query": query, "limit": 10},
                thought=f"Ollama fallback triggered ({fallback_reason}). Falling back to search_live_news tool.",
                is_valid=True
            )
        else:
            fallback = PolicyAction(
                action="finish",
                answer="No detailed information was found.",
                thought=f"Ollama fallback triggered ({fallback_reason}). Routing to local synthesis from retrieved observations.",
                is_valid=True
            )
        return parse_and_validate_policy_action(fallback, tools)


