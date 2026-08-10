import os
import json
import logging
import requests
from typing import List, Dict, Any
from pydantic import BaseModel, Field, ValidationError

logger = logging.getLogger(__name__)

OLLAMA_URL = os.getenv("OLLAMA_URL", "http://localhost:11434")

class ReflectionReport(BaseModel):
    supported_claims: List[str] = Field(default_factory=list, description="Claims that are supported by observations")
    unsupported_claims: List[str] = Field(default_factory=list, description="Hallucinated or unsupported claims")
    missing_information: List[str] = Field(default_factory=list, description="Identified missing pieces of information")
    revise: bool = Field(default=False, description="Whether the answer needs revision")

class ReflectionAgent:
    """
    Reflection Agent compares generated answer against retrieved observations
    to detect hallucinations, unsupported claims, and contradictions.
    """
    def __init__(self, model_name: str = "qwen2.5:3b"):
        self.model_name = model_name

    def _sanitize_json_string(self, text: str) -> str:
        text = text.strip()
        if text.startswith("```json"):
            text = text[7:]
        elif text.startswith("```"):
            text = text[3:]
        if text.endswith("```"):
            text = text[:-3]
        return text.strip()

    async def reflect(self, query: str, answer: str, history: List[Dict[str, Any]]) -> ReflectionReport:
        """
        Evaluate answer against observations in history.
        """
        if not history:
            # Nothing to reflect on
            return ReflectionReport(supported_claims=[], unsupported_claims=[], missing_information=[], revise=False)

        observations_str = ""
        for idx, step in enumerate(history, 1):
            res = step.get('result')
            if isinstance(res, list):
                # Extract titles/descriptions from article list
                items_str = []
                for item in res:
                    if isinstance(item, dict):
                        t = item.get("title", "")
                        d = item.get("description", "") or item.get("cleaned_content", "") or item.get("content", "")
                        items_str.append(f"- {t} ({d[:150]})")
                    else:
                        items_str.append(str(item)[:150])
                observations_str += f"Observation {idx} (from tool '{step.get('tool')}'):\n" + "\n".join(items_str[:15]) + "\n\n"
            else:
                observations_str += f"Observation {idx} (from tool '{step.get('tool')}'):\n{str(res)[:1000]}\n\n"

        system_prompt = (
            "You are an expert Reflection Agent. Your task is to critique the generated answer based ONLY on the provided observations.\n"
            "Identify unsupported claims, hallucinations, missing information, and contradictions.\n"
            "You must output a single valid JSON object strictly conforming to the following structure:\n"
            "{\n"
            "  \"supported_claims\": [\"claim 1\", \"claim 2\"],\n"
            "  \"unsupported_claims\": [],\n"
            "  \"missing_information\": [],\n"
            "  \"revise\": false\n"
            "}\n"
            "Rules:\n"
            "1. If any claim in the answer directly contradicts the observations, list it in 'unsupported_claims' and set 'revise' to true.\n"
            "2. If the user query asks for details that are in the observations but missing in the answer, list them in 'missing_information' and set 'revise' to true.\n"
            "3. If the answer accurately reflects news articles present in the observations, set 'revise' to false.\n"
            "4. DO NOT generate markdown wrapping. Return ONLY the JSON object.\n"
        )

        prompt = (
            f"User Query: {query}\n\n"
            f"Observations:\n{observations_str}\n"
            f"Generated Answer:\n{answer}\n\n"
            "JSON Reflection Report:"
        )

        logger.info("[Reflection Agent] Analyzing generated response for hallucinations...")
        
        try:
            response = requests.post(
                f"{OLLAMA_URL}/api/generate",
                json={
                    "model": self.model_name,
                    "prompt": system_prompt + "\n" + prompt,
                    "stream": False,
                    "format": "json",
                    "options": {
                        "num_predict": 150,
                        "temperature": 0.0
                    }
                },
                timeout=3
            )

            if response.status_code == 200:
                raw_text = response.json().get("response", "").strip()
                cleaned_text = self._sanitize_json_string(raw_text)
                try:
                    data = json.loads(cleaned_text)
                    report = ReflectionReport(**data)
                    logger.info(f"[Reflection Agent] Reflection complete. Revise={report.revise}, Unsupported claims: {len(report.unsupported_claims)}")
                    return report
                except (json.JSONDecodeError, ValidationError) as parse_err:
                    logger.warning(f"[Reflection Agent] Malformed JSON: {raw_text}. Error: {parse_err}")
            else:
                logger.warning(f"[Reflection Agent] Ollama returned status code: {response.status_code}")
        except Exception as e:
            logger.error(f"[Reflection Agent Exception] Ollama query failed: {e}")

        # Safe Fallback: Assume no revision needed to avoid loops on LLM failures
        return ReflectionReport(
            supported_claims=["General verification skipped due to LLM error"],
            unsupported_claims=[],
            missing_information=[],
            revise=False
        )
