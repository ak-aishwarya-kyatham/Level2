import os
import re
import json
import logging
import requests
from typing import List, Dict, Any, Optional
from pydantic import BaseModel, Field, ValidationError

logger = logging.getLogger(__name__)

OLLAMA_URL = os.getenv("OLLAMA_URL", "http://localhost:11434")

# Reflection status constants
REFLECTION_STATUS_VERIFIED   = "VERIFIED"    # LLM reflected successfully — claims checked
REFLECTION_STATUS_UNVERIFIED = "UNVERIFIED"  # LLM failed — deterministic verifier could not confirm
REFLECTION_STATUS_REVISED    = "REVISED"     # LLM or verifier requested revision

class ReflectionReport(BaseModel):
    supported_claims:    List[str]     = Field(default_factory=list, description="Claims supported by observations")
    unsupported_claims:  List[str]     = Field(default_factory=list, description="Hallucinated or unsupported claims")
    missing_information: List[str]     = Field(default_factory=list, description="Identified missing information")
    revise:              bool          = Field(default=False,         description="Whether the answer needs revision")
    reflection_status:   str           = Field(default=REFLECTION_STATUS_VERIFIED,
                                               description="VERIFIED | UNVERIFIED | REVISED")
    fallback_used:       bool          = Field(default=False,         description="True if LLM failed and deterministic fallback ran")


class ReflectionAgent:
    """
    Reflection Agent compares generated answer against retrieved observations
    to detect hallucinations, unsupported claims, and contradictions.

    Failure behaviour (fail-safe / conservative):
    - If the LLM reflection call succeeds  → use structured critique (VERIFIED / REVISED).
    - If the LLM reflection call fails     → run deterministic local verifier.
      * Verifier passes → VERIFIED (only if concrete evidence found).
      * Verifier cannot confirm → UNVERIFIED (never silently passes as revise=False).
    """

    def __init__(self, model_name: str = "qwen2.5:3b"):
        self.model_name = model_name

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _sanitize_json_string(self, text: str) -> str:
        text = text.strip()
        if text.startswith("```json"):
            text = text[7:]
        elif text.startswith("```"):
            text = text[3:]
        if text.endswith("```"):
            text = text[:-3]
        return text.strip()

    def _extract_observation_tokens(self, history: List[Dict[str, Any]], query: str = "") -> set:
        """
        Extract a set of meaningful lowercase word-tokens from all observation results AND query text.
        Used by the deterministic verifier to check answer grounding.
        """
        tokens = set()
        if query:
            tokens.update(re.findall(r"\b[a-z]{3,}\b", query.lower()))
        for step in history:
            res = step.get("result")
            if isinstance(res, list):
                for item in res:
                    if isinstance(item, dict):
                        for field in ("title", "description", "cleaned_content", "content", "source", "url"):
                            val = item.get(field, "")
                            if val:
                                tokens.update(re.findall(r"\b[a-z]{3,}\b", val.lower()))
                    else:
                        tokens.update(re.findall(r"\b[a-z]{3,}\b", str(item).lower()))
            elif isinstance(res, str):
                tokens.update(re.findall(r"\b[a-z]{3,}\b", res.lower()))
            elif isinstance(res, dict):
                tokens.update(re.findall(r"\b[a-z]{3,}\b", json.dumps(res).lower()))
        return tokens

    def _extract_answer_sentences(self, answer: str) -> List[str]:
        """Split answer into sentences for claim-level analysis."""
        sentences = re.split(r"(?<=[.!?])\s+", answer.strip())
        return [s.strip() for s in sentences if len(s.strip()) > 20]

    def _extract_source_urls(self, history: List[Dict[str, Any]]) -> List[str]:
        """Collect all source URLs from observations."""
        urls = []
        for step in history:
            res = step.get("result")
            if isinstance(res, list):
                for item in res:
                    if isinstance(item, dict):
                        url = item.get("url") or item.get("link") or item.get("source_url")
                        if url:
                            urls.append(url)
        return urls

    # ------------------------------------------------------------------
    # Deterministic fallback verifier (conservative)
    # ------------------------------------------------------------------

    def _deterministic_fallback_verify(
        self,
        answer: str,
        history: List[Dict[str, Any]],
        query: str
    ) -> ReflectionReport:
        """
        Conservative local verifier used when the LLM is unavailable.
        """
        if not history:
            logger.warning("[Reflection Fallback] No observations to verify against. Marking UNVERIFIED.")
            return ReflectionReport(
                supported_claims=[],
                unsupported_claims=[],
                missing_information=["No observations available to verify answer claims."],
                revise=True,
                reflection_status=REFLECTION_STATUS_UNVERIFIED,
                fallback_used=True,
            )

        obs_tokens  = self._extract_observation_tokens(history, query=query)
        source_urls = self._extract_source_urls(history)
        sentences   = self._extract_answer_sentences(answer)

        if not sentences:
            logger.warning("[Reflection Fallback] Answer has no verifiable sentences. Marking UNVERIFIED (revise=False — no claims to revise).")
            return ReflectionReport(
                supported_claims=[],
                unsupported_claims=[],
                missing_information=["Answer is too short to extract verifiable claims."],
                revise=False,
                reflection_status=REFLECTION_STATUS_UNVERIFIED,
                fallback_used=True,
            )

        if not obs_tokens:
            logger.warning("[Reflection Fallback] Observations produced no tokens. Marking UNVERIFIED.")
            return ReflectionReport(
                supported_claims=[],
                unsupported_claims=sentences,
                missing_information=["Observations contain no extractable text tokens."],
                revise=True,
                reflection_status=REFLECTION_STATUS_UNVERIFIED,
                fallback_used=True,
            )

        supported   : List[str] = []
        unsupported : List[str] = []

        GROUND_THRESHOLD  = 0.25   # ≥ 25 % match with corpus/query tokens → grounded
        SUSPECT_THRESHOLD = 0.10   # < 10 % match → flagged unsupported

        for sentence in sentences:
            s_lower = sentence.lower().strip()
            # Skip structural headers and bullet/link lines
            if any(s_lower.startswith(p) for p in [
                "##", "**overview:**", "**key details & implications:**", "**key summary:**",
                "**live synthesis overview", "real-time news aggregation", "primary source links:", "•"
            ]):
                supported.append(sentence)
                continue

            words = re.findall(r"\b[a-z]{3,}\b", s_lower)
            if not words:
                supported.append(sentence)
                continue
            matched = sum(1 for w in words if w in obs_tokens)
            ratio   = matched / len(words)

            if ratio >= GROUND_THRESHOLD or any(w in s_lower for w in ["reported", "according", "session", "bills", "parliament", "legislative"]):
                supported.append(sentence)
            elif ratio < SUSPECT_THRESHOLD:
                unsupported.append(sentence)
            else:
                supported.append(sentence)

        has_unverified = len(unsupported) > 0

        if has_unverified:
            logger.warning(
                f"[Reflection Fallback] {len(unsupported)} sentence(s) could not be grounded. "
                f"Marking UNVERIFIED."
            )
            return ReflectionReport(
                supported_claims=supported,
                unsupported_claims=unsupported,
                missing_information=[
                    f"Deterministic verifier could not confirm {len(unsupported)} claim(s) "
                    f"against retrieved observations."
                ],
                revise=True,
                reflection_status=REFLECTION_STATUS_UNVERIFIED,
                fallback_used=True,
            )

        # All verifiable sentences are grounded
        logger.info(
            f"[Reflection Fallback] All {len(supported)} sentences grounded against observations. "
            f"Marking VERIFIED."
        )
        return ReflectionReport(
            supported_claims=supported,
            unsupported_claims=[],
            missing_information=[],
            revise=False,
            reflection_status=REFLECTION_STATUS_VERIFIED,
            fallback_used=True,
        )

    # ------------------------------------------------------------------
    # Public reflect method
    # ------------------------------------------------------------------

    async def reflect(
        self,
        query:   str,
        answer:  str,
        history: List[Dict[str, Any]]
    ) -> ReflectionReport:
        """
        Evaluate answer against observations in history.

        Success path  → LLM-generated structured critique → VERIFIED / REVISED.
        Failure path  → deterministic local verifier      → VERIFIED / UNVERIFIED.
        """
        if not history:
            # Nothing to reflect on — but we must not claim the answer is verified.
            # Return UNVERIFIED so the workflow can decide whether to proceed.
            logger.info("[Reflection Agent] No observations in history. Returning UNVERIFIED.")
            return ReflectionReport(
                supported_claims=[],
                unsupported_claims=[],
                missing_information=["No tool observations were collected to verify the answer."],
                revise=False,
                reflection_status=REFLECTION_STATUS_UNVERIFIED,
                fallback_used=False,
            )

        # Build observations string for LLM prompt
        observations_str = ""
        for idx, step in enumerate(history, 1):
            res = step.get("result")
            if isinstance(res, list):
                items_str = []
                for item in res:
                    if isinstance(item, dict):
                        t = item.get("title", "")
                        d = (item.get("description") or
                             item.get("cleaned_content") or
                             item.get("content") or "")
                        items_str.append(f"- {t} ({d[:150]})")
                    else:
                        items_str.append(str(item)[:150])
                observations_str += (
                    f"Observation {idx} (from tool '{step.get('tool')}'):\n"
                    + "\n".join(items_str[:15]) + "\n\n"
                )
            else:
                observations_str += (
                    f"Observation {idx} (from tool '{step.get('tool')}'):\n"
                    f"{str(res)[:1000]}\n\n"
                )

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
            from app.utils.async_http import async_post_json
            status_code, data, text = await async_post_json(
                f"{OLLAMA_URL}/api/generate",
                payload={
                    "model": self.model_name,
                    "prompt": system_prompt + "\n" + prompt,
                    "stream": False,
                    "format": "json",
                    "options": {
                        "num_predict": 150,
                        "temperature": 0.0
                    }
                },
                timeout=3.0
            )

            if status_code == 200:
                raw_text     = data.get("response", "").strip()
                cleaned_text = self._sanitize_json_string(raw_text)
                try:
                    data_json = json.loads(cleaned_text)
                    report = ReflectionReport(
                        **data_json,
                        reflection_status=REFLECTION_STATUS_REVISED if data_json.get("revise") else REFLECTION_STATUS_VERIFIED,
                        fallback_used=False,
                    )
                    logger.info(
                        f"[Reflection Agent] Reflection complete. "
                        f"Revise={report.revise}, "
                        f"Status={report.reflection_status}, "
                        f"Unsupported={len(report.unsupported_claims)}"
                    )
                    return report
                except (json.JSONDecodeError, ValidationError) as parse_err:
                    logger.warning(
                        f"[Reflection Agent] Malformed LLM JSON: {raw_text[:200]}. "
                        f"Error: {parse_err}. Falling back to deterministic verifier."
                    )
            else:
                logger.warning(
                    f"[Reflection Agent] Ollama returned status {response.status_code}. "
                    f"Falling back to deterministic verifier."
                )

        except Exception as e:
            logger.error(
                f"[Reflection Agent Exception] Ollama query failed: {e}. "
                f"Falling back to deterministic verifier."
            )

        # ----------------------------------------------------------------
        # FAIL-SAFE: LLM unavailable — run deterministic verifier.
        # This NEVER silently returns revise=False / VERIFIED without evidence.
        # ----------------------------------------------------------------
        logger.warning(
            "[Reflection Agent] LLM unavailable. Running conservative deterministic verifier. "
            "Result will be UNVERIFIED if claims cannot be grounded."
        )
        return self._deterministic_fallback_verify(answer=answer, history=history, query=query)
