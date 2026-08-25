import hashlib
import logging
import re
from datetime import datetime
from typing import Any, Dict, List, Optional, Set, Tuple

import numpy as np
import requests

logger = logging.getLogger(__name__)

OLLAMA_URL = "http://localhost:11434"

def generate_fallback_embedding(text: str, dim: int = 1024) -> List[float]:
    """
    Deterministic L2-normalized term-frequency feature vector generator used when
    production ML models (Ollama/BAAI/bge-m3) or network services are unavailable.
    """
    if not text or not text.strip():
        return [0.0] * dim
    words = re.findall(r"\w+", text.lower())
    if not words:
        return [0.0] * dim
    vec = np.zeros(dim, dtype=float)
    for word in words:
        idx = int(hashlib.md5(word.encode("utf-8")).hexdigest(), 16) % dim
        vec[idx] += 1.0
    norm = np.linalg.norm(vec)
    if norm > 0:
        vec = vec / norm
    return vec.tolist()


class DuplicateDetectionAgent:
    """
    Advanced duplicate news detection using title/content semantic embeddings,
    named entity comparison, and publication timestamps.
    """
    def __init__(
        self,
        title_threshold: float = 0.70,
        content_threshold: float = 0.70,
        combined_threshold: float = 0.55,
        embedding_provider: Optional[Any] = None
    ):
        self.title_threshold = title_threshold
        self.content_threshold = content_threshold
        self.combined_threshold = combined_threshold
        self.embedding_provider = embedding_provider

    def cosine_similarity(self, vec1: List[float], vec2: List[float]) -> float:
        if not vec1 or not vec2:
            return 0.0
        v1 = np.array(vec1)
        v2 = np.array(vec2)
        norm1 = np.linalg.norm(v1)
        norm2 = np.linalg.norm(v2)
        if norm1 == 0 or norm2 == 0:
            return 0.0
        return float(np.dot(v1, v2) / (norm1 * norm2))

    async def run(self, target_embedding: List[float], existing_embeddings: List[List[float]], threshold: float = 0.85) -> Tuple[bool, float]:
        """
        Checks if target_embedding is a duplicate against a list of existing vector embeddings.
        Returns (is_duplicate: bool, max_similarity: float).
        """
        if not target_embedding or not existing_embeddings:
            return False, 0.0

        max_sim = 0.0
        for existing in existing_embeddings:
            if existing:
                sim = self.cosine_similarity(target_embedding, existing)
                if sim > max_sim:
                    max_sim = sim

        is_dup = max_sim >= threshold
        return is_dup, max_sim

    def get_ollama_embedding(self, text: str) -> List[float]:
        """Fetch text embeddings using injected provider, Ollama bge-m3 model, or local fallback."""
        if not text or not text.strip():
            return []

        # 1. Dependency injection check
        if self.embedding_provider is not None:
            try:
                if callable(self.embedding_provider):
                    emb = self.embedding_provider(text)
                elif hasattr(self.embedding_provider, "get_embedding_sync"):
                    emb = self.embedding_provider.get_embedding_sync(text)
                elif hasattr(self.embedding_provider, "get_embedding"):
                    emb = self.embedding_provider.get_embedding(text)
                else:
                    emb = []
                if emb:
                    return emb
            except Exception as e:
                logger.warning(f"Injected embedding provider error: {e}")

        # 2. Ollama bge-m3 model lookup
        if not hasattr(self, "_ollama_failed"):
            self._ollama_failed = False

        if not self._ollama_failed:
            try:
                r = requests.post(
                    f"{OLLAMA_URL}/api/embeddings",
                    json={"model": "bge-m3:latest", "prompt": text.strip()[:1000]},
                    timeout=0.3
                )
                if r.status_code == 200:
                    emb = r.json().get("embedding", [])
                    if emb:
                        return emb
            except Exception as e:
                self._ollama_failed = True
                logger.warning(f"Ollama embedding lookup failed ({e}), using local fallback.")

        # 3. Local BAAI/bge-m3 via EmbeddingAgent
        try:
            from app.agents.embedding import EmbeddingAgent
            if not hasattr(self, "_fallback_agent"):
                self._fallback_agent = EmbeddingAgent()
            emb = self._fallback_agent.get_embedding_sync(text)
            if emb:
                return emb
        except Exception as fallback_err:
            logger.error(f"Fallback embedding failed: {fallback_err}")

        # 4. Safe deterministic fallback vector when ML models/network are offline
        return generate_fallback_embedding(text)



    def extract_named_entities(self, text: str) -> Set[str]:
        """Extract coarse named entities (capitalized words) for overlap analysis."""
        if not text:
            return set()
        candidates = re.findall(r"\b[A-Z][a-zA-Z0-9-]+\b", text)
        ignore_words = {
            "Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday",
            "Reuters", "Techcrunch", "Bloomberg", "Gartner", "Engadget", "Wired", "AP", "AFP",
            "The", "A", "An", "In", "On", "At", "For", "With", "By", "To", "From", "Of", "And",
            "Is", "Are", "Was", "Were", "Has", "Have", "Had", "But", "Or", "Not"
        }
        return {c for c in candidates if c not in ignore_words and len(c) > 2}

    def parse_date(self, date_str: str) -> datetime:
        if not date_str:
            return None
        dt = None
        for fmt in ("%Y-%m-%dT%H:%M:%SZ", "%Y-%m-%dT%H:%M:%S", "%Y-%m-%d %H:%M:%S", "%a, %d %b %Y %H:%M:%S"):
            try:
                dt = datetime.strptime(date_str, fmt)
                break
            except Exception:
                continue
        if dt is None:
            try:
                dt = datetime.fromisoformat(date_str.replace("Z", "+00:00"))
            except Exception:
                pass
        if dt is not None:
            if dt.tzinfo is not None:
                from datetime import timezone
                dt = dt.astimezone(timezone.utc).replace(tzinfo=None)
            return dt
        return None

    def date_diff_hours(self, date1: str, date2: str) -> float:
        d1 = self.parse_date(date1)
        d2 = self.parse_date(date2)
        if not d1 or not d2:
            return 72.0  # Assumed far apart if missing
        return abs((d1 - d2).total_seconds()) / 3600.0

    def are_duplicates(self, art1: Dict[str, Any], art2: Dict[str, Any]) -> Tuple[bool, float]:
        """
        Detects if two articles describe the same news event based on semantic,
        entity, and temporal overlap. Uses multi-stage filtering for speed.
        """
        title1 = art1.get("title", "")
        title2 = art2.get("title", "")

        # 1. Semantic Similarity between Titles
        emb_title1 = art1.get("title_emb")
        if emb_title1 is None:
            emb_title1 = self.get_ollama_embedding(title1)
            art1["title_emb"] = emb_title1

        emb_title2 = art2.get("title_emb")
        if emb_title2 is None:
            emb_title2 = self.get_ollama_embedding(title2)
            art2["title_emb"] = emb_title2

        title_sim = self.cosine_similarity(emb_title1, emb_title2)

        # Early exit if titles are completely different to save content embedding API calls
        if title_sim < 0.40:
            return False, title_sim

        content1 = art1.get("content", "")
        content2 = art2.get("content", "")

        # 2. Named Entity Overlap
        ent1 = self.extract_named_entities(title1 + " " + content1[:250])
        ent2 = self.extract_named_entities(title2 + " " + content2[:250])
        common_ent = ent1.intersection(ent2)

        # 3. Publication Timestamp Difference
        hours_diff = self.date_diff_hours(art1.get("published_date"), art2.get("published_date"))

        # 4. Semantic Similarity between Contents (lazily loaded on-demand)
        emb_content1 = art1.get("content_emb")
        if emb_content1 is None:
            emb_content1 = self.get_ollama_embedding(content1[:500])
            art1["content_emb"] = emb_content1

        emb_content2 = art2.get("content_emb")
        if emb_content2 is None:
            emb_content2 = self.get_ollama_embedding(content2[:500])
            art2["content_emb"] = emb_content2

        content_sim = self.cosine_similarity(emb_content1, emb_content2)

        # Deduplication Decision Matrix
        is_dup = False
        max_sim = max(title_sim, content_sim)

        if title_sim >= self.title_threshold:
            is_dup = True
        elif content_sim >= self.content_threshold and len(common_ent) >= 2:
            is_dup = True
        elif title_sim >= self.combined_threshold and content_sim >= self.combined_threshold:
            if hours_diff <= 48.0 or len(common_ent) >= 2:
                is_dup = True

        logger.info(
            f"Deduplication comparison:\n"
            f"  Titles: '{title1[:40]}...' vs '{title2[:40]}...'\n"
            f"  Title Sim: {title_sim:.4f}, Content Sim: {content_sim:.4f}, Entities overlap: {len(common_ent)}, Hours diff: {hours_diff:.1f}h -> Duplicate: {is_dup}"
        )
        return is_dup, max_sim

    def get_source_score(self, source: str) -> int:
        src = (source or "").lower()
        if any(x in src for x in ["reuters", "bloomberg", "gartner", "wsj", "nytimes"]):
            return 3
        if any(x in src for x in ["techcrunch", "engadget", "wired"]):
            return 2
        return 1

    def choose_better_article(self, art1: Dict[str, Any], art2: Dict[str, Any]) -> Dict[str, Any]:
        """
        Merge duplicates by picking the highest-quality source based on:
        - Source credibility
        - Article completeness
        - Freshness
        - Richness of information
        """
        score1 = self.get_source_score(art1.get("source", "")) * 10
        score2 = self.get_source_score(art2.get("source", "")) * 10

        # Completeness (length of content)
        len1 = len(art1.get("content", ""))
        len2 = len(art2.get("content", ""))
        score1 += min(len1 / 200, 5)
        score2 += min(len2 / 200, 5)

        # Freshness (younger article preferred if scores are close)
        d1 = self.parse_date(art1.get("published_date"))
        d2 = self.parse_date(art2.get("published_date"))
        if d1 and d2:
            if d1 > d2:
                score1 += 2
            elif d2 > d1:
                score2 += 2

        # Richness (number of named entities)
        ent1 = len(self.extract_named_entities(art1.get("title", "") + " " + art1.get("content", "")))
        ent2 = len(self.extract_named_entities(art2.get("title", "") + " " + art2.get("content", "")))
        score1 += min(ent1 / 3, 3)
        score2 += min(ent2 / 3, 3)

        if score1 >= score2:
            return art1
        return art2
