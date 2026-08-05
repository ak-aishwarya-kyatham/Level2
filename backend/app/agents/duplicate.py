import logging
import numpy as np
import re
import requests
from typing import List, Dict, Any, Tuple, Set
from datetime import datetime

logger = logging.getLogger(__name__)

OLLAMA_URL = "http://localhost:11434"

class DuplicateDetectionAgent:
    """
    Advanced duplicate news detection using title/content semantic embeddings,
    named entity comparison, and publication timestamps.
    """
    def __init__(self, title_threshold: float = 0.70, content_threshold: float = 0.70, combined_threshold: float = 0.55):
        self.title_threshold = title_threshold
        self.content_threshold = content_threshold
        self.combined_threshold = combined_threshold

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

    def get_ollama_embedding(self, text: str) -> List[float]:
        """Fetch text embeddings using Ollama bge-m3 model."""
        if not text or not text.strip():
            return []
        try:
            r = requests.post(
                f"{OLLAMA_URL}/api/embeddings",
                json={"model": "bge-m3:latest", "prompt": text.strip()[:1000]},
                timeout=5
            )
            if r.status_code == 200:
                return r.json().get("embedding", [])
        except Exception as e:
            logger.warning(f"Ollama embedding lookup failed: {e}")
        return []

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
