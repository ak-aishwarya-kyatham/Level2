import logging
import re

logger = logging.getLogger(__name__)

class CategorizationAgent:
    """
    Classifies news into predefined categories efficiently using word boundary regex matching.
    """
    def __init__(self):
        self.categories = [
            "Politics", "Sports", "Business", "Technology",
            "Entertainment", "Health", "Education", "International",
            "Environment", "World"
        ]

    def _matches_keywords(self, text: str, keywords: list) -> bool:
        for kw in keywords:
            if " " in kw:
                if kw in text:
                    return True
            else:
                if re.search(rf'\b{re.escape(kw)}\b', text):
                    return True
        return False

    def _categorize_sync(self, content: str) -> str:
        """Synchronous version of run() for use in evaluation contexts.
        Identical logic to run() but does not require an event loop."""
        text_lower = content.lower()
        if self._matches_keywords(text_lower, ["tech", "technology", "artificial intelligence", "ai", "software", "apple", "google", "meta", "nvidia", "cybersecurity", "cyber", "semiconductor", "gadgets", "smartphone"]):
            return "Technology"
        elif self._matches_keywords(text_lower, ["climate", "environment", "pollution", "solar", "renewable", "carbon", "emissions", "green energy", "nature"]):
            return "Environment"
        elif self._matches_keywords(text_lower, ["stock", "stocks", "market", "markets", "economy", "economies", "bank", "banks", "billion", "inflation", "finance", "finances", "business", "tax", "earnings", "invest", "investor", "investors", "investment", "investments", "gdp"]):
            return "Business"
        elif self._matches_keywords(text_lower, ["cricket", "football", "match", "cup", "game", "tournament", "olympics", "nba", "tennis", "win", "scored", "champion"]):
            return "Sports"
        elif self._matches_keywords(text_lower, ["health", "hospital", "disease", "vaccine", "cancer", "medical", "doctor", "virus", "treatment", "medicine"]):
            return "Health"
        elif self._matches_keywords(text_lower, ["election", "minister", "government", "policy", "parliament", "congress", "bjp", "biden", "trump", "court", "law", "vote"]):
            return "Politics"
        elif self._matches_keywords(text_lower, ["movie", "film", "star", "actor", "actress", "music", "song", "box office", "show", "hollywood", "bollywood"]):
            return "Entertainment"
        elif self._matches_keywords(text_lower, ["global", "world", "war", "peace", "china", "russia", "ukraine", "summit", "un", "diplomatic", "foreign"]):
            return "World"
        else:
            return "General News"

    async def run(self, content: str) -> str:
        return self._categorize_sync(content)


