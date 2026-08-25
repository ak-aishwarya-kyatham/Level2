import logging
import re

from bs4 import BeautifulSoup

from app.schemas.news import NewsArticleBase

logger = logging.getLogger(__name__)

class CleaningAgent:
    """
    Removes advertisements, HTML tags, scripts, extra spaces, empty paragraphs, and Unicode issues.
    Ensures HTML list items and block elements are separated with proper sentence boundaries.
    """
    def __init__(self):
        pass

    def clean_html(self, text: str) -> str:
        if not text:
            return ""

        soup = BeautifulSoup(text, "html.parser")

        # Remove script, style, and font attribution elements that pollute Google News RSS summaries
        for tag in soup(["script", "style", "font"]):
            tag.decompose()

        # Add period-space separators to block tags (p, li, div, br) so get_text() produces distinct sentences
        for block_tag in soup.find_all(["p", "li", "div", "br", "h1", "h2", "h3"]):
            block_tag.insert_after(". ")

        raw_text = soup.get_text(separator=" ")
        return raw_text

    def clean_text(self, text: str) -> str:
        # Remove HTML structure
        text = self.clean_html(text)

        # Remove repeated periods or double spaces caused by block element separation
        text = re.sub(r'\.\s*\.', '.', text)
        text = re.sub(r'\s+', ' ', text)

        # Remove trailing publisher attributions like "- Engadget", "- TechCrunch", "- BBC"
        text = re.sub(r'\s*[\-\|]\s*(?:Engadget|TechCrunch|Android Police|PPC Land|CoinDesk|Yahoo Finance|CNBC|Barron\'s|Investor\'s Business Daily|BBC|CNN|Reuters|AP News|NDTV News|The Hindu|Indian Express)\b.*?(?=\.|$)', '', text, flags=re.IGNORECASE)

        text = text.strip()
        return text

    async def run(self, article: NewsArticleBase) -> str:
        logger.info(f"Cleaning article: {article.title}")
        cleaned_content = self.clean_text(article.content)
        return cleaned_content
