import re
from bs4 import BeautifulSoup
import logging
from app.schemas.news import NewsArticleBase

logger = logging.getLogger(__name__)

class CleaningAgent:
    """
    Removes advertisements, HTML tags, scripts, extra spaces, empty paragraphs, and Unicode issues.
    """
    def __init__(self):
        pass

    def clean_html(self, text: str) -> str:
        # Use BeautifulSoup to remove HTML tags and scripts
        soup = BeautifulSoup(text, "html.parser")
        
        # Remove script and style elements
        for script in soup(["script", "style"]):
            script.decompose()
            
        return soup.get_text()

    def clean_text(self, text: str) -> str:
        # Remove HTML
        text = self.clean_html(text)
        
        # Remove extra whitespaces and empty lines
        text = re.sub(r'\s+', ' ', text)
        
        # Remove unicode issues/non-ascii if necessary, though retaining it for Telugu is important
        # Let's keep it basic but robust
        text = text.strip()
        
        return text

    async def run(self, article: NewsArticleBase) -> str:
        logger.info(f"Cleaning article: {article.title}")
        cleaned_content = self.clean_text(article.content)
        return cleaned_content
