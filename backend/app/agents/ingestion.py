import asyncio
import logging
import feedparser
import urllib.parse
from typing import List, Tuple
from datetime import datetime
from email.utils import parsedate_to_datetime
from app.schemas.news import NewsArticleBase

logger = logging.getLogger(__name__)

class NewsIngestionAgent:
    """
    Collects live news automatically from comprehensive RSS feeds and dynamic Google News RSS queries.
    All RSS feeds are fetched in parallel via asyncio.gather() for maximum throughput.
    """

    def __init__(self):
        # Live RSS Feeds mapping across top news providers
        self.rss_feeds = {
            "Times of India": "https://timesofindia.indiatimes.com/rssfeedstopstories.cms",
            "TOI Tech": "https://timesofindia.indiatimes.com/rssfeeds/66949542.cms",
            "The Hindu": "https://www.thehindu.com/news/national/feeder/default.rss",
            "Hindustan Times": "https://www.hindustantimes.com/feeds/rss/topnews/rssfeed.xml",
            "Indian Express": "https://indianexpress.com/feed/",
            "BBC News": "http://feeds.bbci.co.uk/news/rss.xml",
            "BBC Tech": "http://feeds.bbci.co.uk/news/technology/rss.xml",
            "TechCrunch": "https://techcrunch.com/feed/",
            "Economic Times": "https://economictimes.indiatimes.com/rssfeedstopstories.cms",
            "NDTV News": "https://feeds.feedburner.com/ndtvnews-top-stories",
            "Google News": "https://news.google.com/rss?hl=en-US&gl=US&ceid=US:en",
            "Google Tech News": "https://news.google.com/rss/headlines/section/topic/TECHNOLOGY?hl=en-US&gl=US&ceid=US:en",
            "Google Business": "https://news.google.com/rss/headlines/section/topic/BUSINESS?hl=en-US&gl=US&ceid=US:en"
        }

    def _parse_entry(self, entry: any, source: str, fallback_url: str) -> NewsArticleBase:
        """Parse a single feed entry into a NewsArticleBase object."""
        try:
            if hasattr(entry, 'published'):
                pub_date = parsedate_to_datetime(entry.published)
            elif hasattr(entry, 'updated'):
                pub_date = parsedate_to_datetime(entry.updated)
            else:
                pub_date = datetime.utcnow()
        except Exception:
            pub_date = datetime.utcnow()

        content = getattr(entry, 'description', '') or getattr(entry, 'summary', '') or getattr(entry, 'title', 'Live news update.')
        title = getattr(entry, 'title', 'Untitled News Article')

        return NewsArticleBase(
            title=title,
            content=content,
            source=source,
            url=getattr(entry, 'link', fallback_url),
            language="English",
            published_date=pub_date
        )

    async def _fetch_single_feed(self, source: str, url: str) -> Tuple[str, List[NewsArticleBase]]:
        """Fetch a single RSS feed and return (source, articles) tuple."""
        try:
            feed = await asyncio.to_thread(feedparser.parse, url)
            articles = [self._parse_entry(entry, source, url) for entry in feed.entries[:8]]
            logger.info(f"Fetched {len(articles)} articles from {source}")
            return source, articles
        except Exception as e:
            logger.error(f"Error fetching live feed for {source}: {str(e)}")
            return source, []

    async def run(self) -> List[NewsArticleBase]:
        """
        Fetch all RSS feeds in parallel using asyncio.gather().
        This reduces total ingestion time from ~13s (sequential) to ~1-2s (parallel).
        """
        logger.info(f"Starting parallel Live News Ingestion across {len(self.rss_feeds)} RSS feeds...")

        # Launch all feed fetches simultaneously
        tasks = [self._fetch_single_feed(source, url) for source, url in self.rss_feeds.items()]
        results = await asyncio.gather(*tasks)

        articles: List[NewsArticleBase] = []
        for _source, feed_articles in results:
            articles.extend(feed_articles)

        logger.info(f"Parallel ingestion completed. Fetched {len(articles)} real articles.")
        return articles

    async def fetch_dynamic_topic_news(self, query: str, limit: int = 10) -> List[NewsArticleBase]:
        """
        Dynamically fetches live articles from Google News RSS for any custom user search query.
        """
        encoded_query = urllib.parse.quote(query)
        feed_url = f"https://news.google.com/rss/search?q={encoded_query}&hl=en-IN&gl=IN&ceid=IN:en"
        logger.info(f"Fetching dynamic news for query '{query}' from: {feed_url}")

        articles = []
        try:
            feed = await asyncio.to_thread(feedparser.parse, feed_url)
            for entry in feed.entries[:limit]:
                try:
                    if hasattr(entry, 'published'):
                        pub_date = parsedate_to_datetime(entry.published)
                    else:
                        pub_date = datetime.utcnow()
                except Exception:
                    pub_date = datetime.utcnow()

                source_title = "Google News"
                if hasattr(entry, 'source') and hasattr(entry.source, 'title'):
                    source_title = entry.source.title

                content = getattr(entry, 'summary', getattr(entry, 'description', entry.title))

                article = NewsArticleBase(
                    title=entry.title,
                    content=content,
                    source=source_title,
                    url=getattr(entry, 'link', '#'),
                    language="English",
                    published_date=pub_date
                )
                articles.append(article)
        except Exception as e:
            logger.error(f"Error fetching dynamic search news for '{query}': {e}")

        return articles
