from datetime import datetime, timezone
from typing import List, Optional

from pydantic import BaseModel, ConfigDict, Field


def _utc_now():
    return datetime.now(timezone.utc)

class NewsArticleBase(BaseModel):
    title: str
    content: str
    source: str
    url: str
    language: str
    published_date: datetime

class NewsArticle(NewsArticleBase):
    id: str = Field(..., alias="_id")
    cleaned_content: Optional[str] = None
    category: Optional[str] = None
    chunks: Optional[List[str]] = None
    embedding: Optional[List[float]] = None
    is_duplicate: bool = False
    created_at: datetime = Field(default_factory=_utc_now)

    model_config = ConfigDict(populate_by_name=True)
