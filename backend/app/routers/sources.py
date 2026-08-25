from fastapi import APIRouter

from app.repositories.news_repository import news_repository

router = APIRouter(prefix="/api/sources", tags=["sources"])

@router.get("/")
async def get_sources():
    sources = news_repository.get_sources()
    return {"sources": sources}
