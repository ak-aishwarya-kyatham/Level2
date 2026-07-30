# NewsIntel AI – Multi-Agent Enterprise Platform

NewsIntel AI is a robust, multi-agent enterprise platform for aggregating, analyzing, and questioning news from various top-tier sources.

## Architecture Diagram

1. **Ingestion Pipeline (Background)**:
   APScheduler -> News Scraper -> Cleaner -> Categorizer (facebook/bart-large-mnli) -> Chunker -> Embedder (BAAI/bge-m3) -> Duplicate Detector -> MongoDB / Qdrant
2. **User Request Pipeline (LangGraph)**:
   User Query -> FastAPI -> Auth & Cache -> Triage Agent -> [Compare/Summary/Search/Trend] -> Retrieval Agent -> Ollama (Qwen) -> Response

## Services
- **Backend**: FastAPI
- **Frontend**: React + Vite + Tailwind CSS + shadcn/ui
- **Database**: MongoDB (Metadata), Qdrant (Vectors), Redis (Caching)
- **AI Models**: Ollama (Qwen), HuggingFace (BART, BGE-M3)

## Setup Guide
1. Ensure Docker and Docker Compose are installed.
2. Clone this repository.
3. Run `docker-compose up --build`
4. The frontend will be available at `http://localhost:5173`
5. The backend will be available at `http://localhost:8000`

## Database Schema
Refer to `backend/app/schemas/news.py` for the NewsArticle model representation.

## Agent Workflow Documentation
Refer to `backend/app/workflows/main_workflow.py` for the LangGraph routing logic.
