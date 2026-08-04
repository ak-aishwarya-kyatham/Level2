import logging
from app.workflows.langgraph_state import AgentState
from app.mcp_client import mcp_client
from app.database.qdrant import QdrantManager
from app.agents.embedding import EmbeddingAgent

logger = logging.getLogger(__name__)

qdrant_manager = QdrantManager()
embedding_agent = EmbeddingAgent()

def search_agent(state: AgentState) -> AgentState:
    logger.info("Search Agent processing request via MCP Client...")
    return state

async def retrieval_agent(state: AgentState) -> AgentState:
    logger.info("Retrieval Agent fetching live context via MCP Server...")
    query = state.get("query", "")
    
    # Check if this is a request for trending news
    if state.get("intent") == "trend":
        logger.info("Fetching top trending articles from dashboard analytics...")
        try:
            stats = await mcp_client.call_tool("get_dashboard_analytics")
            if isinstance(stats, dict) and "trending_topics" in stats:
                trending_topics = stats["trending_topics"]
                retrieved_docs = []
                for topic in trending_topics[:5]:
                    retrieved_docs.append({
                        "title": topic.get("topic", "Trending News Topic"),
                        "content": topic.get("description", "No description available."),
                        "source": topic.get("category", "Trending News"),
                        "url": topic.get("url", "#"),
                        "published_date": ""
                    })
                state["retrieved_documents"] = retrieved_docs
                logger.info(f"Retrieved {len(retrieved_docs)} trending articles.")
                return state
        except Exception as e:
            logger.error(f"Error fetching trending topics: {e}")
            
    # Detect if user provided text directly to be summarized
    import re
    user_provided_text = ""
    lower_query = query.lower().strip()
    prefixes = ["summarize this:", "summarize this text:", "summarize:", "summary of:"]
    for prefix in prefixes:
        if lower_query.startswith(prefix):
            user_provided_text = query[len(prefix):].strip()
            break
            
    if not user_provided_text and len(query) > 150 and "summarize" in lower_query:
        parts = re.split(r'(?i)\bsummarize\b\s*(?:this|the|text|following)?(?:\s+site|\s+article|\s+info)?\s*[:,-]?\s*', query)
        if len(parts) > 1 and len(parts[1].strip()) > 50:
            user_provided_text = parts[1].strip()
            
    if user_provided_text:
        logger.info("Detected user-provided text for direct summarization.")
        state["retrieved_documents"] = [{
            "title": "User Provided Text",
            "content": user_provided_text,
            "source": "User Input",
            "url": "#"
        }]
        return state

    retrieved_docs = []

    # 1. Invoke MCP Tool `search_live_news` via standard MCP protocol interface
    try:
        live_articles = await mcp_client.call_tool("search_live_news", {"query": query, "limit": 5})
        if isinstance(live_articles, list):
            for art in live_articles:
                retrieved_docs.append({
                    "title": art.get("title", "News Update"),
                    "content": art.get("cleaned_content") or art.get("content", ""),
                    "source": art.get("source", "Live Media"),
                    "url": art.get("url", "#"),
                    "published_date": art.get("published_date", "")
                })
    except Exception as e:
        logger.error(f"[MCP Agent Retrieval Error] {e}")

    # 2. Also query Qdrant vector index if active
    if qdrant_manager.client:
        try:
            query_vector = await embedding_agent.run(query)
            qdrant_results = qdrant_manager.search(query_vector=query_vector, top_k=3)
            for result in qdrant_results:
                payload = result.payload or {}
                retrieved_docs.append({
                    "title": payload.get("title", "Vector Match"),
                    "content": payload.get("content", ""),
                    "source": payload.get("source", "Qdrant Vector Index"),
                    "url": payload.get("url", "#")
                })
        except Exception as e:
            logger.warning(f"Qdrant retrieval skipped/failed: {e}")
        
    # Deduplicate retrieved documents by title
    seen_titles = set()
    unique_docs = []
    for doc in retrieved_docs:
        if doc["title"] not in seen_titles:
            seen_titles.add(doc["title"])
            unique_docs.append(doc)

    if not unique_docs:
        # Fallback to reading MCP Resource `news://store/articles`
        try:
            all_live = await mcp_client.read_resource("news://store/articles")
            if isinstance(all_live, list):
                unique_docs = [
                    {
                        "title": a.get("title", "News Headline"),
                        "content": a.get("content", ""),
                        "source": a.get("source", "Live Feeds"),
                        "url": a.get("url", "#")
                    }
                    for a in all_live[:3]
                ]
        except Exception as e:
            logger.error(f"Fallback resource read failed: {e}")

    state["retrieved_documents"] = unique_docs
    logger.info(f"MCP Retrieval complete. Found {len(unique_docs)} context items.")
    return state
