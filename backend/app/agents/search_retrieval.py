import logging
import re
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

def extract_topic_filter(query: str) -> str:
    """
    Extracts specific topic/entity/location filters from conversational trend/feed queries.
    e.g. 'what is the treding topic of andhra pradesh' -> 'andhra pradesh'
    e.g. 'what is the trending topic of telangana state' -> 'telangana'
    e.g. 'trending topics in kerala' -> 'kerala'
    """
    q = query.lower().strip()
    q = re.sub(r"(?i)\b(what is|what are|tell me|give me|show me|latest|top|trending|treding|trednign|trnd|trend|topics?|news|live|feed|feeds|dashboard|nation|world|today|now)\b", "", q)
    q = re.sub(r"(?i)\b(of|in|about|for|on|the|a|an)\b", "", q)
    q = re.sub(r"\s+", " ", q).strip()
    if q.endswith(" state"):
        q = q[:-6].strip()
    if q.isdigit():
        return ""
    return q


async def retrieval_agent(state: AgentState) -> AgentState:
    logger.info("Retrieval Agent fetching live context via MCP Server...")
    query = state.get("query", "")
    extracted_topic = state.get("extracted_topic", "")
    expanded_query = state.get("expanded_query", "") or query
    target_category = state.get("target_category", "All")
    target_url = state.get("target_url", "")
    
    # ── Live Feed intent: fetch latest real-time articles sorted by date ───
    if state.get("intent") == "live_feed":
        logger.info("Fetching latest live feed articles...")
        try:
            from app.repositories.news_repository import news_repository
            articles = news_repository.articles
            if not articles or len(articles) < 5:
                await news_repository.ingest_all_sources()
                articles = news_repository.articles

            # Sort strictly by publication date (newest first)
            recent_articles = sorted(
                articles,
                key=lambda x: str(x.get("published_date", "")),
                reverse=True
            )[:15]

            retrieved_docs = []
            for art in recent_articles:
                retrieved_docs.append({
                    "title": art.get("title", ""),
                    "content": art.get("cleaned_content") or art.get("content") or "",
                    "source": art.get("source", "Live Feed"),
                    "category": art.get("category", "General"),
                    "published_date": art.get("published_date", ""),
                    "url": art.get("url", "#"),
                    "is_live_feed": True,
                })
            state["retrieved_documents"] = retrieved_docs
            state["intent"] = "live_feed"
            logger.info(f"Retrieved {len(retrieved_docs)} live feed articles.")
            return state
        except Exception as e:
            logger.error(f"Error fetching live feed articles: {e}")

    # ── Trend intent: fetch ranked trending topics ───
    if state.get("intent") == "trend":
        topic_filter = extract_topic_filter(query)
        logger.info(f"Processing trend intent (topic_filter='{topic_filter}')...")

        if topic_filter:
            try:
                from app.repositories.news_repository import news_repository
                from datetime import datetime

                matching_arts = await news_repository.search_articles(query=topic_filter, limit=25)
                if len(matching_arts) < 3:
                    try:
                        live_items = await news_repository.ingestion_agent.fetch_dynamic_topic_news(topic_filter, limit=15)
                        existing_urls = {a.get("url") for a in news_repository.articles if a.get("url")}
                        for item in live_items:
                            if item.url not in existing_urls:
                                cleaned_text = await news_repository.cleaning_agent.run(item)
                                cat = await news_repository.categorization_agent.run(cleaned_text)
                                art_dict = {
                                    "id": news_repository._generate_id(item.url, item.title),
                                    "title": item.title,
                                    "content": item.content,
                                    "cleaned_content": cleaned_text,
                                    "source": item.source,
                                    "url": item.url,
                                    "language": item.language,
                                    "category": cat,
                                    "published_date": item.published_date.isoformat() if hasattr(item.published_date, 'isoformat') else str(item.published_date),
                                    "created_at": datetime.utcnow().isoformat()
                                }
                                news_repository.articles.append(art_dict)
                                matching_arts.append(art_dict)
                        news_repository._save_to_disk()
                    except Exception as ex:
                        logger.error(f"Error fetching live topic news for trend '{topic_filter}': {ex}")

                if matching_arts:
                    stop_words = {"the", "a", "an", "in", "on", "of", "for", "to", "and", "is", "at", "with", "by", "from", "as", "after", "over", "new", "top", "today", "says", "said", "has", "have", "been", "was", "were", "are"}
                    clusters = []
                    for art in matching_arts:
                        title = art.get("title", "")
                        if not title:
                            continue
                        words = set(re.findall(r'\b[a-zA-Z]{3,}\b', title.lower())) - stop_words
                        found = False
                        for cl in clusters:
                            if words.intersection(cl["keywords"]) or len(cl["articles"]) == 1:
                                cl["articles"].append(art)
                                cl["keywords"].update(words)
                                found = True
                                break
                        if not found:
                            clusters.append({
                                "representative": art,
                                "articles": [art],
                                "keywords": set(words),
                                "category": art.get("category", "General News")
                            })

                    clusters.sort(key=lambda c: len(c["articles"]), reverse=True)
                    retrieved_docs = []
                    for idx, cl in enumerate(clusters[:10], 1):
                        rep = cl["representative"]
                        cl_count = len(cl["articles"])
                        snippet = (rep.get("cleaned_content") or rep.get("content") or rep.get("title") or "")[:200].strip()
                        retrieved_docs.append({
                            "title": rep.get("title", "Trending Topic"),
                            "content": snippet,
                            "source": rep.get("source", "Live Source"),
                            "category": rep.get("category", "General News"),
                            "url": rep.get("url", "#"),
                            "published_date": rep.get("published_date", ""),
                            "rank": idx,
                            "article_count": cl_count,
                            "velocity": f"+{min(45, max(12, cl_count * 15))}% growth",
                            "is_trending": True,
                            "topic_filter": topic_filter,
                        })
                    state["retrieved_documents"] = retrieved_docs
                    state["intent"] = "trend"
                    logger.info(f"Retrieved {len(retrieved_docs)} topic-filtered trending topics for '{topic_filter}'.")
                    return state
            except Exception as e:
                logger.error(f"Error filtering trending topics for '{topic_filter}': {e}")

        logger.info("Fetching top global trending topics from dashboard analytics...")
        try:
            stats = await mcp_client.call_tool("get_dashboard_analytics")
            if isinstance(stats, dict) and "trending_topics" in stats:
                trending_topics = stats["trending_topics"]
                retrieved_docs = []
                for topic in trending_topics[:10]:
                    retrieved_docs.append({
                        "title": topic.get("topic", "Trending News Topic"),
                        "content": topic.get("description", ""),
                        "source": topic.get("category", "Trending"),
                        "url": topic.get("url", "#"),
                        "published_date": "",
                        "rank": topic.get("rank", 0),
                        "article_count": topic.get("article_count", topic.get("count", 0)),
                        "velocity": topic.get("velocity", ""),
                        "category": topic.get("category", "General News"),
                        "is_trending": True,
                    })
                state["retrieved_documents"] = retrieved_docs
                state["intent"] = "trend"
                logger.info(f"Retrieved {len(retrieved_docs)} trending topics.")
                return state
        except Exception as e:
            logger.error(f"Error fetching trending topics: {e}")

    # Check if this is a request to compare news sources
    if state.get("intent") == "compare" or (" vs " in query.lower() or "compare" in query.lower()):
        logger.info("Executing news source comparison retrieval...")
        known_sources = [
            "Times of India", "The Hindu", "Indian Express", "Hindustan Times",
            "BBC News", "BBC Tech", "TechCrunch", "Economic Times", "NDTV News",
            "Google News", "Google Tech News", "Google Business", "TOI Tech"
        ]
        
        detected_sources = []
        q_lower = query.lower()
        
        if "times of india" in q_lower or "toi" in q_lower:
            detected_sources.append("Times of India")
        if "hindu" in q_lower:
            detected_sources.append("The Hindu")
        if "indian express" in q_lower:
            detected_sources.append("Indian Express")
        if "bbc" in q_lower:
            detected_sources.append("BBC News")
        if "techcrunch" in q_lower:
            detected_sources.append("TechCrunch")
        if "ndtv" in q_lower:
            detected_sources.append("NDTV News")
        if "economic times" in q_lower:
            detected_sources.append("Economic Times")
            
        for src in known_sources:
            if src.lower() in q_lower and src not in detected_sources:
                detected_sources.append(src)
                
        detected_sources = list(dict.fromkeys(detected_sources))
        
        if len(detected_sources) >= 2:
            source1, source2 = detected_sources[0], detected_sources[1]
        elif len(detected_sources) == 1:
            source1 = detected_sources[0]
            source2 = "The Hindu" if source1 != "The Hindu" else "Times of India"
        else:
            source1, source2 = "Times of India", "The Hindu"
            
        logger.info(f"Comparing source coverage: '{source1}' vs '{source2}'")
        
        from app.repositories.news_repository import news_repository
        s1_arts = await news_repository.search_articles(source=source1, limit=10)
        s2_arts = await news_repository.search_articles(source=source2, limit=10)
        
        if len(s1_arts) < 3:
            s1_arts = await news_repository.search_articles(query=source1, limit=10)
        if len(s2_arts) < 3:
            s2_arts = await news_repository.search_articles(query=source2, limit=10)
            
        comparison_docs = []
        for a in s1_arts[:4]:
            doc = dict(a)
            doc["comparison_source"] = source1
            comparison_docs.append(doc)
            
        for a in s2_arts[:4]:
            doc = dict(a)
            doc["comparison_source"] = source2
            comparison_docs.append(doc)
            
        state["retrieved_documents"] = comparison_docs
        state["source1"] = source1
        state["source2"] = source2
        state["intent"] = "compare"
        return state
            
    # Detect if user provided text directly to be summarized
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

    # Handle Target URL Prioritization (Requirement 7)
    if target_url:
        logger.info(f"Target URL detected: {target_url}. Prioritizing direct retrieval.")
        found_art = None
        try:
            all_live = await mcp_client.read_resource("news://store/articles")
            if isinstance(all_live, list):
                for a in all_live:
                    if a.get("url") == target_url:
                        found_art = a
                        break
        except Exception as e:
            logger.error(f"Error checking resource for target URL: {e}")
            
        if not found_art:
            try:
                live_articles = await mcp_client.call_tool("search_live_news", {"query": target_url, "limit": 5})
                if isinstance(live_articles, list) and live_articles:
                    found_art = live_articles[0]
            except Exception as e:
                logger.error(f"Error calling search_live_news for URL: {e}")
                
        if found_art:
            logger.info(f"Found article matching URL: {found_art.get('title')}")
            broader_context = any(k in query.lower() for k in ["broader", "related", "context", "more", "compare", "others"])
            main_doc = {
                "title": found_art.get("title", "Target Article"),
                "content": found_art.get("cleaned_content") or found_art.get("content", ""),
                "source": found_art.get("source", "Direct Link"),
                "url": found_art.get("url", target_url),
                "published_date": found_art.get("published_date", "")
            }
            if not broader_context:
                state["retrieved_documents"] = [main_doc]
                return state
            else:
                retrieved_docs.append(main_doc)
                expanded_query = found_art.get("title", query)

    # If not target url (or broader context requested), fetch articles
    if not target_url or len(retrieved_docs) < 5:
        # Route searches to appropriate news category (Requirement 5)
        try:
            live_articles = await mcp_client.call_tool(
                "search_live_news", 
                {"query": expanded_query, "category": target_category, "limit": 20}
            )
            if isinstance(live_articles, list):
                for art in live_articles:
                    if target_url and art.get("url") == target_url:
                        continue # Skip duplicate of primary article
                    retrieved_docs.append({
                        "title": art.get("title", "News Update"),
                        "content": art.get("cleaned_content") or art.get("content", ""),
                        "source": art.get("source", "Live Media"),
                        "url": art.get("url", "#"),
                        "published_date": art.get("published_date", "")
                    })
        except Exception as e:
            logger.error(f"[MCP Agent Retrieval Error] {e}")

        # Also query Qdrant vector index if active
        if qdrant_manager.client:
            try:
                query_vector = await embedding_agent.run(expanded_query)
                qdrant_results = qdrant_manager.search(query_vector=query_vector, top_k=8)
                for result in qdrant_results:
                    payload = result.payload or {}
                    p_url = payload.get("url", "#")
                    if target_url and p_url == target_url:
                        continue
                    retrieved_docs.append({
                        "title": payload.get("title", "Vector Match"),
                        "content": payload.get("content", ""),
                        "source": payload.get("source", "Qdrant Vector Index"),
                        "url": p_url
                    })
            except Exception as e:
                logger.warning(f"Qdrant retrieval skipped/failed: {e}")

    # Fallback to reading MCP Resource if pool is empty
    if not retrieved_docs:
        try:
            all_live = await mcp_client.read_resource("news://store/articles")
            if isinstance(all_live, list):
                for a in all_live:
                    retrieved_docs.append({
                        "title": a.get("title", "News Headline"),
                        "content": a.get("content", ""),
                        "source": a.get("source", "Live Feeds"),
                        "url": a.get("url", "#")
                    })
        except Exception as e:
            logger.error(f"Fallback resource read failed: {e}")

    # 3. News Event Deduplication using DuplicateDetectionAgent
    from app.agents.duplicate import DuplicateDetectionAgent
    from datetime import datetime
    import requests
    import time
    
    dup_agent = DuplicateDetectionAgent()
    query_emb = []
    
    # ── OPTIMIZATION: Pre-compute ALL embeddings (query + titles + contents) in ONE batch call ──
    # Previously: titles were batched but content embeddings were fetched per-pair (~3s each × 15 pairs = ~45s)
    # Now: single batch call for everything (~5s total)
    if retrieved_docs:
        t_embed_start = time.time()
        logger.info(f"Pre-computing embeddings for query + {len(retrieved_docs)} articles (titles + contents) in single batch...")
        
        # Build the batch: [query, title1, content1, title2, content2, ...]
        texts_to_embed = [expanded_query or ""]
        for art in retrieved_docs:
            texts_to_embed.append((art.get("title") or "News Update")[:500])
            content = (art.get("content") or art.get("cleaned_content") or "")[:500]
            texts_to_embed.append(content if content.strip() else "No content available")
            
        try:
            from app.utils.async_http import async_post_json
            status_code, data, text = await async_post_json(
                "http://localhost:11434/api/embed",
                payload={"model": "bge-m3:latest", "input": texts_to_embed},
                timeout=15.0
            )
            if status_code == 200:
                embeddings = data.get("embeddings", [])
                if len(embeddings) == len(texts_to_embed):
                    query_emb = embeddings[0]
                    for i, art in enumerate(retrieved_docs):
                        art["title_emb"] = embeddings[1 + i * 2]       # title embedding
                        art["content_emb"] = embeddings[1 + i * 2 + 1] # content embedding
                    t_embed_end = time.time()
                    logger.info(f"✅ Batch embeddings populated in {t_embed_end - t_embed_start:.1f}s ({len(retrieved_docs)} articles)")
                else:
                    logger.warning("Batch embedding response length mismatch. Falling back to lazy execution.")
            else:
                logger.warning(f"Batch embedding returned status {status_code}. Falling back to lazy execution.")
        except Exception as e:
            logger.warning(f"Batch embedding failed: {e}. Falling back to lazy execution.")
    
    # Deduplication loop (now uses pre-computed embeddings — no per-pair Ollama calls)
    t_dedup_start = time.time()
    unique_events = []
    for art in retrieved_docs:
        matched_event_idx = -1
        for idx, existing in enumerate(unique_events):
            is_dup, _ = dup_agent.are_duplicates(art, existing)
            if is_dup:
                matched_event_idx = idx
                break
                
        if matched_event_idx != -1:
            better_art = dup_agent.choose_better_article(unique_events[matched_event_idx], art)
            unique_events[matched_event_idx] = better_art
        else:
            unique_events.append(art)
    t_dedup_end = time.time()
    logger.info(f"✅ Deduplication complete in {t_dedup_end - t_dedup_start:.1f}s ({len(retrieved_docs)} → {len(unique_events)} unique articles)")

    # 4. Improved Weighted Relevance Ranking & Relevance Thresholding
    ranked_events = []
    if not query_emb and expanded_query:
        query_emb = await embedding_agent.run(expanded_query)
    query_words = set(re.findall(r'\b\w+\b', expanded_query.lower())) if expanded_query else set()
    
    for idx, art in enumerate(unique_events):
        score = 0.0
        title = (art.get("title") or "").lower()
        content = (art.get("content") or "").lower()
        text = title + " " + content
        
        # 1. Query relevance (semantic and keyword)
        title_emb = art.get("title_emb")
        if title_emb is None:
            title_emb = dup_agent.get_ollama_embedding(art.get("title", ""))
            art["title_emb"] = title_emb
        
        title_sim = dup_agent.cosine_similarity(query_emb, title_emb)
        
        # REQUIREMENT 6: Exclude articles whose relevance score falls below threshold
        # Title similarity threshold is set to 0.35 when valid embeddings exist.
        if query_emb and title_emb and title_sim < 0.35 and not (target_url and art.get("url") == target_url):
            logger.info(f"Skipping article '{art.get('title')}' due to low semantic similarity ({title_sim:.4f})")
            continue
            
        # REQUIREMENT 8: Validate topic/entity membership
        topic_lower = extracted_topic.lower()
        if extracted_topic and topic_lower not in ["technology", "business", "politics", "sports", "health", "international", "general news", "general"]:
            keywords_to_check = [topic_lower]
            raw_words = re.findall(r'\b\w{3,}\b', topic_lower)
            for w in raw_words:
                if w not in ["news", "today", "updates", "update", "latest", "what", "about"]:
                    keywords_to_check.append(w)
                    if len(w) > 4:
                        keywords_to_check.append(w[:4]) # e.g. econ for economic/economy
                        
            # Synonym expansion for common topic query terms
            if any(x in topic_lower for x in ["econ", "market", "finance", "business"]):
                keywords_to_check.extend(["econ", "market", "stock", "trade", "inflation", "finance", "bank", "dollar", "fed", "revenue", "shares"])
            elif any(x in topic_lower for x in ["politi", "government", "election", "parliament"]):
                keywords_to_check.extend(["politi", "govt", "minister", "parliament", "election", "vote", "assembly", "party", "bill", "court"])

            for ent in state.get("extracted_entities", []):
                keywords_to_check.append(ent.lower())
                keywords_to_check.extend(re.findall(r'\b\w{3,}\b', ent.lower()))
            
            keywords_to_check = [k for k in set(keywords_to_check) if k]
            if keywords_to_check and not any(kw in text for kw in keywords_to_check):
                logger.info(f"Skipping article '{art.get('title')}' - unrelated to requested topic '{extracted_topic}'")
                continue
        
        # Semantic similarity weight
        score += (title_sim * 25.0)
        
        # Keyword matches
        if query_words:
            matches = len(query_words.intersection(set(re.findall(r'\b\w+\b', text))))
            score += matches * 1.5
        
        # 2. News importance & Global impact
        if any(k in text for k in ["global", "worldwide", "international", "world", "nationwide", "country"]):
            score += 2.0
            
        # 3. Business impact
        if any(k in text for k in ["acquisition", "revenue", "billion", "shares", "stock", "deal", "merge", "investment", "buyout"]):
            score += 2.0
            
        # 4. Affected users
        if any(k in text for k in ["million", "user", "affect", "subscriber", "customer", "disrupt", "outage"]):
            score += 2.0
            
        # 5. Government involvement
        if any(k in text for k in ["government", "court", "regulate", "levy", "ban", "law", "antitrust", "justice", "commission", "senate", "parliament"]):
            score += 3.0
            
        # 6. Cybersecurity severity
        if any(k in text for k in ["breach", "hack", "leak", "ransomware", "compromise", "vulnerability", "cyberattack"]):
            score += 3.5
            
        # 7. Source authority
        src = (art.get("source") or "").lower()
        src_score = dup_agent.get_source_score(src)
        score += src_score * 2.0
        
        # 8. Publication recency (freshness)
        hours_old = dup_agent.date_diff_hours(art.get("published_date"), datetime.utcnow().strftime("%Y-%m-%dT%H:%M:%SZ"))
        if hours_old < 12:
            score += 3.0
        elif hours_old < 24:
            score += 2.0
        elif hours_old < 48:
            score += 1.0
            
        # 9. Confirmation vs Rumor Status (Rumors penalized)
        is_rumor = any(k in text for k in ["rumor", "reportedly", "alleged", "sources claim", "rumoured", "unconfirmed"])
        if is_rumor:
            score -= 4.0
            
        # Prioritize exact URL source
        if target_url and art.get("url") == target_url:
            score += 50.0
            
        ranked_events.append((score, art))
        
    if not ranked_events and unique_events:
        logger.warning("All articles were filtered out by strict topic checks. Falling back to top unique retrieved articles.")
        final_unique_docs = unique_events[:5]
    else:
        ranked_events.sort(key=lambda x: x[0], reverse=True)
        final_unique_docs = [x[1] for x in ranked_events[:5]]

    state["retrieved_documents"] = final_unique_docs
    logger.info(f"MCP Retrieval complete. Found {len(final_unique_docs)} context items matching topic.")
    return state
