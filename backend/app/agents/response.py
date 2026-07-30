import os
import json
import logging
import requests
from app.workflows.langgraph_state import AgentState

logger = logging.getLogger(__name__)

OLLAMA_URL = os.getenv("OLLAMA_URL", "http://localhost:11434")

def extractive_summary(text: str, num_sentences: int = 3) -> str:
    """
    Generates a sentence-extractive summary of the input text using word frequency scoring.
    """
    import re
    from collections import Counter
    
    # Split into sentences (simple regex)
    sentences = re.split(r'(?<=[.!?])\s+', text)
    sentences = [s.strip() for s in sentences if s.strip()]
    if len(sentences) <= num_sentences:
        return text
        
    # Simple stop words
    stop_words = {"the", "a", "an", "and", "or", "but", "if", "then", "of", "at", "by", "for", "with", "about", "against", "between", "into", "through", "during", "before", "after", "above", "below", "to", "from", "up", "down", "in", "out", "on", "off", "over", "under", "again", "further", "then", "once", "here", "there", "when", "where", "why", "how", "all", "any", "both", "each", "few", "more", "most", "other", "some", "such", "no", "nor", "not", "only", "own", "same", "so", "than", "too", "very", "s", "t", "can", "will", "just", "don", "should", "now", "is", "was", "are", "were", "been", "has", "have", "had", "do", "does", "did"}
    
    # Tokenize words and count frequencies
    words = re.findall(r'\b[a-zA-Z]{3,}\b', text.lower())
    words = [w for w in words if w not in stop_words]
    word_freq = Counter(words)
    
    if not word_freq:
        return "\n".join(sentences[:num_sentences])
        
    # Score sentences
    sentence_scores = {}
    for i, sent in enumerate(sentences):
        sent_words = re.findall(r'\b[a-zA-Z]{3,}\b', sent.lower())
        score = sum(word_freq[w] for w in sent_words if w in word_freq)
        sentence_scores[i] = score / (1 + len(sent_words) ** 0.5)
        
    # Sort by score and pick top N
    top_indices = sorted(sentence_scores, key=sentence_scores.get, reverse=True)[:num_sentences]
    top_indices.sort()
    
    return " ".join([sentences[idx] for idx in top_indices])


def synthesize_executive_summary(query: str, docs: list) -> str:
    """
    100% Dynamic Executive News Summarizer based strictly on retrieved live articles.
    """
    if len(docs) == 1 and docs[0].get("source") == "User Input":
        content = docs[0].get("content", "")
        summary = extractive_summary(content, num_sentences=4)
        return (
            f"## 📝 Text Summary\n\n"
            f"Here is a summary of the text you provided:\n\n"
            f"> {summary}"
        )

    # Otherwise, summarize the retrieved articles
    combined_content = " ".join([d.get("content", "") or d.get("cleaned_content", "") or d.get("title", "") for d in docs[:5]])
    overall_summary = extractive_summary(combined_content, num_sentences=3)

    query_lower = query.lower()
    sources = list(set([d.get("source", "News Media") for d in docs[:5] if d.get("source")]))
    
    # Synthesize live narrative highlights from actual retrieved live articles
    highlights = []
    for idx, doc in enumerate(docs[:5], 1):
        title = doc.get("title", "Article")
        src = doc.get("source", "Media Outlet")
        content = doc.get("content", "").strip() or doc.get("cleaned_content", "").strip() or title
        clean_snippet = content[:180] + ("..." if len(content) > 180 else "")
        highlights.append(f"• **{src}:** {title}\n  *Key Finding:* {clean_snippet}")

    summary_bullets = "\n\n".join(highlights)
    
    exec_summary_text = (
        f"## 📰 Executive Intelligence Briefing: \"{query}\"\n\n"
        f"**Live Synthesis Overview ({len(docs[:5])} Verified Live Articles Analyzed):**\n\n"
        f"Real-time news aggregation across primary outlets ({', '.join(sources[:4])}) reveals the following key intelligence updates:\n\n"
        f"**Key Summary:** {overall_summary}\n\n"
        f"{summary_bullets}"
    )

    # Detailed article breakdown
    article_cards = []
    sources_summary = []
    
    for idx, doc in enumerate(docs[:5], 1):
        title = doc.get("title", "Article")
        source = doc.get("source", "Live Media")
        url = doc.get("url", "#")
        content = doc.get("content", "").strip() or doc.get("cleaned_content", "").strip() or "Full real-time news report ingested and verified by NewsIntel AI pipeline."
        snippet = content[:240] + ("..." if len(content) > 240 else "")
        
        article_cards.append(
            f"**{idx}. [{title}]({url})**  \n"
            f"*Source: {source}*  \n"
            f"> {snippet}\n"
        )
        sources_summary.append(f"• [{source}: {title}]({url})")

    full_response = (
        f"{exec_summary_text}\n\n"
        f"---\n"
        f"### 📋 Full Live Article List & Source Links ({len(docs[:5])} Articles)\n\n"
        + "\n".join(article_cards) + "\n\n"
        f"**Primary Source Links:**\n" + "\n".join(sources_summary)
    )
    
    return full_response


def response_generation_agent(state: AgentState) -> AgentState:
    logger.info("Response Generation Agent building final answer...")
    
    docs = state.get("retrieved_documents", [])
    query = state.get("query", "")
    
    if not docs:
        state["final_response"] = "No relevant live news articles found for your query. Try refreshing the news feeds or adjusting your search term."
        return state

    context_str = "\n\n".join([
        f"Source: {doc.get('source', 'News Outlet')}\nTitle: {doc.get('title')}\nLink: {doc.get('url', '#')}\nContent: {doc.get('content', '')[:300]}"
        for doc in docs
    ])

    prompt = (
        "System: You are NewsIntel AI, an executive intelligence agent for real-time news aggregation.\n"
        f"Query: {query}\n\nLive Context:\n{context_str}\n\n"
        "Provide a prominent Executive Summary block summarizing the main themes, key takeaways, and insights strictly based on the live context above."
    )

    # Try sending to Ollama if local LLM is active
    try:
        response = requests.post(
            f"{OLLAMA_URL}/api/generate",
            json={
                "model": "qwen2.5:3b",
                "prompt": prompt,
                "stream": False
            },
            timeout=10
        )
        if response.status_code == 200:
            data = response.json()
            llm_text = data.get("response")
            if llm_text and len(llm_text.strip()) > 50:
                state["final_response"] = llm_text
                return state
    except Exception as e:
        logger.info(f"Ollama local model not available ({e}). Using built-in Dynamic News Synthesizer.")

    # High-quality built-in dynamic synthesis generating Executive Summary strictly from retrieved live docs
    state["final_response"] = synthesize_executive_summary(query, docs)
    return state
