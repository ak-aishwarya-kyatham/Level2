import os
import json
import logging
import re
import requests
from collections import Counter
from app.workflows.langgraph_state import AgentState

logger = logging.getLogger(__name__)

OLLAMA_URL = os.getenv("OLLAMA_URL", "http://localhost:11434")


# ---------------------------------------------------------------------------
# Sentence Cleaning & Fact Extraction
# ---------------------------------------------------------------------------

def clean_news_sentence(sent: str) -> str:
    """
    Cleans raw news sentences/titles by removing RSS boilerplate, publisher tags,
    trailing site names, LIVE prefixes, and malformed punctuation.
    """
    if not sent:
        return ""
    
    s = sent.strip()
    
    # Remove common RSS / publisher prefixes
    s = re.sub(r'^(?:LIVE|UPDATE|UPDATES|BREAKING|WATCH|JUST IN|EXPLAINER|OPINION):\s*', '', s, flags=re.IGNORECASE)
    s = re.sub(r'^(?:[A-Z0-9\s\-]{2,15}\s*LIVE|\w+\s+LIVE\s+Updates):\s*', '', s, flags=re.IGNORECASE)
    
    # Remove trailing source attribution like "- CNN", "- BBC", "| TechCrunch", " - Reuters"
    s = re.sub(r'\s*[\-\|]\s*(?:CNN|BBC|Reuters|AP News|TechCrunch|NDTV News|The Hindu|Indian Express|Google News|CoinDesk|Yahoo Finance|Investor\'s Business Daily|Bloomberg|WSJ|Engadget)\b.*$', '', s, flags=re.IGNORECASE)
    
    # Clean up double punctuation or awkward whitespace
    s = re.sub(r'\s+', ' ', s).strip()
    
    # Ensure proper sentence termination
    if s and not s.endswith(('.', '!', '?')):
        s += '.'
        
    return s


def extract_clean_article_summary(doc: dict) -> str:
    """
    Extracts the most informative, clean factual sentences from an article.
    Prefers well-formed sentences from article content/snippet over title headers.
    """
    title = doc.get("title") or ""
    content = doc.get("content") or doc.get("cleaned_content") or ""
    
    # Split content into sentences
    raw_sentences = re.split(r'(?<=[.!?])\s+', content)
    sentences = [clean_news_sentence(s) for s in raw_sentences if len(s.strip()) > 30]
    
    # Filter out sentences that are mostly navigation links or copyright disclaimers
    valid_sentences = []
    for s in sentences:
        s_lower = s.lower()
        if any(skip in s_lower for skip in ["subscribe", "click here", "read more", "copyright", "all rights reserved", "follow us"]):
            continue
        valid_sentences.append(s)
        
    if valid_sentences:
        # Take the top 1-2 most informative content sentences
        return " ".join(valid_sentences[:2])
    
    # Fallback to cleaned title if content is missing or short
    cleaned_title = clean_news_sentence(title)
    return cleaned_title if len(cleaned_title) > 20 else ""


# ---------------------------------------------------------------------------
# Grounded Summary Builder
# ---------------------------------------------------------------------------

def generate_grounded_summary(query: str, docs: list) -> str:
    """
    Generates a clean, grammatically correct, retrieval-grounded summary.
    Synthesizes facts from all retrieved articles without broken template stitching.
    """
    if not docs:
        return "No relevant articles retrieved."

    event_summaries = []
    seen_signatures = set()

    for doc in docs:
        summary_text = extract_clean_article_summary(doc)
        if not summary_text:
            continue

        # Simple deduplication check based on sentence signature
        sig = re.sub(r'\W+', '', summary_text[:50].lower())
        if sig in seen_signatures:
            continue
        seen_signatures.add(sig)

        event_summaries.append(summary_text)

    if not event_summaries:
        return "No clear facts could be extracted from the retrieved articles."

    # Join extracted summaries with proper spacing and paragraph coherence
    combined_summary = " ".join(event_summaries)
    
    # Clean up double periods or spaces
    combined_summary = re.sub(r'\.\s*\.', '.', combined_summary)
    combined_summary = re.sub(r'\s+', ' ', combined_summary).strip()

    # Trim to word limit (150-250 words)
    words = combined_summary.split()
    if len(words) > 250:
        trimmed = " ".join(words[:245])
        last_period = trimmed.rfind(".")
        if last_period > 80:
            combined_summary = trimmed[:last_period + 1]
        else:
            combined_summary = trimmed + "..."

    return combined_summary


# ---------------------------------------------------------------------------
# Faithfulness Validator
# ---------------------------------------------------------------------------

FORBIDDEN_BUZZWORDS = [
    "neural processor", "neural-processor", "ai model integration",
    "localized processing", "data residency", "zero-trust", "zero trust",
    "digital transformation", "enterprise modernization", "cloud-native",
    "cloud native", "edge ai", "enterprise organizations to evaluate",
    "pivotal shifts", "ongoing friction between rapid feature deployment",
    "long-term technology roadmaps",
]


def validate_faithfulness(summary: str, docs: list) -> str:
    """
    Validates that every sentence in the summary is grounded in the retrieved docs.
    Removes sentences containing forbidden buzzwords or with low word overlap.
    """
    sentences = re.split(r'(?<=[.!?])\s+', summary)
    sentences = [s.strip() for s in sentences if s.strip()]

    master_source_text = " ".join([
        (doc.get("title", "") + " " + (doc.get("content") or doc.get("cleaned_content") or "")).lower()
        for doc in docs
    ])

    valid_sentences = []

    for sent in sentences:
        sent_lower = sent.lower()

        # Check forbidden buzzwords NOT in sources
        has_forbidden = False
        for buzz in FORBIDDEN_BUZZWORDS:
            if buzz in sent_lower and buzz not in master_source_text:
                logger.warning(f"[Faithfulness] Removed buzzword sentence: {sent[:80]}...")
                has_forbidden = True
                break
        if has_forbidden:
            continue

        # Word overlap check
        sent_words = set(re.findall(r'\b\w{4,}\b', sent_lower))
        source_words = set(re.findall(r'\b\w{4,}\b', master_source_text))
        if not sent_words:
            continue
        overlap = len(sent_words.intersection(source_words)) / len(sent_words)

        if overlap >= 0.40:
            valid_sentences.append(sent)
        else:
            logger.warning(f"[Faithfulness] Discarded low-overlap ({overlap:.0%}): {sent[:80]}...")

    return " ".join(valid_sentences)


# ---------------------------------------------------------------------------
# Extractive Summary (for user-pasted text)
# ---------------------------------------------------------------------------

def extractive_summary(text: str, num_sentences: int = 3) -> str:
    """Sentence-extractive summary using word frequency scoring."""
    sentences = re.split(r'(?<=[.!?])\s+', text)
    sentences = [s.strip() for s in sentences if s.strip()]
    if len(sentences) <= num_sentences:
        return text

    stop_words = {"the", "a", "an", "and", "or", "but", "if", "then", "of", "at", "by", "for", "with", "about", "to", "from", "in", "out", "on", "off", "over", "under", "is", "was", "are", "were", "been", "has", "have", "had", "do", "does", "did"}
    words = re.findall(r'\b[a-zA-Z]{3,}\b', text.lower())
    words = [w for w in words if w not in stop_words]
    word_freq = Counter(words)

    if not word_freq:
        return " ".join(sentences[:num_sentences])

    sentence_scores = {}
    for i, sent in enumerate(sentences):
        sent_words = re.findall(r'\b[a-zA-Z]{3,}\b', sent.lower())
        score = sum(word_freq[w] for w in sent_words if w in word_freq)
        sentence_scores[i] = score / (1 + len(sent_words) ** 0.5)

    top_indices = sorted(sentence_scores, key=sentence_scores.get, reverse=True)[:num_sentences]
    top_indices.sort()
    return " ".join([sentences[idx] for idx in top_indices])


# ---------------------------------------------------------------------------
# Live Feed Briefing
# ---------------------------------------------------------------------------

def synthesize_live_feed_briefing(docs: list, limit: int = 10) -> str:
    """
    Renders the live news feed — matching the UI Live Feed panel display,
    showing latest articles sorted by publication time with source, category, timestamp, title, snippet, and link.
    """
    if not docs:
        return "No live feed items available right now. Please refresh the news feeds."

    limit = max(1, limit)
    selected_docs = docs[:limit]

    lines = [f"## 📡 Real-Time Live News Feed (Top {len(selected_docs)})\n" if limit != 10 else "## 📡 Real-Time Live News Feed\n"]
    if limit > 1:
        lines.append(f"Showing latest **{len(selected_docs)}** live articles from active RSS feeds:\n")

    for idx, doc in enumerate(selected_docs, 1):
        source = doc.get("source", "News Media")
        category = doc.get("category", "General News")
        pub_date = doc.get("published_date", "")
        title = clean_news_sentence(doc.get("title", ""))
        url = doc.get("url", "#")
        snippet = doc.get("content", "") or ""

        # Format publication time if possible
        formatted_time = ""
        if pub_date:
            try:
                from datetime import datetime
                dt = datetime.fromisoformat(str(pub_date).replace("Z", "+00:00"))
                formatted_time = dt.strftime("%b %d, %I:%M %p")
            except Exception:
                formatted_time = str(pub_date)[:16]

        meta_parts = [f"**{source}**", f"`{category}`"]
        if formatted_time:
            meta_parts.append(f"🕒 {formatted_time}")
        
        meta = " · ".join(meta_parts)

        prefix = f"### {idx}. " if limit > 1 else "### "
        block = f"{prefix}[{title}]({url})\n{meta}"
        if snippet and snippet.lower() not in title.lower():
            clean_snip = clean_news_sentence(snippet[:250].strip())
            if clean_snip:
                block += f"\n> {clean_snip}"

        lines.append(block + "\n")

    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Trending Topics Briefing
# ---------------------------------------------------------------------------

def synthesize_trending_briefing(docs: list, limit: int = 10) -> str:
    """
    Renders a ranked list of live trending news topics — same data the dashboard
    shows but presented as a chat-friendly digest.
    """
    if not docs:
        return "No trending topics available right now. Try again in a moment."

    # Filter docs that carry the is_trending flag (rank + article_count)
    trend_docs = [d for d in docs if d.get("is_trending")]
    if not trend_docs:
        trend_docs = docs

    limit = max(1, limit)
    selected_docs = trend_docs[:limit]

    topic_filter = ""
    if selected_docs and selected_docs[0].get("topic_filter"):
        topic_filter = selected_docs[0]["topic_filter"]

    if topic_filter:
        header_title = f"## 🔥 Live Trending Topics in {topic_filter.title()}"
    else:
        header_title = "## 🔥 Live Trending Topics"

    if limit != 10:
        header_title += f" (Top {len(selected_docs)})"

    lines = [header_title + "\n"]

    for doc in selected_docs:
        rank = doc.get("rank", "")
        title = clean_news_sentence(doc.get("title", "Trending Topic"))
        count = doc.get("article_count", 0)
        category = doc.get("category", "")
        url = doc.get("url", "#")
        velocity = doc.get("velocity", "")
        snippet = doc.get("content", "") or ""

        # Clean up snippet
        if snippet:
            snippet = clean_news_sentence(snippet[:180])

        rank_label = f"**#{rank}**" if rank else "**•**"
        feed_label = f"{count} related live feeds" if count else ""
        cat_label = f"[{category}]" if category else ""
        vel_label = f"📈 {velocity}" if velocity else ""

        meta = " · ".join(filter(None, [cat_label, feed_label, vel_label]))

        block = f"{rank_label} [{title}]({url})"
        if meta:
            block += f"  \n  _{meta}_"
        if snippet and snippet.lower() not in title.lower():
            block += f"  \n  {snippet}"

        lines.append(block + "\n")

    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Executive Summary Orchestrator
# ---------------------------------------------------------------------------

def _keyword_overlap_score(title1: str, title2: str) -> float:
    """Returns a keyword overlap ratio between two headlines (0.0–1.0)."""
    stop = {"the", "a", "an", "and", "or", "of", "in", "on", "at", "to", "by",
            "for", "is", "are", "was", "were", "has", "have", "with", "that",
            "this", "from", "its", "as", "be", "it", "how", "what", "vs"}
    def words(t):
        return set(re.findall(r'\b[a-zA-Z]{3,}\b', t.lower())) - stop
    w1, w2 = words(title1), words(title2)
    if not w1 or not w2:
        return 0.0
    return len(w1 & w2) / max(len(w1), len(w2))


def synthesize_comparison_briefing(query: str, docs: list, source1: str = None, source2: str = None) -> str:
    """
    Editorial angle comparison: finds stories covered by BOTH outlets on the same
    topic and shows how each source frames/angles the story differently.
    """
    if not docs:
        return "No articles available for source comparison."

    # Resolve source names from tagged docs
    comparison_sources = list(dict.fromkeys(
        d.get("comparison_source") for d in docs if d.get("comparison_source")
    ))
    if len(comparison_sources) >= 2:
        source1, source2 = comparison_sources[0], comparison_sources[1]
    elif not source1 or not source2:
        seen = list(dict.fromkeys(d.get("source", "Outlet") for d in docs if d.get("source")))
        source1 = seen[0] if len(seen) > 0 else "Times of India"
        source2 = seen[1] if len(seen) > 1 else "The Hindu"

    s1_docs = [d for d in docs if d.get("comparison_source") == source1 or source1.lower() in (d.get("source") or "").lower()]
    s2_docs = [d for d in docs if d.get("comparison_source") == source2 or source2.lower() in (d.get("source") or "").lower()]

    if not s1_docs:
        s1_docs = docs[:len(docs) // 2 or 1]
    if not s2_docs:
        s2_docs = docs[len(docs) // 2:]

    # ── Find shared stories (same topic, different outlet angle) ──────────────
    OVERLAP_THRESHOLD = 0.28
    shared_stories = []   # list of (s1_doc, s2_doc, score)
    used_s1 = set()
    used_s2 = set()

    for i, d1 in enumerate(s1_docs):
        best_score, best_j = 0.0, -1
        for j, d2 in enumerate(s2_docs):
            if j in used_s2:
                continue
            score = _keyword_overlap_score(
                d1.get("title", ""), d2.get("title", "")
            )
            if score > best_score:
                best_score, best_j = score, j
        if best_score >= OVERLAP_THRESHOLD and best_j >= 0:
            shared_stories.append((i, best_j, best_score))
            used_s1.add(i)
            used_s2.add(best_j)

    # ── Exclusive articles (covered by one outlet only) ───────────────────────
    exclusive_s1 = [d for i, d in enumerate(s1_docs) if i not in used_s1]
    exclusive_s2 = [d for j, d in enumerate(s2_docs) if j not in used_s2]

    # ── Build the shared-story comparison blocks ──────────────────────────────
    shared_blocks = []
    for s1_idx, s2_idx, _ in shared_stories[:3]:
        d1 = s1_docs[s1_idx]
        d2 = s2_docs[s2_idx]
        t1 = clean_news_sentence(d1.get("title", "Headline"))
        t2 = clean_news_sentence(d2.get("title", "Headline"))
        sum1 = extract_clean_article_summary(d1)
        sum2 = extract_clean_article_summary(d2)

        # Derive a neutral shared topic label from common keywords
        stop = {"the", "a", "an", "and", "or", "of", "in", "on", "at", "to",
                "by", "for", "is", "are", "was", "were", "has", "have", "with"}
        w1 = [w for w in re.findall(r'\b[A-Za-z]{4,}\b', t1) if w.lower() not in stop]
        topic_label = " ".join(w1[:4]) if w1 else "Shared Story"

        block = (
            f"#### 📌 Shared Story: *{topic_label}*\n"
            f"> **{source1}:** {t1}\n"
            + (f"> _{sum1}_\n" if sum1 and sum1 != t1 else "")
            + f">\n"
            f"> **{source2}:** {t2}\n"
            + (f"> _{sum2}_\n" if sum2 and sum2 != t2 else "")
        )
        shared_blocks.append(block)

    # ── Exclusive story blocks ────────────────────────────────────────────────
    def format_exclusive(docs_list, limit=2):
        items = []
        for doc in docs_list[:limit]:
            title = clean_news_sentence(doc.get("title", "Headline"))
            summary = extract_clean_article_summary(doc)
            if summary and summary != title:
                items.append(f"• **{title}:** {summary}")
            else:
                items.append(f"• **{title}**")
        return "\n".join(items) if items else "• *No exclusive stories found.*"

    excl_s1_block = format_exclusive(exclusive_s1)
    excl_s2_block = format_exclusive(exclusive_s2)

    # ── Coverage analysis summary ─────────────────────────────────────────────
    shared_count = len(shared_stories)
    excl_s1_count = len(exclusive_s1)
    excl_s2_count = len(exclusive_s2)

    if shared_count > 0:
        analysis = (
            f"Both outlets covered **{shared_count} common stor{'y' if shared_count==1 else 'ies'}**, "
            f"but framed them differently. "
            f"**{source1}** ran {excl_s1_count} exclusive headline(s) not covered by {source2}, "
            f"while **{source2}** had {excl_s2_count} exclusive headline(s). "
            f"Compare the wording above to spot editorial angle differences."
        )
    else:
        analysis = (
            f"No overlapping stories were found between **{source1}** and **{source2}** "
            f"in the current live feed. Both outlets appear to be covering distinct topics today. "
            f"**{source1}** published {len(s1_docs)} article(s); **{source2}** published {len(s2_docs)} article(s)."
        )

    # ── Source links ──────────────────────────────────────────────────────────
    link_docs = [s1_docs[i] for i, _, _ in shared_stories] + exclusive_s1[:2] + exclusive_s2[:2]
    sources_summary = []
    for doc in link_docs[:6]:
        title = clean_news_sentence(doc.get("title", "Article"))
        source = doc.get("source", "Live Media")
        url = doc.get("url", "#")
        sources_summary.append(f"• [{source}: {title}]({url})")

    # ── Assemble final output ─────────────────────────────────────────────────
    shared_section = (
        f"### 🔄 Stories Covered by Both Outlets (Editorial Angle Comparison)\n\n"
        + "\n".join(shared_blocks)
        if shared_blocks else
        f"### 🔄 Stories Covered by Both Outlets\n\n*No overlapping stories found in current live feed.*"
    )

    return (
        f"## ⚔️ Editorial Angle Comparison: \"{source1}\" vs \"{source2}\"\n\n"
        f"**Live Analysis ({len(s1_docs) + len(s2_docs)} articles retrieved — "
        f"{shared_count} shared topics, {excl_s1_count + excl_s2_count} exclusive stories)**\n\n"
        f"{shared_section}\n\n"
        f"---\n\n"
        f"### 🅰 Exclusive to {source1}:\n{excl_s1_block}\n\n"
        f"### 🅱 Exclusive to {source2}:\n{excl_s2_block}\n\n"
        f"---\n\n"
        f"**📊 Coverage Analysis:**  \n{analysis}\n\n"
        f"---\n"
        f"**Primary Source Links:**\n" + "\n".join(sources_summary)
    )


def synthesize_executive_summary(query: str, docs: list, llm_summary: str = None, intent: str = "") -> str:
    """
    Orchestrates the Executive Intelligence Briefing generation.
    Uses LLM summary if available, otherwise generates a grounded summary.
    Validates faithfulness before returning.
    """
    # Handle media source comparison
    if intent == "compare" or any("comparison_source" in d for d in docs):
        return synthesize_comparison_briefing(query, docs)

    # Handle user-pasted text
    if len(docs) == 1 and docs[0].get("source") == "User Input":
        content = docs[0].get("content", "")
        summary = extractive_summary(content, num_sentences=4)
        return (
            f"## 📝 Text Summary\n\n"
            f"Here is a summary of the text you provided:\n\n"
            f"> {summary}"
        )

    top_docs = docs[:5]
    sources = list(set([d.get("source", "News Media") for d in top_docs if d.get("source")]))

    # Generate the summary
    if llm_summary and len(llm_summary.strip()) > 50:
        overall_summary = llm_summary.strip()
    else:
        overall_summary = generate_grounded_summary(query, top_docs)

    # Validate faithfulness — remove any hallucinated or ungrounded sentences
    overall_summary = validate_faithfulness(overall_summary, top_docs)

    # Word limit enforcement
    words = overall_summary.split()
    if len(words) > 250:
        overall_summary = " ".join(words[:245]) + "..."

    # Build the briefing layout
    exec_summary_text = (
        f"## 📰 Executive Intelligence Briefing: \"{query}\"\n\n"
        f"**Live Synthesis Overview ({len(top_docs)} Verified Live Articles Analyzed):**\n\n"
        f"Real-time news aggregation across primary outlets ({', '.join(sources[:4])}) reveals the following key intelligence updates:\n\n"
        f"**Key Summary:** {overall_summary}"
    )

    # Simple primary source links
    sources_summary = []
    for doc in top_docs:
        title = doc.get("title", "Article")
        source = doc.get("source", "Live Media")
        url = doc.get("url", "#")
        sources_summary.append(f"• [{source}: {title}]({url})")

    full_response = (
        f"{exec_summary_text}\n\n"
        f"---\n"
        f"**Primary Source Links:**\n" + "\n".join(sources_summary)
    )

    return full_response


# ---------------------------------------------------------------------------
# LangGraph Agent Node
# ---------------------------------------------------------------------------

def response_generation_agent(state: AgentState) -> AgentState:
    import time
    t_total_start = time.time()
    logger.info("Response Generation Agent building final answer...")

    docs = state.get("retrieved_documents", [])
    query = state.get("query", "")
    intent = state.get("intent", "")
    limit = state.get("requested_limit", 10)

    if not docs:
        state["final_response"] = "No relevant live news articles found for your query. Try refreshing the news feeds or adjusting your search term."
        return state

    # --- Short-circuit for live_feed intent: render live feed list ---
    if intent == "live_feed" or any(d.get("is_live_feed") for d in docs):
        state["final_response"] = synthesize_live_feed_briefing(docs, limit=limit)
        return state

    # --- Short-circuit for trend intent: render ranked trending list ---
    if intent == "trend" or any(d.get("is_trending") for d in docs):
        state["final_response"] = synthesize_trending_briefing(docs, limit=limit)
        return state

    # --- Short-circuit for compare intent: no LLM needed ---
    if intent == "compare" or any("comparison_source" in d for d in docs):
        state["final_response"] = synthesize_executive_summary(
            query, docs, llm_summary=None, intent="compare"
        )
        return state

    top_docs = docs[:5]

    context_str = "\n\n".join([
        f"Source: {doc.get('source', 'News Outlet')}\n"
        f"Title: {doc.get('title')}\n"
        f"Content: {doc.get('content', '')[:600]}"
        for doc in top_docs
    ])

    # LLM prompt — strict grounding rules
    prompt = (
        "System: You are a senior intelligence analyst writing an executive briefing.\n"
        "Write a multi-document summary of the articles below. Follow these rules EXACTLY:\n"
        "1. Answer ONLY the user's question. Do not discuss unrelated topics.\n"
        "2. Every sentence MUST be supported by the provided articles. Never invent information.\n"
        "3. Do NOT copy article titles, headlines, or raw snippets. Rewrite facts in your own words.\n"
        "4. Synthesize across all articles into a coherent narrative (150-250 words).\n"
        "5. Explain what happened and why it matters.\n"
        "6. Never mention concepts not in the articles (e.g. neural processors, edge AI, zero trust, data residency).\n"
        "7. Never use generic advisory language or template phrases.\n"
        "8. Be professional, concise, objective, and factual.\n\n"
        f"User's Question: {query}\n\n"
        f"Retrieved Articles:\n{context_str}\n\n"
        "Write ONLY the summary paragraphs. No greetings, headers, or markdown."
    )

    # ── DEBUG: Log the full context going into the summarizer ──
    logger.info("=" * 80)
    logger.info("[RAG CONTEXT DEBUG] Query: %s", query)
    logger.info("[RAG CONTEXT DEBUG] Number of retrieved articles: %d", len(top_docs))
    for i, doc in enumerate(top_docs, 1):
        title = doc.get("title", "N/A")
        source = doc.get("source", "N/A")
        content = doc.get("content", "") or doc.get("cleaned_content", "") or ""
        logger.info(
            "[RAG CONTEXT DEBUG] Article %d:\n"
            "  Source : %s\n"
            "  Title  : %s\n"
            "  Content length: %d chars\n"
            "  Content preview: %s",
            i, source, title, len(content), content[:300].replace("\n", " ")
        )
    logger.info("[RAG CONTEXT DEBUG] Full prompt being sent to LLM:\n%s", prompt)
    logger.info("=" * 80)

    # Try Ollama LLM (timeout reduced from 15s → 3s to fail fast when unavailable)
    llm_summary = None
    t_llm_start = time.time()
    try:
        response = requests.post(
            f"{OLLAMA_URL}/api/generate",
            json={"model": "qwen2.5:3b", "prompt": prompt, "stream": False},
            timeout=3
        )
        t_llm_end = time.time()
        if response.status_code == 200:
            data = response.json()
            llm_text = data.get("response")
            logger.info("[RAG CONTEXT DEBUG] LLM response received in %.1fs (%d chars)",
                        t_llm_end - t_llm_start, len(llm_text) if llm_text else 0)
            if llm_text and len(llm_text.strip()) > 50:
                llm_summary = llm_text
        else:
            logger.warning("[RAG CONTEXT DEBUG] Ollama returned status %d", response.status_code)
    except Exception as e:
        t_llm_end = time.time()
        logger.info(f"Ollama unavailable in {t_llm_end - t_llm_start:.1f}s ({e}). Using built-in grounded summarizer.")

    # Generate the final summary
    t_synth_start = time.time()
    state["final_response"] = synthesize_executive_summary(query, top_docs, llm_summary, intent=intent)
    t_synth_end = time.time()

    t_total_end = time.time()
    logger.info(
        "⏱️ Response Generation timing:\n"
        "  LLM attempt: %.1fs\n"
        "  Synthesis:    %.1fs\n"
        "  Total:        %.1fs",
        t_llm_end - t_llm_start, t_synth_end - t_synth_start, t_total_end - t_total_start
    )
    return state

