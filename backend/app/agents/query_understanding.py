import logging
import re

from app.workflows.langgraph_state import AgentState

logger = logging.getLogger(__name__)

# Query Expansion mapping for common political and technical abbreviations
ABBREVIATIONS = {
    "ncp": {
        "full_name": "Nationalist Congress Party",
        "related": ["Sharad Pawar", "Ajit Pawar", "Maharashtra", "INDIA alliance"],
        "category": "Politics"
    },
    "bjp": {
        "full_name": "Bharatiya Party",
        "related": ["Narendra Modi", "Amit Shah", "NDA", "BJP"],
        "category": "Politics"
    },
    "inc": {
        "full_name": "Indian National Congress",
        "related": ["Rahul Gandhi", "Congress", "Mallikarjun Kharge"],
        "category": "Politics"
    },
    "congress": {
        "full_name": "Indian National Congress",
        "related": ["Rahul Gandhi", "INC", "Mallikarjun Kharge"],
        "category": "Politics"
    },
    "ai": {
        "full_name": "Artificial Intelligence",
        "related": ["Machine Learning", "OpenAI", "ChatGPT", "LLM", "Generative AI"],
        "category": "Technology"
    },
    "rbi": {
        "full_name": "Reserve Bank of India",
        "related": ["Monetary Policy", "Interest Rates", "inflation", "banking"],
        "category": "Business"
    },
    "fed": {
        "full_name": "Federal Reserve",
        "related": ["US economy", "interest rates", "Jerome Powell"],
        "category": "Business"
    },
    "ipl": {
        "full_name": "Indian Premier League",
        "related": ["BCCI", "Cricket", "T20"],
        "category": "Sports"
    }
}

# Rule-based category routing keywords
CATEGORY_KEYWORDS = {
    "Technology": ["tech", "ai", "artificial intelligence", "software", "apple", "iphone", "samsung", "microsoft", "nvidia", "silicon valley", "semiconductor", "cybersecurity", "breach", "hack", "startup", "app", "mobile"],
    "Business": ["economy", "economic", "financial", "finance", "inflation", "fed", "interest rates", "stock", "stocks", "shares", "revenue", "billion", "merger", "acquisition", "rbi", "budget", "markets", "market", "oil", "gold", "trade"],
    "Politics": ["election", "parliament", "congress", "bjp", "ncp", "minister", "government", "modi", "rahul gandhi", "democrats", "republicans", "trump", "biden", "harris", "alliance", "pm", "summons", "levy"],
    "Sports": ["cricket", "ipl", "bcci", "football", "olympics", "match", "cup", "tournament", "player", "fifa", "uefa", "tennis", "sports"],
    "Health": ["covid", "virus", "vaccine", "health", "hospital", "patient", "medical", "fda", "carecloud"],
    "International": ["un", "united nations", "border", "middle east", "ukraine", "russia", "china", "global", "international", "world news", "treaty"]
}

# Common query structure regexes to clean/extract topics
TOPIC_PATTERNS = [
    r"(?i)\blatest\s+news\s+about\s+(.+)",
    r"(?i)\bnews\s+about\s+(.+)",
    r"(?i)\bsearch\s+for\s+(.+)",
    r"(?i)\bsummarize\s+(.+)",
    r"(?i)\bupdates\s+on\s+(.+)",
    r"(?i)\bwhat\s+is\s+happening\s+with\s+(.+)",
    r"(?i)\bwhat\s+are\s+the\s+(.+)"
]

def extract_url(query: str) -> str:
    url_match = re.search(r'https?://[^\s]+', query)
    if url_match:
        return url_match.group(0)
    return ""

def query_understanding_agent(state: AgentState) -> AgentState:
    logger.info("Query Understanding Agent analyzing search query...")
    query = state.get("query", "").strip()

    # 1. URL extraction
    target_url = extract_url(query)
    state["target_url"] = target_url

    # 2. Topic extraction (Rule-based first)
    extracted_topic = ""
    clean_query = query
    if target_url:
        clean_query = query.replace(target_url, "").strip()

    for pattern in TOPIC_PATTERNS:
        match = re.search(pattern, clean_query)
        if match:
            extracted_topic = match.group(1).strip("?:., ")
            break

    if not extracted_topic:
        extracted_topic = clean_query.strip("?:., ")

    # 3. Named Entity Recognition (NER) & Abbreviations Expansion
    entities = []
    expanded_terms = []
    inferred_category = "General News"

    words = re.findall(r'\b\w+\b', clean_query.lower())

    # Check abbreviations
    for word in words:
        if word in ABBREVIATIONS:
            info = ABBREVIATIONS[word]
            entities.append(info["full_name"])
            expanded_terms.append(info["full_name"])
            expanded_terms.extend(info["related"])
            inferred_category = info["category"]

    # NER for capitalized words (excluding common query terms)
    capitalized_words = re.findall(r'\b[A-Z][a-zA-Z]+\b', clean_query)
    for word in capitalized_words:
        if word.lower() not in ["the", "latest", "what", "news", "summarize", "today", "show", "search"]:
            entities.append(word)

    entities = list(set(entities))

    # 4. Route news category
    if inferred_category == "General News":
        max_matches = 0
        for cat, keywords in CATEGORY_KEYWORDS.items():
            matches = sum(1 for kw in keywords if re.search(r'\b' + re.escape(kw) + r'\b', clean_query.lower()))
            if matches > max_matches:
                max_matches = matches
                inferred_category = cat

    # 5. Build expanded query
    expanded_parts = [clean_query]
    if expanded_terms:
        expanded_parts.extend(expanded_terms)
    expanded_query = " ".join(list(set(expanded_parts)))

    # Normalize extracted topic (handle examples specifically)
    topic_lower = extracted_topic.lower()
    topic_words = set(re.findall(r'\b\w+\b', topic_lower))
    for word, info in ABBREVIATIONS.items():
        if word in topic_words:
            extracted_topic = info["full_name"]
            break

    if "tech" in topic_lower:
        extracted_topic = "Technology"
        inferred_category = "Technology"

    state["extracted_topic"] = extracted_topic
    state["extracted_entities"] = entities
    state["expanded_query"] = expanded_query
    state["target_category"] = inferred_category

    logger.info(f"Query Analysis Complete:\n"
                f"  Extracted Topic: {extracted_topic}\n"
                f"  Entities: {entities}\n"
                f"  Expanded Query: {expanded_query}\n"
                f"  Category: {inferred_category}\n"
                f"  Target URL: {target_url}")

    return state
