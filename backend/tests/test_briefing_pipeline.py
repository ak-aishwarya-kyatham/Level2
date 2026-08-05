# No pytest import needed
from app.agents.duplicate import DuplicateDetectionAgent
from app.agents.response import synthesize_executive_summary

def test_duplicate_detection_agent_logic():
    agent = DuplicateDetectionAgent()
    
    art1 = {
        "title": "Samsung bans smart TV apps that expose internet connections",
        "content": "Samsung announced that it will ban any Smart TV application that shares or exposes users' internet connections without explicit authorization.",
        "source": "TechCrunch",
        "published_date": "2026-08-04T10:00:00Z"
    }
    
    art2 = {
        "title": "Samsung Says It's Banning Smart TV Apps That Expose Users' Internet Connections",
        "content": "In a move to secure client data, Samsung is banning TV apps exposing connections.",
        "source": "Engadget",
        "published_date": "2026-08-04T10:15:00Z"
    }

    art3 = {
        "title": "Microsoft discloses major Azure network outage",
        "content": "Microsoft is investigating a global cloud outage affecting Azure networks.",
        "source": "Reuters",
        "published_date": "2026-08-04T10:30:00Z"
    }
    
    # Check that duplicates are detected (similar titles/entities/content)
    is_dup1, _ = agent.are_duplicates(art1, art2)
    assert is_dup1 is True
    
    # Check that non-duplicates are not detected
    is_dup2, _ = agent.are_duplicates(art1, art3)
    assert is_dup2 is False

def test_choose_better_article():
    agent = DuplicateDetectionAgent()
    
    art_low = {
        "title": "Samsung bans smart TV apps",
        "content": "Short text.",
        "source": "GenericBlog",
        "published_date": "2026-08-04T09:00:00Z"
    }
    
    art_high = {
        "title": "Samsung Says It's Banning Smart TV Apps That Expose Users' Internet Connections",
        "content": "Very long and rich complete article content explaining that Samsung announced that it will ban any Smart TV application that shares or exposes users' internet connections.",
        "source": "Reuters",
        "published_date": "2026-08-04T10:00:00Z"
    }
    
    better = agent.choose_better_article(art_low, art_high)
    assert better["source"] == "Reuters"

def test_synthesis_rules():
    docs = [
        {
            "title": "Samsung Says It's Banning Smart TV Apps That Expose Users' Internet Connections",
            "content": "Samsung announced that it will ban any Smart TV application that shares or exposes users' internet connections.",
            "source": "Reuters",
            "published_date": "2026-08-04T10:00:00Z"
        },
        {
            "title": "CareCloud Discloses Cyber Breach",
            "content": "CareCloud has suffered a massive security breach exposing patient medical details to unauthorized parties.",
            "source": "Bloomberg",
            "published_date": "2026-08-04T10:10:00Z"
        }
    ]
    
    summary = synthesize_executive_summary("Samsung and cybersecurity updates", docs)
    
    # Check structure presence
    assert "Executive Intelligence Briefing" in summary
    assert "Live Synthesis Overview" in summary
    assert "Primary Source Links" in summary
    
    key_summary = summary.split("**Key Summary:**")[1].split("---")[0].strip()
    assert "Samsung Says It's Banning" not in key_summary
    assert "developments surrounding" not in key_summary.lower()
    assert "specifically" not in key_summary.lower()
    assert "meanwhile" not in key_summary.lower()

def test_query_understanding_and_category_routing():
    from app.agents.query_understanding import query_understanding_agent
    
    # 1. NCP test
    state1 = {
        "query": "latest news about NCP",
        "intent": "search",
        "extracted_topic": "",
        "extracted_entities": [],
        "expanded_query": "",
        "target_category": "",
        "target_url": ""
    }
    res1 = query_understanding_agent(state1)
    assert "Nationalist Congress Party" in res1["extracted_topic"]
    assert "Nationalist Congress Party" in res1["extracted_entities"]
    assert res1["target_category"] == "Politics"
    
    # 2. Kerala rains test
    state2 = {
        "query": "Kerala rains",
        "intent": "search",
        "extracted_topic": "",
        "extracted_entities": [],
        "expanded_query": "",
        "target_category": "",
        "target_url": ""
    }
    res2 = query_understanding_agent(state2)
    assert res2["extracted_topic"] == "Kerala rains"
    assert "Kerala" in res2["extracted_entities"]
    
    # 3. URL test
    state3 = {
        "query": "summarize https://techcrunch.com/2026/08/apple-ai-news",
        "intent": "summarize",
        "extracted_topic": "",
        "extracted_entities": [],
        "expanded_query": "",
        "target_category": "",
        "target_url": ""
    }
    res3 = query_understanding_agent(state3)
    assert res3["target_url"] == "https://techcrunch.com/2026/08/apple-ai-news"

def test_faithfulness_validation_and_grounding():
    from app.agents.response import generate_grounded_summary, validate_faithfulness
    
    docs = [
        {
            "title": "Kerala rains trigger massive floods and landslides",
            "content": "Heavy monsoon rains in Kerala have caused widespread flooding and mudslides, leading to school closures and transport disruptions across multiple districts. Authorities are setting up relief camps.",
        }
    ]
    
    # 1. Test grounded summary generation produces content related to the query
    summary = generate_grounded_summary("Kerala rains", docs)
    assert len(summary) > 20, "Summary should not be empty"
    summary_lower = summary.lower()
    assert "kerala" in summary_lower or "flood" in summary_lower or "rain" in summary_lower or "weather" in summary_lower
    
    # 2. Test faithfulness validation removes hallucinated buzzwords
    hallucinated_summary = "Heavy monsoon rains in Kerala caused widespread flooding. Local governments are deploying neural processors to optimize cloud-native water drainage systems."
    validated = validate_faithfulness(hallucinated_summary, docs)
    assert "Kerala" in validated
    assert "neural processor" not in validated.lower()
    assert "cloud-native" not in validated.lower()

def test_live_feed_intent_and_synthesis():
    from app.agents.triage import triage_agent
    from app.agents.response import synthesize_live_feed_briefing

    # Test triage routing
    state = {"query": "what is the live feed latest one"}
    res = triage_agent(state)
    assert res["intent"] == "live_feed"

    state2 = {"query": "show live feed"}
    res2 = triage_agent(state2)
    assert res2["intent"] == "live_feed"

    # Test live feed synthesis
    sample_docs = [
        {
            "title": "Government to raise CSAM lapses with Meta",
            "source": "The Hindu",
            "category": "Technology",
            "published_date": "2026-08-04T18:04:00Z",
            "url": "https://thehindu.com/meta",
            "content": "Government officials will meet Meta representatives in Delhi."
        }
    ]
    briefing = synthesize_live_feed_briefing(sample_docs)
    assert "Real-Time Live News Feed" in briefing
    assert "Government to raise CSAM lapses with Meta" in briefing

def test_top_1_limit_handling():
    from app.agents.triage import triage_agent
    from app.agents.response import synthesize_live_feed_briefing

    state = {"query": "top 1"}
    res = triage_agent(state)
    assert res["intent"] == "live_feed"
    assert res["requested_limit"] == 1

    sample_docs = [
        {"title": "Article 1", "source": "Source 1", "published_date": "2026-08-04T18:04:00Z"},
        {"title": "Article 2", "source": "Source 2", "published_date": "2026-08-04T18:00:00Z"},
        {"title": "Article 3", "source": "Source 3", "published_date": "2026-08-04T17:50:00Z"},
    ]

    # Limit=1 should produce exactly 1 article
    briefing1 = synthesize_live_feed_briefing(sample_docs, limit=1)
    assert "Article 1" in briefing1
    assert "Article 2" not in briefing1

    # Limit=2 should produce 2 articles
    briefing2 = synthesize_live_feed_briefing(sample_docs, limit=2)
    assert "Article 1" in briefing2
    assert "Article 2" in briefing2
    assert "Article 3" not in briefing2


