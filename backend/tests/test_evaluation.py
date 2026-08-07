import time
from app.utils.evaluator import evaluate_execution

def test_precision_at_5_real():
    """1. Precision@5: Measures retrieval quality in top 5 RAG search results using actual overlap."""
    query = "AI chips semiconductor"
    retrieved_docs = [
        {"title": "Nvidia releases new AI chip", "content": "semiconductor market updates"},
        {"title": "AMD CPU launch", "content": "details about standard CPUs"},
        {"title": "Intel chips update", "content": "news in silicon and semiconductor technology"},
        {"title": "Weather forecast", "content": "sunny tomorrow"},
        {"title": "Global stock markets", "content": "trading updates"}
    ]
    # 2 docs out of 5 have strong keyword overlap
    metrics = evaluate_execution(
        query=query,
        response="Nvidia and Intel announced new AI semiconductor chips.",
        retrieved_docs=retrieved_docs,
        observations=[],
        intent="search",
        start_time=time.time() - 0.5,
        cache_hit=False
    )
    assert metrics["precision_at_5"] > 0.0

def test_mrr_at_10_real():
    """2. MRR@10: Measures reciprocal rank based on search relevance."""
    query = "AI chips"
    # First relevant doc is at rank 1
    retrieved_docs_1 = [
        {"title": "AI chips are here", "content": "Nvidia GPU"},
        {"title": "Other news", "content": "random"}
    ]
    # First relevant doc is at rank 2
    retrieved_docs_2 = [
        {"title": "Other news", "content": "random"},
        {"title": "AI chips are here", "content": "Nvidia GPU"}
    ]
    
    metrics_1 = evaluate_execution(
        query=query,
        response="answer",
        retrieved_docs=retrieved_docs_1,
        observations=[],
        intent="search",
        start_time=time.time(),
        cache_hit=False
    )
    metrics_2 = evaluate_execution(
        query=query,
        response="answer",
        retrieved_docs=retrieved_docs_2,
        observations=[],
        intent="search",
        start_time=time.time(),
        cache_hit=False
    )
    
    assert metrics_1["mrr_at_10"] == 1.0
    assert metrics_2["mrr_at_10"] == 0.5

def test_faithfulness_real():
    """3. Faithfulness and Groundedness calculations using token overlap."""
    query = "Nvidia performance"
    response = "Nvidia stock rose by 5 percent after launching the new H200 chip."
    observations = [
        {
            "tool": "search_live_news",
            "result": "Nvidia stock rose by 5 percent after launching the new H200 chip."
        }
    ]
    metrics = evaluate_execution(
        query=query,
        response=response,
        retrieved_docs=[],
        observations=observations,
        intent="search",
        start_time=time.time() - 1.0,
        cache_hit=False
    )
    # Since response text matches observations exactly, faithfulness should be high
    assert metrics["faithfulness"] > 0.8
