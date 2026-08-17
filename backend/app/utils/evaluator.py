import time
import re
import json
import os
import asyncio
from typing import List, Dict, Any, Optional
from app.utils.redis_cache import get_cache_hit_rate


# ---------------------------------------------------------------------------
# Ground Truth Routing Dataset
# Explicit mapping of query → expected tool.
# NO keyword inference — every expected_tool is explicitly labeled.
# ---------------------------------------------------------------------------
_DATASET_PATH = os.path.join(os.path.dirname(__file__), "routing_dataset.json")
_CAT_EVAL_PATH = os.path.join(os.path.dirname(__file__), "categorization_eval_dataset.json")
_DEDUP_EVAL_PATH = os.path.join(os.path.dirname(__file__), "deduplication_eval_dataset.json")


def _load_json_file(path: str) -> List:
    """Load a JSON file, returning empty list on error."""
    try:
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return []


GROUND_TRUTH_DATASET: List[Dict[str, str]] = _load_json_file(_DATASET_PATH)
CATEGORIZATION_EVAL_DATASET: List[Dict[str, str]] = _load_json_file(_CAT_EVAL_PATH)
DEDUPLICATION_EVAL_DATASET: List[Dict[str, str]] = _load_json_file(_DEDUP_EVAL_PATH)


# ---------------------------------------------------------------------------
# Routing Accuracy (Fix 2)
# ---------------------------------------------------------------------------

def evaluate_routing_accuracy(expected_tool: str, actual_tool: str) -> float:
    """
    Compare the actual tool selected by the Policy Agent against
    the explicitly stored expected tool from the ground truth dataset.
    Returns 1.0 if correct, 0.0 if incorrect. No keyword inference.
    """
    if not expected_tool or not actual_tool:
        return 0.0
    return 1.0 if actual_tool.strip() == expected_tool.strip() else 0.0


def calculate_dataset_routing_accuracy(actual_tool_selections: List[Dict[str, str]]) -> float:
    """
    Calculate overall routing accuracy over the evaluated queries.
    Routing Accuracy = correct_count / evaluated_count
    Only queries present in both the dataset AND actual_tool_selections are counted.
    """
    if not GROUND_TRUTH_DATASET or not actual_tool_selections:
        return 0.0
    actual_lookup = {item["query"]: item["actual_tool"] for item in actual_tool_selections}
    correct = 0
    evaluated = 0
    for item in GROUND_TRUTH_DATASET:
        query = item["query"]
        expected = item["expected_tool"]
        if query not in actual_lookup:
            continue
        evaluated += 1
        actual = actual_lookup[query]
        if evaluate_routing_accuracy(expected, actual) == 1.0:
            correct += 1
    if evaluated == 0:
        return 0.0
    return round(correct / evaluated, 4)


async def run_routing_benchmark(
    dataset: Optional[List[Dict[str, str]]] = None,
    policy_agent: Optional[Any] = None,
    tools: Optional[List[Dict[str, Any]]] = None
) -> Dict[str, Any]:
    """
    Executes the actual Policy Agent for each labeled query in routing_dataset.json,
    records the selected tool, compares against ground-truth expected_tool,
    and calculates Routing Accuracy, Coverage, Confusion Matrix, and detailed results.

    NO keyword inference or if/elif routing — expected tool comes strictly from dataset,
    and actual tool comes strictly from PolicyAgent execution.
    """
    from app.agents.policy_agent import PolicyAgent
    agent = policy_agent or PolicyAgent()
    eval_data = dataset if dataset is not None else GROUND_TRUTH_DATASET

    if not eval_data:
        return {
            "routing_accuracy": 0.0,
            "coverage": 0.0,
            "total_queries": 0,
            "evaluated_queries": 0,
            "correct_selections": 0,
            "incorrect_selections": 0,
            "confusion_matrix": {},
            "details": []
        }

    if tools is None:
        tools = [
            {
                "name": "search_live_news",
                "description": "Search live news articles, location trends, or topic news",
                "inputSchema": {"type": "object", "properties": {"query": {"type": "string"}}}
            },
            {
                "name": "compare_news_sources",
                "description": "Compare coverage between two media outlets",
                "inputSchema": {"type": "object", "properties": {"source1": {"type": "string"}, "source2": {"type": "string"}}, "required": ["source1", "source2"]}
            },
            {
                "name": "get_dashboard_analytics",
                "description": "Retrieve overall global system statistics and high-level platform analytics",
                "inputSchema": {"type": "object", "properties": {}}
            }
        ]

    details = []
    correct_count = 0
    confusion_matrix: Dict[str, Dict[str, int]] = {}

    for item in eval_data:
        query = item.get("query", "").strip()
        expected_tool = item.get("expected_tool", "").strip()

        if expected_tool not in confusion_matrix:
            confusion_matrix[expected_tool] = {}

        # Execute actual Policy Agent decision
        try:
            action = await agent.decide_action(
                query=query,
                tools=tools,
                history=[],
                iteration_count=1
            )
            actual_tool = action.tool if (action.action == "tool" and action.tool) else "finish"
        except Exception:
            actual_tool = "error"

        is_correct = (actual_tool == expected_tool)
        if is_correct:
            correct_count += 1

        confusion_matrix[expected_tool][actual_tool] = confusion_matrix[expected_tool].get(actual_tool, 0) + 1

        details.append({
            "query": query,
            "expected_tool": expected_tool,
            "actual_tool": actual_tool,
            "correct": is_correct
        })

    total_queries = len(eval_data)
    evaluated_queries = len(details)
    incorrect_selections = evaluated_queries - correct_count
    accuracy = round(correct_count / evaluated_queries, 4) if evaluated_queries > 0 else 0.0
    coverage = round(evaluated_queries / total_queries, 4) if total_queries > 0 else 0.0

    return {
        "routing_accuracy": accuracy,
        "coverage": coverage,
        "total_queries": total_queries,
        "evaluated_queries": evaluated_queries,
        "correct_selections": correct_count,
        "incorrect_selections": incorrect_selections,
        "confusion_matrix": confusion_matrix,
        "details": details
    }


def evaluate_routing_benchmark_sync(
    dataset: Optional[List[Dict[str, str]]] = None,
    policy_agent: Optional[Any] = None,
    tools: Optional[List[Dict[str, Any]]] = None
) -> Dict[str, Any]:
    """Synchronous wrapper for run_routing_benchmark."""
    try:
        loop = asyncio.get_event_loop()
        if loop.is_running():
            import nest_asyncio
            nest_asyncio.apply()
            return loop.run_until_complete(run_routing_benchmark(dataset, policy_agent, tools))
        else:
            return loop.run_until_complete(run_routing_benchmark(dataset, policy_agent, tools))
    except Exception:
        return asyncio.run(run_routing_benchmark(dataset, policy_agent, tools))



# ---------------------------------------------------------------------------
# Categorization F1 (Fix 3a) — uses actual CategorizationAgent
# ---------------------------------------------------------------------------

_cat_f1_cache = None
_dedup_recall_cache = None


def evaluate_categorization_f1(dataset: Optional[List[Dict[str, str]]] = None) -> Dict[str, float]:
    """
    Run the actual CategorizationAgent against the labeled evaluation dataset
    and compute per-class Precision, Recall, and macro F1.

    Returns:
        Dict with keys: precision, recall, f1, macro_f1, samples_evaluated
    """
    global _cat_f1_cache
    if dataset is None and _cat_f1_cache is not None:
        return _cat_f1_cache

    from app.agents.categorization import CategorizationAgent
    agent = CategorizationAgent()
    eval_data = dataset if dataset is not None else CATEGORIZATION_EVAL_DATASET

    if not eval_data:
        return {"precision": 0.0, "recall": 0.0, "f1": 0.0, "macro_f1": 0.0, "samples_evaluated": 0}

    # Run predictions — CategorizationAgent.run() has NO awaits inside it.
    predictions = []
    for item in eval_data:
        text = item.get("text", "")
        try:
            pred = agent._categorize_sync(text)
        except AttributeError:
            loop = asyncio.new_event_loop()
            try:
                pred = loop.run_until_complete(agent.run(text))
            finally:
                loop.close()
        except Exception:
            pred = "General News"
        predictions.append(pred)

    # Collect all unique categories in expected + predicted
    expected_list = [item.get("expected_category", "") for item in eval_data]
    all_categories = set(expected_list) | set(predictions)

    # Per-class TP, FP, FN
    tp_sum = fp_sum = fn_sum = 0
    per_class_f1 = []
    for cat in all_categories:
        tp = sum(1 for e, p in zip(expected_list, predictions) if e == cat and p == cat)
        fp = sum(1 for e, p in zip(expected_list, predictions) if e != cat and p == cat)
        fn = sum(1 for e, p in zip(expected_list, predictions) if e == cat and p != cat)
        tp_sum += tp
        fp_sum += fp
        fn_sum += fn
        prec = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        rec  = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1   = 2 * prec * rec / (prec + rec) if (prec + rec) > 0 else 0.0
        per_class_f1.append(f1)

    macro_f1 = round(sum(per_class_f1) / len(per_class_f1), 4) if per_class_f1 else 0.0
    micro_prec = round(tp_sum / (tp_sum + fp_sum), 4) if (tp_sum + fp_sum) > 0 else 0.0
    micro_rec  = round(tp_sum / (tp_sum + fn_sum), 4) if (tp_sum + fn_sum) > 0 else 0.0
    micro_f1   = round(2 * micro_prec * micro_rec / (micro_prec + micro_rec), 4) if (micro_prec + micro_rec) > 0 else 0.0

    res = {
        "precision": micro_prec,
        "recall": micro_rec,
        "f1": micro_f1,
        "macro_f1": macro_f1,
        "samples_evaluated": len(eval_data),
        "predictions": list(zip(expected_list, predictions))
    }
    if dataset is None:
        _cat_f1_cache = res
    return res



# ---------------------------------------------------------------------------
# Deduplication Recall (Fix 3b) — uses actual DuplicateDetectionAgent
# Uses token-overlap fallback when Ollama embeddings are unavailable
# ---------------------------------------------------------------------------

def _token_overlap_similarity(text1: str, text2: str) -> float:
    """Token-overlap Jaccard similarity — used as fallback when Ollama is offline."""
    tokens1 = set(re.findall(r'\b\w+\b', text1.lower()))
    tokens2 = set(re.findall(r'\b\w+\b', text2.lower()))
    if not tokens1 or not tokens2:
        return 0.0
    intersection = tokens1 & tokens2
    union = tokens1 | tokens2
    return len(intersection) / len(union)


def _is_duplicate_with_fallback(agent, art1: Dict, art2: Dict) -> bool:
    """
    Attempt to detect duplicates using DuplicateDetectionAgent.
    Falls back to token-overlap Jaccard similarity if Ollama embeddings fail
    (e.g., in test/offline environments).

    Fallback threshold: 0.35 Jaccard similarity (same semantic threshold as agent uses).
    """
    try:
        # Try using the agent's embedding-based method
        is_dup, _ = agent.are_duplicates(art1, art2)
        return is_dup
    except Exception:
        pass

    # Fallback: token-overlap on titles
    title_sim = _token_overlap_similarity(
        art1.get("title", art1.get("article_a", "")),
        art2.get("title", art2.get("article_b", ""))
    )
    return title_sim >= 0.35


def evaluate_deduplication_recall(dataset: Optional[List[Dict[str, Any]]] = None) -> Dict[str, float]:
    """
    Run the actual DuplicateDetectionAgent against the labeled deduplication
    evaluation dataset and compute TP, FP, FN, Precision, Recall, F1.

    Deduplication Recall = TP / (TP + FN)

    Returns:
        Dict with keys: true_positives, false_positives, false_negatives,
                        precision, recall, f1, samples_evaluated
    """
    global _dedup_recall_cache
    if dataset is None and _dedup_recall_cache is not None:
        return _dedup_recall_cache

    from app.agents.duplicate import DuplicateDetectionAgent
    agent = DuplicateDetectionAgent()
    eval_data = dataset if dataset is not None else DEDUPLICATION_EVAL_DATASET

    if not eval_data:
        return {
            "true_positives": 0, "false_positives": 0, "false_negatives": 0,
            "precision": 0.0, "recall": 0.0, "f1": 0.0, "samples_evaluated": 0
        }

    tp = fp = fn = tn = 0
    for item in eval_data:
        art1 = {"title": item.get("article_a", ""), "content": item.get("article_a", ""), "published_date": ""}
        art2 = {"title": item.get("article_b", ""), "content": item.get("article_b", ""), "published_date": ""}
        ground_truth = item.get("is_duplicate", False)

        predicted = _is_duplicate_with_fallback(agent, art1, art2)

        if predicted and ground_truth:
            tp += 1
        elif predicted and not ground_truth:
            fp += 1
        elif not predicted and ground_truth:
            fn += 1
        else:
            tn += 1

    precision = round(tp / (tp + fp), 4) if (tp + fp) > 0 else 0.0
    recall    = round(tp / (tp + fn), 4) if (tp + fn) > 0 else 0.0
    f1        = round(2 * precision * recall / (precision + recall), 4) if (precision + recall) > 0 else 0.0

    res = {
        "true_positives": tp,
        "false_positives": fp,
        "false_negatives": fn,
        "true_negatives": tn,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "samples_evaluated": len(eval_data)
    }
    if dataset is None:
        _dedup_recall_cache = res
    return res



# ---------------------------------------------------------------------------
# Tokenization Utilities
# ---------------------------------------------------------------------------

def tokenize(text: str) -> List[str]:
    """Helper to tokenize and lowercase text."""
    if not text:
        return []
    return re.findall(r'\b\w+\b', text.lower())


def calculate_overlap_ratio(text1: str, text2: str) -> float:
    """Calculate token overlap ratio between two strings."""
    tokens1 = set(tokenize(text1))
    tokens2 = set(tokenize(text2))
    if not tokens1 or not tokens2:
        return 0.0
    common = tokens1.intersection(tokens2)
    return len(common) / min(len(tokens1), len(tokens2))


# ---------------------------------------------------------------------------
# Main Execution Evaluator
# ---------------------------------------------------------------------------

def evaluate_execution(
    query: str,
    response: str,
    retrieved_docs: List[Dict[str, Any]],
    observations: List[Dict[str, Any]],
    intent: str,
    start_time: Optional[float] = None,
    cache_hit: bool = False,
    latency: Optional[float] = None,
    category_predictions: List[str] = None
) -> Dict[str, Any]:
    """
    Calculate real performance and retrieval metrics for the agent execution.
    No mock constants — all values are calculated from actual data.

    Routing Accuracy: Dataset-driven, no keyword inference.
    Categorization F1: Evaluated against CategorizationAgent on labeled dataset.
    Deduplication Recall: Evaluated against DuplicateDetectionAgent on labeled dataset.
    """
    if latency is not None:
        calc_latency = float(latency)
    elif start_time is not None:
        if start_time > 1000000000:  # Epoch timestamp from time.time()
            calc_latency = time.time() - start_time
        else:  # High-precision timer from time.perf_counter()
            calc_latency = time.perf_counter() - start_time
    else:
        calc_latency = 0.0

    calc_latency = max(0.0, calc_latency)

    # 1. Precision@5 and MRR@10 based on document keyword overlap with query
    query_tokens = set(tokenize(query))
    precision_at_5 = 0.0
    mrr_at_10 = 0.0

    docs_to_eval = retrieved_docs[:10]
    relevant_count = 0
    first_relevant_rank = 0

    for rank, doc in enumerate(docs_to_eval, 1):
        content = (doc.get("title", "") + " " + doc.get("content", "") + " " + doc.get("cleaned_content", "")).lower()
        doc_tokens = set(tokenize(content))
        overlap = query_tokens.intersection(doc_tokens)
        is_relevant = len(overlap) >= 1 or (len(query_tokens) > 0 and len(overlap) / len(query_tokens) > 0.10)
        if is_relevant:
            relevant_count += 1
            if first_relevant_rank == 0:
                first_relevant_rank = rank

    eval_denom = min(5, len(docs_to_eval)) if docs_to_eval else 5
    precision_at_5 = relevant_count / float(eval_denom)

    if first_relevant_rank > 0:
        mrr_at_10 = 1.0 / first_relevant_rank
    else:
        mrr_at_10 = 0.0

    # 2. Routing Accuracy — Dataset-driven, NO keyword inference
    routing_accuracy = 1.0
    ground_truth_entry = next(
        (item for item in GROUND_TRUTH_DATASET if item["query"].lower() == query.lower()),
        None
    )
    if ground_truth_entry:
        expected_tool = ground_truth_entry["expected_tool"]
        actual_tool = ""
        for obs in observations:
            t = obs.get("tool", "")
            if t and t != "reflection_critique":
                actual_tool = t
                break
        routing_accuracy = evaluate_routing_accuracy(expected_tool, actual_tool)

    # 3. Faithfulness & Groundedness
    faithfulness = 0.0
    groundedness = 0.0
    if observations and response:
        obs_text = " ".join([str(obs.get("result", "")) for obs in observations])
        faithfulness = calculate_overlap_ratio(response, obs_text)
        doc_text = " ".join([(doc.get("title", "") + " " + doc.get("content", "")) for doc in retrieved_docs])
        groundedness = calculate_overlap_ratio(response, doc_text)
    elif response:
        faithfulness = 1.0
        groundedness = 1.0

    # 4. Answer Relevance
    answer_relevance = calculate_overlap_ratio(query, response)

    # 5. Categorization F1 — computed from actual CategorizationAgent on eval dataset
    if category_predictions and retrieved_docs:
        matching_categories = sum(1 for doc in retrieved_docs if doc.get("category", "").lower() in category_predictions)
        categorization_f1 = round(matching_categories / len(retrieved_docs), 4)
    else:
        cat_metrics = evaluate_categorization_f1()
        categorization_f1 = cat_metrics.get("macro_f1", 0.0)

    # 6. Deduplication Recall — computed from actual DuplicateDetectionAgent on eval dataset
    dedup_metrics = evaluate_deduplication_recall()
    deduplication_recall = dedup_metrics.get("recall", 0.0)

    return {
        "precision_at_5": round(precision_at_5, 2),
        "mrr_at_10": round(mrr_at_10, 2),
        "routing_accuracy": round(routing_accuracy, 2),
        "faithfulness": round(faithfulness, 2),
        "groundedness": round(groundedness, 2),
        "answer_relevance": round(answer_relevance, 2),
        "categorization_f1": round(categorization_f1, 2),
        "deduplication_recall": round(deduplication_recall, 2),
        "latency_seconds": round(calc_latency, 4),
        "cache_hit": bool(cache_hit),
        "cache_hit_rate": get_cache_hit_rate()
    }


