from typing import TypedDict, List, Dict, Any

class AgentState(TypedDict):
    """
    Represents the state of our LangGraph workflow.
    """
    user_id: str
    query: str
    intent: str
    cached_response: str
    retrieved_documents: List[Dict[str, Any]]
    specialized_output: str
    final_response: str
    error: str
    extracted_topic: str
    extracted_entities: List[str]
    expanded_query: str
    target_category: str
    target_url: str
    requested_limit: int

