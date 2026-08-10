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
    
    # L2 Agentic Upgrades
    observations: List[Dict[str, Any]]
    iteration_count: int
    reflection_report: Dict[str, Any]
    agent_trace: List[str]
    evaluation_metrics: Dict[str, Any]
    
    # Workflow transition states
    next_action: str
    action_answer: str
    action_thought: str
    action_tool: str
    action_arguments: Dict[str, Any]
    
    # Real Telemetry & Evaluation States
    start_time: float
    cache_hit: bool




