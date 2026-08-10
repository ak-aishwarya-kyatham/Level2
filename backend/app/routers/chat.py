from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
from typing import Optional, List, Dict, Any
from app.workflows.main_workflow import app_graph
from app.workflows.langgraph_state import AgentState
import logging
import time

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api/chat", tags=["chat"])

class ChatRequest(BaseModel):
    query: str
    user_id: Optional[str] = "default_user"

class ChatResponse(BaseModel):
    response: str
    intent: str
    trace: Optional[List[str]] = None
    metrics: Optional[Dict[str, Any]] = None
    observations: Optional[List[Dict[str, Any]]] = None

@router.post("/", response_model=ChatResponse)
async def chat_endpoint(request: ChatRequest):
    t_start = time.time()
    initial_state = AgentState(
        user_id=request.user_id,
        query=request.query,
        intent="",
        cached_response="",
        retrieved_documents=[],
        specialized_output="",
        final_response="",
        error="",
        observations=[],
        iteration_count=0,
        reflection_report={},
        agent_trace=[],
        evaluation_metrics={},
        start_time=time.perf_counter(),
        cache_hit=False
    )

    
    try:
        # Run the LangGraph workflow
        final_state = await app_graph.ainvoke(initial_state)
        
        t_end = time.time()
        logger.info(f"⏱️ TOTAL CHAT RESPONSE TIME: {t_end - t_start:.1f}s for query: \"{request.query}\"")
        logger.info(f"Execution trace:\n" + "\n".join(final_state.get("agent_trace", [])))
        
        return ChatResponse(
            response=final_state.get("final_response") or "No response generated.",
            intent=final_state.get("intent", "search"),
            trace=final_state.get("agent_trace"),
            metrics=final_state.get("evaluation_metrics"),
            observations=final_state.get("observations")
        )
    except Exception as e:
        logger.error(f"Error in chat endpoint: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))


