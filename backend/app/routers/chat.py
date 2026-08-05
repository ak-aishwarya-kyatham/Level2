from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
from typing import Optional
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
        error=""
    )
    
    try:
        # Run the LangGraph workflow
        final_state = await app_graph.ainvoke(initial_state)
        
        t_end = time.time()
        logger.info(f"⏱️ TOTAL CHAT RESPONSE TIME: {t_end - t_start:.1f}s for query: \"{request.query}\"")
        
        return ChatResponse(
            response=final_state.get("final_response", "No response generated."),
            intent=final_state.get("intent", "unknown")
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

