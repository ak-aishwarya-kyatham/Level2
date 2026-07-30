from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
from typing import Optional
from app.workflows.main_workflow import app_graph
from app.workflows.langgraph_state import AgentState

router = APIRouter(prefix="/api/chat", tags=["chat"])

class ChatRequest(BaseModel):
    query: str
    user_id: Optional[str] = "default_user"

class ChatResponse(BaseModel):
    response: str
    intent: str

@router.post("/", response_model=ChatResponse)
async def chat_endpoint(request: ChatRequest):
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
        
        return ChatResponse(
            response=final_state.get("final_response", "No response generated."),
            intent=final_state.get("intent", "unknown")
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
