from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from fastapi import FastAPI, HTTPException
from contextlib import asynccontextmanager
from pydantic import BaseModel
from typing import Optional
import json

from agent import VersatileAgent


# ============================================================================
# API Models
# ============================================================================

class ChatRequest(BaseModel):
    message: str
    session_id: str = "default"
    mode: str = "chat"  # chat, tool, think


class ClearHistoryRequest(BaseModel):
    session_id: str = "default"


class HistoryResponse(BaseModel):
    session_id: str
    history: list[dict[str, str]]


# ============================================================================
# Global Agent Instance
# ============================================================================

agent: Optional[VersatileAgent] = None


# ============================================================================
# Application Lifecycle
# ============================================================================

@asynccontextmanager
async def lifespan(app: FastAPI):
    """애플리케이션 시작/종료 시 실행"""
    global agent
    
    # llama-3.2-Korean-Bllossom-3B
    # model_path = './models/llama-3.2-Korean-Bllossom-3B/f16.gguf'
    # model_path = './models/llama-3.2-Korean-Bllossom-3B/Q8_0.gguf'
    
    # llama-3-Korean-Bllossom-8B
    # model_path = './models/llama-3-Korean-Bllossom-8B/f16.gguf'
    model_path = './models/llama-3-Korean-Bllossom-8B/Q8_0.gguf'
    # model_path = './models/llama-3-Korean-Bllossom-8B/Q4_K_M.gguf'
    # model_path = './models/llama-3-Korean-Bllossom-8B/Open-Ko Q4_K_M.gguf'
    
    # llama-3-Korean-Bllossom-70B
    # model_path = './models/llama-3-Korean-Bllossom-70B/Q4_K_M.gguf'

    # aya-expanse-8b-abliterated
    # model_path = './models/aya-expanse-8b-abliterated/Q8_0.gguf'
    
    print("Loading model...")
    agent = VersatileAgent(model_path)
    print("Model loaded successfully!")
    
    yield
    
    print("Shutting down...")


# ============================================================================
# FastAPI Application
# ============================================================================

app = FastAPI(
    title="Versatile Agent API",
    description="LLM 기반 에이전트",
    version="1.0.0",
    lifespan=lifespan
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ============================================================================
# API Endpoints
# ============================================================================

@app.get("/")
async def root():
    """헬스 체크"""
    return {
        "status": "running",
        "message": "Versatile Agent API is running",
        "version": "1.0.0"
    }


@app.post("/chat")
async def chat(request: ChatRequest):
    """채팅 엔드포인트 (SSE 스트리밍)"""
    if agent is None:
        raise HTTPException(status_code=503, detail="Model not loaded")
    
    async def event_generator():
        try:
            if request.mode == "tool":
                stream = agent.tool_stream(request.message, request.session_id)
            elif request.mode == "think":
                stream = agent.think_and_answer_stream(request.message, request.session_id)
            else:  # chat
                stream = agent.chat_stream(request.message, request.session_id)
            
            async for event in stream:
                event_dict = event.to_dict()
                yield f"data: {json.dumps(event_dict, ensure_ascii=False)}\n\n"
            
            yield "data: [DONE]\n\n"
            
        except Exception as e:
            error_event = {
                "type": "error",
                "message": str(e)
            }
            yield f"data: {json.dumps(error_event, ensure_ascii=False)}\n\n"
    
    return StreamingResponse(
        event_generator(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no"
        }
    )


@app.post("/clear")
async def clear_history(request: ClearHistoryRequest):
    """대화 기록 초기화"""
    if agent is None:
        raise HTTPException(status_code=503, detail="Model not loaded")
    
    agent.clear_history(request.session_id)
    return {
        "status": "success",
        "message": f"History cleared for session: {request.session_id}"
    }


@app.get("/history/{session_id}")
async def get_history(session_id: str = "default"):
    """대화 기록 조회"""
    if agent is None:
        raise HTTPException(status_code=503, detail="Model not loaded")
    
    formatted_history = agent.get_formatted_history(session_id)
    
    return HistoryResponse(
        session_id=session_id,
        history=formatted_history
    )


@app.get("/sessions")
async def list_sessions():
    """활성 세션 목록 조회"""
    if agent is None:
        raise HTTPException(status_code=503, detail="Model not loaded")
    
    sessions = list(agent.sessions.keys())
    return {
        "sessions": sessions,
        "count": len(sessions)
    }


# ============================================================================
# Application Entry Point
# ============================================================================

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
