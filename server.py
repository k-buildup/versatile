from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from fastapi import FastAPI, HTTPException, Depends, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from contextlib import asynccontextmanager
from pydantic import BaseModel, EmailStr
from datetime import datetime, timedelta
from sqlalchemy.orm import Session
from typing import Optional, List
from dotenv import load_dotenv
import json
import os
import jwt

from agent import VersatileAgent

from database import get_db, init_db, get_db_session
import db_operations as db_ops

# 환경 변수 로드
load_dotenv()

# JWT 설정
SECRET_KEY = os.getenv("JWT_SECRET_KEY", "your-secret-key-change-this-in-production")
ALGORITHM = "HS256"
ACCESS_TOKEN_EXPIRE_MINUTES = 60 * 24 * 7  # 7일

security = HTTPBearer()


# ============================================================================
# API Models
# ============================================================================

class UserRegister(BaseModel):
    username: str
    email: EmailStr
    password: str
    display_name: Optional[str] = None


class UserLogin(BaseModel):
    username: str
    password: str


class Token(BaseModel):
    access_token: str
    token_type: str
    user: dict


class UserResponse(BaseModel):
    id: int
    username: str
    email: str
    display_name: str
    created_at: datetime
    last_login: Optional[datetime]


class ChatRequest(BaseModel):
    message: str
    session_id: Optional[str] = None
    mode: str = "chat"


class CreateSessionRequest(BaseModel):
    title: str = "새 채팅"


class UpdateSessionRequest(BaseModel):
    title: str


class SessionResponse(BaseModel):
    session_id: str
    title: str
    created_at: datetime
    updated_at: datetime
    message_count: int


class ClearHistoryRequest(BaseModel):
    session_id: str


class HistoryResponse(BaseModel):
    session_id: str
    history: List[dict]


class FeedbackRequest(BaseModel):
    session_id: str
    message_id: Optional[int] = None
    rating: int
    feedback_text: Optional[str] = None


# ============================================================================
# JWT Functions
# ============================================================================

def create_access_token(data: dict, expires_delta: Optional[timedelta] = None):
    """JWT 액세스 토큰 생성"""
    to_encode = data.copy()
    if expires_delta:
        expire = datetime.utcnow() + expires_delta
    else:
        expire = datetime.utcnow() + timedelta(minutes=ACCESS_TOKEN_EXPIRE_MINUTES)
    
    to_encode.update({"exp": expire})
    encoded_jwt = jwt.encode(to_encode, SECRET_KEY, algorithm=ALGORITHM)
    return encoded_jwt


def verify_token(credentials: HTTPAuthorizationCredentials = Depends(security)) -> dict:
    """JWT 토큰 검증"""
    try:
        token = credentials.credentials
        payload = jwt.decode(token, SECRET_KEY, algorithms=[ALGORITHM])
        return payload
    except jwt.ExpiredSignatureError:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Token has expired"
        )
    except jwt.JWTError:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Could not validate credentials"
        )


def get_current_user(
    token_data: dict = Depends(verify_token),
    db: Session = Depends(get_db_session)
) -> db_ops.User:
    """현재 로그인한 사용자 조회"""
    user_id = token_data.get("user_id")
    if not user_id:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid token"
        )
    
    user = db_ops.get_user_by_id(db, user_id)
    if not user:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="User not found"
        )
    
    return user


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
    
    # 데이터베이스 초기화
    print("Initializing database...")
    try:
        init_db()
        print("Database initialized successfully!")
    except Exception as e:
        print(f"Database initialization error: {e}")
    
    # 모델 로드
    model_path = os.getenv("MODEL_PATH", './models/llama-3-Korean-Bllossom-8B/Q8_0.gguf')
    
    print(f"Loading model from {model_path}...")
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
# Auth Endpoints
# ============================================================================

@app.post("/auth/register", response_model=Token)
async def register(user_data: UserRegister, db: Session = Depends(get_db_session)):
    """회원가입"""
    try:
        if db_ops.get_user_by_username(db, user_data.username):
            raise HTTPException(status_code=400, detail="Username already exists")
        
        if db_ops.get_user_by_email(db, user_data.email):
            raise HTTPException(status_code=400, detail="Email already exists")
        
        user = db_ops.create_user(
            db,
            username=user_data.username,
            email=user_data.email,
            password=user_data.password,
            display_name=user_data.display_name
        )
        
        access_token = create_access_token(data={"user_id": user.id})
        
        return Token(
            access_token=access_token,
            token_type="bearer",
            user={
                "id": user.id,
                "username": user.username,
                "email": user.email,
                "display_name": user.display_name
            }
        )
    
    finally:
        db.close()


@app.post("/auth/login", response_model=Token)
async def login(credentials: UserLogin, db: Session = Depends(get_db_session)):
    """로그인"""
    try:
        user = db_ops.authenticate_user(db, credentials.username, credentials.password)
        
        if not user:
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="Incorrect username or password"
            )
        
        access_token = create_access_token(data={"user_id": user.id})
        
        return Token(
            access_token=access_token,
            token_type="bearer",
            user={
                "id": user.id,
                "username": user.username,
                "email": user.email,
                "display_name": user.display_name
            }
        )
    
    finally:
        db.close()


@app.get("/auth/me", response_model=UserResponse)
async def get_me(current_user: db_ops.User = Depends(get_current_user)):
    """현재 사용자 정보"""
    return UserResponse(
        id=current_user.id,
        username=current_user.username,
        email=current_user.email,
        display_name=current_user.display_name,
        created_at=current_user.created_at,
        last_login=current_user.last_login
    )


# ============================================================================
# Session Management Endpoints
# ============================================================================

@app.post("/sessions/create")
async def create_session(
    request: CreateSessionRequest,
    current_user: db_ops.User = Depends(get_current_user),
    db: Session = Depends(get_db_session)
):
    """새 세션 생성"""
    try:
        session_id = f"user_{current_user.id}_{int(datetime.now().timestamp() * 1000)}"
        
        session = db_ops.create_session(
            db,
            session_id=session_id,
            user_id=current_user.id,
            title=request.title
        )
        
        return {
            "session_id": session.session_id,
            "title": session.title,
            "created_at": session.created_at.isoformat()
        }
    
    finally:
        db.close()


@app.get("/sessions", response_model=List[SessionResponse])
async def list_sessions(
    current_user: db_ops.User = Depends(get_current_user),
    db: Session = Depends(get_db_session)
):
    """사용자의 세션 목록"""
    try:
        sessions = db_ops.get_user_sessions(db, current_user.id)
        
        result = []
        for session in sessions:
            message_count = db.query(db_ops.Message).filter(
                db_ops.Message.session_id == session.session_id
            ).count()
            
            result.append(SessionResponse(
                session_id=session.session_id,
                title=session.title,
                created_at=session.created_at,
                updated_at=session.updated_at,
                message_count=message_count
            ))
        
        return result
    
    finally:
        db.close()


@app.put("/sessions/{session_id}/title")
async def update_session_title(
    session_id: str,
    request: UpdateSessionRequest,
    current_user: db_ops.User = Depends(get_current_user),
    db: Session = Depends(get_db_session)
):
    """세션 제목 업데이트"""
    try:
        session = db_ops.get_session(db, session_id)
        
        if not session or session.user_id != current_user.id:
            raise HTTPException(status_code=404, detail="Session not found")
        
        updated_session = db_ops.update_session_title(db, session_id, request.title)
        
        return {
            "session_id": updated_session.session_id,
            "title": updated_session.title
        }
    
    finally:
        db.close()


@app.delete("/sessions/{session_id}")
async def delete_session(
    session_id: str,
    current_user: db_ops.User = Depends(get_current_user),
    db: Session = Depends(get_db_session)
):
    """세션 삭제"""
    try:
        success = db_ops.delete_session(db, session_id, current_user.id)
        
        if not success:
            raise HTTPException(status_code=404, detail="Session not found")
        
        return {"status": "success", "message": "Session deleted"}
    
    finally:
        db.close()


# ============================================================================
# Chat Endpoints
# ============================================================================

@app.post("/chat")
async def chat(
    request: ChatRequest,
    current_user: db_ops.User = Depends(get_current_user),
    db: Session = Depends(get_db_session)
):
    """채팅 엔드포인트"""
    if agent is None:
        raise HTTPException(status_code=503, detail="Model not loaded")
    
    try:
        # 세션 처리
        if not request.session_id:
            session_id = f"user_{current_user.id}_{int(datetime.now().timestamp() * 1000)}"
            title = request.message[:30] + "..." if len(request.message) > 30 else request.message
            db_ops.create_session(db, session_id, current_user.id, title)
        else:
            session_id = request.session_id
            session = db_ops.get_session(db, session_id)
            if not session or session.user_id != current_user.id:
                raise HTTPException(status_code=403, detail="Access denied")
        
        # 사용자 메시지 저장
        db_ops.create_message(db, session_id, "user", request.message, request.mode)
        
    except HTTPException:
        raise
    except Exception as e:
        db.close()
        raise HTTPException(status_code=500, detail=str(e))
    
    async def event_generator():
        assistant_response = ""
        thinking_process = []  # 사고 과정 저장
        current_todo = None
        
        try:
            if request.mode == "tool":
                stream = agent.tool_stream(request.message, session_id)
            elif request.mode == "think":
                stream = agent.think_and_answer_stream(request.message, session_id)
            else:
                stream = agent.chat_stream(request.message, session_id)
            
            async for event in stream:
                if hasattr(event, 'to_dict'):
                    event_dict = event.to_dict()
                else:
                    event_dict = {
                        "type": getattr(event, 'type', getattr(event, 'event_type', 'unknown')),
                        "content": getattr(event, 'content', ''),
                        "mode": getattr(event, 'mode', ''),
                        "is_start": getattr(event, 'is_start', False)
                    }
                
                event_type = event_dict.get("type")
                mode = event_dict.get("mode")
                content = event_dict.get("content", "")
                is_start = event_dict.get("is_start", False)
                
                # Think 모드 사고 과정 수집
                if request.mode == "think":
                    if mode == "todo" and is_start and content:
                        # 새 todo 시작
                        if current_todo:
                            thinking_process.append(current_todo)
                        current_todo = {"todo": content, "content": ""}
                    
                    elif mode == "result" and event_type == "stream" and content:
                        # result 내용 수집
                        if current_todo:
                            current_todo["content"] += content
                
                # 최종 응답 수집
                if event_type in ["text", "stream"] and mode == "basic":
                    if content:
                        assistant_response += content
                
                yield f"data: {json.dumps(event_dict, ensure_ascii=False)}\n\n"
            
            # 마지막 todo 저장
            if current_todo:
                thinking_process.append(current_todo)
            
            # 어시스턴트 응답 저장 (사고 과정 포함)
            if assistant_response.strip():
                try:
                    with get_db() as db_context:
                        db_ops.create_message(
                            db_context,
                            session_id,
                            "assistant",
                            assistant_response,
                            request.mode,
                            thinking_process=thinking_process if thinking_process else None
                        )
                except Exception as db_error:
                    print(f"Failed to save assistant message: {db_error}")
            
            # 세션 ID 전송
            if not request.session_id:
                yield f"data: {json.dumps({'type': 'session_created', 'session_id': session_id}, ensure_ascii=False)}\n\n"
            
            yield "data: [DONE]\n\n"
            
        except Exception as e:
            import traceback
            print(f"Stream error: {e}")
            print(traceback.format_exc())
            error_event = {"type": "error", "message": str(e)}
            yield f"data: {json.dumps(error_event, ensure_ascii=False)}\n\n"
        
        finally:
            db.close()
    
    return StreamingResponse(
        event_generator(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no"
        }
    )


@app.get("/history/{session_id}")
async def get_history(
    session_id: str,
    current_user: db_ops.User = Depends(get_current_user),
    db: Session = Depends(get_db_session)
):
    """대화 기록 조회"""
    try:
        session = db_ops.get_session(db, session_id)
        if not session or session.user_id != current_user.id:
            raise HTTPException(status_code=404, detail="Session not found")
        
        formatted_history = db_ops.get_formatted_history(db, session_id, include_thinking=True)
        
        return HistoryResponse(
            session_id=session_id,
            history=formatted_history
        )
    
    finally:
        db.close()


@app.post("/clear")
async def clear_history(
    request: ClearHistoryRequest,
    current_user: db_ops.User = Depends(get_current_user),
    db: Session = Depends(get_db_session)
):
    """대화 기록 초기화"""
    if agent is None:
        raise HTTPException(status_code=503, detail="Model not loaded")
    
    try:
        result = db_ops.clear_session_history(db, request.session_id, current_user.id)
        
        if "error" in result:
            raise HTTPException(status_code=403, detail=result["error"])
        
        agent.clear_history(request.session_id)
        
        return {
            "status": "success",
            "message": f"History cleared for session: {request.session_id}",
            "details": result
        }
    
    finally:
        db.close()


# ============================================================================
# Other Endpoints
# ============================================================================

@app.get("/")
async def root():
    """헬스 체크"""
    return {
        "status": "running",
        "message": "Versatile Agent API",
        "version": "1.0.0"
    }


if __name__ == "__main__":
    import uvicorn
    
    host = os.getenv("API_HOST", "0.0.0.0")
    port = int(os.getenv("API_PORT", "8000"))
    
    uvicorn.run(app, host=host, port=port)
