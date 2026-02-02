from typing import List, Optional, Dict
from sqlalchemy.orm import Session
from sqlalchemy import desc, func
from datetime import datetime
import bcrypt

from database import User, ChatSession, Message, ToolLog, UserFeedback


# ============================================================================
# User Operations
# ============================================================================

def get_password_hash(password: str) -> str:
    """비밀번호 해싱"""
    # bcrypt는 최대 72바이트까지만 지원
    password_bytes = password.encode('utf-8')[:72]
    salt = bcrypt.gensalt()
    hashed = bcrypt.hashpw(password_bytes, salt)
    return hashed.decode('utf-8')


def verify_password(plain_password: str, hashed_password: str) -> bool:
    """비밀번호 검증"""
    try:
        # bcrypt는 최대 72바이트까지만 지원
        password_bytes = plain_password.encode('utf-8')[:72]
        hashed_bytes = hashed_password.encode('utf-8')
        return bcrypt.checkpw(password_bytes, hashed_bytes)
    except Exception as e:
        print(f"Password verification error: {e}")
        return False


def create_user(db: Session, username: str, email: str, password: str, display_name: str = None) -> User:
    """새 사용자 생성"""
    password_hash = get_password_hash(password)
    
    user = User(
        username=username,
        email=email,
        password_hash=password_hash,
        display_name=display_name or username
    )
    
    db.add(user)
    db.commit()
    db.refresh(user)
    return user


def get_user_by_username(db: Session, username: str) -> Optional[User]:
    """사용자명으로 사용자 조회"""
    return db.query(User).filter(User.username == username).first()


def get_user_by_email(db: Session, email: str) -> Optional[User]:
    """이메일로 사용자 조회"""
    return db.query(User).filter(User.email == email).first()


def get_user_by_id(db: Session, user_id: int) -> Optional[User]:
    """ID로 사용자 조회"""
    return db.query(User).filter(User.id == user_id).first()


def authenticate_user(db: Session, username: str, password: str) -> Optional[User]:
    """사용자 인증"""
    user = get_user_by_username(db, username)
    if not user:
        return None
    if not verify_password(password, user.password_hash):
        return None
    
    # 마지막 로그인 시간 업데이트
    user.last_login = datetime.now()
    db.commit()
    
    return user


def update_user(db: Session, user_id: int, **kwargs) -> Optional[User]:
    """사용자 정보 업데이트"""
    user = get_user_by_id(db, user_id)
    if not user:
        return None
    
    for key, value in kwargs.items():
        if hasattr(user, key):
            setattr(user, key, value)
    
    db.commit()
    db.refresh(user)
    return user


# ============================================================================
# Chat Session Operations
# ============================================================================

def create_session(db: Session, session_id: str, user_id: int, title: str = "새 채팅") -> ChatSession:
    """새 채팅 세션 생성"""
    db_session = ChatSession(
        session_id=session_id,
        user_id=user_id,
        title=title
    )
    db.add(db_session)
    db.commit()
    db.refresh(db_session)
    return db_session


def get_session(db: Session, session_id: str) -> Optional[ChatSession]:
    """세션 조회"""
    return db.query(ChatSession).filter(ChatSession.session_id == session_id).first()


def get_or_create_session(db: Session, session_id: str, user_id: int, title: str = "새 채팅") -> ChatSession:
    """세션 조회 또는 생성"""
    session = get_session(db, session_id)
    if not session:
        session = create_session(db, session_id, user_id, title)
    return session


def get_user_sessions(db: Session, user_id: int, limit: int = 50) -> List[ChatSession]:
    """사용자의 세션 목록"""
    return (
        db.query(ChatSession)
        .filter(ChatSession.user_id == user_id)
        .order_by(desc(ChatSession.updated_at))
        .limit(limit)
        .all()
    )


def update_session_title(db: Session, session_id: str, title: str) -> Optional[ChatSession]:
    """세션 제목 업데이트"""
    session = get_session(db, session_id)
    if session:
        session.title = title
        db.commit()
        db.refresh(session)
    return session


def delete_session(db: Session, session_id: str, user_id: int) -> bool:
    """세션 삭제 (권한 확인)"""
    session = get_session(db, session_id)
    if session and session.user_id == user_id:
        db.delete(session)
        db.commit()
        return True
    return False


# ============================================================================
# Message Operations
# ============================================================================

def create_message(
    db: Session,
    session_id: str,
    role: str,
    content: str,
    mode: str = "chat"
) -> Message:
    """메시지 생성"""
    message = Message(
        session_id=session_id,
        role=role,
        content=content,
        mode=mode
    )
    db.add(message)
    db.commit()
    db.refresh(message)
    return message


def get_session_messages(
    db: Session,
    session_id: str,
    limit: Optional[int] = None
) -> List[Message]:
    """세션의 메시지 조회"""
    query = db.query(Message).filter(Message.session_id == session_id).order_by(Message.created_at)
    
    if limit:
        query = query.limit(limit)
    
    return query.all()


def get_recent_messages(
    db: Session,
    session_id: str,
    count: int = 10
) -> List[Message]:
    """최근 메시지 조회"""
    return (
        db.query(Message)
        .filter(Message.session_id == session_id)
        .order_by(desc(Message.created_at))
        .limit(count)
        .all()
    )[::-1]


def delete_session_messages(db: Session, session_id: str) -> int:
    """세션의 모든 메시지 삭제"""
    count = db.query(Message).filter(Message.session_id == session_id).delete()
    db.commit()
    return count


# ============================================================================
# Tool Log Operations
# ============================================================================

def create_tool_log(
    db: Session,
    session_id: str,
    tool_name: str,
    tool_input: str,
    tool_output: str,
    success: bool = True,
    error_message: Optional[str] = None,
    message_id: Optional[int] = None
) -> ToolLog:
    """도구 사용 로그 생성"""
    log = ToolLog(
        session_id=session_id,
        message_id=message_id,
        tool_name=tool_name,
        tool_input=tool_input,
        tool_output=tool_output,
        success=success,
        error_message=error_message
    )
    db.add(log)
    db.commit()
    db.refresh(log)
    return log


def get_tool_logs(
    db: Session,
    session_id: Optional[str] = None,
    tool_name: Optional[str] = None,
    limit: int = 100
) -> List[ToolLog]:
    """도구 로그 조회"""
    query = db.query(ToolLog)
    
    if session_id:
        query = query.filter(ToolLog.session_id == session_id)
    
    if tool_name:
        query = query.filter(ToolLog.tool_name == tool_name)
    
    return query.order_by(desc(ToolLog.created_at)).limit(limit).all()


# ============================================================================
# User Feedback Operations
# ============================================================================

def create_feedback(
    db: Session,
    session_id: str,
    rating: int,
    feedback_text: Optional[str] = None,
    message_id: Optional[int] = None
) -> UserFeedback:
    """사용자 피드백 생성"""
    feedback = UserFeedback(
        session_id=session_id,
        message_id=message_id,
        rating=rating,
        feedback_text=feedback_text
    )
    db.add(feedback)
    db.commit()
    db.refresh(feedback)
    return feedback


def get_session_feedback(db: Session, session_id: str) -> List[UserFeedback]:
    """세션의 피드백 조회"""
    return (
        db.query(UserFeedback)
        .filter(UserFeedback.session_id == session_id)
        .order_by(desc(UserFeedback.created_at))
        .all()
    )


def get_feedback_stats(db: Session, user_id: Optional[int] = None) -> Dict:
    """피드백 통계"""
    query = db.query(UserFeedback)
    
    if user_id:
        # 특정 사용자의 세션에 대한 피드백만
        query = query.join(ChatSession).filter(ChatSession.user_id == user_id)
    
    total = query.count()
    avg_rating = query.with_entities(func.avg(UserFeedback.rating)).scalar()
    
    return {
        "total_feedback": total or 0,
        "average_rating": float(avg_rating) if avg_rating else 0.0
    }


# ============================================================================
# History Management
# ============================================================================

def get_formatted_history(db: Session, session_id: str) -> List[Dict[str, str]]:
    """포맷된 대화 기록 반환"""
    messages = get_session_messages(db, session_id)
    
    return [
        {
            "role": msg.role,
            "content": msg.content,
            "timestamp": msg.created_at.isoformat()
        }
        for msg in messages
    ]


def clear_session_history(db: Session, session_id: str, user_id: int) -> Dict:
    """세션 기록 삭제 (권한 확인)"""
    session = get_session(db, session_id)
    if not session or session.user_id != user_id:
        return {"error": "Unauthorized or session not found"}
    
    # 메시지 삭제
    message_count = delete_session_messages(db, session_id)
    
    # 도구 로그 삭제
    tool_log_count = db.query(ToolLog).filter(ToolLog.session_id == session_id).delete()
    
    # 피드백 삭제
    feedback_count = db.query(UserFeedback).filter(UserFeedback.session_id == session_id).delete()
    
    db.commit()
    
    return {
        "messages_deleted": message_count,
        "tool_logs_deleted": tool_log_count,
        "feedback_deleted": feedback_count
    }
