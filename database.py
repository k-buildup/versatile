from sqlalchemy import create_engine, Column, Integer, String, Text, Boolean, TIMESTAMP, Enum, ForeignKey
from sqlalchemy.orm import sessionmaker, Session, relationship
from sqlalchemy.ext.declarative import declarative_base
from contextlib import contextmanager
from urllib.parse import quote_plus
from sqlalchemy.sql import func
from dotenv import load_dotenv
from typing import Generator
import os

# 환경 변수 로드
load_dotenv()

# ============================================================================
# Database Configuration
# ============================================================================

DB_HOST = os.getenv("DB_HOST", "localhost")
DB_PORT = os.getenv("DB_PORT", "3306")
DB_USER = os.getenv("DB_USER", "agent_user")
DB_PASSWORD = os.getenv("DB_PASSWORD", "agentpass123")
DB_NAME = os.getenv("DB_NAME", "agent_db")

DB_PASSWORD_ENCODED = quote_plus(DB_PASSWORD)
DATABASE_URL = f"mysql+pymysql://{DB_USER}:{DB_PASSWORD_ENCODED}@{DB_HOST}:{DB_PORT}/{DB_NAME}?charset=utf8mb4"

# SQLAlchemy 엔진 생성
engine = create_engine(
    DATABASE_URL,
    pool_pre_ping=True,
    pool_recycle=3600,
    echo=False
)

# 세션 팩토리
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)

# Base 클래스
Base = declarative_base()


# ============================================================================
# Database Models
# ============================================================================

class User(Base):
    __tablename__ = "users"
    
    id = Column(Integer, primary_key=True, autoincrement=True)
    username = Column(String(100), unique=True, nullable=False, index=True)
    email = Column(String(255), unique=True, nullable=False, index=True)
    password_hash = Column(String(255), nullable=False)
    display_name = Column(String(100))
    created_at = Column(TIMESTAMP, server_default=func.current_timestamp())
    updated_at = Column(TIMESTAMP, server_default=func.current_timestamp(), onupdate=func.current_timestamp())
    last_login = Column(TIMESTAMP, nullable=True)
    is_active = Column(Boolean, default=True)
    
    # Relationships
    sessions = relationship("ChatSession", back_populates="user", cascade="all, delete-orphan")


class ChatSession(Base):
    __tablename__ = "chat_sessions"
    
    id = Column(Integer, primary_key=True, autoincrement=True)
    session_id = Column(String(255), unique=True, nullable=False, index=True)
    user_id = Column(Integer, ForeignKey('users.id', ondelete='CASCADE'), nullable=False, index=True)
    title = Column(String(255), default='새 채팅')
    created_at = Column(TIMESTAMP, server_default=func.current_timestamp(), index=True)
    updated_at = Column(TIMESTAMP, server_default=func.current_timestamp(), onupdate=func.current_timestamp())
    
    # Relationships
    user = relationship("User", back_populates="sessions")
    messages = relationship("Message", back_populates="session", cascade="all, delete-orphan")


class Message(Base):
    __tablename__ = "messages"
    
    id = Column(Integer, primary_key=True, autoincrement=True)
    session_id = Column(String(255), ForeignKey('chat_sessions.session_id', ondelete='CASCADE'), nullable=False, index=True)
    role = Column(Enum('user', 'assistant', 'system'), nullable=False)
    content = Column(Text, nullable=False)
    mode = Column(String(50), default='chat')
    created_at = Column(TIMESTAMP, server_default=func.current_timestamp(), index=True)
    
    # Relationships
    session = relationship("ChatSession", back_populates="messages")


class ToolLog(Base):
    __tablename__ = "tool_logs"
    
    id = Column(Integer, primary_key=True, autoincrement=True)
    session_id = Column(String(255), ForeignKey('chat_sessions.session_id', ondelete='CASCADE'), nullable=False, index=True)
    message_id = Column(Integer, ForeignKey('messages.id', ondelete='SET NULL'), nullable=True)
    tool_name = Column(String(100), index=True)
    tool_input = Column(Text)
    tool_output = Column(Text)
    success = Column(Boolean, default=True)
    error_message = Column(Text)
    created_at = Column(TIMESTAMP, server_default=func.current_timestamp(), index=True)


class UserFeedback(Base):
    __tablename__ = "user_feedback"
    
    id = Column(Integer, primary_key=True, autoincrement=True)
    session_id = Column(String(255), ForeignKey('chat_sessions.session_id', ondelete='CASCADE'), nullable=False, index=True)
    message_id = Column(Integer, ForeignKey('messages.id', ondelete='SET NULL'), nullable=True)
    rating = Column(Integer)
    feedback_text = Column(Text)
    created_at = Column(TIMESTAMP, server_default=func.current_timestamp())


# ============================================================================
# Database Functions
# ============================================================================

def init_db():
    """데이터베이스 테이블 생성"""
    Base.metadata.create_all(bind=engine)
    print("Database tables created successfully!")


@contextmanager
def get_db() -> Generator[Session, None, None]:
    """데이터베이스 세션 컨텍스트 매니저"""
    db = SessionLocal()
    try:
        yield db
        db.commit()
    except Exception:
        db.rollback()
        raise
    finally:
        db.close()


def get_db_session() -> Session:
    """데이터베이스 세션 생성 (FastAPI dependency 용)"""
    return SessionLocal()
