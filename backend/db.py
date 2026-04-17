"""
db.py
-----
Hybrid Database functionality for SQLite.
Stores Chat History and saved Retention Strategies.
"""

import os
from sqlalchemy import create_engine, Column, Integer, String, Text, DateTime
from sqlalchemy.orm import declarative_base, sessionmaker
from datetime import datetime
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

DATABASE_URL = os.environ.get("DATABASE_URL", "sqlite:///./bank_churn_rag.db")

# SQLAlchemy 1.4+ requires 'postgresql' prefix instead of 'postgres'
if DATABASE_URL.startswith("postgres://"):
    DATABASE_URL = DATABASE_URL.replace("postgres://", "postgresql://", 1)

connect_args = {}
if "sqlite" in DATABASE_URL:
    connect_args["check_same_thread"] = False

engine = create_engine(
    DATABASE_URL, 
    connect_args=connect_args,
    pool_pre_ping=True,
    pool_recycle=300
)
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)


Base = declarative_base()

class ChatMessage(Base):
    __tablename__ = "chat_messages"

    id = Column(Integer, primary_key=True, index=True)
    session_id = Column(String, index=True, default="default")
    role = Column(String)  # 'user' or 'ai'
    content = Column(Text)
    timestamp = Column(DateTime, default=datetime.utcnow)

class RetentionStrategy(Base):
    __tablename__ = "retention_strategies"

    id = Column(Integer, primary_key=True, index=True)
    context = Column(String) # e.g., 'Germany, female, low tenure'
    strategy_text = Column(Text)
    created_at = Column(DateTime, default=datetime.utcnow)

def init_db():
    Base.metadata.create_all(bind=engine)

_db_initialized = False

def get_db():
    global _db_initialized
    if not _db_initialized:
        logger.info("Lazily initializing database tables...")
        init_db()
        _db_initialized = True
        
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()
