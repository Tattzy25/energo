from __future__ import annotations
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
from sqlalchemy.pool import NullPool
import os
import re

DATABASE_URL = os.getenv("DATABASE_URL", "postgresql+psycopg://user:pass@localhost:5432/energo")

# Auto-detect Neon pooled connections and optimize for serverless
def _create_engine():
    url = DATABASE_URL
    # Convert postgres:// to postgresql+psycopg:// for compatibility
    if url.startswith("postgres://"):
        url = url.replace("postgres://", "postgresql+psycopg://", 1)
    
    # For Vercel serverless + Neon pooled connections: use NullPool to avoid double-pooling
    if "-pooler." in url or "serverless" in os.getenv("VERCEL_ENV", ""):
        return create_engine(url, poolclass=NullPool, pool_pre_ping=True)
    else:
        return create_engine(url, pool_pre_ping=True)

engine = _create_engine()
SessionLocal = sessionmaker(bind=engine, autoflush=False, autocommit=False)


def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        try:
            db.close()
        except Exception:
            pass
