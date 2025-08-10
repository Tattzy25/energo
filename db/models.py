from __future__ import annotations
from sqlalchemy import Column, Integer, String, DateTime, Text, Boolean, ForeignKey
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.sql import func
from sqlalchemy.orm import relationship
from datetime import datetime

Base = declarative_base()


class Agent(Base):
    __tablename__ = "agents"
    
    id = Column(Integer, primary_key=True, index=True)
    name = Column(String(100), unique=True, nullable=False)
    personality = Column(Text)
    created_at = Column(DateTime, default=func.now())
    is_active = Column(Boolean, default=True)
    
    conversations = relationship("Conversation", back_populates="agent")


class Conversation(Base):
    __tablename__ = "conversations"
    
    id = Column(Integer, primary_key=True, index=True)
    agent_id = Column(Integer, ForeignKey("agents.id"))
    prompt = Column(Text, nullable=False)
    response = Column(Text)
    provider = Column(String(50))
    model = Column(String(100))
    created_at = Column(DateTime, default=func.now())
    
    agent = relationship("Agent", back_populates="conversations")


class MCPServer(Base):
    __tablename__ = "mcp_servers"
    
    id = Column(Integer, primary_key=True, index=True)
    name = Column(String(100), unique=True, nullable=False)
    url = Column(String(500))
    status = Column(String(20), default="offline")
    created_at = Column(DateTime, default=func.now())


class Tool(Base):
    __tablename__ = "tools"
    
    id = Column(Integer, primary_key=True, index=True)
    name = Column(String(100), unique=True, nullable=False)
    description = Column(Text)
    category = Column(String(50))
    is_active = Column(Boolean, default=True)
    created_at = Column(DateTime, default=func.now())