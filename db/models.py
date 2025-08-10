from __future__ import annotations
from sqlalchemy import Column, Integer, String, DateTime, Text, Boolean, ForeignKey, Float, JSON
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


# ===== MCP PROVIDER PLATFORM MODELS =====

class MCPTool(Base):
    """Registry of available MCP tools/capabilities"""
    __tablename__ = "mcp_tools"
    
    id = Column(Integer, primary_key=True, index=True)
    name = Column(String(100), unique=True, nullable=False, index=True)
    display_name = Column(String(200))
    description = Column(Text)
    category = Column(String(50), index=True)  # e.g., "image", "video", "text", "music"
    version = Column(String(20), default="1.0.0")
    
    # Connection details
    endpoint_url = Column(String(500))
    connection_type = Column(String(20), default="websocket")  # websocket, stdio, sse
    
    # Pricing and limits
    pricing_model = Column(String(20), default="free")  # free, per_call, subscription
    price_per_call = Column(Float, default=0.0)
    rate_limit_per_hour = Column(Integer, default=1000)
    
    # Metadata
    tags = Column(JSON)  # ["ai", "image-generation", "stable-diffusion"]
    config_schema = Column(JSON)  # JSON schema for tool configuration
    
    # Status
    health_status = Column(String(20), default="unknown")  # healthy, degraded, down, unknown
    last_health_check = Column(DateTime)
    
    is_active = Column(Boolean, default=True)
    is_public = Column(Boolean, default=True)  # whether external clients can discover
    created_at = Column(DateTime, default=func.now())
    updated_at = Column(DateTime, default=func.now(), onupdate=func.now())
    
    # Relationships
    sessions = relationship("MCPSession", back_populates="tool")
    usage_logs = relationship("MCPUsageLog", back_populates="tool")


class MCPSession(Base):
    """Active MCP sessions (short-lived tokens for client connections)"""
    __tablename__ = "mcp_sessions"
    
    id = Column(Integer, primary_key=True, index=True)
    session_token = Column(String(100), unique=True, nullable=False, index=True)
    tool_id = Column(Integer, ForeignKey("mcp_tools.id"), nullable=False)
    
    # Client info
    client_id = Column(String(100))  # optional client identifier
    client_ip = Column(String(45))
    user_agent = Column(String(500))
    
    # Session lifecycle
    created_at = Column(DateTime, default=func.now())
    expires_at = Column(DateTime, nullable=False)
    last_used_at = Column(DateTime)
    is_active = Column(Boolean, default=True)
    
    # Usage tracking
    calls_made = Column(Integer, default=0)
    bytes_transferred = Column(Integer, default=0)
    
    # Relationships
    tool = relationship("MCPTool", back_populates="sessions")
    usage_logs = relationship("MCPUsageLog", back_populates="session")


class MCPUsageLog(Base):
    """Per-call usage tracking for billing and analytics"""
    __tablename__ = "mcp_usage_logs"
    
    id = Column(Integer, primary_key=True, index=True)
    session_id = Column(Integer, ForeignKey("mcp_sessions.id"))
    tool_id = Column(Integer, ForeignKey("mcp_tools.id"), nullable=False)
    
    # Call details
    method_name = Column(String(100))
    request_size_bytes = Column(Integer, default=0)
    response_size_bytes = Column(Integer, default=0)
    duration_ms = Column(Integer)
    
    # Status
    status_code = Column(Integer)  # 200, 429, 500, etc.
    error_message = Column(Text)
    
    # Billing
    billable_units = Column(Float, default=1.0)  # e.g., 1 call = 1 unit, or based on tokens/pixels
    cost_usd = Column(Float, default=0.0)
    
    created_at = Column(DateTime, default=func.now())
    
    # Relationships
    session = relationship("MCPSession", back_populates="usage_logs")
    tool = relationship("MCPTool", back_populates="usage_logs")


class MCPTenant(Base):
    """Multi-tenant support for rate limits and billing"""
    __tablename__ = "mcp_tenants"
    
    id = Column(Integer, primary_key=True, index=True)
    name = Column(String(100), unique=True, nullable=False)
    email = Column(String(200))
    
    # API access
    api_key = Column(String(100), unique=True, nullable=False, index=True)
    
    # Limits and quotas
    monthly_call_limit = Column(Integer, default=10000)
    calls_this_month = Column(Integer, default=0)
    rate_limit_per_minute = Column(Integer, default=60)
    
    # Billing
    billing_email = Column(String(200))
    stripe_customer_id = Column(String(100))
    subscription_tier = Column(String(50), default="free")  # free, pro, enterprise
    
    # Status
    is_active = Column(Boolean, default=True)
    created_at = Column(DateTime, default=func.now())
    last_reset_at = Column(DateTime, default=func.now())  # for monthly quotas


# Legacy models (keeping for backward compatibility)
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