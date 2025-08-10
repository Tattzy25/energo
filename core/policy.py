from __future__ import annotations
from typing import Any, Dict, Optional
from pydantic import BaseModel
from loguru import logger


class ActionRecord(BaseModel):
    action: str
    target: Optional[str] = None
    allowed: bool = True
    reason: str = ""
    metadata: Dict[str, Any] = {}


class PolicyEnforcer:
    """
    Simple policy layer to support privacy, transparency, and accountability.
    - Redacts sensitive fields in logs
    - Provides pre_action checks (allow/deny with reason)
    - Audits actions for accountability
    """

    SENSITIVE_KEYS = {"api_key", "apikey", "token", "secret", "password", "authorization"}

    def __init__(self, enabled: bool = True) -> None:
        self.enabled = enabled
        self.audit_log: list[ActionRecord] = []

    def redact(self, data: Any) -> Any:
        try:
            if isinstance(data, dict):
                return {k: ("***" if k.lower() in self.SENSITIVE_KEYS else self.redact(v)) for k, v in data.items()}
            if isinstance(data, (list, tuple)):
                t = type(data)
                return t(self.redact(v) for v in data)
            return data
        except Exception:
            return data

    def pre_action(self, action: str, target: Optional[str] = None, metadata: Optional[Dict[str, Any]] = None) -> ActionRecord:
        # Extend with richer allow/deny logic as needed
        record = ActionRecord(action=action, target=target, allowed=True, reason="policy_ok", metadata=self.redact(metadata or {}))
        return record

    def audit(self, record: ActionRecord) -> None:
        self.audit_log.append(record)
        # Transparent logging
        logger.info("POLICY action={} target={} allowed={} reason={} meta={}", record.action, record.target, record.allowed, record.reason, record.metadata)
