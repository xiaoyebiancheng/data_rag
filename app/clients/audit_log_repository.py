import os
from datetime import datetime
from typing import Any, Dict, Optional

from dotenv import load_dotenv
from pymongo import ASCENDING, DESCENDING, MongoClient

from app.core.logger import logger

load_dotenv()


class AuditLogRepository:
    def __init__(self):
        mongo_url = os.getenv("MONGO_URL")
        db_name = os.getenv("MONGO_DB_NAME")
        if not mongo_url or not db_name:
            raise RuntimeError("缺少MongoDB配置，请检查 MONGO_URL / MONGO_DB_NAME")
        self.client = MongoClient(mongo_url)
        self.db = self.client[db_name]
        self.audit_log = self.db["audit_log"]
        self._ensure_indexes()

    def _ensure_indexes(self) -> None:
        self.audit_log.create_index([("timestamp", DESCENDING)])
        self.audit_log.create_index([("user_id", ASCENDING), ("timestamp", DESCENDING)])
        self.audit_log.create_index([("tenant_id", ASCENDING), ("timestamp", DESCENDING)])
        self.audit_log.create_index([("action", ASCENDING), ("timestamp", DESCENDING)])

    @staticmethod
    def _utcnow() -> datetime:
        return datetime.utcnow()

    def write_log(self, payload: Dict[str, Any]) -> None:
        safe_payload = dict(payload)
        safe_payload.pop("authorization", None)
        safe_payload.pop("token", None)
        safe_payload["timestamp"] = safe_payload.get("timestamp") or self._utcnow()
        self.audit_log.insert_one(safe_payload)


_audit_log_repository: Optional[AuditLogRepository] = None


def get_audit_log_repository() -> AuditLogRepository:
    global _audit_log_repository
    if _audit_log_repository is None:
        _audit_log_repository = AuditLogRepository()
        logger.info("AuditLogRepository 初始化成功")
    return _audit_log_repository


def write_audit_log(
    *,
    user_id: str,
    tenant_id: str,
    action: str,
    resource_type: str,
    resource_id: str,
    success: bool,
    error_message: str = "",
    department_id: str = "",
    extra: Optional[Dict[str, Any]] = None,
) -> None:
    try:
        repository = get_audit_log_repository()
        payload = {
            "user_id": user_id,
            "tenant_id": tenant_id,
            "department_id": department_id,
            "action": action,
            "resource_type": resource_type,
            "resource_id": resource_id,
            "success": success,
            "error_message": error_message[:1000] if error_message else "",
        }
        if extra:
            payload["extra"] = extra
        repository.write_log(payload)
    except Exception as exc:
        # 优化: 优化的原因是审计日志是安全可观测能力，不能因为审计存储短暂异常反向打断查询或导入主流程。
        logger.warning(f"审计日志写入失败，已降级不影响主流程: {exc}")
