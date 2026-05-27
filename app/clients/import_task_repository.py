import os
from datetime import datetime
from typing import Any, Dict, List, Optional

from dotenv import load_dotenv
from pymongo import ASCENDING, DESCENDING, MongoClient

from app.core.logger import logger

load_dotenv()


class ImportTaskStatus:
    PENDING = "PENDING"
    RUNNING = "RUNNING"
    SUCCESS = "SUCCESS"
    FAILED = "FAILED"
    DUPLICATED = "DUPLICATED"
    REPLACED = "REPLACED"


class ImportTaskRepository:
    def __init__(self):
        mongo_url = os.getenv("MONGO_URL")
        db_name = os.getenv("MONGO_DB_NAME")
        if not mongo_url or not db_name:
            raise RuntimeError("缺少MongoDB配置，请检查 MONGO_URL / MONGO_DB_NAME")
        self.client = MongoClient(mongo_url)
        self.db = self.client[db_name]
        self.tasks = self.db["import_task"]
        self.node_logs = self.db["import_task_node_log"]
        self._ensure_indexes()

    def _ensure_indexes(self) -> None:
        self.tasks.create_index([("task_id", ASCENDING)], unique=True)
        self.tasks.create_index([("status", ASCENDING), ("updated_at", DESCENDING)])
        self.tasks.create_index([("doc_id", ASCENDING), ("updated_at", DESCENDING)])
        self.tasks.create_index([("file_hash", ASCENDING), ("updated_at", DESCENDING)])
        self.node_logs.create_index([("task_id", ASCENDING), ("start_time", DESCENDING)])
        self.node_logs.create_index([("task_id", ASCENDING), ("node_name", ASCENDING), ("status", ASCENDING)])

    @staticmethod
    def _utcnow() -> datetime:
        return datetime.utcnow()

    def upsert_task(self, payload: Dict[str, Any]) -> None:
        now = self._utcnow()
        data = dict(payload)
        data.setdefault("created_at", now)
        data["updated_at"] = now
        created_at = data["created_at"]
        update_fields = dict(data)
        update_fields.pop("created_at", None)
        self.tasks.update_one(
            {"task_id": data["task_id"]},
            {"$set": update_fields, "$setOnInsert": {"created_at": created_at}},
            upsert=True,
        )

    def update_task_fields(self, task_id: str, **fields: Any) -> None:
        payload = {key: value for key, value in fields.items() if value is not None}
        if not payload:
            return
        payload["updated_at"] = self._utcnow()
        self.tasks.update_one({"task_id": task_id}, {"$set": payload}, upsert=True)

    def get_task(self, task_id: str) -> Optional[Dict[str, Any]]:
        doc = self.tasks.find_one({"task_id": task_id})
        return self._serialize(doc)

    def start_node(self, task_id: str, node_name: str, retry_count: int = 0) -> None:
        now = self._utcnow()
        self.node_logs.insert_one(
            {
                "task_id": task_id,
                "node_name": node_name,
                "status": "RUNNING",
                "retry_count": retry_count,
                "start_time": now,
                "end_time": None,
                "latency_ms": None,
                "error_message": "",
            }
        )
        self.update_task_fields(task_id, current_node=node_name)

    def finish_node(self, task_id: str, node_name: str, status: str = "SUCCESS", error_message: str = "") -> None:
        now = self._utcnow()
        running_log = self.node_logs.find_one(
            {"task_id": task_id, "node_name": node_name, "status": "RUNNING"},
            sort=[("start_time", DESCENDING)],
        )
        if running_log:
            start_time = running_log.get("start_time", now)
            latency_ms = int((now - start_time).total_seconds() * 1000)
            self.node_logs.update_one(
                {"_id": running_log["_id"]},
                {
                    "$set": {
                        "status": status,
                        "end_time": now,
                        "latency_ms": latency_ms,
                        "error_message": error_message,
                    }
                },
            )
        self.update_task_fields(task_id, current_node=node_name if status == "FAILED" else "")

    def increment_retry(self, task_id: str) -> int:
        self.tasks.update_one({"task_id": task_id}, {"$inc": {"retry_count": 1}, "$set": {"updated_at": self._utcnow()}})
        task = self.get_task(task_id) or {}
        return int(task.get("retry_count", 0))

    def list_node_logs(self, task_id: str) -> List[Dict[str, Any]]:
        cursor = self.node_logs.find({"task_id": task_id}).sort([("start_time", ASCENDING)])
        return [self._serialize(doc) for doc in cursor]

    @staticmethod
    def _serialize(document: Optional[Dict[str, Any]]) -> Optional[Dict[str, Any]]:
        if document is None:
            return None
        doc = dict(document)
        doc.pop("_id", None)
        return doc


_import_task_repository: Optional[ImportTaskRepository] = None


def get_import_task_repository() -> ImportTaskRepository:
    global _import_task_repository
    if _import_task_repository is None:
        _import_task_repository = ImportTaskRepository()
        logger.info("ImportTaskRepository 初始化成功")
    return _import_task_repository
