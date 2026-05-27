import os
from datetime import datetime
from typing import Any, Dict, List, Optional

from dotenv import load_dotenv
from pymongo import ASCENDING, DESCENDING, MongoClient

from app.core.logger import logger

load_dotenv()


class DocumentStatus:
    ACTIVE = "ACTIVE"
    DELETED = "DELETED"
    REPLACED = "REPLACED"
    FAILED = "FAILED"
    DUPLICATED = "DUPLICATED"


class ChunkStatus:
    ACTIVE = "ACTIVE"
    DELETED = "DELETED"


class DocumentMetaRepository:
    def __init__(self):
        mongo_url = os.getenv("MONGO_URL")
        db_name = os.getenv("MONGO_DB_NAME")
        if not mongo_url or not db_name:
            raise RuntimeError("缺少MongoDB配置，请检查 MONGO_URL / MONGO_DB_NAME")

        self.client = MongoClient(mongo_url)
        self.db = self.client[db_name]
        self.document_meta = self.db["document_meta"]
        self.chunk_meta = self.db["chunk_meta"]
        self._ensure_indexes()

    def _ensure_indexes(self) -> None:
        # 增: 增的原因是文档元数据需要支撑按doc_id、file_hash、file_title、状态的高频查询和版本排序。
        self.document_meta.create_index([("doc_id", ASCENDING)], unique=True)
        self.document_meta.create_index([("file_hash", ASCENDING), ("status", ASCENDING)])
        self.document_meta.create_index([("file_title", ASCENDING), ("version", DESCENDING)])
        self.document_meta.create_index([("status", ASCENDING), ("updated_at", DESCENDING)])
        self.document_meta.create_index([("tenant_id", ASCENDING), ("visibility", ASCENDING), ("updated_at", DESCENDING)])
        self.document_meta.create_index([("doc_type", ASCENDING), ("product_line", ASCENDING)])

        # 增: 增的原因是chunk元数据需要支撑按doc_id、chunk_id、状态批量清理和列表查询。
        self.chunk_meta.create_index([("chunk_id", ASCENDING)], unique=True)
        self.chunk_meta.create_index([("doc_id", ASCENDING), ("status", ASCENDING)])
        self.chunk_meta.create_index([("file_hash", ASCENDING), ("status", ASCENDING)])
        self.chunk_meta.create_index([("tenant_id", ASCENDING), ("visibility", ASCENDING), ("status", ASCENDING)])
        self.chunk_meta.create_index([("doc_type", ASCENDING), ("section_type", ASCENDING), ("status", ASCENDING)])

    @staticmethod
    def _utcnow() -> datetime:
        return datetime.utcnow()

    def find_active_by_file_hash(self, file_hash: str) -> Optional[Dict[str, Any]]:
        return self.document_meta.find_one({
            "file_hash": file_hash,
            "status": DocumentStatus.ACTIVE,
        })

    def find_active_by_doc_id(self, doc_id: str, extra_filters: Optional[Dict[str, Any]] = None) -> Optional[Dict[str, Any]]:
        query = {
            "doc_id": doc_id,
            "status": {"$in": [DocumentStatus.ACTIVE, DocumentStatus.REPLACED, DocumentStatus.DELETED, DocumentStatus.FAILED]},
        }
        if extra_filters:
            query.update(extra_filters)
        return self.document_meta.find_one(query)

    def find_latest_by_file_title(self, file_title: str) -> Optional[Dict[str, Any]]:
        return self.document_meta.find_one(
            {"file_title": file_title},
            sort=[("version", DESCENDING), ("updated_at", DESCENDING)],
        )

    def list_documents(self, status: Optional[str] = None, extra_filters: Optional[Dict[str, Any]] = None) -> List[Dict[str, Any]]:
        query: Dict[str, Any] = {}
        if status:
            query["status"] = status
        if extra_filters:
            query.update(extra_filters)
        cursor = self.document_meta.find(query).sort([("updated_at", DESCENDING), ("version", DESCENDING)])
        return [self._serialize_document_meta(doc) for doc in cursor]

    def list_versions(self, doc_id: str, extra_filters: Optional[Dict[str, Any]] = None) -> List[Dict[str, Any]]:
        query = {"doc_id": doc_id}
        if extra_filters:
            query.update(extra_filters)
        current = self.document_meta.find_one(query)
        if not current:
            return []
        list_query = {"file_title": current["file_title"]}
        if extra_filters:
            list_query.update(extra_filters)
        cursor = self.document_meta.find(list_query).sort([("version", DESCENDING)])
        return [self._serialize_document_meta(doc) for doc in cursor]

    def list_chunks(self, doc_id: str, extra_filters: Optional[Dict[str, Any]] = None) -> List[Dict[str, Any]]:
        query = {"doc_id": doc_id}
        if extra_filters:
            query.update(extra_filters)
        cursor = self.chunk_meta.find(query).sort([("created_at", ASCENDING)])
        return [self._serialize_chunk_meta(doc) for doc in cursor]

    def upsert_document_meta(self, document: Dict[str, Any]) -> None:
        now = self._utcnow()
        doc = dict(document)
        doc.setdefault("created_at", now)
        doc["updated_at"] = now
        created_at = doc["created_at"]
        update_fields = dict(doc)
        update_fields.pop("created_at", None)
        self.document_meta.update_one(
            {"doc_id": doc["doc_id"]},
            {"$set": update_fields, "$setOnInsert": {"created_at": created_at}},
            upsert=True,
        )

    def create_failed_document_meta(self, document: Dict[str, Any]) -> None:
        failed_doc = dict(document)
        failed_doc["status"] = DocumentStatus.FAILED
        failed_doc.setdefault("chunk_count", 0)
        failed_doc.setdefault("minio_urls", [])
        self.upsert_document_meta(failed_doc)

    def mark_document_status(self, doc_id: str, status: str) -> None:
        self.document_meta.update_one(
            {"doc_id": doc_id},
            {"$set": {"status": status, "updated_at": self._utcnow()}},
        )

    def mark_latest_active_replaced(self, file_title: str, exclude_doc_id: Optional[str] = None) -> List[str]:
        query: Dict[str, Any] = {
            "file_title": file_title,
            "status": DocumentStatus.ACTIVE,
        }
        if exclude_doc_id:
            query["doc_id"] = {"$ne": exclude_doc_id}
        docs = list(self.document_meta.find(query, {"doc_id": 1}))
        if docs:
            self.document_meta.update_many(
                query,
                {"$set": {"status": DocumentStatus.REPLACED, "updated_at": self._utcnow()}},
            )
        return [doc["doc_id"] for doc in docs]

    def upsert_chunk_metas(self, chunks: List[Dict[str, Any]]) -> None:
        now = self._utcnow()
        for chunk in chunks:
            data = dict(chunk)
            data.setdefault("created_at", now)
            data["updated_at"] = now
            created_at = data["created_at"]
            update_fields = dict(data)
            update_fields.pop("created_at", None)
            self.chunk_meta.update_one(
                {"chunk_id": data["chunk_id"]},
                {"$set": update_fields, "$setOnInsert": {"created_at": created_at}},
                upsert=True,
            )

    def mark_chunks_deleted_by_doc_id(self, doc_id: str) -> int:
        result = self.chunk_meta.update_many(
            {"doc_id": doc_id, "status": {"$ne": ChunkStatus.DELETED}},
            {"$set": {"status": ChunkStatus.DELETED, "updated_at": self._utcnow()}},
        )
        return result.modified_count

    def mark_chunks_deleted_by_file_hash(self, file_hash: str) -> int:
        result = self.chunk_meta.update_many(
            {"file_hash": file_hash, "status": {"$ne": ChunkStatus.DELETED}},
            {"$set": {"status": ChunkStatus.DELETED, "updated_at": self._utcnow()}},
        )
        return result.modified_count

    def get_document_for_duplicate(self, file_hash: str) -> Optional[Dict[str, Any]]:
        doc = self.find_active_by_file_hash(file_hash)
        return self._serialize_document_meta(doc) if doc else None

    @staticmethod
    def _serialize_document_meta(document: Optional[Dict[str, Any]]) -> Optional[Dict[str, Any]]:
        if document is None:
            return None
        doc = dict(document)
        doc.pop("_id", None)
        return doc

    @staticmethod
    def _serialize_chunk_meta(document: Optional[Dict[str, Any]]) -> Optional[Dict[str, Any]]:
        if document is None:
            return None
        doc = dict(document)
        doc.pop("_id", None)
        return doc


_document_meta_repository: Optional[DocumentMetaRepository] = None


def get_document_meta_repository() -> DocumentMetaRepository:
    global _document_meta_repository
    if _document_meta_repository is None:
        _document_meta_repository = DocumentMetaRepository()
        logger.info("DocumentMetaRepository 初始化成功")
    return _document_meta_repository
