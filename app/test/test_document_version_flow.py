import json
import shutil
import unittest
import ast
import time
from pathlib import Path
from typing import Any, Dict, List

from app.clients.document_meta_repository import get_document_meta_repository
from app.clients.milvus_utils import get_milvus_client, delete_by_doc_id, delete_by_file_title
from app.conf.milvus_config import milvus_config
from app.import_process.agent.main_graph import kb_import_app
from app.import_process.agent.state import get_default_state
from app.utils.escape_milvus_string_utils import escape_milvus_string


class DocumentVersionFlowIntegrationTest(unittest.TestCase):
    FILE_TITLE = "version_case"

    @classmethod
    def setUpClass(cls):
        cls.repo = get_document_meta_repository()
        cls.milvus = get_milvus_client()
        cls.base_dir = Path("/Users/liwenye/PycharmProjects/study_agent/dataset_rag/tmp/version_regression")
        cls.v1_dir = cls.base_dir / "v1"
        cls.v2_dir = cls.base_dir / "v2"
        cls._prepare_test_files()

    @classmethod
    def tearDownClass(cls):
        if cls.base_dir.exists():
            shutil.rmtree(cls.base_dir, ignore_errors=True)
        if getattr(cls.repo, "client", None) is not None:
            cls.repo.client.close()

    @classmethod
    def _prepare_test_files(cls) -> None:
        cls.v1_dir.mkdir(parents=True, exist_ok=True)
        cls.v2_dir.mkdir(parents=True, exist_ok=True)
        (cls.v1_dir / "images").mkdir(exist_ok=True)
        (cls.v2_dir / "images").mkdir(exist_ok=True)

        # 增: 增的原因是版本管理回归测试需要稳定可控的同名不同内容样本，避免依赖现有业务文档产生偶发差异。
        (cls.v1_dir / f"{cls.FILE_TITLE}.md").write_text(
            "# 版本联调样例\n\n"
            "## 产品说明\n"
            "版本联调样例设备的额定功率为 100W。\n\n"
            "## 注意事项\n"
            "请在断电状态下维护。\n",
            encoding="utf-8",
        )
        (cls.v2_dir / f"{cls.FILE_TITLE}.md").write_text(
            "# 版本联调样例\n\n"
            "## 产品说明\n"
            "版本联调样例设备的额定功率为 200W。\n\n"
            "## 注意事项\n"
            "请在断电状态下维护，并佩戴绝缘手套。\n",
            encoding="utf-8",
        )

    def setUp(self):
        self._cleanup_existing()

    def _cleanup_existing(self) -> None:
        docs = self.repo.list_documents()
        target_docs = [doc for doc in docs if doc.get("file_title") == self.FILE_TITLE]
        for doc in target_docs:
            doc_id = doc["doc_id"]
            delete_by_doc_id(self.milvus, milvus_config.chunks_collection, doc_id)
            delete_by_doc_id(self.milvus, milvus_config.item_name_collection, doc_id)
            self.repo.mark_chunks_deleted_by_doc_id(doc_id)
            self.repo.mark_document_status(doc_id, "DELETED")
        delete_by_file_title(self.milvus, milvus_config.chunks_collection, self.FILE_TITLE)
        delete_by_file_title(self.milvus, milvus_config.item_name_collection, self.FILE_TITLE)
        # 优化: 优化的原因是回归脚本需要保证每次从version=1开始验证，因此测试数据必须彻底清空，不能只做软删除。
        self.repo.document_meta.delete_many({"file_title": self.FILE_TITLE})
        self.repo.chunk_meta.delete_many({"title": {"$exists": True}, "doc_id": {"$in": [doc["doc_id"] for doc in target_docs]}})

    def _run_import(self, task_id: str, file_path: Path, local_dir: Path) -> Dict[str, Any]:
        state = get_default_state()
        state["task_id"] = task_id
        state["local_file_path"] = str(file_path)
        state["local_dir"] = str(local_dir)
        final_state = dict(state)
        for event in kb_import_app.stream(state, stream_mode="updates"):
            for _, result in event.items():
                if isinstance(result, dict):
                    final_state.update(result)
        return final_state

    @staticmethod
    def _rows_len(rows: Any) -> int:
        if rows is None:
            return 0
        if isinstance(rows, list):
            return len(rows)
        data = getattr(rows, "data", None)
        if isinstance(data, list):
            return len(data)
        text = str(rows)
        if text.startswith("data: "):
            try:
                payload = text[len("data: "):].split(", extra_info:", 1)[0]
                parsed = ast.literal_eval(payload)
                if isinstance(parsed, list):
                    return len(parsed)
            except Exception:
                pass
        try:
            return len(list(rows))
        except Exception:
            return 0

    def _count_chunks_by_doc_id(self, doc_id: str) -> int:
        rows = self.milvus.query(
            collection_name=milvus_config.chunks_collection,
            filter=f'doc_id=="{escape_milvus_string(doc_id)}"',
            output_fields=["chunk_id"],
        )
        return self._rows_len(rows)

    def _count_item_names_by_doc_id(self, doc_id: str) -> int:
        rows = self.milvus.query(
            collection_name=milvus_config.item_name_collection,
            filter=f'doc_id=="{escape_milvus_string(doc_id)}"',
            output_fields=["pk"],
        )
        return self._rows_len(rows)

    def _delete_doc(self, doc_id: str) -> None:
        delete_by_doc_id(self.milvus, milvus_config.chunks_collection, doc_id)
        delete_by_doc_id(self.milvus, milvus_config.item_name_collection, doc_id)
        self.repo.mark_document_status(doc_id, "DELETED")
        self.repo.mark_chunks_deleted_by_doc_id(doc_id)

    # 优化: 优化的原因是Milvus删除后同进程短时间内可能存在可见性延迟，回归脚本需要轮询确认最终状态，避免把一致性抖动误判为业务失败。
    def _wait_until_count(self, counter, target: int, *, retries: int = 10, sleep_seconds: float = 0.3) -> int:
        last_value = counter()
        for _ in range(retries):
            if last_value == target:
                return last_value
            time.sleep(sleep_seconds)
            last_value = counter()
        return last_value

    def test_document_version_flow(self):
        v1_result = self._run_import(
            task_id="version_regression_v1",
            file_path=self.v1_dir / f"{self.FILE_TITLE}.md",
            local_dir=self.v1_dir,
        )
        v1_doc_id = v1_result["doc_id"]
        v1_meta = self.repo.find_active_by_doc_id(v1_doc_id)

        self.assertEqual(v1_result["document_status"], "ACTIVE")
        self.assertEqual(v1_result["version"], 1)
        self.assertEqual(len(v1_result.get("chunks", [])), 3)
        self.assertIsNotNone(v1_meta)
        self.assertEqual(v1_meta["status"], "ACTIVE")

        duplicated_result = self._run_import(
            task_id="version_regression_dup",
            file_path=self.v1_dir / f"{self.FILE_TITLE}.md",
            local_dir=self.v1_dir,
        )
        self.assertEqual(duplicated_result["document_status"], "DUPLICATED")
        self.assertEqual(duplicated_result["doc_id"], v1_doc_id)
        self.assertEqual(duplicated_result["version"], 1)

        v2_result = self._run_import(
            task_id="version_regression_v2",
            file_path=self.v2_dir / f"{self.FILE_TITLE}.md",
            local_dir=self.v2_dir,
        )
        v2_doc_id = v2_result["doc_id"]
        v2_meta = self.repo.find_active_by_doc_id(v2_doc_id)
        versions = self.repo.list_versions(v2_doc_id)
        version_status_map = {item["version"]: item["status"] for item in versions}

        self.assertEqual(v2_result["document_status"], "ACTIVE")
        self.assertEqual(v2_result["version"], 2)
        self.assertEqual(v2_result["old_doc_id"], v1_doc_id)
        self.assertIsNotNone(v2_meta)
        self.assertEqual(v2_meta["status"], "ACTIVE")
        self.assertEqual(version_status_map[1], "REPLACED")
        self.assertEqual(version_status_map[2], "ACTIVE")
        self.assertEqual(self._count_chunks_by_doc_id(v1_doc_id), 0)
        self.assertEqual(self._count_chunks_by_doc_id(v2_doc_id), 3)

        self._delete_doc(v2_doc_id)
        deleted_meta = self.repo.find_active_by_doc_id(v2_doc_id)
        deleted_chunk_meta = self.repo.list_chunks(v2_doc_id)
        chunk_count_after_delete = self._wait_until_count(lambda: self._count_chunks_by_doc_id(v2_doc_id), 0)
        item_name_count_after_delete = self._wait_until_count(lambda: self._count_item_names_by_doc_id(v2_doc_id), 0)

        self.assertIsNotNone(deleted_meta)
        self.assertEqual(deleted_meta["status"], "DELETED")
        self.assertEqual(chunk_count_after_delete, 0)
        self.assertEqual(item_name_count_after_delete, 0)
        self.assertEqual(sorted({item["status"] for item in deleted_chunk_meta}), ["DELETED"])

        print(json.dumps({
            "v1_doc_id": v1_doc_id,
            "v2_doc_id": v2_doc_id,
            "versions": [
                {
                    "doc_id": item["doc_id"],
                    "version": item["version"],
                    "status": item["status"],
                    "chunk_count": item.get("chunk_count"),
                }
                for item in versions
            ],
            "delete_check": {
                "chunk_count_after_delete": chunk_count_after_delete,
                "item_name_count_after_delete": item_name_count_after_delete,
            }
        }, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    unittest.main()
