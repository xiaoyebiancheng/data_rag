import traceback
from typing import Any, Dict, List, Optional

from app.clients.import_task_repository import ImportTaskStatus, get_import_task_repository
from app.core.logger import logger
from .sse_utils import push_to_session

# ---------------------------
# 内存态任务追踪（单进程）
# ---------------------------
_tasks_running_list: Dict[str, List[str]] = {}
_tasks_done_list: Dict[str, List[str]] = {}
_tasks_status: Dict[str, str] = {}
_tasks_result: Dict[str, Dict[str, Any]] = {}

TASK_STATUS_PENDING = "pending"
TASK_STATUS_PROCESSING = "processing"
TASK_STATUS_COMPLETED = "completed"
TASK_STATUS_FAILED = "failed"

_IMPORT_NODE_NAMES = {
    "upload_file",
    "node_entry",
    "node_pdf_to_md",
    "node_md_img",
    "node_item_name_recognition",
    "node_document_split",
    "node_bge_embedding",
    "node_import_milvus",
}

_NODE_NAME_TO_CN: Dict[str, str] = {
    "upload_file": "开始上传文件",
    "node_entry": "检查文件",
    "node_pdf_to_md": "PDF转Markdown",
    "node_md_img": "Markdown图片处理",
    "node_item_name_recognition": "主体名称识别",
    "node_document_split": "文档切分",
    "node_bge_embedding": "向量生成",
    "node_import_kg": "导入知识图谱",
    "node_import_milvus": "导入向量库",
    "__end__": "处理完成",
    "END": "处理完成",
    "node_item_name_confirm": "确认问题产品",
    "node_answer_output": "生成答案",
    "node_rerank": "重排序",
    "node_rrf": "倒排融合",
    "node_web_search_mcp": "网络搜索",
    "node_search_embedding": "切片搜索",
    "node_search_embedding_hyde": "切片搜索(假设性文档)",
    "node_multi_search": "多路搜索",
    "node_query_kg": "查询知识图谱",
    "node_join": "多路搜索合并",
}


def _ensure_task(task_id: str) -> None:
    if task_id not in _tasks_running_list:
        _tasks_running_list[task_id] = []
    if task_id not in _tasks_done_list:
        _tasks_done_list[task_id] = []
    if task_id not in _tasks_result:
        _tasks_result[task_id] = {}


def _to_cn(node_name: str) -> str:
    return _NODE_NAME_TO_CN.get(node_name, node_name)


def _safe_import_repo():
    try:
        return get_import_task_repository()
    except Exception as exc:
        logger.warning(f"ImportTaskRepository 不可用，回退到内存态任务追踪: {exc}")
        return None


def _is_import_task(task_id: str, node_name: Optional[str] = None) -> bool:
    if node_name and node_name in _IMPORT_NODE_NAMES:
        return True
    repo = _safe_import_repo()
    if repo is None:
        return False
    return repo.get_task(task_id) is not None


def init_import_task(
    task_id: str,
    *,
    file_title: str,
    source_path: str,
    local_dir: str,
    status: str = ImportTaskStatus.PENDING,
    max_retry: int = 1,
) -> None:
    repo = _safe_import_repo()
    if repo is None:
        return
    repo.upsert_task(
        {
            "task_id": task_id,
            "doc_id": "",
            "file_hash": "",
            "file_title": file_title,
            "status": status,
            "current_node": "",
            "retry_count": 0,
            "max_retry": max_retry,
            "error_stack": "",
            "finished_at": None,
            "source_path": source_path,
            "local_dir": local_dir,
        }
    )


def update_import_task_fields(task_id: str, **fields: Any) -> None:
    repo = _safe_import_repo()
    if repo is None:
        return
    repo.update_task_fields(task_id, **fields)


def get_import_task(task_id: str) -> Optional[Dict[str, Any]]:
    repo = _safe_import_repo()
    if repo is None:
        return None
    return repo.get_task(task_id)


def list_import_task_node_logs(task_id: str) -> List[Dict[str, Any]]:
    repo = _safe_import_repo()
    if repo is None:
        return []
    return repo.list_node_logs(task_id)


def increment_import_task_retry(task_id: str) -> int:
    repo = _safe_import_repo()
    if repo is None:
        return 0
    return repo.increment_retry(task_id)


def add_running_task(task_id: str, node_name: str, is_stream: bool = False) -> None:
    _ensure_task(task_id)
    running = _tasks_running_list[task_id]
    if node_name not in running:
        running.append(node_name)
    if _is_import_task(task_id, node_name):
        repo = _safe_import_repo()
        if repo is not None:
            task = repo.get_task(task_id) or {}
            repo.start_node(task_id, node_name, retry_count=int(task.get("retry_count", 0)))
            repo.update_task_fields(task_id, current_node=node_name, status=ImportTaskStatus.RUNNING)
    if is_stream:
        task_push_queue(task_id)


def add_done_task(task_id: str, node_name: str, is_stream: bool = False) -> None:
    _ensure_task(task_id)
    running = _tasks_running_list[task_id]
    _tasks_running_list[task_id] = [n for n in running if n != node_name]
    done = _tasks_done_list[task_id]
    if node_name not in done:
        done.append(node_name)
    if _is_import_task(task_id, node_name):
        repo = _safe_import_repo()
        if repo is not None:
            repo.finish_node(task_id, node_name, status="SUCCESS")
    if is_stream:
        task_push_queue(task_id)


def fail_running_import_node(task_id: str, error_message: str) -> None:
    repo = _safe_import_repo()
    if repo is None:
        return
    running = _tasks_running_list.get(task_id, [])
    node_name = running[-1] if running else ""
    if node_name:
        repo.finish_node(task_id, node_name, status="FAILED", error_message=error_message)
        repo.update_task_fields(task_id, current_node=node_name)


def set_task_result(task_id: str, key: str, value: Any) -> None:
    _ensure_task(task_id)
    _tasks_result[task_id][key] = value
    if _is_import_task(task_id):
        mapped_fields = {}
        if key == "doc_id":
            mapped_fields["doc_id"] = value
        elif key == "document_status":
            mapped_fields["status"] = value
        elif key == "error":
            mapped_fields["error_stack"] = value
        elif key == "file_hash":
            mapped_fields["file_hash"] = value
        elif key == "file_title":
            mapped_fields["file_title"] = value
        if mapped_fields:
            update_import_task_fields(task_id, **mapped_fields)


def get_task_result(task_id: str, key: str, default: Any = "") -> Any:
    _ensure_task(task_id)
    return _tasks_result.get(task_id, {}).get(key, default)


def get_task_status(task_id: str) -> str:
    return _tasks_status.get(task_id, "")


def get_done_task_list(task_id: str) -> List[str]:
    _ensure_task(task_id)
    return [_to_cn(n) for n in _tasks_done_list.get(task_id, [])]


def get_running_task_list(task_id: str) -> List[str]:
    _ensure_task(task_id)
    return [_to_cn(n) for n in _tasks_running_list.get(task_id, [])]


def update_task_status(task_id: str, status_name: str, push_queue: bool = False) -> None:
    _tasks_status[task_id] = status_name
    if _is_import_task(task_id):
        import_status = {
            TASK_STATUS_PENDING: ImportTaskStatus.PENDING,
            TASK_STATUS_PROCESSING: ImportTaskStatus.RUNNING,
            TASK_STATUS_COMPLETED: ImportTaskStatus.SUCCESS,
            TASK_STATUS_FAILED: ImportTaskStatus.FAILED,
        }.get(status_name, status_name.upper())
        update_import_task_fields(task_id, status=import_status, finished_at=None if status_name == TASK_STATUS_PROCESSING else None)
    if push_queue:
        task_push_queue(task_id)


def mark_import_task_finished(task_id: str, status: str, error_stack: str = "") -> None:
    repo = _safe_import_repo()
    if repo is None:
        return
    repo.update_task_fields(
        task_id,
        status=status,
        error_stack=error_stack,
        finished_at=repo._utcnow(),
    )


def record_import_task_failure(task_id: str, exc: Exception) -> None:
    error_stack = traceback.format_exc()
    fail_running_import_node(task_id, str(exc))
    update_import_task_fields(task_id, status=ImportTaskStatus.FAILED, error_stack=error_stack)


def task_push_queue(task_id: str):
    push_to_session(task_id, "progress", {
        "status": get_task_status(task_id),
        "done_list": get_done_task_list(task_id),
        "running_list": get_running_task_list(task_id),
    })


def clear_task(task_id: str):
    _tasks_running_list.pop(task_id, None)
    _tasks_done_list.pop(task_id, None)
    _tasks_status.pop(task_id, None)
    _tasks_result.pop(task_id, None)
