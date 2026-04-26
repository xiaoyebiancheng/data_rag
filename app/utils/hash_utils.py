import hashlib
from typing import Any, Dict


def calculate_file_sha256(file_path: str, chunk_size: int = 1024 * 1024) -> str:
    """
    计算文件内容的SHA256，作为文档内容唯一标识。
    """
    sha256 = hashlib.sha256()
    with open(file_path, "rb") as file_obj:
        while True:
            chunk = file_obj.read(chunk_size)
            if not chunk:
                break
            sha256.update(chunk)
    return sha256.hexdigest()


def calculate_chunk_sha256(chunk: Dict[str, Any]) -> str:
    """
    计算切片核心内容的SHA256，作为chunk去重与版本比对标识。
    """
    title = str(chunk.get("title", "")).strip()
    parent_title = str(chunk.get("parent_title", "")).strip()
    content = str(chunk.get("content", "")).strip()
    part = str(chunk.get("part", "")).strip()
    raw = "\n".join([title, parent_title, part, content])
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()
