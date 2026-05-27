from __future__ import annotations

import re
from typing import Any, Dict, List, Tuple


DEFAULT_DOC_METADATA = {
    "doc_type": "other",
    "product_line": "",
    "language": "zh-CN",
    "source_priority": 50,
    "tenant_id": "default",
    "department_id": "default",
    "visibility": "tenant",
}

_DOC_TYPE_RULES = [
    ("faq", [r"faq", r"常见问题", r"问答"]),
    ("troubleshooting", [r"故障", r"排查", r"维修"]),
    ("specification", [r"规格", r"参数", r"spec"]),
    ("guide", [r"指南", r"guide", r"教程"]),
    ("manual", [r"手册", r"说明书", r"manual"]),
]

_SECTION_TYPE_RULES = [
    ("warning", [r"警告", r"注意事项", r"安全"]),
    ("parameter", [r"参数", r"规格", r"配置"]),
    ("installation", [r"安装", r"接线", r"装配"]),
    ("operation", [r"操作", r"使用", r"步骤", r"设置"]),
    ("troubleshooting", [r"故障", r"异常", r"排查", r"报错"]),
    ("overview", [r"简介", r"概述", r"介绍", r"overview"]),
]


def _match_type(text: str, rules: List[Tuple[str, List[str]]], default: str) -> str:
    lowered = text.lower()
    for target_type, patterns in rules:
        if any(re.search(pattern, lowered, re.IGNORECASE) for pattern in patterns):
            return target_type
    return default


def infer_doc_type(file_title: str, chunks: List[Dict[str, Any]]) -> str:
    combined = file_title + "\n" + "\n".join(f"{chunk.get('title', '')}\n{chunk.get('content', '')[:200]}" for chunk in chunks[:5])
    return _match_type(combined, _DOC_TYPE_RULES, "other")


def infer_section_type(title: str, content: str) -> str:
    text = f"{title}\n{content[:200]}"
    return _match_type(text, _SECTION_TYPE_RULES, "other")


def infer_product_line(file_title: str, item_name: str) -> str:
    source = item_name or file_title
    match = re.match(r"([A-Za-z\u4e00-\u9fa5]+)", source.strip())
    return match.group(1) if match else ""


def enrich_document_and_chunks(file_title: str, item_name: str, chunks: List[Dict[str, Any]]) -> Tuple[Dict[str, Any], List[Dict[str, Any]]]:
    doc_metadata = dict(DEFAULT_DOC_METADATA)
    doc_metadata["doc_type"] = infer_doc_type(file_title, chunks)
    doc_metadata["product_line"] = infer_product_line(file_title, item_name)

    enriched_chunks: List[Dict[str, Any]] = []
    for chunk in chunks:
        current = dict(chunk)
        current.setdefault("doc_type", doc_metadata["doc_type"])
        current.setdefault("product_line", doc_metadata["product_line"])
        current.setdefault("language", doc_metadata["language"])
        current.setdefault("source_priority", doc_metadata["source_priority"])
        current.setdefault("tenant_id", doc_metadata["tenant_id"])
        current.setdefault("department_id", doc_metadata["department_id"])
        current.setdefault("visibility", doc_metadata["visibility"])
        current["section_type"] = infer_section_type(current.get("title", ""), current.get("content", ""))
        enriched_chunks.append(current)
    return doc_metadata, enriched_chunks
