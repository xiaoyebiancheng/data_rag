from __future__ import annotations

import re
from dataclasses import asdict, dataclass, field
from typing import Any, Dict, List


@dataclass
class RetrievalConfig:
    dense_weight: float
    sparse_weight: float
    top_k: int
    use_hyde: bool
    use_rerank: bool
    rerank_top_n: int
    score_threshold: float
    metadata_filters: Dict[str, Any] = field(default_factory=dict)


@dataclass
class QueryProfile:
    query_type: str
    reason: str
    is_ambiguous: bool
    retrieval_config: RetrievalConfig

    def to_log_dict(self) -> Dict[str, Any]:
        return {
            "query_type": self.query_type,
            "reason": self.reason,
            "is_ambiguous": self.is_ambiguous,
            "retrieval_config": asdict(self.retrieval_config),
        }


DEFAULT_RETRIEVAL_CONFIG = RetrievalConfig(
    dense_weight=0.9,
    sparse_weight=0.1,
    top_k=5,
    use_hyde=True,
    use_rerank=True,
    rerank_top_n=10,
    score_threshold=0.6,
    metadata_filters={},
)

_EXACT_MODEL_PATTERNS = [
    r"\b[A-Za-z]{1,8}[- ]?\d{2,}[A-Za-z0-9-]*\b",
    r"故障码",
    r"错误码",
    r"参数",
    r"编号",
    r"型号",
    r"\d+\s?(W|V|A|Hz|mm|cm|kg|寸|英寸|mAh)\b",
]
_HOWTO_PATTERNS = [r"怎么", r"如何", r"步骤", r"使用", r"操作", r"设置", r"教程", r"说明"]
_TROUBLESHOOTING_PATTERNS = [r"故障", r"报错", r"异常", r"无法", r"不能", r"失效", r"排查", r"维修"]
_BROAD_PATTERNS = [r"是什么", r"介绍", r"原理", r"区别", r"作用", r"优缺点", r"说明一下"]
_AMBIGUOUS_PATTERNS = [r"这个", r"这个东西", r"这个产品", r"它", r"这款", r"那个", r"那个产品"]


def _matches_any(text: str, patterns: List[str]) -> bool:
    return any(re.search(pattern, text, re.IGNORECASE) for pattern in patterns)


def infer_query_type(original_query: str, rewritten_query: str, item_names: List[str]) -> tuple[str, str]:
    text = f"{original_query} {rewritten_query}".strip()
    if not item_names and _matches_any(text, _AMBIGUOUS_PATTERNS):
        return "ambiguous", "命中指代词且未识别到明确主体"
    if _matches_any(text, _TROUBLESHOOTING_PATTERNS):
        return "troubleshooting", "命中故障/异常/排查类关键词"
    if _matches_any(text, _EXACT_MODEL_PATTERNS):
        return "exact_model_or_param", "命中型号/参数/故障码类关键词"
    if _matches_any(text, _HOWTO_PATTERNS):
        return "semantic_howto", "命中操作/使用/步骤类关键词"
    if _matches_any(text, _BROAD_PATTERNS):
        return "broad_explanation", "命中宽泛解释类关键词"
    if not item_names and len(text) <= 10:
        return "ambiguous", "问题较短且主体不明确"
    return "broad_explanation", "未命中特殊规则，回退到宽泛解释类"


def build_retrieval_config(query_type: str) -> RetrievalConfig:
    if query_type == "exact_model_or_param":
        return RetrievalConfig(
            dense_weight=0.35,
            sparse_weight=0.65,
            top_k=4,
            use_hyde=False,
            use_rerank=True,
            rerank_top_n=6,
            score_threshold=0.72,
            metadata_filters={"section_type": ["parameter"], "doc_type": ["manual", "specification"]},
        )
    if query_type == "semantic_howto":
        return RetrievalConfig(
            dense_weight=0.85,
            sparse_weight=0.15,
            top_k=5,
            use_hyde=False,
            use_rerank=True,
            rerank_top_n=8,
            score_threshold=0.62,
            metadata_filters={"section_type": ["operation", "installation"], "doc_type": ["manual", "guide"]},
        )
    if query_type == "troubleshooting":
        return RetrievalConfig(
            dense_weight=0.7,
            sparse_weight=0.3,
            top_k=6,
            use_hyde=True,
            use_rerank=True,
            rerank_top_n=10,
            score_threshold=0.58,
            metadata_filters={"section_type": ["troubleshooting", "warning"], "doc_type": ["troubleshooting", "manual", "faq"]},
        )
    if query_type == "ambiguous":
        return RetrievalConfig(
            dense_weight=DEFAULT_RETRIEVAL_CONFIG.dense_weight,
            sparse_weight=DEFAULT_RETRIEVAL_CONFIG.sparse_weight,
            top_k=3,
            use_hyde=False,
            use_rerank=False,
            rerank_top_n=3,
            score_threshold=0.8,
            metadata_filters={},
        )
    return RetrievalConfig(
        dense_weight=0.9,
        sparse_weight=0.1,
        top_k=6,
        use_hyde=True,
        use_rerank=True,
        rerank_top_n=10,
        score_threshold=0.6,
        metadata_filters={"doc_type": ["manual", "faq", "guide"]},
    )


def build_query_profile(original_query: str, rewritten_query: str, item_names: List[str]) -> QueryProfile:
    query_type, reason = infer_query_type(original_query, rewritten_query, item_names)
    retrieval_config = build_retrieval_config(query_type)
    return QueryProfile(
        query_type=query_type,
        reason=reason,
        is_ambiguous=query_type == "ambiguous",
        retrieval_config=retrieval_config,
    )
