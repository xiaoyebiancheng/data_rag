from __future__ import annotations

import re
from typing import Any, Dict, List, Tuple

from pydantic import BaseModel, Field


class AnswerSource(BaseModel):
    doc_id: str = ""
    chunk_id: str = ""
    file_title: str = ""
    title: str = ""
    parent_title: str = ""
    item_name: str = ""
    version: int = 0
    score: float = 0.0
    rerank_score: float = 0.0
    evidence_text: str = ""
    image_urls: List[str] = Field(default_factory=list)
    image_summary_warnings: List[str] = Field(default_factory=list)


class TrustedAnswer(BaseModel):
    answer: str = ""
    sources: List[AnswerSource] = Field(default_factory=list)
    confidence: float = 0.0
    need_clarification: bool = False
    refusal_reason: str = ""
    unsupported_claims: List[str] = Field(default_factory=list)
    retrieval_trace_id: str = ""


_STOPWORDS = {"请问", "这个", "那个", "什么", "如何", "怎么", "一下", "可以", "是否", "需要", "进行", "产品", "设备"}
_IMAGE_MARKDOWN_RE = re.compile(r"!\[(.*?)\]\((.*?)\)")


def extract_image_evidence(text: str) -> Tuple[List[str], List[str]]:
    image_urls: List[str] = []
    warnings: List[str] = []
    if not text:
        return image_urls, warnings
    for alt_text, url in _IMAGE_MARKDOWN_RE.findall(text):
        if url and url not in image_urls:
            image_urls.append(url)
        if "低置信图片摘要" in alt_text or "图片摘要失败" in alt_text:
            warnings.append(alt_text)
    if "image_summary_quality" in text and not warnings:
        warnings.append("该证据包含低置信或失败的图片摘要，请结合原图核验")
    return image_urls, warnings


def build_sources_from_reranked_docs(reranked_docs: List[Dict[str, Any]], top_n: int = 5) -> List[AnswerSource]:
    sources: List[AnswerSource] = []
    for doc in reranked_docs:
        if doc.get("source") != "local":
            continue
        chunk_id = doc.get("chunk_id")
        if chunk_id in (None, ""):
            continue
        evidence_text = doc.get("text", "") or ""
        image_urls, image_summary_warnings = extract_image_evidence(evidence_text)
        sources.append(
            AnswerSource(
                doc_id=str(doc.get("doc_id", "") or ""),
                chunk_id=str(chunk_id),
                file_title=doc.get("file_title", "") or "",
                title=doc.get("title", "") or "",
                parent_title=doc.get("parent_title", "") or "",
                item_name=doc.get("item_name", "") or "",
                version=int(doc.get("version", 0) or 0),
                score=float(doc.get("score", 0.0) or 0.0),
                rerank_score=float(doc.get("rerank_score", doc.get("score", 0.0)) or 0.0),
                evidence_text=evidence_text[:1200],
                image_urls=image_urls,
                image_summary_warnings=image_summary_warnings,
            )
        )
        if len(sources) >= top_n:
            break
    return sources


def assess_answer_gate(
    state: Dict[str, Any],
    sources: List[AnswerSource],
    score_threshold: float,
) -> Tuple[float, bool, str]:
    if state.get("answer") and ("没有匹配" in state.get("answer", "") or "请重新提问" in state.get("answer", "")):
        return 0.1, True, "item_name 不明确，当前无法确认检索主体"
    if state.get("answer") and "您是想咨询以下哪个商品" in state.get("answer", ""):
        return 0.2, True, "item_name 存在多个候选，需要用户澄清"
    if not sources:
        return 0.1, True, "没有检索到可作为证据的本地切片"

    top1 = sources[0].rerank_score if sources else 0.0
    confidence = min(0.99, max(0.0, top1))
    if top1 < score_threshold:
        return confidence, False, f"top1 rerank 分数低于阈值 {score_threshold}"

    if len(sources) >= 2:
        top2 = sources[1].rerank_score
        if abs(top1 - top2) <= 0.03 and sources[0].item_name and sources[1].item_name and sources[0].item_name != sources[1].item_name:
            return max(0.2, confidence - 0.25), True, "Top结果主体冲突明显，建议先澄清问题主体"

    return confidence, False, ""


def _extract_keywords(sentence: str) -> List[str]:
    words = re.findall(r"[\u4e00-\u9fa5A-Za-z0-9]{2,}", sentence)
    return [word for word in words if word not in _STOPWORDS]


def verify_answer_support(answer: str, sources: List[AnswerSource]) -> List[str]:
    if not answer or not sources:
        return [answer] if answer else []
    evidence_text = "\n".join(source.evidence_text for source in sources)
    unsupported_claims: List[str] = []
    sentences = [sentence.strip() for sentence in re.split(r"[。！？\n]", answer) if sentence.strip()]
    for sentence in sentences:
        keywords = _extract_keywords(sentence)
        if not keywords:
            continue
        hit_count = sum(1 for keyword in keywords if keyword in evidence_text)
        if hit_count == 0:
            unsupported_claims.append(sentence)
    return unsupported_claims


def build_refusal_text(refusal_reason: str, need_clarification: bool) -> str:
    if need_clarification:
        return f"当前信息还不足以给出可靠答案。{refusal_reason}。请补充更明确的产品名称、参数或问题场景。"
    return f"当前检索到的证据不足以支持可靠回答。{refusal_reason}。"
