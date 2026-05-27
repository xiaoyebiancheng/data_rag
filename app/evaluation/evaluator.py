from __future__ import annotations

import json
import time
from dataclasses import dataclass, field
from typing import Any, Dict, Iterable, List, Optional, Sequence

from pymilvus import AnnSearchRequest

from app.clients.milvus_utils import (
    create_hybrid_search_requests,
    get_milvus_client,
    hybrid_search,
)
from app.conf.milvus_config import milvus_config
from app.core.load_prompt import load_prompt
from app.core.logger import logger
from app.evaluation.dataset_schema import EvalSample
from app.evaluation.metrics import average, hit_at_k, mean_reciprocal_rank, ndcg_at_k, percentile, recall_at_k
from app.lm.embedding_utils import generate_embeddings
from app.lm.lm_utils import get_llm_client
from app.lm.reranker_utils import get_reranker_model
from app.prompts.prompt_registry import get_prompt_definition
from app.query_process.agent.nodes.node_rrf import step_3_reciprocal_rank_fusion
from app.query_process.agent.nodes.node_search_embedding_hyde import step_1_create_hyde_doc


DEFAULT_STRATEGIES = [
    "dense_only",
    "sparse_only",
    "hybrid",
    "hybrid_rrf",
    "hybrid_rrf_rerank",
    "hyde_hybrid_rrf_rerank",
]


@dataclass
class JudgeScore:
    score: Optional[float]
    reason: str
    error: str = ""


@dataclass
class SampleEvalResult:
    sample: EvalSample
    strategy: str
    retrieved_docs: List[Dict[str, Any]] = field(default_factory=list)
    answer: str = ""
    hit_at_k: Optional[float] = None
    recall_at_k: Optional[float] = None
    mrr: Optional[float] = None
    ndcg_at_k: Optional[float] = None
    latency_ms: float = 0.0
    answer_relevance: Optional[float] = None
    answer_relevance_reason: str = ""
    faithfulness: Optional[float] = None
    faithfulness_reason: str = ""
    prompt_versions: Dict[str, Dict[str, str]] = field(default_factory=dict)
    error: str = ""


@dataclass
class StrategySummary:
    strategy: str
    sample_results: List[SampleEvalResult]
    top_k: int

    def overall_metrics(self) -> Dict[str, Optional[float]]:
        latencies = [result.latency_ms for result in self.sample_results]
        return {
            "Hit@K": average(result.hit_at_k for result in self.sample_results),
            "Recall@K": average(result.recall_at_k for result in self.sample_results),
            "MRR": average(result.mrr for result in self.sample_results),
            "NDCG@K": average(result.ndcg_at_k for result in self.sample_results),
            "Faithfulness": average(result.faithfulness for result in self.sample_results),
            "Answer Relevance": average(result.answer_relevance for result in self.sample_results),
            "Avg Latency": average(latencies),
            "P50 Latency": percentile(latencies, 50),
            "P95 Latency": percentile(latencies, 95),
        }

    def metrics_by_category(self) -> Dict[str, Dict[str, Optional[float]]]:
        grouped: Dict[str, List[SampleEvalResult]] = {}
        for result in self.sample_results:
            grouped.setdefault(result.sample.category, []).append(result)

        category_metrics: Dict[str, Dict[str, Optional[float]]] = {}
        for category, results in grouped.items():
            latencies = [result.latency_ms for result in results]
            category_metrics[category] = {
                "Hit@K": average(result.hit_at_k for result in results),
                "Recall@K": average(result.recall_at_k for result in results),
                "MRR": average(result.mrr for result in results),
                "NDCG@K": average(result.ndcg_at_k for result in results),
                "Faithfulness": average(result.faithfulness for result in results),
                "Answer Relevance": average(result.answer_relevance for result in results),
                "Avg Latency": average(latencies),
                "P95 Latency": percentile(latencies, 95),
            }
        return category_metrics


class RAGEvaluator:
    def __init__(self, top_k: int = 5, prompt_versions: Optional[Dict[str, str]] = None):
        # 增: 增的原因是离线评测需要一个独立入口对象，统一组织“检索 -> 生成 -> Judge -> 聚合”全流程，避免污染线上查询链路。
        self.top_k = top_k
        self.milvus_client = get_milvus_client()
        self.output_fields = ["chunk_id", "content", "file_title", "title", "parent_title", "item_name"]
        # 增: 增的原因是评测需要在不改线上默认版本的前提下，显式指定参与评测的 Prompt 版本，方便做 Prompt 回归对比。
        self.prompt_versions = dict(prompt_versions or {})

    def evaluate(self, samples: Sequence[EvalSample], strategies: Sequence[str]) -> List[StrategySummary]:
        summaries: List[StrategySummary] = []
        for strategy in strategies:
            logger.info(f"[Evaluation] 开始执行策略评测: {strategy}")
            sample_results: List[SampleEvalResult] = []
            for sample in samples:
                sample_results.append(self._evaluate_single_sample(sample, strategy))
            summaries.append(StrategySummary(strategy=strategy, sample_results=sample_results, top_k=self.top_k))
        return summaries

    def _evaluate_single_sample(self, sample: EvalSample, strategy: str) -> SampleEvalResult:
        result = SampleEvalResult(sample=sample, strategy=strategy)
        result.prompt_versions = self._build_prompt_version_snapshot()
        start_time = time.perf_counter()
        try:
            docs = self._run_strategy(sample, strategy)
            result.retrieved_docs = docs
        except Exception as exc:
            logger.exception(f"[Evaluation] 策略 {strategy} 检索失败: {exc}")
            result.error = str(exc)
            result.retrieved_docs = []
        finally:
            result.latency_ms = (time.perf_counter() - start_time) * 1000

        retrieved_chunk_ids = [self._extract_chunk_id(doc) for doc in result.retrieved_docs if self._extract_chunk_id(doc)]
        result.hit_at_k = hit_at_k(retrieved_chunk_ids, sample.golden_chunk_ids, self.top_k)
        result.recall_at_k = recall_at_k(retrieved_chunk_ids, sample.golden_chunk_ids, self.top_k)
        result.mrr = mean_reciprocal_rank(retrieved_chunk_ids, sample.golden_chunk_ids)
        result.ndcg_at_k = ndcg_at_k(retrieved_chunk_ids, sample.golden_chunk_ids, self.top_k)

        result.answer = self._generate_answer(sample, result.retrieved_docs)

        relevance = self._judge_answer_relevance(sample, result.answer)
        result.answer_relevance = relevance.score
        result.answer_relevance_reason = relevance.reason or relevance.error

        faithfulness = self._judge_faithfulness(sample, result.answer, result.retrieved_docs)
        result.faithfulness = faithfulness.score
        result.faithfulness_reason = faithfulness.reason or faithfulness.error
        return result

    def _run_strategy(self, sample: EvalSample, strategy: str) -> List[Dict[str, Any]]:
        if strategy == "dense_only":
            return self._search_dense(sample.question, sample.item_names, self.top_k)
        if strategy == "sparse_only":
            return self._search_sparse(sample.question, sample.item_names, self.top_k)
        if strategy == "hybrid":
            return self._search_hybrid(sample.question, sample.item_names, self.top_k)
        if strategy == "hybrid_rrf":
            return self._search_hybrid_rrf(sample.question, sample.item_names, self.top_k)
        if strategy == "hybrid_rrf_rerank":
            return self._search_hybrid_rrf_rerank(sample.question, sample.item_names, self.top_k)
        if strategy == "hyde_hybrid_rrf_rerank":
            return self._search_hyde_hybrid_rrf_rerank(sample.question, sample.item_names, self.top_k)
        raise ValueError(f"不支持的评测策略: {strategy}")

    def _search_dense(self, question: str, item_names: Sequence[str], top_k: int) -> List[Dict[str, Any]]:
        embedding = generate_embeddings([question])
        # 增: 增的原因是离线评测需要拆分 dense / sparse 单路检索，证明混合检索是否真的带来收益。
        response = self.milvus_client.search(
            collection_name=milvus_config.chunks_collection,
            data=[embedding["dense"][0]],
            anns_field="dense_vector",
            filter=self._build_item_filter(item_names),
            limit=top_k,
            output_fields=self.output_fields,
            search_params={"metric_type": "COSINE"},
        )
        return self._normalize_milvus_hits(response[0] if response else [])

    def _search_sparse(self, question: str, item_names: Sequence[str], top_k: int) -> List[Dict[str, Any]]:
        embedding = generate_embeddings([question])
        response = self.milvus_client.search(
            collection_name=milvus_config.chunks_collection,
            data=[embedding["sparse"][0]],
            anns_field="sparse_vector",
            filter=self._build_item_filter(item_names),
            limit=top_k,
            output_fields=self.output_fields,
            search_params={"metric_type": "IP"},
        )
        return self._normalize_milvus_hits(response[0] if response else [])

    def _search_hybrid(self, question: str, item_names: Sequence[str], top_k: int) -> List[Dict[str, Any]]:
        embedding = generate_embeddings([question])
        requests = create_hybrid_search_requests(
            dense_vector=embedding["dense"][0],
            sparse_vector=embedding["sparse"][0],
            expr=self._build_item_filter(item_names),
            limit=top_k,
        )
        response = hybrid_search(
            client=self.milvus_client,
            collection_name=milvus_config.chunks_collection,
            reqs=requests,
            ranker_weights=(0.9, 0.1),
            norm_score=True,
            limit=top_k,
            output_fields=self.output_fields,
        )
        return self._normalize_milvus_hits(response[0] if response else [])

    def _search_hybrid_rrf(self, question: str, item_names: Sequence[str], top_k: int) -> List[Dict[str, Any]]:
        dense_docs = self._search_dense(question, item_names, top_k)
        sparse_docs = self._search_sparse(question, item_names, top_k)
        hybrid_docs = self._search_hybrid(question, item_names, top_k)
        # 增: 增的原因是当前项目线上主链路没有独立的 dense/sparse 多路对比模式，这里补一个评测适配层，专门用于做策略效果对比。
        fused = step_3_reciprocal_rank_fusion(
            [
                (dense_docs, 1.0),
                (sparse_docs, 1.0),
                (hybrid_docs, 1.0),
            ],
            top_k=top_k,
        )
        return self._normalize_rrf_docs(fused)

    def _search_hybrid_rrf_rerank(self, question: str, item_names: Sequence[str], top_k: int) -> List[Dict[str, Any]]:
        fused_docs = self._search_hybrid_rrf(question, item_names, max(top_k * 2, top_k))
        return self._rerank_docs(question, fused_docs, top_k)

    def _search_hyde_hybrid_rrf_rerank(self, question: str, item_names: Sequence[str], top_k: int) -> List[Dict[str, Any]]:
        hyde_doc = step_1_create_hyde_doc(question)
        hyde_hybrid_docs = self._search_hybrid(f"{question}\n{hyde_doc}", item_names, max(top_k * 2, top_k))
        dense_docs = self._search_dense(question, item_names, top_k)
        sparse_docs = self._search_sparse(question, item_names, top_k)
        hybrid_docs = self._search_hybrid(question, item_names, top_k)
        fused = step_3_reciprocal_rank_fusion(
            [
                (dense_docs, 1.0),
                (sparse_docs, 1.0),
                (hybrid_docs, 1.0),
                (hyde_hybrid_docs, 1.0),
            ],
            top_k=max(top_k * 2, top_k),
        )
        normalized_docs = self._normalize_rrf_docs(fused)
        return self._rerank_docs(question, normalized_docs, top_k)

    def _rerank_docs(self, question: str, docs: Sequence[Dict[str, Any]], top_k: int) -> List[Dict[str, Any]]:
        if not docs:
            return []
        reranker = get_reranker_model()
        pairs = [[question, doc.get("text", "")] for doc in docs]
        scores = reranker.compute_score(pairs, normalize=True)

        reranked_docs: List[Dict[str, Any]] = []
        for score, doc in zip(scores, docs):
            current = dict(doc)
            # 优化: 优化的原因是离线评测需要保留统一 score 字段，方便策略间做排序质量与 bad case 对比。
            current["score"] = float(score)
            reranked_docs.append(current)
        reranked_docs.sort(key=lambda item: item.get("score", 0.0), reverse=True)
        return reranked_docs[:top_k]

    def _generate_answer(self, sample: EvalSample, docs: Sequence[Dict[str, Any]]) -> str:
        prompt = self._build_answer_prompt(sample, docs)
        try:
            llm = get_llm_client()
            response = llm.invoke(prompt)
            return str(response.content).strip()
        except Exception as exc:
            logger.exception(f"[Evaluation] 生成答案失败: {exc}")
            return f"[answer_generation_failed] {exc}"

    def _judge_answer_relevance(self, sample: EvalSample, answer: str) -> JudgeScore:
        prompt_name = "eval_answer_relevance_judge"
        prompt = load_prompt(
            prompt_name,
            version=self.prompt_versions.get(prompt_name),
            question=sample.question,
            answer=answer,
            golden_answer=sample.golden_answer or "无参考答案",
        )
        return self._invoke_judge(prompt)

    def _judge_faithfulness(self, sample: EvalSample, answer: str, docs: Sequence[Dict[str, Any]]) -> JudgeScore:
        context = "\n\n".join(
            f"[{index}] {doc.get('title', '')}\n{doc.get('text', '')}"
            for index, doc in enumerate(docs, start=1)
        ) or "无检索上下文"
        prompt_name = "eval_faithfulness_judge"
        prompt = load_prompt(
            prompt_name,
            version=self.prompt_versions.get(prompt_name),
            question=sample.question,
            answer=answer,
            context=context,
        )
        return self._invoke_judge(prompt)

    def _invoke_judge(self, prompt: str) -> JudgeScore:
        try:
            judge = get_llm_client(json_mode=True)
            response = judge.invoke(prompt)
            content = str(response.content).strip()
            if content.startswith("```json"):
                content = content.replace("```json", "", 1).replace("```", "").strip()
            payload = json.loads(content)
            score = payload.get("score")
            reason = str(payload.get("reason", "")).strip()
            if score is None:
                return JudgeScore(score=None, reason=reason, error="judge_missing_score")
            return JudgeScore(score=float(score), reason=reason)
        except Exception as exc:
            # 优化: 优化的原因是评测中 Judge 偶发失败不能中断整批任务，否则离线评测无法稳定产出完整报告。
            logger.exception(f"[Evaluation] LLM Judge 调用失败: {exc}")
            return JudgeScore(score=None, reason="", error=str(exc))

    def _build_answer_prompt(self, sample: EvalSample, docs: Sequence[Dict[str, Any]]) -> str:
        context_blocks: List[str] = []
        total_chars = 0
        for index, doc in enumerate(docs, start=1):
            block = (
                f"[{index}][source={doc.get('source', 'local')}][title={doc.get('title', '')}]"
                f"[score={doc.get('score', 0.0)}]\n\n[text={doc.get('text', '')}]"
            )
            if total_chars + len(block) > 12000:
                break
            context_blocks.append(block)
            total_chars += len(block)
        prompt_name = "answer_out"
        return load_prompt(
            prompt_name,
            version=self.prompt_versions.get(prompt_name),
            context="\n\n".join(context_blocks) or "无可用上下文",
            history="没有历史对话记录",
            item_names=",".join(sample.item_names),
            question=sample.question,
        )

    def _build_prompt_version_snapshot(self) -> Dict[str, Dict[str, str]]:
        snapshot: Dict[str, Dict[str, str]] = {}
        for prompt_name in ["answer_out", "eval_answer_relevance_judge", "eval_faithfulness_judge"]:
            definition = get_prompt_definition(prompt_name, version=self.prompt_versions.get(prompt_name))
            snapshot[prompt_name] = {
                "prompt_name": definition.prompt_name,
                "prompt_version": definition.version,
            }
        return snapshot

    def _build_item_filter(self, item_names: Sequence[str]) -> str:
        cleaned = [name.replace('"', '\\"') for name in item_names if name]
        if not cleaned:
            return ""
        item_name_expr = ", ".join(f'"{name}"' for name in cleaned)
        return f"item_name in [{item_name_expr}]"

    def _normalize_milvus_hits(self, hits: Sequence[Dict[str, Any]]) -> List[Dict[str, Any]]:
        normalized: List[Dict[str, Any]] = []
        for hit in hits:
            entity = dict(hit.get("entity", {}))
            chunk_id = entity.get("chunk_id") or hit.get("id")
            normalized.append(
                {
                    "chunk_id": str(chunk_id) if chunk_id is not None else "",
                    "id": hit.get("id", chunk_id),
                    "score": float(hit.get("distance", 0.0)),
                    "text": entity.get("content", ""),
                    "title": entity.get("title", ""),
                    "item_name": entity.get("item_name", ""),
                    "file_title": entity.get("file_title", ""),
                    "parent_title": entity.get("parent_title", ""),
                    "source": "local",
                    "entity": entity,
                }
            )
        return normalized

    def _normalize_rrf_docs(self, docs: Sequence[Dict[str, Any]]) -> List[Dict[str, Any]]:
        normalized: List[Dict[str, Any]] = []
        for doc in docs:
            entity = dict(doc.get("entity", {}))
            chunk_id = entity.get("chunk_id") or doc.get("chunk_id") or doc.get("id")
            normalized.append(
                {
                    "chunk_id": str(chunk_id) if chunk_id is not None else "",
                    "id": doc.get("id", chunk_id),
                    "score": float(doc.get("score", doc.get("distance", 0.0))),
                    "text": entity.get("content", doc.get("text", "")),
                    "title": entity.get("title", doc.get("title", "")),
                    "item_name": entity.get("item_name", doc.get("item_name", "")),
                    "file_title": entity.get("file_title", doc.get("file_title", "")),
                    "parent_title": entity.get("parent_title", doc.get("parent_title", "")),
                    "source": doc.get("source", "local"),
                    "entity": entity or doc.get("entity", {}),
                }
            )
        return normalized

    def _extract_chunk_id(self, doc: Dict[str, Any]) -> str:
        chunk_id = doc.get("chunk_id") or doc.get("id")
        return str(chunk_id) if chunk_id is not None else ""
