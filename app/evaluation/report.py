from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Sequence

from app.evaluation.evaluator import SampleEvalResult, StrategySummary


def _format_metric(value):
    if value is None:
        return "-"
    if isinstance(value, float):
        return f"{value:.4f}"
    return str(value)


def _build_strategy_table(summaries: Sequence[StrategySummary]) -> List[str]:
    lines = [
        "| strategy | Hit@K | Recall@K | MRR | NDCG@K | Faithfulness | Answer Relevance | Avg Latency(ms) | P50 Latency(ms) | P95 Latency(ms) |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for summary in summaries:
        metrics = summary.overall_metrics()
        lines.append(
            "| "
            + " | ".join(
                [
                    summary.strategy,
                    _format_metric(metrics["Hit@K"]),
                    _format_metric(metrics["Recall@K"]),
                    _format_metric(metrics["MRR"]),
                    _format_metric(metrics["NDCG@K"]),
                    _format_metric(metrics["Faithfulness"]),
                    _format_metric(metrics["Answer Relevance"]),
                    _format_metric(metrics["Avg Latency"]),
                    _format_metric(metrics["P50 Latency"]),
                    _format_metric(metrics["P95 Latency"]),
                ]
            )
            + " |"
        )
    return lines


def _build_category_tables(summaries: Sequence[StrategySummary]) -> List[str]:
    lines: List[str] = []
    for summary in summaries:
        lines.append(f"### `{summary.strategy}`")
        lines.append("")
        lines.append("| category | Hit@K | Recall@K | MRR | NDCG@K | Faithfulness | Answer Relevance | Avg Latency(ms) | P95 Latency(ms) |")
        lines.append("|---|---:|---:|---:|---:|---:|---:|---:|---:|")
        for category, metrics in sorted(summary.metrics_by_category().items()):
            lines.append(
                "| "
                + " | ".join(
                    [
                        category,
                        _format_metric(metrics["Hit@K"]),
                        _format_metric(metrics["Recall@K"]),
                        _format_metric(metrics["MRR"]),
                        _format_metric(metrics["NDCG@K"]),
                        _format_metric(metrics["Faithfulness"]),
                        _format_metric(metrics["Answer Relevance"]),
                        _format_metric(metrics["Avg Latency"]),
                        _format_metric(metrics["P95 Latency"]),
                    ]
                )
                + " |"
            )
        lines.append("")
    return lines


def _collect_bad_cases(summaries: Sequence[StrategySummary]) -> List[SampleEvalResult]:
    bad_cases: List[SampleEvalResult] = []
    for summary in summaries:
        for result in summary.sample_results:
            retrieval_bad = result.hit_at_k is not None and result.hit_at_k < 1.0
            answer_bad = (
                (result.answer_relevance is not None and result.answer_relevance < 0.6)
                or (result.faithfulness is not None and result.faithfulness < 0.6)
            )
            if result.error or retrieval_bad or answer_bad:
                bad_cases.append(result)
    return bad_cases


def _build_bad_case_section(summaries: Sequence[StrategySummary]) -> List[str]:
    lines = ["## Bad Cases", ""]
    bad_cases = _collect_bad_cases(summaries)
    if not bad_cases:
        lines.append("本次评测未发现明显 bad case。")
        lines.append("")
        return lines

    for index, result in enumerate(bad_cases, start=1):
        lines.append(f"### Case {index} - `{result.strategy}` / `{result.sample.category}`")
        lines.append(f"- Question: {result.sample.question}")
        lines.append(f"- Item Names: {result.sample.item_names}")
        lines.append(f"- Golden Chunk IDs: {result.sample.golden_chunk_ids}")
        lines.append(f"- Retrieved Chunk IDs: {[doc.get('chunk_id') for doc in result.retrieved_docs]}")
        lines.append(f"- Hit@K: {_format_metric(result.hit_at_k)}")
        lines.append(f"- Recall@K: {_format_metric(result.recall_at_k)}")
        lines.append(f"- MRR: {_format_metric(result.mrr)}")
        lines.append(f"- NDCG@K: {_format_metric(result.ndcg_at_k)}")
        lines.append(f"- Faithfulness: {_format_metric(result.faithfulness)}")
        lines.append(f"- Faithfulness Reason: {result.faithfulness_reason or '-'}")
        lines.append(f"- Answer Relevance: {_format_metric(result.answer_relevance)}")
        lines.append(f"- Answer Relevance Reason: {result.answer_relevance_reason or '-'}")
        lines.append(f"- Prompt Versions: {result.prompt_versions or '-'}")
        lines.append(f"- Error: {result.error or '-'}")
        lines.append(f"- Answer: {result.answer or '-'}")
        lines.append("")
    return lines


def _build_recommendations(summaries: Sequence[StrategySummary]) -> List[str]:
    lines = ["## 推荐优化建议", ""]
    if not summaries:
        lines.append("- 无可用评测结果。")
        return lines

    ranked = []
    for summary in summaries:
        metrics = summary.overall_metrics()
        retrieval_score = sum(
            value for value in [metrics["Hit@K"], metrics["Recall@K"], metrics["MRR"], metrics["NDCG@K"]] if value is not None
        )
        answer_score = sum(
            value for value in [metrics["Faithfulness"], metrics["Answer Relevance"]] if value is not None
        )
        ranked.append((summary.strategy, retrieval_score, answer_score, metrics["P95 Latency"]))

    ranked.sort(key=lambda item: (item[1], item[2]), reverse=True)
    best_strategy = ranked[0][0]
    lines.append(f"- 综合当前数据，优先推荐继续迭代 `{best_strategy}`。")

    high_latency = [item for item in ranked if item[3] is not None and item[3] > 3000]
    if high_latency:
        lines.append("- 部分策略 P95 延迟偏高，建议优先排查 HyDE 生成和 Reranker 批处理耗时。")

    poor_faithfulness = []
    for summary in summaries:
        faithfulness = summary.overall_metrics()["Faithfulness"]
        if faithfulness is not None and faithfulness < 0.7:
            poor_faithfulness.append(summary.strategy)
    if poor_faithfulness:
        lines.append(f"- `{', '.join(poor_faithfulness)}` 的忠实度偏低，建议增加引用约束和答案后验校验。")

    low_recall = []
    for summary in summaries:
        recall_value = summary.overall_metrics()["Recall@K"]
        if recall_value is not None and recall_value < 0.5:
            low_recall.append(summary.strategy)
    if low_recall:
        lines.append(f"- `{', '.join(low_recall)}` 的 Recall 偏低，建议优化 item_name 过滤与 chunk 粒度。")

    lines.append("")
    return lines


def build_markdown_report(
    summaries: Sequence[StrategySummary],
    dataset_path: str,
    top_k: int,
    prompt_versions: Dict[str, str] | None = None,
) -> str:
    # 增: 增的原因是离线评测需要标准化 Markdown 报告，方便你后续直接做策略复盘、PR 说明和阶段性汇报。
    lines: List[str] = [
        "# RAG 离线评测报告",
        "",
        f"- Dataset: `{dataset_path}`",
        f"- Top K: `{top_k}`",
        f"- Strategy Count: `{len(summaries)}`",
        f"- Prompt Versions: `{prompt_versions or {}}`",
        "",
        "## 总体指标",
        "",
    ]
    lines.extend(_build_strategy_table(summaries))
    lines.append("")
    lines.append("## 按 Category 分组指标")
    lines.append("")
    lines.extend(_build_category_tables(summaries))
    lines.extend(_build_bad_case_section(summaries))
    lines.extend(_build_recommendations(summaries))
    return "\n".join(lines).strip() + "\n"


def write_markdown_report(output_path: str | Path, markdown: str) -> Path:
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(markdown, encoding="utf-8")
    return path
