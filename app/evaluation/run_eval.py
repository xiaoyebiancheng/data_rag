from __future__ import annotations

import argparse

from app.core.logger import logger
from app.evaluation.dataset_schema import load_eval_dataset
from app.evaluation.evaluator import DEFAULT_STRATEGIES, RAGEvaluator
from app.evaluation.report import build_markdown_report, write_markdown_report


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="RAG 离线评测")
    parser.add_argument("--dataset", required=True, help="jsonl 评测数据路径")
    parser.add_argument("--strategies", default=",".join(DEFAULT_STRATEGIES), help="策略列表，逗号分隔")
    parser.add_argument("--top-k", type=int, default=5, help="Top K")
    parser.add_argument("--output", default="reports/rag_eval_report.md", help="Markdown 报告输出路径")
    parser.add_argument(
        "--prompt-versions",
        default="",
        help="Prompt 版本覆盖，格式如 answer_out=v1,eval_answer_relevance_judge=v1",
    )
    return parser.parse_args()


def parse_prompt_versions(raw_value: str) -> dict[str, str]:
    mapping: dict[str, str] = {}
    if not raw_value:
        return mapping
    for item in raw_value.split(","):
        current = item.strip()
        if not current:
            continue
        if "=" not in current:
            raise ValueError(f"无效的 prompt 版本参数：{current}")
        prompt_name, version = current.split("=", 1)
        mapping[prompt_name.strip()] = version.strip()
    return mapping


def main() -> None:
    args = parse_args()
    strategies = [item.strip() for item in args.strategies.split(",") if item.strip()]
    dataset = load_eval_dataset(args.dataset)
    prompt_versions = parse_prompt_versions(args.prompt_versions)

    # 增: 增的原因是需要提供一个可直接命令行运行的评测入口，降低不同检索策略做横向对比的使用成本。
    evaluator = RAGEvaluator(top_k=args.top_k, prompt_versions=prompt_versions)
    summaries = evaluator.evaluate(dataset, strategies)

    markdown = build_markdown_report(summaries, args.dataset, args.top_k, prompt_versions=prompt_versions)
    report_path = write_markdown_report(args.output, markdown)
    logger.info(f"离线评测完成，报告已输出到: {report_path}")


if __name__ == "__main__":
    main()
