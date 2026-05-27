from __future__ import annotations

import argparse
import json
import re
from datetime import datetime
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont


DEFAULT_FONT = "/System/Library/Fonts/Hiragino Sans GB.ttc"


def parse_report(report_path: Path) -> dict[str, str]:
    text = report_path.read_text(encoding="utf-8")
    match = re.search(
        r"\| hybrid_rrf_rerank \| ([0-9.]+) \| ([0-9.]+) \| ([0-9.]+) \| "
        r"([0-9.]+) \| ([0-9.]+) \| ([0-9.]+) \| ([0-9.]+) \| ([0-9.]+) \| ([0-9.]+) \|",
        text,
    )
    if not match:
        raise ValueError(f"Cannot find hybrid_rrf_rerank metrics in {report_path}")

    (
        hit,
        recall,
        mrr,
        ndcg,
        faithfulness,
        answer_relevance,
        avg_latency,
        p50_latency,
        p95_latency,
    ) = map(float, match.groups())

    return {
        "hit_at_5": f"{hit * 100:.2f}%",
        "recall_at_5": f"{recall * 100:.2f}%",
        "mrr": f"{mrr:.4f}",
        "ndcg_at_5": f"{ndcg:.4f}",
        "faithfulness": f"{faithfulness * 100:.2f}%",
        "answer_relevance": f"{answer_relevance * 100:.2f}%",
        "avg_latency": f"{avg_latency / 1000:.2f}s",
        "p50_latency": f"{p50_latency / 1000:.2f}s",
        "p95_latency": f"{p95_latency / 1000:.2f}s",
    }


def count_jsonl(path: Path) -> int:
    with path.open("r", encoding="utf-8") as f:
        return sum(1 for line in f if line.strip())


def font(size: int, bold: bool = False) -> ImageFont.FreeTypeFont:
    return ImageFont.truetype(DEFAULT_FONT, size=size)


def centered_text(
    draw: ImageDraw.ImageDraw,
    xy: tuple[int, int],
    text: str,
    text_font: ImageFont.FreeTypeFont,
    fill: tuple[int, int, int],
) -> None:
    x, y = xy
    bbox = draw.textbbox((0, 0), text, font=text_font)
    draw.text((x - (bbox[2] - bbox[0]) / 2, y), text, font=text_font, fill=fill)


def build_png(metrics: dict[str, str], sample_count: int, output: Path) -> None:
    width, height = 1800, 520
    scale = 2
    img = Image.new("RGB", (width * scale, height * scale), (248, 250, 252))
    draw = ImageDraw.Draw(img)

    def s(v: int) -> int:
        return v * scale

    def box(coords, radius, fill, outline=None, width=1):
        draw.rounded_rectangle(
            tuple(s(int(v)) for v in coords),
            radius=s(radius),
            fill=fill,
            outline=outline,
            width=s(width),
        )

    navy = (29, 43, 76)
    muted = (105, 116, 138)
    blue = (73, 95, 230)
    teal = (33, 168, 168)
    line = (219, 226, 239)
    white = (255, 255, 255)

    box((44, 34, 1756, 486), 24, white, line, 2)
    draw.text((s(78), s(62)), "多模态 RAG 知识库评测结果", font=font(s(34)), fill=navy)
    draw.text(
        (s(78), s(112)),
        "50 条离线测试集 · hybrid RRF + rerank · TopK=5 · 指标来自本地评测报告",
        font=font(s(22)),
        fill=muted,
    )

    metrics_to_draw = [
        ("50", "retrieval eval 样本", "固定测试用例"),
        (metrics["hit_at_5"], "Hit@5 / Recall@5", "golden chunk 命中率"),
        (metrics["mrr"], "MRR", "相关证据排序质量"),
        (metrics["faithfulness"], "Faithfulness", "回答忠实度"),
        (metrics["p95_latency"], "P95 latency", "端到端问答延迟"),
    ]

    start_x, card_w, gap = 78, 314, 22
    y0, y1 = 178, 384
    for idx, (value, label, caption) in enumerate(metrics_to_draw):
        x0 = start_x + idx * (card_w + gap)
        x1 = x0 + card_w
        box((x0, y0, x1, y1), 18, (250, 252, 255), (226, 232, 244), 2)
        accent = blue if idx in {0, 1, 2} else teal
        draw.rounded_rectangle((s(x0), s(y0), s(x1), s(y0 + 8)), radius=s(8), fill=accent)
        centered_text(draw, (s((x0 + x1) // 2), s(y0 + 42)), value, font(s(46)), navy)
        centered_text(draw, (s((x0 + x1) // 2), s(y0 + 107)), label, font(s(22)), navy)
        centered_text(draw, (s((x0 + x1) // 2), s(y0 + 147)), caption, font(s(18)), muted)

    footnote = (
        f"N={sample_count}; Answer Relevance={metrics['answer_relevance']}; "
        f"NDCG@5={metrics['ndcg_at_5']}; Avg latency={metrics['avg_latency']}; "
        f"generated {datetime.now().strftime('%Y-%m-%d')}"
    )
    draw.text((s(78), s(426)), footnote, font=font(s(19)), fill=muted)

    img = img.resize((width, height), Image.Resampling.LANCZOS)
    output.parent.mkdir(parents=True, exist_ok=True)
    img.save(output)


def build_svg(metrics: dict[str, str], sample_count: int, output: Path) -> None:
    payload = {
        "sample_count": sample_count,
        **metrics,
        "generated_at": datetime.now().strftime("%Y-%m-%d"),
    }
    output.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--report", default="reports/rag_eval_report_current.md")
    parser.add_argument("--dataset", default="data/eval/rag_eval_sample.current.jsonl")
    parser.add_argument("--png", default="reports/resume_rag_metrics.png")
    parser.add_argument("--json", default="reports/resume_rag_metrics.json")
    args = parser.parse_args()

    metrics = parse_report(Path(args.report))
    sample_count = count_jsonl(Path(args.dataset))
    build_png(metrics, sample_count, Path(args.png))
    build_svg(metrics, sample_count, Path(args.json))
    print(f"Generated {args.png}")
    print(f"Generated {args.json}")


if __name__ == "__main__":
    main()
