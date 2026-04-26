from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional


@dataclass
class EvalSample:
    question: str
    item_names: List[str] = field(default_factory=list)
    golden_chunk_ids: List[str] = field(default_factory=list)
    golden_answer: str = ""
    category: str = "未分类"
    metadata: Dict[str, Any] = field(default_factory=dict)

    # 增: 增的原因是评测样本需要统一做基础清洗，避免下游检索与指标计算反复做空值处理。
    @classmethod
    def from_dict(cls, payload: Dict[str, Any]) -> "EvalSample":
        # 优化: 优化的原因是评测数据后续可能同时存在 golden_answer / reference_answer 两种字段命名，这里做兼容以避免数据格式切换导致评测入口报错。
        answer_text = payload.get("golden_answer")
        if answer_text in (None, ""):
            answer_text = payload.get("reference_answer", "")
        return cls(
            question=str(payload.get("question", "")).strip(),
            item_names=[str(x).strip() for x in payload.get("item_names", []) if str(x).strip()],
            golden_chunk_ids=[str(x).strip() for x in payload.get("golden_chunk_ids", []) if str(x).strip()],
            golden_answer=str(answer_text).strip(),
            category=str(payload.get("category", "未分类")).strip() or "未分类",
            metadata=dict(payload.get("metadata", {})),
        )


def load_eval_dataset(dataset_path: str | Path) -> List[EvalSample]:
    path = Path(dataset_path)
    samples: List[EvalSample] = []
    with path.open("r", encoding="utf-8") as file_obj:
        for line_no, raw_line in enumerate(file_obj, start=1):
            line = raw_line.strip()
            if not line:
                continue
            payload = json.loads(line)
            sample = EvalSample.from_dict(payload)
            if not sample.question:
                raise ValueError(f"评测数据第{line_no}行缺少 question")
            samples.append(sample)
    return samples


def iter_categories(samples: Iterable[EvalSample]) -> List[str]:
    return sorted({sample.category for sample in samples})


def safe_json_dumps(data: Dict[str, Any]) -> str:
    return json.dumps(data, ensure_ascii=False, sort_keys=True)
