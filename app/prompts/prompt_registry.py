from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Optional, Tuple

from pydantic import BaseModel, Field

from app.utils.path_util import PROJECT_ROOT


class PromptDefinition(BaseModel):
    prompt_name: str = Field(..., description="Prompt 名称")
    version: str = Field(..., description="Prompt 版本")
    task_type: str = Field(..., description="任务类型")
    template_path: str = Field(..., description="Prompt 模板路径")
    description: str = Field(..., description="Prompt 说明")
    created_at: str = Field(..., description="创建时间")
    updated_at: str = Field(..., description="更新时间")
    is_default: bool = Field(True, description="是否默认版本")


def _build_prompt_definition(
    prompt_name: str,
    version: str,
    task_type: str,
    description: str,
    *,
    created_at: str = "2026-05-20T00:00:00+08:00",
    updated_at: str = "2026-05-20T00:00:00+08:00",
    is_default: bool = True,
) -> PromptDefinition:
    return PromptDefinition(
        prompt_name=prompt_name,
        version=version,
        task_type=task_type,
        template_path=f"prompts/{prompt_name}.prompt",
        description=description,
        created_at=created_at,
        updated_at=updated_at,
        is_default=is_default,
    )


_PROMPT_REGISTRY: Dict[Tuple[str, str], PromptDefinition] = {
    ("answer_out", "v1"): _build_prompt_definition(
        "answer_out",
        "v1",
        "answer_generation",
        "问答主链路答案生成 Prompt，强调只基于参考内容作答并限制无关扩写。",
    ),
    ("eval_answer_relevance_judge", "v1"): _build_prompt_definition(
        "eval_answer_relevance_judge",
        "v1",
        "evaluation_judge",
        "离线评测 Answer Relevance Judge Prompt，评估答案是否真正回答问题。",
    ),
    ("eval_faithfulness_judge", "v1"): _build_prompt_definition(
        "eval_faithfulness_judge",
        "v1",
        "evaluation_judge",
        "离线评测 Faithfulness Judge Prompt，评估答案是否被上下文支持。",
    ),
    ("hyde_prompt", "v1"): _build_prompt_definition(
        "hyde_prompt",
        "v1",
        "query_expansion",
        "HyDE 假设性答案生成 Prompt，用于增强召回。",
    ),
    ("image_summary", "v1"): _build_prompt_definition(
        "image_summary",
        "v1",
        "multimodal_summary",
        "图片摘要 Prompt，用于把 Markdown 图片转成可检索文本描述。",
    ),
    ("item_name_recognition", "v1"): _build_prompt_definition(
        "item_name_recognition",
        "v1",
        "document_entity_recognition",
        "导入链路主体识别 Prompt，用于识别文档对应 item_name。",
    ),
    ("product_recognition_system", "v1"): _build_prompt_definition(
        "product_recognition_system",
        "v1",
        "document_entity_recognition",
        "主体识别系统 Prompt，约束 item_name 识别输出边界。",
    ),
    ("rewritten_query_and_itemnames", "v1"): _build_prompt_definition(
        "rewritten_query_and_itemnames",
        "v1",
        "query_rewrite",
        "查询链路 Query Rewrite + item_name 提取 Prompt。",
    ),
}


_DEFAULT_VERSION_BY_NAME: Dict[str, str] = {}
for (_prompt_name, _version), _definition in _PROMPT_REGISTRY.items():
    if _definition.is_default:
        _DEFAULT_VERSION_BY_NAME[_prompt_name] = _version


def get_prompt_definition(name: str, version: Optional[str] = None) -> PromptDefinition:
    selected_version = version or _DEFAULT_VERSION_BY_NAME.get(name)
    if not selected_version:
        raise KeyError(f"未在 Prompt Registry 中找到默认版本：{name}")
    definition = _PROMPT_REGISTRY.get((name, selected_version))
    if definition is None:
        raise KeyError(f"未在 Prompt Registry 中找到 Prompt：name={name}, version={selected_version}")
    return definition


def list_prompt_definitions(prompt_name: Optional[str] = None) -> List[PromptDefinition]:
    if prompt_name:
        return [definition for (name, _), definition in _PROMPT_REGISTRY.items() if name == prompt_name]
    return list(_PROMPT_REGISTRY.values())


def render_prompt(name: str, *, version: Optional[str] = None, **kwargs) -> str:
    definition = get_prompt_definition(name, version=version)
    prompt_path = PROJECT_ROOT / definition.template_path
    if not prompt_path.exists():
        raise FileNotFoundError(f"Prompt 模板不存在：{prompt_path}")
    raw_prompt = prompt_path.read_text(encoding="utf-8")
    if kwargs:
        return raw_prompt.format(**kwargs)
    return raw_prompt
