from __future__ import annotations

from typing import Any, Dict, Iterable, List

from app.utils.escape_milvus_string_utils import escape_milvus_string


def _quote(value: Any) -> str:
    return f'"{escape_milvus_string(str(value))}"'


def build_item_name_expr(item_names: Iterable[str]) -> str:
    normalized = [item_name for item_name in item_names if item_name]
    if not normalized:
        return ""
    item_name_str = ", ".join(_quote(item_name) for item_name in normalized)
    return f"item_name in [{item_name_str}]"


def build_metadata_expr(metadata_filters: Dict[str, Any]) -> str:
    expr_list: List[str] = []
    for field_name, field_value in (metadata_filters or {}).items():
        if field_value is None or field_value == "":
            continue
        if isinstance(field_value, (list, tuple, set)):
            values = [value for value in field_value if value not in (None, "")]
            if not values:
                continue
            value_expr = ", ".join(_quote(value) for value in values)
            expr_list.append(f"{field_name} in [{value_expr}]")
            continue
        expr_list.append(f'{field_name} == {_quote(field_value)}')
    return " and ".join(expr_list)


def combine_expr(*parts: str) -> str:
    normalized = [part.strip() for part in parts if part and part.strip()]
    return " and ".join(normalized)
