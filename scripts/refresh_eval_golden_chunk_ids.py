from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Sequence

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from app.clients.milvus_utils import create_hybrid_search_requests, get_milvus_client, hybrid_search
from app.conf.milvus_config import milvus_config
from app.lm.embedding_utils import generate_embeddings
from app.utils.escape_milvus_string_utils import escape_milvus_string


OUTPUT_FIELDS = ["chunk_id", "content", "file_title", "title", "parent_title", "item_name"]


def _load_jsonl(path: Path) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    for raw_line in path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if line:
            rows.append(json.loads(line))
    return rows


def _write_jsonl(path: Path, rows: Sequence[Dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    content = "\n".join(json.dumps(row, ensure_ascii=False, separators=(",", ":")) for row in rows)
    path.write_text(content + "\n", encoding="utf-8")


def _load_alias_map(path: str) -> Dict[str, List[str]]:
    if not path:
        return {}
    alias_path = Path(path)
    if not alias_path.exists():
        return {}
    payload = json.loads(alias_path.read_text(encoding="utf-8"))
    return {
        str(key): [str(item) for item in value if str(item).strip()]
        for key, value in payload.items()
        if isinstance(value, list)
    }


def _expand_item_names(item_names: Sequence[str], alias_map: Dict[str, List[str]]) -> List[str]:
    expanded: List[str] = []
    for item_name in item_names:
        if item_name and item_name not in expanded:
            expanded.append(item_name)
        for alias in alias_map.get(item_name, []):
            if alias and alias not in expanded:
                expanded.append(alias)
    return expanded


def _item_expr(item_names: Sequence[str]) -> str:
    cleaned = [item for item in item_names if item]
    if not cleaned:
        return ""
    quoted = ", ".join(f'"{escape_milvus_string(item)}"' for item in cleaned)
    return f"item_name in [{quoted}]"


def _search(client, query_text: str, item_names: Sequence[str], limit: int) -> List[Dict[str, Any]]:
    embeddings = generate_embeddings([query_text])
    reqs = create_hybrid_search_requests(
        dense_vector=embeddings["dense"][0],
        sparse_vector=embeddings["sparse"][0],
        expr=_item_expr(item_names),
        limit=limit,
    )
    response = hybrid_search(
        client=client,
        collection_name=milvus_config.chunks_collection,
        reqs=reqs,
        ranker_weights=(0.9, 0.1),
        norm_score=True,
        limit=limit,
        output_fields=OUTPUT_FIELDS,
    )
    return list(response[0]) if response else []


def _normalize_hits(hits: Sequence[Dict[str, Any]]) -> List[Dict[str, Any]]:
    normalized: List[Dict[str, Any]] = []
    for hit in hits:
        entity = dict(hit.get("entity", {}))
        chunk_id = entity.get("chunk_id") or hit.get("id")
        normalized.append(
            {
                "chunk_id": str(chunk_id),
                "item_name": entity.get("item_name", ""),
                "file_title": entity.get("file_title", ""),
                "title": entity.get("title", ""),
                "score": float(hit.get("distance", 0.0)),
            }
        )
    return normalized


def refresh_rows(
    rows: Sequence[Dict[str, Any]],
    limit: int,
    update_item_names: bool,
    alias_map: Dict[str, List[str]],
) -> List[Dict[str, Any]]:
    client = get_milvus_client()
    if client is None:
        raise RuntimeError("Milvus client is not available")

    refreshed: List[Dict[str, Any]] = []
    for index, row in enumerate(rows, start=1):
        question = str(row.get("question", "")).strip()
        answer = str(row.get("golden_answer") or row.get("reference_answer") or "").strip()
        item_names = [str(item).strip() for item in row.get("item_names", []) if str(item).strip()]
        search_item_names = _expand_item_names(item_names, alias_map)
        query_text = f"{question}\n{answer}".strip()

        hits = _search(client, query_text, search_item_names, limit)
        fallback_used = False
        if not hits and search_item_names:
            hits = _search(client, query_text, [], limit)
            fallback_used = True

        normalized_hits = _normalize_hits(hits)
        golden_top_n = max(1, len(row.get("golden_chunk_ids") or []))
        selected = normalized_hits[:golden_top_n]
        selected_ids = [hit["chunk_id"] for hit in selected if hit.get("chunk_id")]
        matched_item_names = []
        for hit in selected:
            item_name = hit.get("item_name")
            if item_name and item_name not in matched_item_names:
                matched_item_names.append(item_name)

        current = dict(row)
        metadata = dict(current.get("metadata") or {})
        metadata["golden_refresh"] = {
            "refreshed_at": time.strftime("%Y-%m-%d %H:%M:%S"),
            "source": "current_milvus_hybrid_search",
            "old_golden_chunk_ids": row.get("golden_chunk_ids", []),
            "old_item_names": item_names,
            "search_item_names": search_item_names,
            "fallback_without_item_filter": fallback_used,
            "matched_item_names": matched_item_names,
            "candidates": normalized_hits[: min(limit, 5)],
        }
        current["metadata"] = metadata
        current["golden_chunk_ids"] = selected_ids
        if update_item_names and matched_item_names:
            current["item_names"] = matched_item_names

        refreshed.append(current)
        print(
            f"[{index}/{len(rows)}] {question} -> "
            f"golden={selected_ids or '[]'}, item_names={current.get('item_names', [])}, fallback={fallback_used}"
        )
    return refreshed


def main() -> None:
    parser = argparse.ArgumentParser(description="Refresh eval golden chunk ids against the current Milvus corpus.")
    parser.add_argument("--input", default="data/eval/rag_eval_sample.jsonl")
    parser.add_argument("--output", default="data/eval/rag_eval_sample.current.jsonl")
    parser.add_argument("--limit", type=int, default=8)
    parser.add_argument("--alias-map", default="data/eval/item_name_aliases.json")
    parser.add_argument("--keep-item-names", action="store_true", help="Do not replace item_names with matched current-corpus names.")
    args = parser.parse_args()

    rows = _load_jsonl(Path(args.input))
    alias_map = _load_alias_map(args.alias_map)
    refreshed = refresh_rows(rows, limit=args.limit, update_item_names=not args.keep_item_names, alias_map=alias_map)
    _write_jsonl(Path(args.output), refreshed)
    print(f"Refreshed eval dataset written to {args.output}")


if __name__ == "__main__":
    main()
