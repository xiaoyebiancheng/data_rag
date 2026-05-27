from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import Any, Dict, Iterable, List

import requests


SUPPORTED_SUFFIXES = {".pdf", ".md"}


def _now() -> str:
    return time.strftime("%Y-%m-%d %H:%M:%S")


def _load_existing(json_path: Path) -> Dict[str, Any]:
    if not json_path.exists():
        return {"files": []}
    return json.loads(json_path.read_text(encoding="utf-8"))


def _successful_paths(report: Dict[str, Any]) -> set[str]:
    ok_status = {"ACTIVE", "DUPLICATED"}
    paths: set[str] = set()
    for item in report.get("files", []):
        document_status = str(item.get("document_status", "")).upper()
        if item.get("upload_ok") and document_status in ok_status:
            paths.add(item.get("path", ""))
    return paths


def _iter_files(doc_dir: Path, include_unsupported: bool) -> Iterable[Path]:
    for path in sorted(doc_dir.iterdir(), key=lambda current: current.name.lower()):
        if not path.is_file():
            continue
        if include_unsupported or path.suffix.lower() in SUPPORTED_SUFFIXES:
            yield path


def upload_file(import_url: str, file_path: Path, headers: Dict[str, str]) -> Dict[str, Any]:
    started = time.perf_counter()
    with file_path.open("rb") as file_obj:
        response = requests.post(
            f"{import_url.rstrip('/')}/upload",
            files=[("files", (file_path.name, file_obj))],
            headers=headers,
            timeout=180,
        )
    latency_ms = round((time.perf_counter() - started) * 1000, 2)
    response.raise_for_status()
    payload = response.json()
    return {
        "latency_ms": latency_ms,
        "payload": payload,
        "task_ids": payload.get("task_ids", []),
    }


def poll_status(import_url: str, task_id: str, headers: Dict[str, str], timeout_seconds: int, interval_seconds: float) -> Dict[str, Any]:
    deadline = time.time() + timeout_seconds
    last_payload: Dict[str, Any] = {}
    while time.time() < deadline:
        response = requests.get(
            f"{import_url.rstrip('/')}/status/{task_id}",
            headers=headers,
            timeout=30,
        )
        response.raise_for_status()
        last_payload = response.json()
        status = str(last_payload.get("status", "")).lower()
        document_status = str((last_payload.get("result") or {}).get("document_status", "")).upper()
        if status in {"completed", "failed"} or document_status in {"ACTIVE", "DUPLICATED", "FAILED"}:
            return last_payload
        time.sleep(interval_seconds)
    raise TimeoutError(f"import task did not finish within {timeout_seconds}s: {task_id}")


def _summarize_task(status_payload: Dict[str, Any]) -> Dict[str, Any]:
    result = status_payload.get("result") or {}
    task_meta = status_payload.get("task_meta") or {}
    node_logs = status_payload.get("node_logs") or []
    return {
        "task_id": status_payload.get("task_id", ""),
        "status": status_payload.get("status", ""),
        "doc_id": result.get("doc_id", ""),
        "document_status": result.get("document_status", ""),
        "error": result.get("error") or task_meta.get("error_stack") or "",
        "done_list": status_payload.get("done_list", []),
        "running_list": status_payload.get("running_list", []),
        "node_log_count": len(node_logs),
    }


def _write_reports(json_path: Path, markdown_path: Path, report: Dict[str, Any]) -> None:
    json_path.parent.mkdir(parents=True, exist_ok=True)
    markdown_path.parent.mkdir(parents=True, exist_ok=True)
    json_path.write_text(json.dumps(report, ensure_ascii=False, indent=2, default=str), encoding="utf-8")

    files = report.get("files", [])
    skipped = report.get("skipped", [])
    success = [
        item for item in files
        if item.get("upload_ok") and str(item.get("document_status", "")).upper() in {"ACTIVE", "DUPLICATED"}
    ]
    failed = [item for item in files if item not in success]
    lines = [
        "# Doc Corpus Import Report",
        "",
        f"- Generated At: `{_now()}`",
        f"- Import URL: `{report.get('import_url', '')}`",
        f"- Doc Dir: `{report.get('doc_dir', '')}`",
        f"- Dry Run: `{report.get('dry_run', False)}`",
        f"- Total Files In Doc: `{report.get('total_files', 0)}`",
        f"- Supported Files: `{report.get('supported_files', 0)}`",
        f"- Imported/Duplicated: `{len(success)}`",
        f"- Failed: `{len(failed)}`",
        f"- Skipped Unsupported: `{len(skipped)}`",
        "",
        "## Imported",
        "",
        "| file | task_id | document_status | doc_id | upload_ms |",
        "| --- | --- | --- | --- | ---: |",
    ]
    for item in success:
        lines.append(
            f"| {Path(item.get('path', '')).name} | `{item.get('task_id', '')}` | "
            f"`{item.get('document_status', '')}` | `{item.get('doc_id', '')}` | {item.get('upload_latency_ms', '')} |"
        )

    planned = report.get("planned_files", [])
    if planned:
        lines.extend(["", "## Planned Files", "", "| file | suffix |", "| --- | --- |"])
        for item in planned:
            lines.append(f"| {Path(item.get('path', '')).name} | `{item.get('suffix', '')}` |")

    if failed:
        lines.extend(["", "## Failed", "", "| file | task_id | status | error |", "| --- | --- | --- | --- |"])
        for item in failed:
            error = str(item.get("error", "")).replace("\n", " ")[:300]
            lines.append(
                f"| {Path(item.get('path', '')).name} | `{item.get('task_id', '')}` | "
                f"`{item.get('document_status') or item.get('status', '')}` | {error} |"
            )

    if skipped:
        lines.extend(["", "## Skipped Unsupported", "", "| file | suffix |", "| --- | --- |"])
        for item in skipped:
            lines.append(f"| {Path(item.get('path', '')).name} | `{item.get('suffix', '')}` |")

    markdown_path.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description="Import all supported files under doc/ into the RAG knowledge base.")
    parser.add_argument("--doc-dir", default="doc")
    parser.add_argument("--import-url", default="http://127.0.0.1:8000")
    parser.add_argument("--poll-timeout", type=int, default=1800)
    parser.add_argument("--poll-interval", type=float, default=5.0)
    parser.add_argument("--json-output", default="reports/doc_corpus_import_report.json")
    parser.add_argument("--markdown-output", default="reports/doc_corpus_import_report.md")
    parser.add_argument("--resume", action="store_true", help="Skip files that already imported successfully in the JSON report.")
    parser.add_argument("--include-unsupported", action="store_true", help="Upload unsupported suffixes too; normally they will fail in node_entry.")
    parser.add_argument("--limit", type=int, default=0, help="Optional max number of files to upload.")
    parser.add_argument("--dry-run", action="store_true", help="Only write the planned import report without calling the import API.")
    parser.add_argument("--user-id", default="eval-user")
    parser.add_argument("--tenant-id", default="default")
    parser.add_argument("--department-id", default="default")
    args = parser.parse_args()

    doc_dir = Path(args.doc_dir)
    if not doc_dir.exists():
        raise FileNotFoundError(doc_dir)

    json_path = Path(args.json_output)
    markdown_path = Path(args.markdown_output)
    existing = _load_existing(json_path) if args.resume else {"files": []}
    done_paths = _successful_paths(existing) if args.resume else set()
    headers = {
        "X-User-Id": args.user_id,
        "X-Tenant-Id": args.tenant_id,
        "X-Department-Id": args.department_id,
    }

    all_files = [path for path in sorted(doc_dir.iterdir(), key=lambda current: current.name.lower()) if path.is_file()]
    skipped = [
        {"path": str(path), "suffix": path.suffix.lower()}
        for path in all_files
        if path.suffix.lower() not in SUPPORTED_SUFFIXES
    ]
    candidates = [path for path in _iter_files(doc_dir, args.include_unsupported)]
    if args.limit > 0:
        candidates = candidates[: args.limit]

    report: Dict[str, Any] = {
        "generated_at": _now(),
        "import_url": args.import_url,
        "doc_dir": str(doc_dir),
        "total_files": len(all_files),
        "supported_files": sum(1 for path in all_files if path.suffix.lower() in SUPPORTED_SUFFIXES),
        "skipped": [] if args.include_unsupported else skipped,
        "files": list(existing.get("files", [])) if args.resume else [],
    }

    if args.dry_run:
        report["dry_run"] = True
        report["planned_files"] = [{"path": str(path), "suffix": path.suffix.lower()} for path in candidates]
        _write_reports(json_path, markdown_path, report)
        print(f"Dry-run import report written to {markdown_path}")
        return

    for index, file_path in enumerate(candidates, start=1):
        if str(file_path) in done_paths:
            print(f"[{index}/{len(candidates)}] skip already imported: {file_path}")
            continue
        item: Dict[str, Any] = {
            "path": str(file_path),
            "suffix": file_path.suffix.lower(),
            "upload_ok": False,
            "started_at": _now(),
        }
        print(f"[{index}/{len(candidates)}] uploading: {file_path}")
        try:
            upload = upload_file(args.import_url, file_path, headers)
            item["upload_ok"] = True
            item["upload_latency_ms"] = upload["latency_ms"]
            task_ids: List[str] = upload.get("task_ids", [])
            item["task_ids"] = task_ids
            if task_ids:
                status_payload = poll_status(
                    args.import_url,
                    task_ids[0],
                    headers,
                    args.poll_timeout,
                    args.poll_interval,
                )
                item.update(_summarize_task(status_payload))
        except Exception as exc:
            item["error"] = str(exc)
            print(f"failed: {file_path}: {exc}")
        finally:
            item["finished_at"] = _now()
            report["files"] = [existing_item for existing_item in report["files"] if existing_item.get("path") != str(file_path)]
            report["files"].append(item)
            _write_reports(json_path, markdown_path, report)

    _write_reports(json_path, markdown_path, report)
    print(f"Import report written to {markdown_path}")


if __name__ == "__main__":
    main()
