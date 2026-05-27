from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import Any, Dict

import requests


def _http_session() -> requests.Session:
    session = requests.Session()
    session.trust_env = False
    return session


def _request_json(method: str, url: str, **kwargs) -> Dict[str, Any]:
    with _http_session() as session:
        response = session.request(method, url, timeout=kwargs.pop("timeout", 120), **kwargs)
    response.raise_for_status()
    if not response.content:
        return {}
    return response.json()


def check_query_health(query_url: str) -> Dict[str, Any]:
    started = time.perf_counter()
    payload = _request_json("GET", f"{query_url.rstrip('/')}/health", timeout=10)
    return {
        "ok": payload.get("status") == "ok",
        "latency_ms": round((time.perf_counter() - started) * 1000, 2),
        "payload": payload,
    }


def upload_file(import_url: str, file_path: Path) -> Dict[str, Any]:
    started = time.perf_counter()
    with file_path.open("rb") as file_obj:
        with _http_session() as session:
            response = session.post(
                f"{import_url.rstrip('/')}/upload",
                files=[("files", (file_path.name, file_obj))],
                timeout=120,
            )
    response.raise_for_status()
    payload = response.json()
    return {
        "latency_ms": round((time.perf_counter() - started) * 1000, 2),
        "payload": payload,
    }


def poll_import_status(import_url: str, task_id: str, timeout_seconds: int, interval_seconds: float) -> Dict[str, Any]:
    deadline = time.time() + timeout_seconds
    last_payload: Dict[str, Any] = {}
    while time.time() < deadline:
        last_payload = _request_json("GET", f"{import_url.rstrip('/')}/status/{task_id}", timeout=10)
        status = str(last_payload.get("status", "")).lower()
        document_status = str((last_payload.get("result") or {}).get("document_status", "")).upper()
        if status in {"completed", "failed"} or document_status in {"ACTIVE", "DUPLICATED", "FAILED"}:
            return last_payload
        time.sleep(interval_seconds)
    raise TimeoutError(f"import task did not finish within {timeout_seconds}s: {task_id}")


def query_once(query_url: str, query: str, session_id: str) -> Dict[str, Any]:
    started = time.perf_counter()
    payload = _request_json(
        "POST",
        f"{query_url.rstrip('/')}/query",
        json={"query": query, "session_id": session_id, "is_stream": False},
        timeout=180,
    )
    return {
        "latency_ms": round((time.perf_counter() - started) * 1000, 2),
        "payload": payload,
    }


def write_report(output_path: Path, report: Dict[str, Any]) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    lines = [
        "# E2E Smoke Report",
        "",
        f"- Generated At: `{time.strftime('%Y-%m-%d %H:%M:%S')}`",
        f"- Query URL: `{report['query_url']}`",
        f"- Import URL: `{report['import_url']}`",
        "",
        "## Health",
        "",
        f"- OK: `{report['health'].get('ok')}`",
        f"- Latency(ms): `{report['health'].get('latency_ms')}`",
        "",
    ]
    if report.get("upload"):
        lines.extend([
            "## Upload",
            "",
            f"- File: `{report['upload_file']}`",
            f"- Latency(ms): `{report['upload'].get('latency_ms')}`",
            "```json",
            json.dumps(report["upload"].get("payload"), ensure_ascii=False, indent=2, default=str),
            "```",
            "",
            "## Import Status",
            "",
            "```json",
            json.dumps(report.get("import_status"), ensure_ascii=False, indent=2, default=str),
            "```",
            "",
        ])
    lines.extend([
        "## Query",
        "",
        f"- Session ID: `{report['session_id']}`",
        f"- Query: `{report['query']}`",
        f"- Latency(ms): `{report['query_result'].get('latency_ms')}`",
        "",
        "```json",
        json.dumps(report["query_result"].get("payload"), ensure_ascii=False, indent=2, default=str),
        "```",
        "",
    ])
    output_path.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description="Run a lightweight HTTP E2E smoke check.")
    parser.add_argument("--import-url", default="http://127.0.0.1:8000")
    parser.add_argument("--query-url", default="http://127.0.0.1:8001")
    parser.add_argument("--upload-file", default="")
    parser.add_argument("--query", default="HAK180 使用什么电源规格？")
    parser.add_argument("--session-id", default="e2e-smoke-session")
    parser.add_argument("--poll-timeout", type=int, default=900)
    parser.add_argument("--poll-interval", type=float, default=3.0)
    parser.add_argument("--output", default="reports/e2e_smoke_report.md")
    args = parser.parse_args()

    report: Dict[str, Any] = {
        "import_url": args.import_url,
        "query_url": args.query_url,
        "session_id": args.session_id,
        "query": args.query,
        "health": check_query_health(args.query_url),
    }

    if args.upload_file:
        file_path = Path(args.upload_file)
        if not file_path.exists():
            raise FileNotFoundError(file_path)
        upload = upload_file(args.import_url, file_path)
        report["upload_file"] = str(file_path)
        report["upload"] = upload
        task_ids = upload.get("payload", {}).get("task_ids", [])
        if task_ids:
            report["import_status"] = poll_import_status(
                args.import_url,
                task_ids[0],
                args.poll_timeout,
                args.poll_interval,
            )

    report["query_result"] = query_once(args.query_url, args.query, args.session_id)
    write_report(Path(args.output), report)
    print(f"E2E smoke report written to {args.output}")


if __name__ == "__main__":
    main()
