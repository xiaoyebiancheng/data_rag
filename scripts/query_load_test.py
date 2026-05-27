from __future__ import annotations

import argparse
import concurrent.futures
import json
import statistics
import time
import uuid
from pathlib import Path
from typing import Any, Dict, List

import requests


DEFAULT_QUERIES = [
    "HAK180 使用什么电源规格？",
    "打印机卡纸后应该怎么处理？",
    "路由器恢复出厂设置怎么操作？",
    "设备首次安装需要注意哪些步骤？",
    "无线网络连接失败时应该检查哪些配置？",
]


def _http_session() -> requests.Session:
    session = requests.Session()
    session.trust_env = False
    return session


def _percentile(values: List[float], percentile: float) -> float:
    if not values:
        return 0.0
    ordered = sorted(values)
    index = min(len(ordered) - 1, max(0, int(round((percentile / 100) * (len(ordered) - 1)))))
    return round(ordered[index], 2)


def _load_queries(path: str) -> List[str]:
    if not path:
        return DEFAULT_QUERIES
    queries: List[str] = []
    for line in Path(path).read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            payload = json.loads(line)
            query = payload.get("query") or payload.get("question")
        except json.JSONDecodeError:
            query = line
        if query:
            queries.append(str(query))
    return queries or DEFAULT_QUERIES


def query_once(query_url: str, query: str, timeout: int) -> Dict[str, Any]:
    started = time.perf_counter()
    session_id = f"load-{uuid.uuid4()}"
    try:
        with _http_session() as session:
            response = session.post(
                f"{query_url.rstrip('/')}/query",
                json={"query": query, "session_id": session_id, "is_stream": False},
                timeout=timeout,
            )
        latency_ms = round((time.perf_counter() - started) * 1000, 2)
        ok = response.ok
        payload: Any
        try:
            payload = response.json()
        except ValueError:
            payload = response.text[:500]
        return {
            "ok": ok,
            "status_code": response.status_code,
            "latency_ms": latency_ms,
            "query": query,
            "session_id": session_id,
            "error": "" if ok else str(payload)[:500],
        }
    except Exception as exc:
        return {
            "ok": False,
            "status_code": 0,
            "latency_ms": round((time.perf_counter() - started) * 1000, 2),
            "query": query,
            "session_id": session_id,
            "error": str(exc)[:500],
        }


def build_summary(results: List[Dict[str, Any]], started_at: float, finished_at: float) -> Dict[str, Any]:
    latencies = [float(item["latency_ms"]) for item in results if item.get("ok")]
    failures = [item for item in results if not item.get("ok")]
    total = len(results)
    return {
        "total": total,
        "success": total - len(failures),
        "failed": len(failures),
        "success_rate": round((total - len(failures)) / total, 4) if total else 0,
        "duration_seconds": round(finished_at - started_at, 2),
        "avg_latency_ms": round(statistics.mean(latencies), 2) if latencies else 0,
        "p50_latency_ms": _percentile(latencies, 50),
        "p95_latency_ms": _percentile(latencies, 95),
        "max_latency_ms": round(max(latencies), 2) if latencies else 0,
    }


def write_reports(output_path: Path, report: Dict[str, Any]) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    json_path = output_path.with_suffix(".json")
    json_path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")

    summary = report["summary"]
    lines = [
        "# Query Load Test Report",
        "",
        f"- Generated At: `{time.strftime('%Y-%m-%d %H:%M:%S')}`",
        f"- Query URL: `{report['query_url']}`",
        f"- Requests: `{summary['total']}`",
        f"- Concurrency: `{report['concurrency']}`",
        f"- Timeout(s): `{report['timeout']}`",
        "",
        "## Summary",
        "",
        f"- Success: `{summary['success']}`",
        f"- Failed: `{summary['failed']}`",
        f"- Success Rate: `{summary['success_rate']}`",
        f"- Duration(s): `{summary['duration_seconds']}`",
        f"- Avg Latency(ms): `{summary['avg_latency_ms']}`",
        f"- P50 Latency(ms): `{summary['p50_latency_ms']}`",
        f"- P95 Latency(ms): `{summary['p95_latency_ms']}`",
        f"- Max Latency(ms): `{summary['max_latency_ms']}`",
        "",
    ]
    failures = [item for item in report["results"] if not item.get("ok")]
    if failures:
        lines.extend(["## Failures", ""])
        for item in failures[:10]:
            lines.append(f"- `{item['query']}` -> status `{item['status_code']}`: {item['error']}")
        lines.append("")
    output_path.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description="Run a lightweight query API load test.")
    parser.add_argument("--query-url", default="http://127.0.0.1:8001")
    parser.add_argument("--queries-file", default="")
    parser.add_argument("--requests", type=int, default=10)
    parser.add_argument("--concurrency", type=int, default=2)
    parser.add_argument("--timeout", type=int, default=180)
    parser.add_argument("--output", default="reports/query_load_test_report.md")
    args = parser.parse_args()

    queries = _load_queries(args.queries_file)
    workload = [queries[index % len(queries)] for index in range(args.requests)]

    started_at = time.perf_counter()
    with concurrent.futures.ThreadPoolExecutor(max_workers=args.concurrency) as executor:
        futures = [executor.submit(query_once, args.query_url, query, args.timeout) for query in workload]
        results = [future.result() for future in concurrent.futures.as_completed(futures)]
    finished_at = time.perf_counter()

    report = {
        "query_url": args.query_url,
        "concurrency": args.concurrency,
        "timeout": args.timeout,
        "summary": build_summary(results, started_at, finished_at),
        "results": results,
    }
    write_reports(Path(args.output), report)
    print(f"Query load test report written to {args.output}")


if __name__ == "__main__":
    main()
