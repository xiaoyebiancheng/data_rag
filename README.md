# Dataset RAG

面向设备手册、产品文档的多模态 RAG 知识库问答系统。核心链路包括文档导入、图片摘要、标题语义切片、BGE-M3 dense/sparse 混合检索、HyDE、RRF、Rerank、可信答案、多租户过滤和离线评测。

## 1. 本地环境

项目要求 Python 3.12。首次启动前先准备环境变量：

```bash
cp .env.example .env
```

至少需要按实际环境填写：

```text
OPENAI_API_KEY
OPENAI_BASE_URL
MINERU_API_TOKEN
MCP_DASHSCOPE_BASE_URL_STREAMABLE
```

## 2. 启动基础设施

启动 MongoDB、MinIO、Milvus、Attu：

```bash
docker compose up -d
```

默认端口：

```text
MongoDB: 27017
MinIO API: 9000
MinIO Console: 9001
Milvus MinIO API: 9002
Milvus MinIO Console: 9003
Milvus: 19530
Attu: 8000
```

如果要同时用容器启动两个 API 服务：

```bash
docker compose --profile api up -d --build
```

API 容器模式下端口为：

```text
Import API: 18000 -> container 8000
Query API: 18001 -> container 8001
```

## 3. 本地直接启动 API

导入服务：

```bash
.venv/bin/python -m uvicorn app.import_process.api.import_server:app --host 127.0.0.1 --port 8000
```

查询服务：

```bash
.venv/bin/python -m uvicorn app.query_process.api.query_server:app --host 127.0.0.1 --port 8001
```

查询健康检查：

```bash
curl http://127.0.0.1:8001/health
```

## 4. 端到端 smoke

在导入服务和查询服务都启动后，可以运行 HTTP smoke 脚本。它会检查健康状态、可选上传文档、执行一次查询，并输出 Markdown 报告。

只测查询：

```bash
.venv/bin/python scripts/e2e_smoke.py --query "HAK180 使用什么电源规格？"
```

带文件上传：

```bash
.venv/bin/python scripts/e2e_smoke.py --upload-file "doc/hak180产品安全手册.pdf" --query "HAK180 使用什么电源规格？"
```

默认报告输出：

```text
reports/e2e_smoke_report.md
```

批量导入 `doc` 目录中的全部可支持文件：

```bash
.venv/bin/python scripts/import_doc_corpus.py --doc-dir doc --resume
```

当前批量导入脚本只导入 `.pdf`、`.md`。其他后缀会被记录到报告的 skipped 列表，不计入成功入库。

## 5. 离线评测

项目内置固定评测样本：

```text
data/eval/rag_eval_sample.jsonl
```

运行评测：

```bash
.venv/bin/python -m app.evaluation.run_eval \
  --dataset data/eval/rag_eval_sample.jsonl \
  --strategies hybrid,hybrid_rrf,hybrid_rrf_rerank,hyde_hybrid_rrf_rerank \
  --top-k 5 \
  --output reports/rag_eval_report.md
```

报告会包含 Hit@K、Recall@K、MRR、NDCG@K、Faithfulness、Answer Relevance、平均延迟和 P95 延迟。

## 6. 快速单测

不依赖外部服务的核心逻辑测试：

```bash
.venv/bin/python -m unittest \
  app.test.test_evaluation_metrics \
  app.test.test_metadata_tagging \
  app.test.test_query_profile \
  app.test.test_trusted_answer \
  app.test.test_prompt_registry \
  app.test.test_auth_context
```

## 7. 当前生产化边界

- 导入重试当前是全链路重试，还不是节点级断点恢复。
- Docker Compose 适合本地联调，不等同于生产 K8s 编排。
- API 多实例部署时，SSE 队列和实时任务进度需要外置到 Redis 或使用 sticky session。
- 完整链路依赖外部 LLM、MinerU、Milvus、MongoDB、MinIO；上线前需要补监控、告警、限流和熔断。
