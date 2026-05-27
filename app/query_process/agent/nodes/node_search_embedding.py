import sys

from app.clients.milvus_utils import create_hybrid_search_requests, get_milvus_client, hybrid_search
from app.core.logger import logger
from app.lm.embedding_utils import generate_embeddings
from app.query_process.retrieval.milvus_filtering import build_item_name_expr, build_metadata_expr, combine_expr
from app.query_process.retrieval.query_profile import DEFAULT_RETRIEVAL_CONFIG
from app.utils.task_utils import add_running_task, add_done_task
from dotenv import find_dotenv, load_dotenv

from app.conf.milvus_config import milvus_config

load_dotenv(find_dotenv())


def _get_retrieval_config(state):
    retrieval_config = dict(DEFAULT_RETRIEVAL_CONFIG.__dict__)
    retrieval_config.update(state.get("retrieval_config", {}) or {})
    retrieval_config.setdefault("metadata_filters", {})
    return retrieval_config


def _search_chunks(rewritten_query, item_names, retrieval_config, security_filter_expr=""):
    embeddings = generate_embeddings([rewritten_query])
    item_name_expr = build_item_name_expr(item_names)
    metadata_expr = build_metadata_expr(retrieval_config.get("metadata_filters", {}))
    full_expr = combine_expr(security_filter_expr, item_name_expr, metadata_expr)
    logger.info(
        f"动态检索配置已生效: query_type={retrieval_config.get('query_type', '')}, "
        f"expr={full_expr or item_name_expr or '<empty>'}, retrieval_config={retrieval_config}"
    )

    hybrid_search_requests = create_hybrid_search_requests(
        dense_vector=embeddings["dense"][0],
        sparse_vector=embeddings["sparse"][0],
        expr=full_expr or combine_expr(security_filter_expr, item_name_expr) or None,
        limit=int(retrieval_config.get("top_k", DEFAULT_RETRIEVAL_CONFIG.top_k)),
    )
    milvus_client = get_milvus_client()
    resp = hybrid_search(
        client=milvus_client,
        collection_name=milvus_config.chunks_collection,
        reqs=hybrid_search_requests,
        ranker_weights=(
            float(retrieval_config.get("dense_weight", DEFAULT_RETRIEVAL_CONFIG.dense_weight)),
            float(retrieval_config.get("sparse_weight", DEFAULT_RETRIEVAL_CONFIG.sparse_weight)),
        ),
        norm_score=True,
        limit=int(retrieval_config.get("top_k", DEFAULT_RETRIEVAL_CONFIG.top_k)),
        output_fields=[
            "chunk_id",
            "doc_id",
            "content",
            "file_title",
            "title",
            "parent_title",
            "item_name",
            "version",
            "doc_type",
            "section_type",
            "product_line",
            "language",
            "source_priority",
            "tenant_id",
            "department_id",
            "visibility",
        ],
    )
    result = resp[0] if resp else []
    if result or not metadata_expr:
        return result

    # 优化: 优化的原因是旧文档可能没有新增业务metadata字段，但安全过滤不能被放松，所以回退时只移除业务标签过滤，保留租户/部门/可见性约束。
    logger.info("metadata过滤未命中结果，回退到仅保留安全过滤和item_name的兼容检索逻辑")
    security_filters = {
        key: value
        for key, value in (retrieval_config.get("metadata_filters", {}) or {}).items()
        if key in {"tenant_id", "department_id", "visibility"}
    }
    fallback_expr = combine_expr(security_filter_expr, item_name_expr, build_metadata_expr(security_filters))
    fallback_requests = create_hybrid_search_requests(
        dense_vector=embeddings["dense"][0],
        sparse_vector=embeddings["sparse"][0],
        expr=fallback_expr or combine_expr(security_filter_expr, item_name_expr) or None,
        limit=int(retrieval_config.get("top_k", DEFAULT_RETRIEVAL_CONFIG.top_k)),
    )
    fallback_resp = hybrid_search(
        client=milvus_client,
        collection_name=milvus_config.chunks_collection,
        reqs=fallback_requests,
        ranker_weights=(
            float(retrieval_config.get("dense_weight", DEFAULT_RETRIEVAL_CONFIG.dense_weight)),
            float(retrieval_config.get("sparse_weight", DEFAULT_RETRIEVAL_CONFIG.sparse_weight)),
        ),
        norm_score=True,
        limit=int(retrieval_config.get("top_k", DEFAULT_RETRIEVAL_CONFIG.top_k)),
        output_fields=[
            "chunk_id",
            "doc_id",
            "content",
            "file_title",
            "title",
            "parent_title",
            "item_name",
            "version",
            "doc_type",
            "section_type",
            "product_line",
            "language",
            "source_priority",
            "tenant_id",
            "department_id",
            "visibility",
        ],
    )
    return fallback_resp[0] if fallback_resp else []


def node_search_embedding(state):
    """
    节点功能：进行向量内容检索
    主要作用: 问题 -> 查询chunks切片
    达到目标: {"embedding_chunks":[chunks]}
    """
    print("---量内容检索 开始处理---")
    add_running_task(state["session_id"], sys._getframe().f_code.co_name, state.get("is_stream"))
    rewritten_query = state.get("rewritten_query")
    item_names = state.get("item_names")
    retrieval_config = _get_retrieval_config(state)
    retrieval_config["query_type"] = state.get("query_type", "")

    embedding_chunks = _search_chunks(
        rewritten_query,
        item_names,
        retrieval_config,
        state.get("security_filter_expr", ""),
    )
    logger.info(f"向量混合检索结果:{embedding_chunks}")
    add_done_task(state["session_id"], sys._getframe().f_code.co_name, state.get("is_stream"))

    print("---量内容检索 处理结束---")
    return {"embedding_chunks": embedding_chunks}


if __name__ == "__main__":
    test_state = {
        "session_id": "test_search_embedding_001",
        "rewritten_query": "HAK 180 烫金机使用说明",
        "item_names": ["HAK 180 烫金机"],
        "is_stream": False,
    }

    print("\n>>> 开始测试 node_search_embedding 节点...")
    try:
        result = node_search_embedding(test_state)
        logger.info(f"检索结果汇总：{result}")
    except Exception as e:
        logger.error(f"测试运行失败: {e}", exc_info=True)
