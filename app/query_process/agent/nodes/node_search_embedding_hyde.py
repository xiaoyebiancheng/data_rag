# HyDE节点
import sys

from langchain_core.messages import HumanMessage

from app.clients.milvus_utils import create_hybrid_search_requests, get_milvus_client, hybrid_search
from app.core.load_prompt import load_prompt
from app.core.logger import logger
from app.lm.embedding_utils import generate_embeddings
from app.lm.lm_utils import get_llm_client
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


def step_1_create_hyde_doc(rewritten_query):
    """
    调用模型根据问题,生成一份答案
    """
    llm = get_llm_client()
    hyde_prompt = load_prompt("hyde_prompt", rewritten_query=rewritten_query)
    response = llm.invoke([HumanMessage(content=hyde_prompt)])
    hyde_doc = response.content
    logger.info(f"使用模型生成假设性答案:问题:{rewritten_query},答案:{hyde_doc}")
    return hyde_doc


def step_2_search_embedding_hyde(rewritten_query, hyde_doc, item_names, retrieval_config, security_filter_expr=""):
    query_str = rewritten_query + hyde_doc
    embeddings = generate_embeddings([query_str])
    item_name_expr = build_item_name_expr(item_names)
    metadata_expr = build_metadata_expr(retrieval_config.get("metadata_filters", {}))
    full_expr = combine_expr(security_filter_expr, item_name_expr, metadata_expr)

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
        logger.info(f"假设性问题混合查询结果:{result}")
        return result

    logger.info("HyDE metadata过滤未命中结果，回退到仅保留安全过滤和item_name的兼容检索逻辑")
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
    result = fallback_resp[0] if fallback_resp else []
    logger.info(f"假设性问题混合查询回退结果:{result}")
    return result


def node_search_embedding_hyde(state):
    """
    假设性答案:问题 -> lm -> 给一个假设性答案 -> 问题+假设性答案 -> 搜索
    节点功能：HyDE (Hypothetical Document Embedding)
    """
    print("---HyDE 开始处理---")
    add_running_task(state["session_id"], sys._getframe().f_code.co_name, state.get("is_stream"))
    rewritten_query = state.get("rewritten_query")
    item_names = state.get("item_names")
    retrieval_config = _get_retrieval_config(state)
    retrieval_config["query_type"] = state.get("query_type", "")
    logger.info(
        f"HyDE动态策略: query_type={state.get('query_type', '')}, "
        f"use_hyde={retrieval_config.get('use_hyde')}, retrieval_config={retrieval_config}"
    )
    if not retrieval_config.get("use_hyde", DEFAULT_RETRIEVAL_CONFIG.use_hyde):
        add_done_task(state["session_id"], sys._getframe().f_code.co_name, state.get("is_stream"))
        print("---HyDE 跳过处理---")
        return {"hyde_embedding_chunks": []}

    hyde_doc = step_1_create_hyde_doc(rewritten_query)
    resp = step_2_search_embedding_hyde(
        rewritten_query,
        hyde_doc,
        item_names,
        retrieval_config,
        state.get("security_filter_expr", ""),
    )
    add_done_task(state["session_id"], sys._getframe().f_code.co_name, state.get("is_stream"))

    print("---HyDE 处理结束---")
    return {"hyde_embedding_chunks": resp}
