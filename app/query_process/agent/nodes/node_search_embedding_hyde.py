# HyDE节点
import sys

from langchain_core.messages import HumanMessage

from app.utils.task_utils import add_running_task, add_done_task
from app.lm.lm_utils import *
from app.lm.embedding_utils import *
from app.clients.milvus_utils import *
from app.core.logger import logger
from app.core.load_prompt import load_prompt
from dotenv import load_dotenv, find_dotenv

load_dotenv(find_dotenv())


def step_1_create_hyde_doc(rewritten_query):
    """
    调用模型根据问题,生成一份答案
    :param rewritten_query: 问题
    :return: 答案字符串
    """
    llm = get_llm_client()

    # 加载提示词
    hyde_prompt = load_prompt("hyde_prompt", rewritten_query=rewritten_query)

    message = [
        HumanMessage(content=hyde_prompt)
    ]
    # 发起请求
    response = llm.invoke(message)
    hyde_doc = response.content
    logger.info(f"使用模型生成假设性答案:问题:{rewritten_query},答案:{hyde_doc}")
    return hyde_doc


def step_2_search_embedding_hyde(rewritten_query, hyde_doc, item_names):
    """
    根据问题+假设性答案查询向量数据库,进行混合索引
    :param rewritten_query:
    :param hyde_doc:
    :param item_names:
    :return: [[] -> 结果 id 分数 实体列信息]
    """
    # 1.拼接重写问题,lm生成的假设性答案
    query_str = rewritten_query + hyde_doc
    # 2.拼接查询字符串生成对应向量
    embeddings = generate_embeddings([query_str])
    # 3.生成查询AnnSearchRequest列表
    item_name_str = ', '.join(f'"{item_name}"' for item_name in item_names)
    hybrid_search_requests = create_hybrid_search_requests(
        dense_vector=embeddings['dense'][0],
        sparse_vector=embeddings['sparse'][0],
        expr=f"item_name in [{item_name_str}]",
    )
    # 4.进行混合查询处理
    milvus_client = get_milvus_client()
    resp = hybrid_search(
        client=milvus_client,
        collection_name=milvus_config.chunks_collection,
        reqs=hybrid_search_requests,
        ranker_weights=(0.9, 0.1),
        norm_score=True,
        output_fields=["chunk_id", "content", "file_title", "title", "parent_title", "item_name"],
    )
    # 5.处理返回结果
    result = resp[0] if resp else []
    logger.info(f"假设性问题混合查询结果:{result}")
    return result


def node_search_embedding_hyde(state):
    """
    假设性答案:问题 -> lm -> 给一个假设性答案 -> 问题+假设性答案 -> 搜索
    节点功能：HyDE (Hypothetical Document Embedding)
    先让 LLM 生成假设性答案，再对答案进行向量检索，提高召回率。
    """
    print("---HyDE 开始处理---")
    add_running_task(state["session_id"], sys._getframe().f_code.co_name, state.get("is_stream"))
    # 1. 提取参数(item_names || rewritten_query)
    rewritten_query = state.get("rewritten_query")
    item_names = state.get("item_names")
    # 2. 调用LLM生成假设性答案rewritten_query
    hyde_doc = step_1_create_hyde_doc(rewritten_query)
    # 3. 问题+答案,进行向量检索(混合检索)
    resp = step_2_search_embedding_hyde(rewritten_query, hyde_doc, item_names)
    # 4. 赋值和返回结果 hyde_embedding_chunks
    add_done_task(state["session_id"], sys._getframe().f_code.co_name, state.get("is_stream"))

    print("---HyDE 处理结束---")
    return {"hyde_embedding_chunks": resp}
