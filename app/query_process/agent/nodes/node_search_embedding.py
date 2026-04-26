import sys
import os
from app.utils.task_utils import add_running_task, add_done_task
from app.lm.embedding_utils import generate_embeddings
from app.clients.milvus_utils import create_hybrid_search_requests, hybrid_search, get_milvus_client
from app.core.logger import logger
from dotenv import load_dotenv, find_dotenv

from conf.milvus_config import milvus_config

load_dotenv(find_dotenv())


def node_search_embedding(state):
    """
    节点功能：进行向量内容检索
    主要作用: 问题 -> 查询chunks切片
    达到目标: {"embedding_chunks":[chunks]}
    需要参数:
            {
                rewritten_query:重写的问题 -> 根据它查询
                item_name:[]  -> 明确的主体
            }
    """
    print("---量内容检索 开始处理---")
    add_running_task(state["session_id"], sys._getframe().f_code.co_name, state.get("is_stream"))
    # 搜索假设性答案
    # 1.先从state获取参数数据
    rewritten_query = state.get("rewritten_query")
    item_names = state.get("item_names")
    # 2.将重写问题生成对应的向量[稠密和稀疏]
    embeddings = generate_embeddings([rewritten_query])
    # 3.进行向量数据库的混合查询
    # 3.1 创建混合查询请求对象AnnSearchRequest
    # 查询条件:1.向量索引 2.item_name一定要在item_name里 混合查询的查询条件 item_name(字段) in [item_names]
    # 因为 item_name 是字符串字段。Milvus 的过滤表达式里，字符串值必须带引号。
    # 给每个 item_name 外面套上双引号,再把它们拼成逗号分隔的字符串
    item_name_str = ', '.join(f'"{item_name}"' for item_name in item_names)
    hybrid_search_requests = create_hybrid_search_requests(
        dense_vector=embeddings['dense'][0],
        sparse_vector=embeddings['sparse'][0],
        expr=f"item_name in [{item_name_str}]",
    )
    # 3.2 调用混合查询方法
    milvus_client = get_milvus_client()
    resp = hybrid_search(
        client=milvus_client,
        collection_name=milvus_config.chunks_collection,
        reqs=hybrid_search_requests,
        ranker_weights=(0.9, 0.1),
        norm_score=True,
        limit=5,
        output_fields=["chunk_id", "content", "file_title", "title", "parent_title", "item_name"],
    )
    # 得到结果格式
    """
        [
            [
                id,
                distance,
                entity:
                {
                    "chunk_id", "content", "file_title","title", "parent_title", "item_name"
                }
            ]
        ]
    """

    # 4.处理查询结果赋值 embedding_chunks属性即可
    embedding_chunks = resp[0] if resp else []
    logger.info(f"向量混合检索结果:{embedding_chunks}")
    # ...
    add_done_task(state["session_id"], sys._getframe().f_code.co_name, state.get("is_stream"))

    print("---量内容检索 处理结束---")
    return {"embedding_chunks": embedding_chunks}


if __name__ == "__main__":
    # 模拟测试数据
    test_state = {
        "session_id": "test_search_embedding_001",
        "rewritten_query": "HAK 180 烫金机使用说明",  # 模拟改写后的查询
        "item_names": ["HAK 180 烫金机"],  # 模拟已确认的商品名
        "is_stream": False
    }

    print("\n>>> 开始测试 node_search_embedding 节点...")
    try:
        # 执行节点函数
        result = node_search_embedding(test_state)
        logger.info(f"检索结果汇总：{result}")
        # 验证结果
        chunks = result.get("embedding_chunks", [])
        print(f"\n>>> 测试完成！检索到 {len(chunks)} 条结果")

        if chunks:
            print("\n>>> Top 1 结果详情:")
            top1 = chunks[0]
            # 打印关键字段（注意：entity字段可能包含具体业务数据）
            print(f"ID: {top1.get('id')}")
            print(f"Distance: {top1.get('distance')}")
            entity = top1.get('entity', {})
            print(f"Item Name: {entity.get('item_name')}")
            print(f"Content Preview: {entity.get('content', '')[:100]}...")
        else:
            print("\n>>> 警告：未检索到任何结果，请检查 Milvus 数据或 item_names 是否匹配")

    except Exception as e:
        logger.error(f"测试运行失败: {e}", exc_info=True)
