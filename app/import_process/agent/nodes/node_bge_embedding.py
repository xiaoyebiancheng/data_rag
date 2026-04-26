import sys
import os
from typing import Any, List, Dict

from app.import_process.agent.state import ImportGraphState
from app.lm.embedding_utils import get_bge_m3_ef, generate_embeddings
from app.utils.task_utils import add_running_task,add_done_task
from app.core.logger import logger

# ==========================================
# BGE-M3向量化核心节点
# 核心能力：将文本切片转换为稠密/稀疏双向量，为Milvus向量检索提供数据基础
# 依赖模型：BAAI/bge-m3（多语言、多粒度，同时支持语义/关键词检索）
# 向量说明：
#   1. 稠密向量：1024维固定长度，记录文本深层语义信息，用于语义相似度匹配
#   2. 稀疏向量：变长键值对，记录文本关键词/特征位置，用于关键词精准匹配
# 核心设计：
#   - 单例模型：避免重复加载模型，节省显存/时间
#   - 批量处理：分批生成向量，防止大批次导致的显存溢出
#   - 文本增强：拼接商品名+切片内容，强化核心特征，提升检索准确性
# ==========================================
def node_bge_embedding(state: ImportGraphState) -> ImportGraphState:
    """
    LangGraph核心节点：BGE-M3文本向量化处理
    主流程（串行执行，全流程异常隔离）：
        1. 输入校验：验证chunks有效性，核心数据缺失则终止当前节点
        2. 模型初始化：获取BGE-M3单例模型实例，避免重复加载
        3. 批量向量化：分批拼接文本、生成双向量，为切片绑定向量字段
        4. 状态更新：将带向量的chunks更新回全局状态，供下游Milvus入库节点使用
    参数：
        state: ImportGraphState - 流程全局状态对象，包含上游传入的chunks、task_id等数据
    返回：
        ImportGraphState - 更新后的状态对象，chunks字段新增dense_vector/sparse_vector
    异常处理：
        节点内所有异常均捕获，不终止整体LangGraph流程，仅记录错误日志
    """
    # 获取当前节点名称，用于日志和任务状态记录
    current_node = sys._getframe().f_code.co_name
    logger.info(f">>> 开始执行LangGraph节点：{current_node}")

    # 标记任务运行状态，用于任务监控/前端进度展示
    add_running_task(state.get("task_id", ""), current_node)
    logger.info("--- BGE-M3 文本向量化处理启动 ---")

    try:
        # 获取要生成向量的chunks
        chunks=state['chunks']
        if not chunks or not isinstance(chunks, list):
            logger.warning("chunks数据无效，请检查输入数据")
            raise ValueError("chunks数据无效，请检查输入数据")
        # 给每个chunk生成向量
        # 获取嵌入式模型的客户端
        # 批量生成向量
        """
        1. 什么内容需要生产向量
        假如只把chunks生成向量, (item_name 华为手机 title 开机方式 content 长按开机)
                                        华为手机        充电器         什么开机的问题
        导致容易匹配失败,要加入其他数据(让主谓宾都有) item_name  content(title+内容)
        f"商品名:item_name,介绍:content"
        有中文尽量使用中文的标点符号!
        原则:核心词前置(前置集中力,权重 前128token)
        2. 列表能放一个字符串
         for - chunk - [content,content,content] -> 生成向量
         8192 - token -5
        """
        final_chunks = [] #存储处理完的chunks ->带有向量
        batch_size = 5 # 需要迅速按  上下文窗口(token)/块的大小
        for i in range(0, len(chunks), batch_size):
            batch_items = chunks[i:i+batch_size]

            current_texts = []
            for item in batch_items:
                # 获取当前chunk的文本内容
                item_name = item['item_name']
                item_content = item['content']
                # 添加商品名称
                item_text = f"商品名:{item_name},内容介绍:{item_content}"
                # 添加到当前批次的文本列表
                current_texts.append(item_text)
            # 当前批次生产的向量
            result = generate_embeddings(current_texts)
            for i, chunk in enumerate(batch_items):
                # 完善chunk的属性添加稠密和稀疏向量
                chunk_item=chunk.copy()
                chunk_item['dense_vector'] = result['dense'][i]
                chunk_item['sparse_vector'] = result['sparse'][i]
                final_chunks.append(chunk_item)

        state['chunks'] = final_chunks
        logger.info(f"--- BGE-M3 向量化处理完成，共处理 {len(final_chunks)} 条文本切片 ---")
        add_done_task(state.get("task_id", ""), current_node)
    except Exception as e:
        # 捕获节点所有异常，记录错误堆栈，不中断整体流程
        logger.error(f"BGE-M3向量化节点执行失败：{str(e)}", exc_info=True)

    # 返回更新后的状态对象，传递至下游节点
    return state