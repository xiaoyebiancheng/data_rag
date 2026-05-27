# 导入核心依赖：数据类、环境变量读取、路径处理
import sys

from dotenv import load_dotenv

from app.core.logger import logger
from app.utils.task_utils import add_running_task, add_done_task
from app.lm.reranker_utils import get_reranker_model
from app.query_process.retrieval.query_profile import DEFAULT_RETRIEVAL_CONFIG

# 提前加载.env配置文件（保持和原代码一致，只需执行一次）
load_dotenv()

# -----------------------------
# Rerank / TopK 全局常量（不从 state 读取）
# -----------------------------
# 动态 TopK 硬上限：最多取前 N 条（<=10）
RERANK_MAX_TOPK: int = 10
# 最小 TopK：至少保留前 N 条（>=1，且 <= RERANK_MAX_TOPK）
RERANK_MIN_TOPK: int = 1
# 断崖阈值（相对）当下一个结果比前一个结果低了 25% 及以上时，认为从这里开始质量明显下降，可以截断
# 原理:看相邻 gap：
# 0.92 -> 0.88: (0.04 / 0.92) = 0.043
# 0.88 -> 0.84: (0.04 / 0.88) = 0.045
# 0.84 -> 0.60: (0.24 / 0.84) = 0.286
# 这里第三个 gap 超过了 0.25，说明：前三条是一档,从第四条开始明显掉了一层,所以可以只保留前三条。
RERANK_GAP_RATIO: float = 0.25
# 断崖阈值（绝对）
RERANK_GAP_ABS: float = 0.5  # 0.93 , 0.92 , 0.91 ,[断崖] 0.39 (只取前三个)


def step_1_merge_rrf_mcp(state):
    """
    进行rrf+mcp的数据整合
    :param state:
    :return:
    """
    # 1.state获取不同路的数据
    rrf_chunks = state.get("rrf_chunks", [])
    web_search_docs = state.get("web_search_docs", [])
    # 2.准备一个列表容器
    chunks_list = []
    # 3.循环进行数据添加
    # 3.1 local rrf
    for chunk in rrf_chunks:
        entity = chunk.get("entity")
        chunk_id = entity.get("chunk_id")
        content = entity.get("content")
        title = entity.get("title")
        parent_title = entity.get("parent_title", "")
        file_title = entity.get("file_title", "")
        item_name = entity.get("item_name", "")
        doc_id = entity.get("doc_id", "")
        version = entity.get("version", 0)
        raw_score = chunk.get("distance", 0.0)
        chunks_list.append({
            "text": content,
            "chunk_id": chunk_id,
            "title": title,
            "parent_title": parent_title,
            "file_title": file_title,
            "item_name": item_name,
            "doc_id": doc_id,
            "version": version,
            "url": "",
            "source": "local"
            ,
            "retrieval_score": raw_score,
        })
    # 3.2 web mcp
    for doc in web_search_docs:
        text = doc.get("snippet")
        url = doc.get("url")
        title = doc.get("title")
        chunks_list.append({
            "text": text,
            "chunk_id": "",
            "title": title,
            "url": url,
            "source": "web"
        })
    logger.info(f"多路数据融合.最终结果为: {chunks_list}")
    return chunks_list


def step_2_rerank_doc_list(doc_list, state):
    """
    使用rerank进行精排
    :param doc_list:
    :param state:
    :return:
    """
    # 1.获取原有的问题
    rewritten_query = state.get("rewritten_query") or state.get("original_query")
    # 2.获取问题对应的所有答案
    text_list = [doc.get("text") for doc in doc_list]
    # 3.加载rerank模型
    rerank = get_reranker_model()
    # 4.处理数据 设置 问题+答案 成对-> 装到列表中,调用打分方法
    questions_pairs = [[rewritten_query, text] for text in text_list]
    # normalize默认为False, True表示归一化,缩放到0-1
    scores = rerank.compute_score(questions_pairs, normalize=True)
    # 5.将原有的数据添加到对应的分
    doc_list_with_score = []

    for score, item in zip(scores, doc_list):
        item["score"] = score
        item["rerank_score"] = score
        doc_list_with_score.append(item)
    doc_list_with_score.sort(key=lambda x: x["score"], reverse=True)
    logger.info(f"多路数据排序和打分.最终结果为: {doc_list_with_score}")
    return doc_list_with_score


def step_3_topk_and_gap(rerank_score_list):
    """
    进行再次的算法筛选,取出动态的topk元素
    :param rerank_score_list:
    :return:
    """
    max_topk = RERANK_MAX_TOPK  # 至多获取的元素数量
    min_topk = RERANK_MIN_TOPK  # 至少获取的元素数量
    gap_ratio = RERANK_GAP_RATIO  # 断崖阈值百分比（相对）
    gap_abs = RERANK_GAP_ABS  # 断崖阈值分差（绝对）
    # 思路:两个两个比,所以用双指针
    # 1.思考最大截取数量
    # topk不应该大于列表长度
    topk = min(max_topk, len(rerank_score_list))
    # 2.循环处理数据列表,进行双指针处理和比较
    if topk > min_topk:
        # 正常循环min-1,topk-1
        for index in range(min_topk - 1, topk - 1):
            # 双指针
            score_1 = rerank_score_list[index].get("score", 0.0)
            score_2 = rerank_score_list[index + 1].get("score", 0.0)
            gap = score_1 - score_2
            # 除法分母不为0   score不能为负
            rel = gap / (abs(score_1) + 1e-6)
            if rel >= gap_ratio or gap >= gap_abs:
                # 断崖,跳出循环
                logger.info(f"数据集合{index}和{index + 1}的位置发生了断崖,结束循环!")
                topk = index + 1  # index下标是从0开始的,topk对应的是截取长度
                break
    # 3.截取确定的数量topk
    topk_doc_list = rerank_score_list[:topk]
    logger.info(f"最终截取长度为:{topk},截取内容为: {topk_doc_list}")
    return topk_doc_list


def step_3_topk_by_config(doc_list, rerank_top_n):
    topk = max(1, min(int(rerank_top_n), len(doc_list)))
    final_doc_list = doc_list[:topk]
    logger.info(f"根据动态配置截取TopK,长度为:{topk},内容为:{final_doc_list}")
    return final_doc_list


def _get_retrieval_config(state):
    retrieval_config = dict(DEFAULT_RETRIEVAL_CONFIG.__dict__)
    retrieval_config.update(state.get("retrieval_config", {}) or {})
    return retrieval_config


def node_rerank(state):
    """
    节点作用 : rrf + mcp -> 精排序rerank ->chunk - 打分 -> 算法 -> top k
    算法理解: 算法 (最多10条  最少1条  相对0.25  绝对 0.5 )
    节点功能：使用 Cross-Encoder 模型对 RRF 后的结果进行精确打分重排。
    """
    print("---Rerank处理---")
    add_running_task(state["session_id"], sys._getframe().f_code.co_name, state.get("is_stream"))
    retrieval_config = _get_retrieval_config(state)
    logger.info(
        f"Rerank动态策略: query_type={state.get('query_type', '')}, "
        f"use_rerank={retrieval_config.get('use_rerank')}, retrieval_config={retrieval_config}"
    )
    # 1.非同源路的结果合并(rrf+mcp) 放到一个集合中
    """
        [
            rrf = {id:chunk_id,distance:0.x,entity:{chunk_id,content,title}}
            mcp = {snippet:内容,title:标题,url:关联的文章或图片的地址}
            {
                text:content  snippet:内容
                chunk_id:rrf-chunk_id  mcp None
                title:title
                url:rrf None    mcp url
                source: local->rrf web->mcp 
            }
        ]   
        
    """
    doc_list = step_1_merge_rrf_mcp(state)

    if not retrieval_config.get("use_rerank", DEFAULT_RETRIEVAL_CONFIG.use_rerank):
        # 优化: 优化的原因是部分查询没有必要走最重的精排链路，直接复用RRF融合结果可减少时延，同时保持主流程节点不变。
        final_doc_list = step_3_topk_by_config(doc_list, retrieval_config.get("top_k", DEFAULT_RETRIEVAL_CONFIG.top_k))
        state["reranked_docs"] = final_doc_list
        add_done_task(state['session_id'], sys._getframe().f_code.co_name, state.get("is_stream"))
        return state

    # 2.启用rerank进行精排 (数据和分)
    rerank_score_list = step_2_rerank_doc_list(doc_list, state)

    """
        {
            text:content  snippet:内容
            chunk_id:rrf-chunk_id  mcp None
            title:title
            url:rrf None    mcp url
            source: local->rrf web->mcp 
            score:rerank打的分
        }
    """

    # 3.启动算法进行防断崖以及top_k处理
    final_doc_list = step_3_topk_and_gap(rerank_score_list)
    rerank_top_n = retrieval_config.get("rerank_top_n", DEFAULT_RETRIEVAL_CONFIG.rerank_top_n)
    final_doc_list = final_doc_list[:max(1, int(rerank_top_n))]
    state["reranked_docs"] = final_doc_list
    add_done_task(state['session_id'], sys._getframe().f_code.co_name, state.get("is_stream"))
    return state


if __name__ == "__main__":
    print("\n" + "=" * 50)
    print(">>> 启动 node_rerank 本地测试")
    print("=" * 50)

    # 1. 模拟数据
    # 1.1 RRF 本地文档数据
    mock_rrf_chunks = [
        {"entity": {"chunk_id": "local_1", "content": "RRF是一种倒数排名融合算法", "title": "算法介绍", "score": 0.9}},
        {"entity": {"chunk_id": "local_2", "content": "BGE是一个强大的重排序模型", "title": "模型介绍", "score": 0.8}},
        {"entity": {"chunk_id": "local_3", "content": "无关的测试文档内容", "title": "测试文档", "score": 0.1}}  # 预期低分
    ]

    # 1.2 MCP 联网搜索数据
    mock_web_docs = [
        {"title": "Rerank技术详解", "url": "http://web.com/1", "snippet": "Rerank即重排序，常用于RAG系统的第二阶段"},
        {"title": "无关网页", "url": "http://web.com/2", "snippet": "今天天气不错，适合出去游玩"}  # 预期低分
    ]

    mock_state = {
        "session_id": "test_rerank_session",
        "rewritten_query": "什么是RRF和Rerank？",  # 查询意图：想了解这两个算法
        "rrf_chunks": mock_rrf_chunks,
        "web_search_docs": mock_web_docs,
        "is_stream": False
    }

    try:
        # 运行节点
        result = node_rerank(mock_state)
        reranked = result.get("reranked_docs", [])

        print("\n" + "=" * 50)
        print(">>> 测试结果摘要:")
        print(f"输入文档总数: {len(mock_rrf_chunks) + len(mock_web_docs)}")
        print(f"输出文档总数: {len(reranked)}")
        print("-" * 30)

        print("最终排名:")
        for i, doc in enumerate(reranked, 1):
            print(f"Rank {i}: Source={doc.get('source')}, Score={doc.get('score'):.4f}, Text={doc.get('text')[:20]}...")

        # 验证逻辑：
        # 预期 "local_1", "local_2", "Rerank技术详解" 分数较高
        # 预期 "local_3", "无关网页" 分数较低，可能被截断或排在最后

        top1_score = reranked[0].get("score")
        if top1_score > 0:
            print("\n[PASS] Rerank 打分正常")
        else:
            print("\n[FAIL] Rerank 打分异常 (均为0或负数)")

        print("=" * 50)

    except Exception as e:
        logger.exception(f"测试运行期间发生未捕获异常: {e}")
