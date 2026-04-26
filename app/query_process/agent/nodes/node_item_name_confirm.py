import sys
import os
import json
import logging
from typing import List, Dict, Any, Optional
from langchain_core.messages import SystemMessage, HumanMessage
from mpmath import limit

from app.core.load_prompt import load_prompt
from app.query_process.agent.state import QueryGraphState
from app.utils.task_utils import add_running_task, add_done_task
from app.clients.mongo_history_utils import get_recent_messages, save_chat_message, update_message_item_names
from app.lm.lm_utils import get_llm_client
from app.lm.embedding_utils import generate_embeddings
from app.clients.milvus_utils import get_milvus_client, create_hybrid_search_requests, hybrid_search
from dotenv import load_dotenv, find_dotenv
from app.core.logger import logger
from conf.milvus_config import milvus_config

load_dotenv(find_dotenv())


def step_3_llm_item_name_and_rewrite_query(original_query, history_chats):
    """
    根据历史记录 -> 识别item_names 和重写问题
    :param original_query:原有问题
    :param history_chats:历史记录
    :return: {item_name =[],rewritten_query:问题}
    """
    # 1.准备提示词
    history_text = ""
    for chat in history_chats:
        history_text += (f"聊天角色:{chat['role']},回答内容:{chat['text']},"
                         f"重写问题:{chat['rewritten_query']},"
                         f"关联主体:{','.join(chat.get('item_names', []))},时间:{chat['ts']}\n")
    prompt = load_prompt("rewritten_query_and_itemnames", history_text=history_text, query=original_query)
    # 2.调用模型
    lm_client = get_llm_client(json_mode=True)
    # system -> 模型的角色边界 -> 应该是不变的! [角色,规则,格式]
    # user  -> 每次任务提示 -> 多条动态调整 ! [提问/聊天]
    # 事实上,不嫌麻烦,可以把模型的角色和边界写到user功能也是一样的
    message = [
        HumanMessage(content=prompt)
    ]
    response = lm_client.invoke(message)
    # 怎么确保,模型一定返回格式化数据?json! 1.设置json格式化 2.提示词中明确  3.一定给模型参考示例 4.做好返回格式的校验
    # 3.解析结果
    content = response.content
    # json -> ```json  json  ```
    if content.startswith("```json"):
        content = content.replace("```json", "").replace("```", "")
    dict_content = json.loads(content)

    if "item_names" not in dict_content:
        dict_content["item_names"] = []
    if "rewritten_query" not in dict_content:
        dict_content["rewritten_query"] = original_query  # 原问题
    # 4.封装返回
    logger.info(f"已完成问题的重写和item_name的提取!结果为:{dict_content}")
    return dict_content


def step_4_query_milvus_item_names(item_names):
    # 查询向量数据库,进行item_name的确定
    final_results = []
    # 1. 获取milvus的客户端
    milvus_client = get_milvus_client()
    # 2. 将item_names转成向量(稠密和稀疏) [循环]
    embeddings = generate_embeddings(item_names)
    # 3. 混合查询(创建稠密和稀疏的AnnSearchRequest || 设置权重重排 || 进行混合查询 )
    for index, item_name in enumerate(item_names):
        # 3.1 获取当前item_name对应的向量
        dense_vector = embeddings["dense"][index]
        sparse_vector = embeddings["sparse"][index]
        # 3.2 拼对应的AnnSearchRequest
        reqs = create_hybrid_search_requests(dense_vector, sparse_vector)
        # 3.3 定义权重重排,进行混合检索
        response = hybrid_search(
            client=milvus_client,
            collection_name=milvus_config.item_name_collection,
            reqs=reqs,
            ranker_weights=(0.8, 0.2),
            norm_score=True  # 0-1
        )
        # 3.4 结果解析
        match = []  # 当前item对应的匹配结果
        if response and len(response) > 0:
            for hit in response[0]:
                entity = hit.get("entity", {})
                hit_name = entity.get("item_name", "")
                # pymilvus Hit 中分数字段是 distance（等同 score 属性），不是 dict key "score"
                score = hit.get("distance", 0)
                if hit_name:
                    match.append({
                        "item_name": hit_name,
                        "score": score
                    })
    # 4. 提取查询结果封装返回的数据格式
        final_results.append({
            "extracted": item_name, # 模型给的
            "match": match  # 查询到的
        })
    # 5. 封装返回数据
    logger.info(f"已完成item_name的查询和结果返回!结果为:{final_results}")
    return final_results


def step_5_confirmed_and_optional_item_names(query_milvus_results):
    """
    通过向量数据库查询的item_name,根据分数归纳确定和可选的item_name列表
    :param query_milvus_results:
    :return:{confirmed_item_names:[],option_item_names:[]}
    评分规则: 用0.85 和 0.60 (根据开发实际情况)
    思路 :1.循环处理每个item_name列表和分 2.高分只要1个 3.可选 可以要2   4.不区分extracted:item_name装到对应的确认或者可选集合中
    """
    # 1.准备两个列表 确认  可选的
    confirmed_item_names=[] #确认
    options_item_names = [] #可选
    # 2.循环处理元数据 query_milvus_results
    for item_name_meta in query_milvus_results:
        extracted_name = item_name_meta.get("extracted")
        matches = item_name_meta.get("match", [])
        # 3.进行分数排序(倒序) || 列表推导式 提取0.85 || 0.6
        matches.sort(key=lambda x: x.get("score", 0), reverse=True)
        high_score_matches = [x for x in matches if x.get("score",0)>=0.85]
        middle_score_matches = [x for x in matches if x.get("score", 0) >= 0.6]
        # 4.处理高分的列表 只有一个则获取一个1 || 多个,若与extract中item_name一致则获取or 选最高分的1个
        if len(high_score_matches) ==1:
            confirmed_item_names.append(high_score_matches[0].get("item_name"))
            continue
        elif len(high_score_matches)>1:
            # 同名不一定分最高,优先考虑同名的
            same_name_item =None
            for item in high_score_matches:
                if item.get("item_name") == extracted_name:
                    same_name_item= item
                    break
            if not same_name_item:
                same_name_item = high_score_matches[0]
            confirmed_item_names.append(same_name_item.get("item_name"))
            continue
        # 5.处理可选分数列表,给用户返回提示,可以多带几个!选2个
        if len(middle_score_matches)>0:
            for item in middle_score_matches[:2]:
                options_item_names.append(item.get("item_name",""))
            continue
        logger.info(f"未找到匹配的item_name!忽略:{extracted_name}")
    # 6.处理返回结果(set 去重)
    result = {
        "confirmed_item_names":list(set(confirmed_item_names)),
        "options_item_names":list(set(options_item_names))
    }
    logger.info(f"处理结果:{result}")
    return result

def step_6_deal_list(state,item_results,history_chats,rewritten_query):
    """
    根据集合类型中数据,判定是否要赋值answer内容
    :param state:
    :param item_results:
    :param history_chats:
    :param rewritten_query:
    :return:
    """
    # 1.先获取两个集合(确认|可选)
    confirmed_item_names = item_results.get("confirmed_item_names", [])
    options_item_names = item_results.get("options_item_names", [])
    # 2.确认集合有数据(处理)
    if len(confirmed_item_names)>0:
        # 2.1 更新下聊天记录 -> item_names ->confirmed_item_names(空着)
        # 2.2 修改和存储state状态
        state['item_names'] = confirmed_item_names
        state['rewritten_query'] = rewritten_query
        state['history'] = history_chats
        if "answer" in state:
            del state["answer"]
        logger.info(f"有确定的item_name:{confirmed_item_names}")
        return state
    # 3.确认集合没数据,处理可选集合
    if len(options_item_names)>0:
        option_names = ''.join(options_item_names)
        answer = f"您是想咨询以下哪个商品:{option_names}?请下次提问明确商品名称!"
        state['answer'] = answer
        logger.info(f"有可选的item_name:{options_item_names}")
        return state
    # 4.都没数据 (处理)
    answer = "没有匹配的商品名,请重新提问!"
    state['answer'] = answer
    logger.info(f"没有匹配的item_name")
    return state

def node_item_name_confirm(state):
    """
    节点功能：确认用户问题中的核心商品名称。
    # 核心目标: 1. 提取[item_name] (大模型从历史对话+本次提问 获取 -> item_name ->向量数据库 ->打分 ->ABC)
              2.利用模型重写用户的问题,确保后续查询召回率更高!!
    # 核心参数: state['original_query' -> 用户的原问题] || session_id
    # 响应数据: item_names:List[str] # 提取出的商品名称
    #          rewritten_query:str #改写后的问题
    #          history:list        #历史对话记录
    # 1.获取历史条件记录(作为依据)
    # 2.保存当前次的聊天记录
    # 3.利用模型lm -> 1.提取item_name  2.重写提问问题
    # 4.进行item_name的向量数据库查询
    # 5.对item_name结果进行打分分类处理 A[确认集合] B[可选集合]
    # 6.处理确认和可选集合! 有确认 -> 继续下个节点  || 有可选or无item_name -> answer赋值结果
    # 7.补充state状态item_names  rewritten_query   history
    """
    print(f"---node_item_name_confirm---开始处理")
    # 记录任务开始
    add_running_task(state["session_id"], sys._getframe().f_code.co_name, state["is_stream"])

    # 1.获取历史条件记录(作为依据)
    history_chats = get_recent_messages(state["session_id"], limit=10)

    # 3.利用模型lm -> 1.提取item_name  2.重写提问问题
    # 参数: state['original_query'] || history_chats
    # 响应:{item_name:[],rewritten_query:str}
    # 为啥要重写?
    """
        1.消除指代歧义  他 ta 它 不明确! 明确查询主体,item_name
        2.补全上下文   他的问题需要有历史记录支持!
        3.去掉口语和冗余
        4.润色问题增加召回率   模型查询的时候也会更加精准
    """
    item_names_and_rewritten_query = step_3_llm_item_name_and_rewrite_query(state["original_query"], history_chats)
    item_names = item_names_and_rewritten_query.get("item_names", [])
    rewritten_query = item_names_and_rewritten_query.get("rewritten_query", "")
    # 4.进行item_name的向量数据库查询
    # milvus向量查询item_names -> 模型提取  不一定跟我们向量数据库的完全相同
    # 参数: item_names = [华为,苹果]
    # 返回: 华为 -> 向量数据库中item_names(向量查询)   苹果-> 向量数据库中item_names(向量查询)
    # 返回格式 : [
    #           {extracted:(模型提取的item_name),match:[{item_name:向量数据库中item_name,score:0.9},{...,...}]},
    #           {extracted:(模型提取的item_name),match:[{item_name:向量数据库中item_name,score:0.9},{...,...}]}
    #            ]
    item_results={}
    if len(item_names) > 0:
        query_milvus_results = step_4_query_milvus_item_names(item_names)
        # 5.对item_name结果进行打分分类处理 A[确认集合] B[可选集合]
        item_results = step_5_confirmed_and_optional_item_names(query_milvus_results)

    # 6.处理确认和可选集合! 有确认 -> 继续下个节点  || 有可选or无item_name -> answer赋值结果
    state= step_6_deal_list(state,item_results,history_chats,rewritten_query)
    # # 7.记录本次的聊天对话(answer回答) !!!!挪到output节点写

    # 保存当前次的聊天记录(聊天记录一定会保存的! 提问内容)
    save_chat_message(
        session_id=state["session_id"],
        role="user",
        text=state["original_query"],
        rewritten_query=state.get("rewritten_query", ""),
        item_names=state.get("item_names", []),
        image_urls=[]
    )
    # 记录任务结束
    add_done_task(state["session_id"], sys._getframe().f_code.co_name, state["is_stream"])
    print(f"---node_item_name_confirm---处理结束")

    return state

if __name__ == "__main__":
    # 模拟输入状态
    mock_state = {
        "session_id": "test_session_001",
        "original_query": "他好用吗？哈哈哈 哈哈哈",
        "is_stream": False
    }

    print(">>> 开始测试 node_item_name_confirm...")
    try:
        # 运行节点
        result_state = node_item_name_confirm(mock_state)

        print("\n>>> 测试完成！最终状态:")
        print(json.dumps(result_state, indent=2, ensure_ascii=False,default=str))

        # 简单验证
        if result_state.get("item_names"):
            print(f"\n[PASS] 成功提取并确认商品名: {result_state['item_names']}")
        else:
            print(f"\n[WARN] 未确认到商品名 (可能是向量库无匹配或LLM未提取)")

    except Exception as e:
        print(f"\n[FAIL] 测试运行出错: {e}")
