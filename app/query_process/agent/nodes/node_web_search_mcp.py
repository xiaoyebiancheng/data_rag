import asyncio
import os
import json
import sys
from agents.mcp import MCPServerSse  # pip install openai-agents
from agents.mcp import MCPServerStreamableHttp  # pip install openai-agents

from app.conf.bailian_mcp_config import mcp_config
from app.utils.task_utils import add_running_task, add_done_task
from app.core.logger import logger

DASHSCOPE_BASE_URL_STREAMABLE = mcp_config.mcp_base_url
DASHSCOPE_API_KEY = mcp_config.api_key


async def mcp_call_streamable(query):
    """
    调用百炼的网络搜索工具
    :param query:
    :return:
    """
    # 1.创建MCPServerStreamableHttp对象
    search_mcp = MCPServerStreamableHttp(
        name="search_mcp",
        params={
            # 核心参数
            "url": DASHSCOPE_BASE_URL_STREAMABLE,
            "headers": {"Authorization": f"Bearer {DASHSCOPE_API_KEY}"},
            "timeout": 10,  # 连接超时时间
        },
        max_retry_attempts=3
    )
    # 2.连接 - 调用 - 关闭
    try:
        # 连接
        await search_mcp.connect()
        # 若需要获取工具
        # tool = await search_mcp.list_tools()
        # 调用
        result = await search_mcp.call_tool(
            tool_name="bailian_web_search",
            arguments={
                "query": query,
                "count": 5,
            }
        )
        return result
    finally:
        await search_mcp.cleanup()


def node_web_search_mcp(state):
    """
    节点功能，调用外部搜索引擎补充信息
    :param state:
    :return:
    """
    add_running_task(state["session_id"], sys._getframe().f_code.co_name, state["is_stream"])
    print("---node-web-search-mcp处理---")

    # 1. 获取问题 (rewritten_query)
    query = state.get("rewritten_query")

    # 2.调用streamable网络搜索方法
    result = asyncio.run(mcp_call_streamable(query))
    # 3.结果处理即可
    """
    {
        "isError": false,
        "content": [
            {
                "text": "{\"pages\":[{\"snippet\":\"一、整体盈利比例 1. 散户盈利比例 2025年A股散户整体盈利比例约为18.9%,亏损比例高达81.1%。这一数据在多个权威平台统计中保持一致,反映出市场“赚了指数亏了钱”的结构性特征[3][4]。 2. 全市场盈利比例 若包含机构投资者,全市场盈利账户比例为32.3%,平均收益率5.8%,人均盈利1.96-3.2万元;但散户平均收益率为-23.6%,人均亏损约2.1万元[4]。 二、盈利分化特征 1. 资金规模差异 - 小额账户(1万元以下)亏损比例接近99.9%,1万-10万元账户亏损比例98.7%。 - 高净值投资者(500万元以上)盈利比例高达97%-99.1%,显著高于散户[9][10]。 2. 行业与策略分化 - 科技、高端制造等高景气板块贡献主要盈利,部分龙头股全年涨幅超100%。 - 结构性行情下,约40%个股下跌,投资者收益高度依赖选股能力[5][8]。 三、市场背景与趋势 1. 指数表现与实际收益背离 2025年上证指数全年上涨约19%,但多数散户未能分享涨幅,凸显市场有效性不足[3][4]。 2. 政策与资金影响 监管层推动分红新规、退市制度完善等措施,但中小投资者因信息劣势和交易习惯,盈利难度仍较高[11]。\",
                                        \"hostname\":\"东方财富网\",
                                        \"hostlogo\":\"https://search-operate.cdn.bcebos.com/bd185a0552b248a6154062a682af6c7c.png\",
                                        \"title\":\"2025年元旦至今,还有盈利的应该没有几个人了! \",
                                        \"url\":\"https://caifuhao.eastmoney.com/news/20260330095509378458170\"},
                                    {\"snippet\":\"和讯首页|手机和讯 登录注册 股票客户端 Android 股票客户端 iPhone\",\"hostname\":\"和讯网\",\"hostlogo\":\"https://img.alicdn.com/imgextra/i3/O1CN01VcUfI91cc0kCH3Gt2_!!6000000003620-73-tps-32-32.ico\",\"title\":\"行情中心-和讯网 国内全面的即时行情数据服务中心\",\"url\":\"https://quote.hexun.com/\"},{\"snippet\":\"4月15日和16日,A股确实迎来了高光时刻,深成指和创业板都刷新了阶段新高,创业板更是直接突破3600点,创下2015年6月以来的近11年最高点位。\",\"hostname\":\"网易\",\"hostlogo\":\"https://ss1.baidu.com/6ONXsjip0QIZ8tyhnq/it/u=1534926245,1016405979&fm=195&app=88&f=JPEG?w=200&h=200\",\"title\":\"创业板创11年新高!本轮牛市普通人没赚到钱?核心逻辑一次性说透\",\"url\":\"https://www.163.com/dy/article/KR31K8G105568PET.html\"}],\"request_id\":\"8daf14ae-dbbc-43df-8f60-f8c0149f473f\",\"tools\":[],\"status\":0}",
                "type": "text"
            }
        ]
    }
    """
    web_documents = json.loads(result.content[0].text).get("pages", [])

    logger.info(f"百炼mcp搜索结果:{web_documents}")
    add_done_task(state["session_id"], sys._getframe().f_code.co_name, state["is_stream"])
    print("---node-web-search-mcp处理结束---")
    return {"web_search_docs": web_documents}

from dotenv import load_dotenv

if __name__ == '__main__':
    load_dotenv()
    test_state = {
        "session_id": "test_web_search_mcp_001",
        "rewritten_query": "HAK 180 在出厂默认状态下，若想在纸张上只把烫金膜转印到顶部 50 mm–170 mm 的局部区域，应在操作面板上如何设置",
        "is_stream":True
    }

    # 调用 websearch_node 函数
    result_state = node_web_search_mcp(test_state)

    # 验证结果
    print("测试结果:")
    print(f"查询内容: {test_state.get('rewritten_query')}")

    # 输出搜索结果
    search_results = result_state.get('web_search_docs', [])
    print(f"搜索结果数量: {len(search_results)}")
    print("search_results", search_results)