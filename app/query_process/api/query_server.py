from pathlib import Path
import uuid
import uvicorn
from fastapi import FastAPI, BackgroundTasks, Depends, HTTPException, Request
from fastapi.responses import FileResponse, StreamingResponse
from pydantic import BaseModel, Field
from starlette.middleware.cors import CORSMiddleware
from app.core.logger import logger
from app.clients.audit_log_repository import write_audit_log
from app.security.auth import AuthContext, get_auth_context

from app.utils.task_utils import *
from app.utils.sse_utils import create_sse_queue, SSEEvent, sse_generator
from app.clients.mongo_history_utils import *
from app.query_process.agent.main_graph import query_app
from app.query_process.agent.state import create_query_default_state
from app.utils.path_util import PROJECT_ROOT

# 6个接口   健康状态  返回页面 发起提问  sse长连接   查看历史消息   清空历史对话

# 定义fastapi对象
app = FastAPI(title="query service", description="掌柜智库查询服务！")
_session_security: dict[str, dict[str, str]] = {}

# 跨域配置
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


# 健康状态
@app.get("/health")
async def health():
    logger.info(f"触发后台检测检查接口,数据一切正常")
    return {"status": "ok"}


# 返回chat.html
@app.get("/chat.html")
async def chat_html():
    # 查找chat.html页面的地址
    chat_html_path = PROJECT_ROOT / "app" / "query_process" / "page" / "chat.html"
    if not chat_html_path.exists():
        raise HTTPException(status_code=404, detail="chat.html not found")
    return FileResponse(chat_html_path)


# 发起提问接口
# 接收参数的类型
class QueryRequest(BaseModel):
    query: str = Field(..., title="查询内容,必须传递")
    session_id: str = Field(..., title="会话id,可以不传递,uuid生成一个!")
    is_stream: bool = Field(False, title="是否流式返回结果")


def run_query_graph(query: str, session_id: str, is_stream: bool, auth_context: dict):
    # 一会调用main_graph执行
    # 本次任务开启了!is_stream=True 把结果加入到队列中,sse可以取到
    update_task_status(session_id, "processing", is_stream)
    state = create_query_default_state(session_id=session_id,
                                       original_query=query, is_stream=is_stream, **auth_context)
    try:
        query_app.invoke(state)
        update_task_status(session_id, "completed", is_stream)
        write_audit_log(
            user_id=auth_context.get("user_id", ""),
            tenant_id=auth_context.get("tenant_id", "default"),
            department_id=auth_context.get("department_id", "default"),
            action="query",
            resource_type="session",
            resource_id=session_id,
            success=True,
            extra={"is_stream": is_stream, "query_type": state.get("query_type", "")},
        )
    except Exception as e:
        logger.exception(f"---session_id ={session_id},查询流程出现异常!!{str(e)}")
        update_task_status(session_id, "failed", is_stream) # 更新后端状态
        # 推送指定类型的事件
        push_to_session(session_id, SSEEvent.ERROR, {"error": str(e)}) # 通知前端事件
        write_audit_log(
            user_id=auth_context.get("user_id", ""),
            tenant_id=auth_context.get("tenant_id", "default"),
            department_id=auth_context.get("department_id", "default"),
            action="query",
            resource_type="session",
            resource_id=session_id,
            success=False,
            error_message=str(e),
            extra={"is_stream": is_stream},
        )


def _bind_session_auth(session_id: str, auth: AuthContext) -> None:
    _session_security[session_id] = {
        "user_id": auth.user_id,
        "tenant_id": auth.tenant_id,
        "department_id": auth.department_id,
    }


def _assert_session_access(session_id: str, auth: AuthContext) -> None:
    session_auth = _session_security.get(session_id)
    if not session_auth:
        return
    if (
        session_auth.get("tenant_id") != auth.tenant_id
        or session_auth.get("department_id") != auth.department_id
        or session_auth.get("user_id") != auth.user_id
    ):
        write_audit_log(
            user_id=auth.user_id,
            tenant_id=auth.tenant_id,
            department_id=auth.department_id,
            action="query",
            resource_type="session",
            resource_id=session_id,
            success=False,
            error_message="session access denied",
        )
        raise HTTPException(status_code=403, detail="无权限访问该会话")


@app.post("/query")  # 客户端 -> 问题 -> grapg开启了 ->查到rag的结果 -> 返回即可!
async def query(request: QueryRequest, background_tasks: BackgroundTasks, auth: AuthContext = Depends(get_auth_context)):
    """

    :param request: 请求参数
    :param background_tasks:异步执行函数  is_stream = True
    :return:
    """
    query = request.query
    session_id = request.session_id or str(uuid.uuid4())
    is_stream = request.is_stream
    auth_context = auth.to_state_fields()
    _bind_session_auth(session_id, auth)
    logger.info(f"query鉴权上下文: session_id={session_id}, auth={auth.to_log_dict()}, has_authorization={auth.has_authorization}")

    # 判断是不是流式处理 (是 -> 异步 -> 先返回一个结果  开始处理 | 后台运行图,结果向前端推送)
    if is_stream:
        # 只要开启流式处理,我们业务中就是将数据插入到队列中! {session_id,queue[update_task_state,add_running_task,add_done_list]}
        # 创建当前session_id 对应的队列 -> _session_stream
        create_sse_queue(session_id)
        # 异步执行  立即返回结果给前端 || 中间的过程sse 一点一点推送给前端
        background_tasks.add_task(run_query_graph, query, session_id, is_stream, auth_context)
        logger.info(f"query:{query}已经开启了异步和流式处理!!")
        return {
            "session_id": session_id,
            "message": "本次查询正在处理中...",
        }
    else:
        # 同步执行
        run_query_graph(query, session_id, is_stream, auth_context)
        # 获取最后一个节点插入的结果! node_answer_output(answer)
        answer = get_task_result(session_id, "answer")  # task_utils封装的一个存储会话结果函数
        metadata = get_task_result(session_id, "metadata", {})
        # 返回对应的json数据即可
        logger.info(f"query:{query}已经开启了同步处理!处理结果为:{answer}!")
        return {
            "answer": answer,
            "metadata": metadata,
            "session_id": session_id,
            "message": "本次查询处理完毕!",
            "done_list": []
        }


@app.get("/stream/{session_id}")
async def stream(session_id: str,request: Request, auth: AuthContext = Depends(get_auth_context)):
    _assert_session_access(session_id, auth)
    logger.info(f"session_id={session_id}客户端,已经和后台建立了长连接")
    return StreamingResponse(
        sse_generator(session_id,request),
        media_type="text/event-stream"
    )

@app.get("/history/{session_id}")
async def history(session_id: str,limit: int = 10, auth: AuthContext = Depends(get_auth_context)):
    """
    :param session_id:
    :param limit: 切割数量
    :return:
    """
    _assert_session_access(session_id, auth)
    # 获取历史对话
    chats = get_recent_messages(session_id,limit)
    # chat mogodb_id -> ObjectId -> 不能直接序列化 json
    logger.info(f"session_id={session_id}获取历史对话成功!,查询数据为:{chats}")
    return {
        "session_id": session_id,
        "items": chats
    }

@app.delete("/history/{session_id}")
async def delete_history(session_id: str, auth: AuthContext = Depends(get_auth_context)):
    """
    :param session_id:
    :return:
    """
    _assert_session_access(session_id, auth)
    # 删除历史对话
    delete_count =  clear_history(session_id)
    logger.info(f"session_id={session_id}删除历史对话成功!,删除数量:{delete_count}")
    return {
        "session_id": session_id,
        "message": f"{session_id}聊天记录删除成功!"
    }

if __name__ == '__main__':
    uvicorn.run(app, host="127.0.0.1", port=8001)
