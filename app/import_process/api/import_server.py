import os
import shutil
import uuid
from typing import List, Dict, Any
from datetime import datetime
import uvicorn
# 第三方库
from fastapi import FastAPI, UploadFile, File, BackgroundTasks, Depends, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
# 项目内部工具/配置/客户端
from app.clients.audit_log_repository import write_audit_log
from app.clients.document_meta_repository import (
    get_document_meta_repository,
    DocumentStatus,
)
from app.clients.milvus_utils import get_milvus_client, delete_by_doc_id
from app.security.auth import AuthContext, build_mongo_security_query, get_auth_context
from app.utils.path_util import PROJECT_ROOT
from app.utils.task_utils import (
    add_running_task,
    add_done_task,
    clear_task,
    get_done_task_list,
    get_running_task_list,
    get_import_task,
    increment_import_task_retry,
    init_import_task,
    list_import_task_node_logs,
    mark_import_task_finished,
    record_import_task_failure,
    update_task_status,
    get_task_status,
    set_task_result,
    get_task_result,
    update_import_task_fields,
)
from app.clients.import_task_repository import ImportTaskStatus
from app.import_process.agent.state import get_default_state
from app.import_process.agent.main_graph import kb_import_app  # LangGraph全流程编译实例
from app.core.logger import logger  # 项目统一日志工具
from app.conf.milvus_config import milvus_config

# 初始化FastAPI应用实例
# 标题和描述会在Swagger文档(http://ip:port/docs)中展示
app = FastAPI(
    title="File Import Service",
    description="Web service for uploading files to Knowledge Base (PDF/MD → 解析 → 切分 → 向量化 → Milvus入库)"
)

# 跨域中间件配置：解决前端调用后端接口的跨域限制
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # 允许所有前端域名访问（生产环境建议指定具体域名）
    allow_credentials=True,  # 允许携带Cookie等认证信息
    allow_methods=["*"],  # 允许所有HTTP方法（GET/POST/PUT/DELETE等）
    allow_headers=["*"],  # 允许所有请求头
)


# 8080/import  --> import.html
@app.get("/import", response_class=FileResponse)
async def get_import_page():
    """
    提供静态文件：前端页面
    """
    import_html_path = PROJECT_ROOT / "app" / "import_process" / "page" / "import.html"
    if not import_html_path.exists():
        raise HTTPException(status_code=404, detail="File not found")
    return FileResponse(path=import_html_path, media_type="text/html")


# 定义调用import_graph的函数,像之前测试一样执行的图!
# 需要local_file_path(str)  task_id  local_dir(str)
def run_import_graph(task_id:str, local_file_path:str, local_dir:str, auth_context: dict | None = None):
    """
    开启图的执行和调用
    :return:
    """
    try:
        # 本次任务的总状态
        # _task_status:Dict[str,str]={}
        # key = task_id
        # value = task_id 任务状态
        update_task_status(task_id, "processing")
        init_state = get_default_state()
        init_state["task_id"] = task_id
        init_state["local_file_path"] = local_file_path
        init_state["local_dir"] = local_dir
        if auth_context:
            init_state.update(auth_context)
        update_import_task_fields(task_id, source_path=local_file_path, local_dir=local_dir)
        final_state = dict(init_state)
        # 执行我们的图
        for event in kb_import_app.stream(init_state,stream_mode="updates"):
        # event {节点名:state}
            for node_name,result in event.items():
                logger.info(f"{node_name}节点执行完毕,执行结果为:{result}")
                if isinstance(result, dict):
                    final_state.update(result)
                # add_done_task(task_id,node_name)
        if final_state.get("document_status") == DocumentStatus.DUPLICATED:
            set_task_result(task_id, "doc_id", final_state.get("doc_id", ""))
            set_task_result(task_id, "document_status", DocumentStatus.DUPLICATED)
            set_task_result(task_id, "file_hash", final_state.get("file_hash", ""))
            set_task_result(task_id, "file_title", final_state.get("file_title", ""))
            mark_import_task_finished(task_id, ImportTaskStatus.DUPLICATED)
            if auth_context:
                write_audit_log(
                    user_id=auth_context.get("created_by", ""),
                    tenant_id=auth_context.get("tenant_id", "default"),
                    department_id=auth_context.get("department_id", "default"),
                    action="document_import",
                    resource_type="import_task",
                    resource_id=task_id,
                    success=True,
                    extra={"document_status": DocumentStatus.DUPLICATED, "doc_id": final_state.get("doc_id", "")},
                )
        elif final_state.get("doc_id"):
            set_task_result(task_id, "doc_id", final_state.get("doc_id", ""))
            set_task_result(task_id, "document_status", final_state.get("document_status", DocumentStatus.ACTIVE))
            set_task_result(task_id, "file_hash", final_state.get("file_hash", ""))
            set_task_result(task_id, "file_title", final_state.get("file_title", ""))
            final_status = final_state.get("document_status", DocumentStatus.ACTIVE)
            if final_status == DocumentStatus.REPLACED:
                mark_import_task_finished(task_id, ImportTaskStatus.REPLACED)
            else:
                mark_import_task_finished(task_id, ImportTaskStatus.SUCCESS)
            if auth_context:
                write_audit_log(
                    user_id=auth_context.get("created_by", ""),
                    tenant_id=auth_context.get("tenant_id", "default"),
                    department_id=auth_context.get("department_id", "default"),
                    action="document_import",
                    resource_type="document",
                    resource_id=final_state.get("doc_id", ""),
                    success=True,
                    extra={"task_id": task_id, "document_status": final_status},
                )
                if final_state.get("old_doc_id"):
                    write_audit_log(
                        user_id=auth_context.get("created_by", ""),
                        tenant_id=auth_context.get("tenant_id", "default"),
                        department_id=auth_context.get("department_id", "default"),
                        action="document_replace",
                        resource_type="document",
                        resource_id=final_state.get("doc_id", ""),
                        success=True,
                        extra={"old_doc_id": final_state.get("old_doc_id", ""), "task_id": task_id},
                    )
        update_task_status(task_id,"completed")
    except Exception as e:
        logger.error(f"=====图执行失败!发生异常=====")
        set_task_result(task_id, "error", str(e))
        record_import_task_failure(task_id, e)
        update_task_status(task_id, "failed")
        if auth_context:
            write_audit_log(
                user_id=auth_context.get("created_by", ""),
                tenant_id=auth_context.get("tenant_id", "default"),
                department_id=auth_context.get("department_id", "default"),
                action="document_import",
                resource_type="import_task",
                resource_id=task_id,
                success=False,
                error_message=str(e),
            )


# 8080/upload post -> 文件上传 + 开启导入流程
"""
    1.接收文件存储到output文件夹! /output/当天的日期/uuid(taskid)/文件夹
    2.异步开启,import_graph图的执行 1. 整个任务的状态(开始和结束) 2.每个节点的状态(add_running  add_done)
"""
@app.post("/upload")
async def upload_file(background_tasks: BackgroundTasks, files: List[UploadFile] = File(...), auth: AuthContext = Depends(get_auth_context)):
    """
    文件上传核心接口
    1. 接收前端上传的多文件（PDF/MD为主）
    2. 按「日期/任务ID」分层保存到本地输出目录，避免文件冲突
    3. 将文件上传至MinIO对象存储，做持久化保存
    4. 为每个文件生成唯一TaskID，启动独立的LangGraph后台处理任务
    5. 实时更新任务状态，供前端轮询监控进度

    :param background_tasks: FastAPI后台任务对象，用于异步执行LangGraph流程
    :param files: 前端上传的文件列表（form-data格式）
    :return: 包含上传结果和所有任务ID的JSON响应
    """
    # 1. 整理下输出位置output/日期文件夹
    today_str = datetime.now().strftime("%Y%m%d")
    base_out_path = PROJECT_ROOT/"output"/today_str
    auth_context = auth.to_state_fields()
    logger.info(f"upload鉴权上下文: auth={auth.to_log_dict()}, has_authorization={auth.has_authorization}")
    # 2. 记录下每个文件上传的任务id[taskid,taskid]
    task_ids = []
    # 3. 循环处理每个上传的文件(存储到本地) + 进行异步图任务调用
    for file in files:
        # file -> UploadFile (.file上传文件的输入流  .filename上传文件名
        #                     .read可直接读取  .contenttype 获取我们文件minetype类型)
        task_id = str(uuid.uuid4())
        task_ids.append(task_id)
        # 文件的dir_path
        dir_path = base_out_path/task_id
        dir_path.mkdir(parents=True, exist_ok=True)
        # 文件的local_file_path
        local_file_path = dir_path/file.filename
        init_import_task(
            task_id,
            file_title=os.path.splitext(file.filename)[0],
            source_path=str(local_file_path),
            local_dir=str(dir_path),
            status=ImportTaskStatus.PENDING,
            max_retry=1,
        )
        update_import_task_fields(
            task_id,
            tenant_id=auth.tenant_id,
            department_id=auth.department_id,
            created_by=auth.user_id,
            visibility=auth.visibility,
        )
        # 记录下进行文件上传了
        add_running_task(task_id, "upload_file")
        # 将上传的文件写入到local_file_path
        with local_file_path.open("wb") as buffer:
            shutil.copyfileobj(file.file, buffer) #把流写入指定位置,此工具好处在于可以处理大文件(自动分开写)
        add_done_task(task_id, "upload_file")

        # 异步执行
        # 参数1:run_import_graph执行的方法
        # 参数2: *args,参数列表task_id, local_file_path, dir_path --> run_import_graph
        background_tasks.add_task(run_import_graph, task_id, str(local_file_path), str(dir_path), auth_context)
        logger.info(f"任务ID:{task_id}上传文件成功,并开启了对应的异步任务!")
    # 4. 最终返回结果即可
    return {
        "code":200,
        "message":f"完成了文件上传,并开启了异步任务!文件数量为:{len(files)}",
        "task_ids":task_ids
    }


# --------------------------
# 核心接口：任务状态查询接口
# 前端轮询此接口获取单个任务的处理进度和状态
# 访问地址：http://localhost:8000/status/{task_id} （GET请求）
# --------------------------
@app.get("/status/{task_id}", summary="任务状态查询", description="根据TaskID查询单个文件的处理进度和全局状态")
async def get_task_progress(task_id: str):
    """
    任务状态查询接口
    前端轮询此接口（如每秒1次），获取任务的实时处理进度
    返回数据均来自内存中的任务管理字典（task_utils.py），高性能无IO

    :param task_id: 全局唯一任务ID（由/upload接口返回）
    :return: 包含任务全局状态、已完成节点、运行中节点的JSON响应
    """
    # 构造任务状态返回体
    task_status_info: Dict[str, Any] = {
        "code": 200,
        "task_id": task_id,
        "status": get_task_status(task_id),  # 任务全局状态：pending/processing/completed/failed
        "done_list": get_done_task_list(task_id),  # 已完成的节点/阶段列表
        "running_list": get_running_task_list(task_id),  # 正在运行的节点/阶段列表
        "result": {
            "doc_id": get_task_result(task_id, "doc_id"),
            "document_status": get_task_result(task_id, "document_status"),
            "error": get_task_result(task_id, "error"),
        },
        "task_meta": get_import_task(task_id),
        "node_logs": list_import_task_node_logs(task_id),
    }
    # 记录状态查询日志，方便追踪前端轮询情况
    logger.info(
        f"[{task_id}] 任务状态查询，当前状态：{task_status_info['status']}，已完成节点：{task_status_info['done_list']}")
    return task_status_info


@app.get("/documents", summary="查询文档列表")
async def list_documents(status: str = "", auth: AuthContext = Depends(get_auth_context)):
    repository = get_document_meta_repository()
    documents = repository.list_documents(status=status or None, extra_filters=build_mongo_security_query(auth))
    return {
        "code": 200,
        "count": len(documents),
        "documents": documents,
    }


@app.get("/documents/{doc_id}/versions", summary="查询文档版本")
async def list_document_versions(doc_id: str, auth: AuthContext = Depends(get_auth_context)):
    repository = get_document_meta_repository()
    versions = repository.list_versions(doc_id, extra_filters=build_mongo_security_query(auth))
    if not versions:
        raise HTTPException(status_code=404, detail="文档不存在")
    return {
        "code": 200,
        "doc_id": doc_id,
        "versions": versions,
    }


@app.get("/documents/{doc_id}/chunks", summary="查询文档chunks")
async def list_document_chunks(doc_id: str, auth: AuthContext = Depends(get_auth_context)):
    repository = get_document_meta_repository()
    chunks = repository.list_chunks(doc_id, extra_filters=build_mongo_security_query(auth))
    if not chunks:
        return {
            "code": 200,
            "doc_id": doc_id,
            "chunks": [],
        }
    return {
        "code": 200,
        "doc_id": doc_id,
        "chunks": chunks,
    }


@app.delete("/documents/{doc_id}", summary="删除文档")
async def delete_document(doc_id: str, auth: AuthContext = Depends(get_auth_context)):
    repository = get_document_meta_repository()
    document = repository.find_active_by_doc_id(doc_id, extra_filters=build_mongo_security_query(auth))
    if not document:
        raise HTTPException(status_code=404, detail="文档不存在")

    milvus_client = get_milvus_client()
    delete_by_doc_id(milvus_client, milvus_config.chunks_collection, doc_id)
    delete_by_doc_id(milvus_client, milvus_config.item_name_collection, doc_id)

    repository.mark_document_status(doc_id, DocumentStatus.DELETED)
    repository.mark_chunks_deleted_by_doc_id(doc_id)
    logger.info(f"文档删除完成，doc_id={doc_id}, file_title={document.get('file_title')}")
    write_audit_log(
        user_id=auth.user_id,
        tenant_id=auth.tenant_id,
        department_id=auth.department_id,
        action="document_delete",
        resource_type="document",
        resource_id=doc_id,
        success=True,
    )
    return {
        "code": 200,
        "message": "文档删除成功",
        "doc_id": doc_id,
    }


@app.post("/documents/{doc_id}/reimport", summary="重新导入文档")
async def reimport_document(doc_id: str, background_tasks: BackgroundTasks, auth: AuthContext = Depends(get_auth_context)):
    repository = get_document_meta_repository()
    document = repository.find_active_by_doc_id(doc_id, extra_filters=build_mongo_security_query(auth))
    if not document:
        raise HTTPException(status_code=404, detail="文档不存在")

    local_file_path = document.get("source_path", "")
    local_dir = document.get("local_dir", "")
    if not local_file_path or not os.path.exists(local_file_path):
        raise HTTPException(status_code=400, detail="原始文件不存在，无法重新导入")
    if not local_dir:
        local_dir = str(PROJECT_ROOT / "output" / datetime.now().strftime("%Y%m%d") / str(uuid.uuid4()))
        os.makedirs(local_dir, exist_ok=True)

    task_id = str(uuid.uuid4())
    init_import_task(
        task_id,
        file_title=document.get("file_title", ""),
        source_path=local_file_path,
        local_dir=local_dir,
        status=ImportTaskStatus.PENDING,
        max_retry=1,
    )
    update_import_task_fields(
        task_id,
        tenant_id=auth.tenant_id,
        department_id=auth.department_id,
        created_by=auth.user_id,
        visibility=auth.visibility,
        doc_id=doc_id,
    )
    background_tasks.add_task(run_import_graph, task_id, local_file_path, local_dir, auth.to_state_fields())
    logger.info(f"文档重新导入任务已创建，原doc_id={doc_id}, 新task_id={task_id}")
    return {
        "code": 200,
        "message": "已创建重新导入任务",
        "task_id": task_id,
        "doc_id": doc_id,
    }


@app.post("/import/tasks/{task_id}/retry", summary="重试导入任务")
async def retry_import_task(task_id: str, background_tasks: BackgroundTasks, auth: AuthContext = Depends(get_auth_context)):
    task = get_import_task(task_id)
    if not task:
        raise HTTPException(status_code=404, detail="任务不存在")
    if task.get("tenant_id") != auth.tenant_id or task.get("department_id") != auth.department_id or task.get("visibility") != auth.visibility:
        raise HTTPException(status_code=403, detail="无权限重试该任务")
    source_path = task.get("source_path", "")
    local_dir = task.get("local_dir", "")
    if not source_path or not os.path.exists(source_path):
        raise HTTPException(status_code=400, detail="原始文件不存在，无法重试")
    retry_count = int(task.get("retry_count", 0))
    max_retry = int(task.get("max_retry", 1))
    if retry_count >= max_retry:
        raise HTTPException(status_code=400, detail="已达到最大重试次数")

    clear_task(task_id)
    retry_count = increment_import_task_retry(task_id)
    update_import_task_fields(
        task_id,
        status=ImportTaskStatus.PENDING,
        current_node="",
        error_stack="fallback为全链路重试，当前架构未开启节点级恢复",
        finished_at=None,
        created_by=auth.user_id,
    )
    background_tasks.add_task(run_import_graph, task_id, source_path, local_dir, auth.to_state_fields())
    logger.info(f"导入任务重试已创建，task_id={task_id}, retry_count={retry_count}")
    write_audit_log(
        user_id=auth.user_id,
        tenant_id=auth.tenant_id,
        department_id=auth.department_id,
        action="retry_import_task",
        resource_type="import_task",
        resource_id=task_id,
        success=True,
        extra={"retry_count": retry_count},
    )
    return {
        "code": 200,
        "message": "已创建重试任务，当前采用全链路重试模式",
        "task_id": task_id,
        "retry_count": retry_count,
    }

if __name__ == '__main__':
    uvicorn.run(app, host="127.0.0.1", port=8000)
