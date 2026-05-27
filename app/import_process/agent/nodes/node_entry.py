import os.path
import sys
import uuid

from app.clients.document_meta_repository import get_document_meta_repository, DocumentStatus
from app.clients.milvus_utils import get_milvus_client
from app.conf.milvus_config import milvus_config
from app.core.logger import logger
from app.import_process.agent.state import ImportGraphState
from app.utils.hash_utils import calculate_file_sha256
from app.utils.escape_milvus_string_utils import escape_milvus_string
from app.utils.task_utils import add_running_task, add_done_task, set_task_result, update_import_task_fields


def _has_legacy_same_title(file_title_value: str) -> bool:
    milvus_client = get_milvus_client()
    if milvus_client is None or not milvus_client.has_collection(milvus_config.chunks_collection):
        return False
    safe_file_title = escape_milvus_string(file_title_value)
    result = milvus_client.query(
        collection_name=milvus_config.chunks_collection,
        filter=f'file_title=="{safe_file_title}"',
        output_fields=["file_title"],
        limit=1,
    )
    return bool(result)


def node_entry(state: ImportGraphState) -> ImportGraphState:
    """
    节点: 入口节点 (node_entry)
    为什么叫这个名字: 作为图的 Entry Point，负责接收外部输入并决定流程走向。
    未来要实现:
    1.  进入节点的日志输出[节点+核心参数]
        记录任务状态,哪个任务开始了 -> 给前端推送信息 (埋点)
    2.  参数校验,local_file_path -> 没有传入文件 ->end | local_dir 没有传入输出文件 -> 创建临时文件夹
    3.  判断文件类型 (PDF/MD)。设置 state 中的路由标记 (is_pdf_read_enabled / is_md_read_enabled)。
    4.  结束节点的日志输出[节点+核心参数] ,记录任务状态[哪个任务结束了] -> 给前端推送信息(埋点)
    """
    # 1.进入节点的日志输出[节点+核心参数]
    # sys._getframe().f_code.co_name 会返回当前正在执行的函数的名称（字符串形式）
    function_name = sys._getframe().f_code.co_name
    logger.info(f">>> [{function_name}] 开始执行了!现在状态为: {state}")
    add_running_task(state['task_id'], function_name)  # tool中的函数, 用于记录任务状态与前端交互

    # 2.参数非空校验
    local_file_path = state.get("local_file_path")
    if not local_file_path:
        logger.error(f">>> [{function_name}] 检查发现没有输入文件!无法进行解析")
        return state
    if not os.path.exists(local_file_path):
        logger.error(f">>> [{function_name}] 检查发现输入文件不存在!路径为:{local_file_path}")
        return state

    # 3.判定并且完成state属性赋值
    # endswith表示文件后缀
    suffix = os.path.splitext(local_file_path)[1].lower()
    if suffix == ".pdf":
        state["is_pdf_read_enabled"] = True
        # 统一补齐下游依赖字段，避免PDF节点读取pdf_path时报KeyError
        state["pdf_path"] = local_file_path
        state["file_type"] = "pdf"
    elif suffix == ".md":
        state["is_md_read_enabled"] = True
        # MD直读路径下，md_path直接使用输入文件
        state["md_path"] = local_file_path
        state["file_type"] = "md"
    elif suffix == ".txt":
        state["is_md_read_enabled"] = True
        state["md_path"] = local_file_path
        state["file_type"] = "txt"
    else:
        logger.error(f">>> [{function_name}] 检查发现输入文件格式不支持!请检查文件后缀")
        return state

    # 提取file_title,xx/xxx/aaa.pdf  -> aaa ->为了后续大模型没有识别出来当前文件对应item_name ->file_title兜底
    file_title = os.path.basename(local_file_path).split(".")[0]
    # file_title = Path(local_file_path).stem # 去掉后缀的文件名  .suffix 取后缀
    state["file_title"] = str(file_title)
    set_task_result(state["task_id"], "file_title", state["file_title"])

    # 增: 增的原因是版本管理和重复上传识别都依赖稳定的文件内容哈希，必须在导入最前面完成计算。
    file_hash = calculate_file_sha256(local_file_path)
    state["file_hash"] = file_hash
    set_task_result(state["task_id"], "file_hash", file_hash)
    update_import_task_fields(state["task_id"], file_hash=file_hash, file_title=state["file_title"])

    # 增: 增的原因是导入流程需要在入口就完成重复上传/同名新版本判定，避免重复执行解析、切片和向量化。
    repository = get_document_meta_repository()
    duplicated_doc = repository.find_active_by_file_hash(file_hash)
    if duplicated_doc:
        state["doc_id"] = duplicated_doc["doc_id"]
        state["version"] = duplicated_doc.get("version", 1)
        state["document_status"] = DocumentStatus.DUPLICATED
        state["duplicated_doc_id"] = duplicated_doc["doc_id"]
        state["duplicate_reason"] = "file_hash_matched"
        state["is_pdf_read_enabled"] = False
        state["is_md_read_enabled"] = False
        logger.info(
            f">>> [{function_name}] 检测到重复上传，直接复用已存在文档。doc_id={state['doc_id']}, file_hash={file_hash}"
        )
        logger.info(f">>> [{function_name}] 执行结束了!现在状态为: {state}")
        add_done_task(state['task_id'], function_name)
        return state

    latest_doc = repository.find_latest_by_file_title(state["file_title"])
    state["doc_id"] = str(uuid.uuid4())
    state["version"] = 1
    state["document_status"] = DocumentStatus.ACTIVE
    if latest_doc:
        state["version"] = int(latest_doc.get("version", 0)) + 1
        state["old_doc_id"] = latest_doc.get("doc_id", "")
        state["old_version"] = int(latest_doc.get("version", 0))
        state["old_file_hash"] = latest_doc.get("file_hash", "")
        state["previous_doc_ids"] = [latest_doc.get("doc_id", "")]
        logger.info(
            f">>> [{function_name}] 检测到同名文档新版本，旧doc_id={state['old_doc_id']}，新版本号={state['version']}"
        )
    elif _has_legacy_same_title(state["file_title"]):
        # 增: 增的原因是现网Milvus中可能已经有旧版历史数据但尚未建立Mongo元数据，需要提供一次性兼容升级能力。
        state["version"] = 2
        state["old_version"] = 1
        logger.info(
            f">>> [{function_name}] 检测到Milvus中存在同名历史数据但缺少元数据，按新版本导入处理。file_title={state['file_title']}"
        )

    # 4.结束节点的日志输出[节点+核心参数] ,记录任务状态[哪个任务结束了] -> 给前端推送信息(埋点)
    logger.info(f">>> [{function_name}] 执行结束了!现在状态为: {state}")
    add_done_task(state['task_id'], function_name)  # tool中的函数, 用于记录任务状态与前端交互
    return state
