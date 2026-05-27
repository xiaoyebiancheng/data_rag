# 导入Milvus相关依赖
import sys
from typing import List, Dict, Any

from pymilvus import DataType

from app.clients.document_meta_repository import (
    get_document_meta_repository,
    DocumentStatus,
    ChunkStatus,
)
from app.clients.milvus_utils import (
    get_milvus_client,
    delete_by_doc_id,
    delete_by_file_title,
    delete_by_file_title_version,
    delete_by_item_name,
)
from app.conf.milvus_config import milvus_config
from app.core.logger import logger
# 导入自定义模块
from app.import_process.agent.state import ImportGraphState
from app.import_process.metadata_tagging import enrich_document_and_chunks
from app.utils.escape_milvus_string_utils import escape_milvus_string
from app.utils.task_utils import add_running_task, add_done_task, set_task_result

# 从配置文件读取切片集合名称，与配置解耦，便于环境切换
CHUNKS_COLLECTION_NAME = milvus_config.chunks_collection


def step_2_prepare_collections(state):
    # 1. 获取milvus的客户端
    milvus_client = get_milvus_client()
    # 2.判断是否存在集合(表),存在创建集合(表)

    if not milvus_client.has_collection(collection_name=milvus_config.chunks_collection):
        # 创建集合要先创建列,再创建索引,最后才能创建集合
        # 2.1 创建集合对应的列信息
        schema = milvus_client.create_schema(
            auto_id=True,  # 主键自增长
            enable_dynamic_field=True,  # 动态字段
        )
        # 2.2 add filed to  schema
        # pk  file_title  item_name  dense_vector  sparse_vector
        schema.add_field(field_name="chunk_id", datatype=DataType.INT64, is_primary=True, auto_id=True)
        schema.add_field(field_name="file_title", datatype=DataType.VARCHAR, max_length=65535)
        schema.add_field(field_name="item_name", datatype=DataType.VARCHAR, max_length=65535)
        schema.add_field(field_name="content", datatype=DataType.VARCHAR, max_length=65535)
        schema.add_field(field_name="title", datatype=DataType.VARCHAR, max_length=65535)
        schema.add_field(field_name="parent_title", datatype=DataType.VARCHAR, max_length=65535)
        schema.add_field(field_name="part", datatype=DataType.INT8)
        schema.add_field(field_name="dense_vector", datatype=DataType.FLOAT_VECTOR, dim=1024)
        schema.add_field(field_name="sparse_vector", datatype=DataType.SPARSE_FLOAT_VECTOR)
        # 2.3 查询快,配置索引
        """
        稠密索引:
        IVF算法(基于桶结构):(适合数据量大,对延迟和精度要求不太高)
            1.把所有向量聚类,找出若干中心点
            2.每个向量丢到自己最近的中心点的桶中
            3.查询的时候 -> 查询中心点所在的桶 ->桶中查询向量
            优点:结构简单,占内存小,创建索引快,适合海量数据处理
            缺点:针对桶内查询,精度较差
        HNSW算法(基于图结构):(适合数据量小,对延迟和精度要求高的)
            1.构建多层图
            2.图中的向量进行连接
            3.分层查询,从上至下
            4.上层比较少,下层数据比较细
            5.logn 时间复杂度
            优点:查询速度快,精度高,高纬度向量(768+)
            缺点:数据量大 -> 占服务器,内存
        """
        index_params = milvus_client.prepare_index_params()
        """
        M：最终保留多少邻居.M 决定图的“宽度”
        efConstruction：为了挑这些邻居，前期会考察多少候选.efConstruction 决定建图时的“认真程度”
            10000  M=16  efConstruction:200,
            50000  M=32  efConstruction:300,
            100000  M=64  efConstruction:400,
        """
        index_params.add_index(
            field_name="dense_vector",  # 给哪个列创建索引 稠密
            index_name="dense_vector_index",  # 索引的名字
            index_type="HNSW",  # 配置超找所用的算法
            metric_type="COSINE",  # 配置向量匹配和对比方法 IP CONSINE
            params={
                "M": 32,
                "efConstruction": 300,
            }
        )
        """
        稀疏索引
        """
        index_params.add_index(
            field_name="sparse_vector",  # 给哪个列创建索引 稀疏
            index_type="SPARSE_INVERTED_INDEX",  # 配置超找所用的算法
            index_name="sparse_vector_index",
            metric_type="IP",  # 配置向量匹配和对比方法 IP CONSINE
            # 只计算可能得高分的向量,跳过大量的0
            params={"inverted_index_algo": "DAAT_MAXSCORE"}
        )

        # 创建集合
        milvus_client.create_collection(
            collection_name=milvus_config.chunks_collection,
            schema=schema,
            index_params=index_params,
        )
    return milvus_client


def step_3_delete_old_data(milvus_client, state):
    # 优化: 优化的原因是旧逻辑按item_name删除会误伤不同文件但同主体的数据，版本管理必须优先按doc_id/file_title+version精准清理。
    old_doc_id = state.get("old_doc_id")
    file_title = state.get("file_title", "")
    old_version = state.get("old_version", 0)
    item_name = state.get("item_name", "")

    if old_doc_id:
        delete_by_doc_id(milvus_client, CHUNKS_COLLECTION_NAME, old_doc_id)
    elif file_title and old_version and state.get("old_file_hash"):
        delete_by_file_title_version(milvus_client, CHUNKS_COLLECTION_NAME, file_title, old_version)
    elif file_title and old_version:
        delete_by_file_title(milvus_client, CHUNKS_COLLECTION_NAME, file_title)
    elif item_name:
        delete_by_item_name(milvus_client, CHUNKS_COLLECTION_NAME, item_name)
    milvus_client.load_collection(collection_name=CHUNKS_COLLECTION_NAME)


def step_4_insert_collections(milvus_client, chunks):
    """
    插入集合数据
    :param chunks:
    :return: chunks -> 主键回显
    """
    insert_result=milvus_client.insert(collection_name=CHUNKS_COLLECTION_NAME,data=chunks)
    # 成功插入了几条
    insert_count= insert_result.get("insert_count",0)
    logger.info(f"完成了数据插入,成功插入了{insert_count}条数据")
    # 获取回显的ids
    """
    返回值 insert_result
    Milvus 插入后通常会返回一个结果字典，
    里面可能包含：成功插入了多少条, 返回的主键 ID 列表
        例如可能像这样：
        insert_result = {
            "insert_count": 3,
            "ids": [101, 102, 103]
            }
    """
    ids = insert_result.get("ids",[])
    if ids and len(ids) == len(chunks):
        for index,chunk in enumerate(chunks):
            chunk['chunk_id'] = ids[index]
    return chunks


def step_3_1_fill_milvus_metadata(state: ImportGraphState, chunks: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    # 增: 增的原因是Milvus中的切片需要携带doc_id/file_hash/version以及metadata标签，后续才能支持按版本精确删除、标签过滤和审计。
    for chunk in chunks:
        chunk["doc_id"] = state.get("doc_id", "")
        chunk["file_hash"] = state.get("file_hash", "")
        chunk["version"] = int(state.get("version", 1))
        chunk["status"] = ChunkStatus.ACTIVE
        chunk["tenant_id"] = state.get("tenant_id", "default")
        chunk["department_id"] = state.get("department_id", "default")
        chunk["visibility"] = state.get("visibility", "internal")
    return chunks


def step_5_write_metadata(state: ImportGraphState) -> None:
    repository = get_document_meta_repository()
    chunks = state.get("chunks", [])
    document_meta = {
        "doc_id": state.get("doc_id", ""),
        "file_title": state.get("file_title", ""),
        "file_hash": state.get("file_hash", ""),
        "file_type": state.get("file_type", ""),
        "item_name": state.get("item_name", ""),
        "version": int(state.get("version", 1)),
        "status": DocumentStatus.ACTIVE,
        "chunk_count": len(chunks),
        "minio_urls": state.get("minio_urls", []),
        "source_path": state.get("local_file_path", ""),
        "local_dir": state.get("local_dir", ""),
        "doc_type": state.get("doc_type", "other"),
        "product_line": state.get("product_line", ""),
        "language": state.get("language", "zh-CN"),
        "source_priority": int(state.get("source_priority", 50)),
        "tenant_id": state.get("tenant_id", "default"),
        "department_id": state.get("department_id", "default"),
        "visibility": state.get("visibility", "internal"),
        "created_by": state.get("created_by", ""),
    }
    repository.upsert_document_meta(document_meta)

    replaced_doc_ids = []
    if state.get("old_doc_id"):
        replaced_doc_ids = repository.mark_latest_active_replaced(
            state.get("file_title", ""),
            exclude_doc_id=state.get("doc_id", ""),
        )
    for old_doc_id in replaced_doc_ids:
        repository.mark_chunks_deleted_by_doc_id(old_doc_id)
        delete_by_doc_id(get_milvus_client(), milvus_config.item_name_collection, old_doc_id)
    if not replaced_doc_ids and state.get("old_version") and not state.get("old_doc_id"):
        delete_by_file_title(get_milvus_client(), milvus_config.item_name_collection, state.get("file_title", ""))

    chunk_metas = []
    for chunk in chunks:
        chunk_metas.append({
            "chunk_id": chunk.get("chunk_id"),
            "doc_id": state.get("doc_id", ""),
            "file_hash": state.get("file_hash", ""),
            "chunk_hash": chunk.get("chunk_hash", ""),
            "title": chunk.get("title", ""),
            "item_name": chunk.get("item_name", ""),
            "milvus_collection": CHUNKS_COLLECTION_NAME,
            "status": ChunkStatus.ACTIVE,
            "doc_type": chunk.get("doc_type", state.get("doc_type", "other")),
            "section_type": chunk.get("section_type", "other"),
            "product_line": chunk.get("product_line", state.get("product_line", "")),
            "language": chunk.get("language", state.get("language", "zh-CN")),
            "source_priority": int(chunk.get("source_priority", state.get("source_priority", 50))),
            "tenant_id": chunk.get("tenant_id", state.get("tenant_id", "default")),
            "department_id": chunk.get("department_id", state.get("department_id", "default")),
            "visibility": chunk.get("visibility", state.get("visibility", "internal")),
            "created_by": state.get("created_by", ""),
        })
    repository.upsert_chunk_metas(chunk_metas)


def step_6_mark_failed(state: ImportGraphState) -> None:
    if not state.get("doc_id"):
        return
    repository = get_document_meta_repository()
    repository.create_failed_document_meta({
        "doc_id": state.get("doc_id", ""),
        "file_title": state.get("file_title", ""),
        "file_hash": state.get("file_hash", ""),
        "file_type": state.get("file_type", ""),
        "item_name": state.get("item_name", ""),
        "version": int(state.get("version", 1)),
        "minio_urls": state.get("minio_urls", []),
        "source_path": state.get("local_file_path", ""),
        "local_dir": state.get("local_dir", ""),
        "doc_type": state.get("doc_type", "other"),
        "product_line": state.get("product_line", ""),
        "language": state.get("language", "zh-CN"),
        "source_priority": int(state.get("source_priority", 50)),
        "tenant_id": state.get("tenant_id", "default"),
        "department_id": state.get("department_id", "default"),
        "visibility": state.get("visibility", "internal"),
        "created_by": state.get("created_by", ""),
    })

def node_import_milvus(state: ImportGraphState) -> ImportGraphState:
    """
    节点: 导入向量库 (node_import_milvus)
    为什么叫这个名字: 将处理好的向量数据写入 Milvus 数据库。
    未来要实现:
    1. 连接 Milvus。
    2. 根据 item_name 删除旧数据 (幂等性)。
    3. 批量插入新的向量数据。
    """
    function_name = sys._getframe().f_code.co_name
    logger.info(f">>> [{function_name}] 开始执行了!现在状态为: {state}")
    add_running_task(state['task_id'], function_name)

    try:
        # 1.获取要生成向量的chunks
        chunks = state['chunks']
        if not chunks or not isinstance(chunks, list):
            logger.warning("chunks数据无效，请检查输入数据")
            raise ValueError("chunks数据无效，请检查输入数据")

        # 2.没有集合,要创建集合collection(filed,index,collection)
        milvus_client = step_2_prepare_collections(state)
        # 增: 增的原因是新增metadata标签需要尽量在导入末端统一推断，避免改动前序切片和识别节点结构。
        doc_metadata, chunks = enrich_document_and_chunks(state.get("file_title", ""), state.get("item_name", ""), chunks)
        state["doc_type"] = doc_metadata.get("doc_type", "other")
        state["product_line"] = doc_metadata.get("product_line", "")
        state["language"] = doc_metadata.get("language", "zh-CN")
        state["source_priority"] = int(doc_metadata.get("source_priority", 50))
        state["tenant_id"] = doc_metadata.get("tenant_id", "default")
        state["department_id"] = doc_metadata.get("department_id", "default")
        state["visibility"] = doc_metadata.get("visibility", "internal")
        chunks = step_3_1_fill_milvus_metadata(state, chunks)
        # 3.删除旧数据(根据doc_id / file_title+version / item_name 逐级清理)
        step_3_delete_old_data(milvus_client, state)
        # 4.插入chunks的数据即可
        with_id_chunks = step_4_insert_collections(milvus_client, chunks)
        state['chunks'] = with_id_chunks
        step_5_write_metadata(state)
        set_task_result(state["task_id"], "doc_id", state.get("doc_id", ""))
        set_task_result(state["task_id"], "document_status", DocumentStatus.ACTIVE)
    except Exception as e:
        logger.error(f">>> [{function_name}] 导入chunks对应的向量数据库异常,异常信息为: {e}")
        step_6_mark_failed(state)
        set_task_result(state["task_id"], "document_status", DocumentStatus.FAILED)
        raise
    finally:
        # 结束节点的日志输出[节点+核心参数] ,记录任务状态[哪个任务结束了] -> 给前端推送信息(埋点)
        logger.info(f">>> [{function_name}] 执行结束了!现在状态为: {state}")
        add_done_task(state['task_id'], function_name)  # tool中的函数, 用于记录任务状态与前端交互
    return state


if __name__ == '__main__':
    # --- 单元测试 ---
    # 目的：验证 Milvus 导入节点的完整流程，包括连接、创建集合、清理旧数据和插入新数据。
    import sys
    import os
    from dotenv import load_dotenv

    # 加载环境变量 (自动寻找项目根目录的 .env)
    current_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.dirname(os.path.dirname(current_dir))
    load_dotenv(os.path.join(project_root, ".env"))

    # 构造测试数据
    dim = 1024
    test_state = {
        "task_id": "test_milvus_task",
        "chunks": [
            {
                "content": "Milvus 测试文本 1",
                "title": "测试标题",
                "item_name": "测试项目_Milvus",  # 必须有 item_name，用于幂等清理
                "parent_title":"test.pdf",
                "part":1,
                "file_title": "test.pdf",
                "dense_vector": [0.1] * dim,  # 模拟 Dense Vector
                "sparse_vector": {1: 0.5, 10: 0.8}  # 模拟 Sparse Vector
            }
        ],
        "chunks": [
            {
                "content": "Milvus 测试文本 2",
                "title": "测试标题2",
                "item_name": "测试项目_Milvus2",  # 必须有 item_name，用于幂等清理
                "parent_title": "test.pdf2",
                "part": 1,
                "file_title": "test.pdf2",
                "dense_vector": [0.1] * dim,  # 模拟 Dense Vector
                "sparse_vector": {1: 0.5, 10: 0.8}  # 模拟 Sparse Vector
            }
        ]
    }

    print("正在执行 Milvus 导入节点测试...")
    try:
        # 检查必要的环境变量
        if not os.getenv("MILVUS_URL"):
            print("❌ 未设置 MILVUS_URL，无法连接 Milvus")
        elif not os.getenv("CHUNKS_COLLECTION"):
            print("❌ 未设置 CHUNKS_COLLECTION")
        else:
            # 执行节点函数
            result_state = node_import_milvus(test_state)

            # 验证结果
            chunks = result_state.get("chunks", [])
            if chunks and chunks[0].get("chunk_id"):
                print(f"✅ Milvus 导入测试通过，生成 ID: {chunks[0]['chunk_id']}")
            else:
                print("❌ 测试失败：未能获取 chunk_id")

    except Exception as e:
        print(f"❌ 测试失败: {e}")
