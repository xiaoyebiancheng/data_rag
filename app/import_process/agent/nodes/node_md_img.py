import os
import re
import sys
import base64
from pathlib import Path
from typing import Any, Dict, List, Tuple
from collections import deque

from minio.deleteobjects import DeleteObject

# 【核心改造1：移除原生OpenAI，导入LangChain工具类和多模态消息模块】
from app.clients.minio_utils import get_minio_client
from app.import_process.agent.state import ImportGraphState
from app.utils.task_utils import add_running_task, add_done_task
# LLM客户端工具类（核心复用，替换原生OpenAI调用）
from app.lm.lm_utils import get_llm_client
# LangChain多模态依赖（消息构造+异常捕获）
from langchain.messages import HumanMessage
from langchain_core.exceptions import LangChainException
# 项目配置
from app.conf.minio_config import minio_config
from app.conf.lm_config import lm_config
# 项目日志工具（统一使用）
from app.core.logger import logger
# api访问限速工具
from app.utils.rate_limit_utils import apply_api_rate_limit
# 提示词加载工具
from app.core.load_prompt import load_prompt

# MinIO支持的图片格式集合（小写后缀，统一匹配标准）
IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".gif", ".bmp", ".webp"}
SUMMARY_STATUS_OK = "ok"
SUMMARY_STATUS_LOW_CONFIDENCE = "low_confidence"
SUMMARY_STATUS_FAILED = "failed"
DEFAULT_FAILED_SUMMARY = "图片摘要生成失败，请以原图为准"
LOW_CONFIDENCE_PREFIX = "低置信图片摘要"
FAILED_PREFIX = "图片摘要失败"
GENERIC_SUMMARIES = {"图片", "图片描述", "示意图", "结构图", "截图", "设备图片", "产品图片"}
UNCERTAIN_PHRASES = ("看不清", "无法判断", "无法识别", "无法可靠识别", "不确定", "可能是", "似乎是", "大概", "疑似")

"""
    主要目标: 将md中的图片进行单独处理,方便后续去模型和识别图片的含义 !
    主要动作: 图片 ->文件服务器 ->图片网络地址  (上文100) 图片(下文100) -> 视觉模型 ->图片总结
             ---> [图片总结](图片网络地址) -> state -> md_content = 新内容(图片处理后的) || md_path = 处理后的md地址
    主要技术:
        minio    视觉模型
    总结步骤:
    1. 校验并且获取本次操作的数据
        参数： state -> md_path md_content
        响应： 1. 校验后的md_content  2.md路径对象  3. 获取图片的文件夹 images
    2. 识别md中使用过的图片，采取做下一步（进行图片总结）
        参数： 1. md_content 2. images图片的文件夹地址
        响应： [(图片名,图片地址,(上文,下文))]
    3. 进行图片内容的总结和处理（视觉模型）
        参数： 第二次的响应 [(图片名,图片地址,(上文,下文))]  || md文件的名称（提示词中 md文件名就是存储图片images的文件名）
        响应： {图片名:总结,......}
    4. 上传图片minio以及更新md的内容
        参数： minio_client || {图片名:总结,......} || [(图片名,图片地址,(上文,下文)) (minio)] || md_content 旧 || md文件的名称（提示词中 md文件名就是存储图片images的文件名）
        响应： new_md_content
        state[md_content] = new_md_content
    5. 进行数据的最终处理和备份
        参数： new_md_content , 原md地址 -> xx.md -> xx_new.md
        响应： 新的md的地址 new_md_path
        state[md_path] = new_new_md_path

    return state 
"""


def is_supported_image(filename: str) -> bool:
    """
    判断文件是否为MinIO支持的图片格式（后缀不区分大小写）
    :param filename: 文件名（含后缀）
    :return: 支持返回True，否则False
    """
    return os.path.splitext(filename)[1].lower() in IMAGE_EXTENSIONS


def step_1_get_content(state: ImportGraphState) -> Tuple[str, Path, Path]:
    """
    提取内容
    :param state:
    :return:
    """
    # 1.获取md的地址
    md_file_path = state.get("md_path")
    if not md_file_path:
        raise ValueError("md_path不能为空")
    md_path_obj = Path(md_file_path)
    if not md_path_obj.exists():
        raise FileNotFoundError(f"md_path: {md_file_path}文件不存在")
    # 要读取md_content文件
    md_content = state.get("md_content")
    if not state.get("md_content"):
        with open(md_path_obj, "r", encoding="utf-8") as f:
            md_content = f.read()
        state["md_content"] = md_content
    # 获取图片的文件夹
    # 注意:自己传入的md -> 你的图片文件夹也必须叫images
    images_dir_obj = md_path_obj.parent / "images"
    return md_content, md_path_obj, images_dir_obj


def find_image_in_md_content(md_content, image_file, content_length=100):
    """
    从md_content中识别图片的上下文
    :param md_content:
    :param image_file:
    :param content_length:
    :return:
    """
    # 定义正则表达式 r表示不能转译, .表示任意字符，*表示任意次数 ,?表示非贪婪模式,\让字符保持原样

    pattern = re.compile(rf"!\[.*?\]\(.*?{re.escape(image_file)}.*?\)")

    match = pattern.search(md_content)
    if not match:
        logger.warning(f"图片文件 {image_file} 未在 md 中找到引用，无需处理")
        return None

    start, end = match.span()
    pre_text = md_content[max(0, start - content_length):start]
    post_text = md_content[end:min(len(md_content), end + content_length)]
    content = (pre_text, post_text)

    logger.info(f"图片 {image_file} 上下文: {content}")
    return content


def step_2_scan_images(md_content, images_dir_obj) -> List[Tuple[str, str, Tuple[str, str]]]:
    """
    进行md中图片识别,并且截取图片对应的上下文环境
    :param md_content:
    :param images_dir_obj:
    :return:
    """
    targets = []
    # 1. 循环读取images文件夹中的所有图片,校验在md中是否使用,使用了就截取上下文
    for image_file in os.listdir(images_dir_obj):
        # 检查图片是否可用
        if not is_supported_image(image_file):
            logger.warning(f"图片文件 {image_file} 不支持，无需处理")
            continue
        # 是图片,看是否存在于md,存在则截取上下文
        content_data = find_image_in_md_content(md_content, image_file)
        if not content_data:
            logger.warning(f"图片文件 {image_file} 不在md中，无需处理")
            continue
        targets.append((image_file, str(images_dir_obj / image_file), content_data))
    return targets


def encode_image_to_base64(image_path: str) -> str:
    """
    将本地图片文件编码为Base64字符串（用于多模态大模型输入）
    :param image_path: 图片本地完整路径
    :return: 图片的Base64编码字符串（UTF-8解码）
    """
    with open(image_path, "rb") as img_file:
        base64_str = base64.b64encode(img_file.read()).decode("utf-8")
    logger.debug(f"图片Base64编码完成，文件：{image_path}，编码后长度：{len(base64_str)}")
    return base64_str


def _failed_image_summary(reason: str) -> Dict[str, Any]:
    return {
        "summary": DEFAULT_FAILED_SUMMARY,
        "status": SUMMARY_STATUS_FAILED,
        "confidence": 0.0,
        "reason": reason,
    }


def normalize_image_summary(raw_summary: str) -> Dict[str, Any]:
    """
    对 VLM 图片摘要做轻量质量门控。
    这里不把低质量摘要直接当成事实证据，而是显式标记状态，后续写入 Markdown 和 sources。
    """
    summary = (raw_summary or "").strip().replace("\n", " ")
    summary = re.sub(r"\s+", " ", summary)
    if not summary:
        return _failed_image_summary("empty_summary")

    summary = summary[:80]
    reasons: List[str] = []
    if summary in GENERIC_SUMMARIES or len(summary) < 6:
        reasons.append("generic_or_too_short")
    if any(phrase in summary for phrase in UNCERTAIN_PHRASES):
        reasons.append("uncertain_expression")

    if reasons:
        return {
            "summary": summary,
            "status": SUMMARY_STATUS_LOW_CONFIDENCE,
            "confidence": 0.35,
            "reason": ",".join(reasons),
        }
    return {
        "summary": summary,
        "status": SUMMARY_STATUS_OK,
        "confidence": 0.9,
        "reason": "",
    }


def _coerce_summary_result(summary_result: Any) -> Dict[str, Any]:
    if isinstance(summary_result, dict):
        result = dict(summary_result)
        result.setdefault("summary", DEFAULT_FAILED_SUMMARY)
        result.setdefault("status", SUMMARY_STATUS_FAILED)
        result.setdefault("confidence", 0.0)
        result.setdefault("reason", "")
        return result
    return normalize_image_summary(str(summary_result or ""))


def summarize_image(image_path: str, root_folder: str, image_content: Tuple[str, str]) -> Dict[str, Any]:
    """
    调用多模态大模型生成图片内容摘要（适配LangChain工具类，复用项目统一LLM客户端）
    生成的摘要用于Markdown图片标题，严格控制50字以内中文描述
    :param image_path: 图片本地完整路径
    :param root_folder: 文档所属文件夹/主名，为大模型提供上下文
    :param image_content: 图片在MD中的上下文元组，格式(上文文本, 下文文本)
    :return: 图片摘要质量结果，包含 summary/status/confidence/reason
    """
    # 将图片编码为Base64，适配多模态大模型输入要求
    base64_image = encode_image_to_base64(image_path)
    try:
        # 1. 获取项目统一LLM客户端（自动缓存，传入多模态模型名）
        vm_model = get_llm_client(model=lm_config.lv_model)

        # 加载并渲染提示词（核心：传入所有占位符对应的变量）
        prompt_text = load_prompt(
            name="image_summary",  # 提示词文件名（不带.prompt）
            root_folder=root_folder,  # 对应{root_folder}
            image_content=image_content  # 对应{image_content[0]}、{image_content[1]}
        )

        # 2. 构造LangChain标准多模态HumanMessage（兼容千问/OpenAI等视觉模型）
        messages = [
            HumanMessage(
                content=[
                    # 文本提示词：携带上下文，限定摘要规则
                    {
                        "type": "text",
                        "text": prompt_text
                    },
                    # 多模态核心：Base64编码图片数据
                    {
                        "type": "image_url",
                        "image_url": {
                            # base64图片转后的字符串 jpg -> image/jpeg
                            "url": f"data:image/jpeg;base64,{base64_image}"
                        }
                    }
                ]
            )
        ]

        # 3. LangChain标准调用：invoke方法（工具类已封装超时/重试等参数）
        response = vm_model.invoke(messages)

        # 4. 解析响应（LangChain统一返回content字段，统一格式无需多层解析）
        summary = response.content.strip().replace("\n", "")  # strip()去空格,replace("\n", "")去换行
        result = normalize_image_summary(summary)
        logger.info(f"图片摘要生成成功：{image_path}，摘要质量：{result}")
        return result

    except LangChainException as e:
        logger.error(f"图片摘要生成失败（LangChain框架异常）：{image_path}，错误信息：{str(e)}")
        return _failed_image_summary("langchain_exception")
    except Exception as e:
        logger.error(f"图片摘要生成失败（系统异常）：{image_path}，错误信息：{str(e)}")
        return _failed_image_summary("system_exception")


def step_3_generate_img_summaries(targets, stem):
    """
    步骤3：批量为待处理图片生成内容摘要，带API速率限制防止触发大模型限流
    :param stem: 文档文件名（不含后缀），作为大模型prompt上下文
    :param targets: 待处理图片列表，元素为(图片文件名, 图片完整路径, 图片上下文)
    :param requests_per_minute: 每分钟最大API请求数，默认9次（按大模型限制调整）
    :return: 图片摘要字典，键：图片文件名，值：图片内容摘要
    """
    summaries = {}
    request_times = deque()  # 外部初始化请求时间队列，跨循环复用

    for img_file, image_path, context in targets:
        # 直接调用抽离的公共工具方法，参数和原逻辑完全一致
        apply_api_rate_limit(request_times, max_requests=9, window_seconds=60)
        logger.debug(f"开始生成图片摘要：{image_path}")
        summaries[img_file] = summarize_image(image_path, root_folder=stem, image_content=context)

    logger.info(f"图片摘要批量生成完成，共处理{len(summaries)}张图片")
    return summaries


def _build_markdown_image_replacement(summary_result: Dict[str, Any], url: str) -> str:
    summary = str(summary_result.get("summary") or DEFAULT_FAILED_SUMMARY)
    status = str(summary_result.get("status") or SUMMARY_STATUS_FAILED)
    confidence = float(summary_result.get("confidence") or 0.0)
    reason = str(summary_result.get("reason") or "")

    if status == SUMMARY_STATUS_FAILED:
        alt_text = f"{FAILED_PREFIX}：{summary}"
    elif status == SUMMARY_STATUS_LOW_CONFIDENCE:
        alt_text = f"{LOW_CONFIDENCE_PREFIX}：{summary}"
    else:
        alt_text = summary

    markdown = f"![{alt_text}]({url})"
    if status != SUMMARY_STATUS_OK:
        markdown += (
            "\n\n<details>\n"
            "<summary>image_summary_quality</summary>\n\n"
            f"status: {status}\n"
            f"confidence: {confidence:.2f}\n"
            f"reason: {reason or '-'}\n"
            f"source_url: {url}\n"
            "</details>"
        )
    return markdown


def step_4_upload_images_and_update_md(summaries, targets, md_content, stem):
    """
    将我们图片传递到minio服务器 ,看minio官网api
    替换原md中的图片描述和描述
    :param summaries: 图片名:描述
    :param targets: (图片名,原地址,(上,下))
    :param md_content: 原md内容
    :param stem: 文件名
    :return: 新md, 图片摘要审计信息
    """
    # 理解minio存储结果: 桶 / upload-images/文件夹名字/图片.jpg
    minio_client = get_minio_client()
    # 1. 删除minio中的对应文件的图片
    # 1.1 获取要删除的对象
    # Object object_name
    # 注意:{minio_config.minio_img_dir[1:]}  去掉前缀 /
    object_list = minio_client.list_objects(minio_config.bucket_name,
                                            prefix=f"{minio_config.minio_img_dir[1:]}/{stem}",
                                            recursive=True)
    delete_object_list = [DeleteObject(obj.object_name) for obj in object_list]
    # 1.2 调用方法进行删除.固定用法,调一下删一下
    errors = minio_client.remove_objects(minio_config.bucket_name, delete_object_list)
    for errors in errors:
        logger.error(f"删除对象失败:{errors}")
    logger.info(f"已经完成{stem}下的对象清空,本次删除了:{len(delete_object_list)}个对象!")
    # 2. 上传图片到minio服务器
    # 声明记录图片上传结果的字典
    images_url = {}
    # targets: (图片名,原地址,(上,下))
    for image_file, image_path, _ in targets:
        try:
            minio_client.fput_object(
                bucket_name=minio_config.bucket_name,
                object_name=f"{minio_config.minio_img_dir[1:]}/{stem}/{image_file}",  # 传入minio 桶后面的命名 xx.png
                file_path=image_path,
                content_type="image/jpeg",
            )
            # 上传完毕以后记录
            # 图片地址 = 协议+断电+对象名 http://127.0.0.1:9000/桶名/对象名
            endpoint = minio_config.endpoint
            if not endpoint.startswith(("http://", "https://")):
                endpoint = f"http://{endpoint}"
            images_url[
                image_file] = f"{endpoint}/{minio_config.bucket_name}{minio_config.minio_img_dir}/{stem}/{image_file}"
            logger.info(f"完成图片{image_file}上传,访问地址为:{images_url[image_file]}")
        except Exception as e:
            logger.error(f"上传图片失败:{image_file},失败原因{e}")

    # 3. 替换原md中的图片和描述
    # summaries = 图片名:描述
    # images_url = 图片名:url地址
    # 汇总:{图片名:(描述,url地址)}
    image_infos = {}
    for image_file, summary in summaries.items():
        if url := images_url.get(image_file):
            image_infos[image_file] = (_coerce_summary_result(summary), url)
    logger.info(f"图片处理的汇总的汇总结果:{image_infos}")
    image_summary_audit = []
    if image_infos:
        """
        xxxxx
        xxx ![xx](图片地址/image_file) -> ![summary](minio的url)
        xxxx
        """
        for image_file, (summary_result, url) in image_infos.items():
            # 使用正则
            rep = re.compile(r"!\[.*?\]\(.*?"+re.escape(image_file)+r".*?\)")
            md_content = rep.sub(_build_markdown_image_replacement(summary_result, url), md_content)
            image_summary_audit.append(
                {
                    "image_file": image_file,
                    "source_url": url,
                    "summary": summary_result.get("summary", ""),
                    "status": summary_result.get("status", SUMMARY_STATUS_FAILED),
                    "confidence": summary_result.get("confidence", 0.0),
                    "reason": summary_result.get("reason", ""),
                }
            )
        logger.info(f"图片处理完毕,新的md文件内容为:{md_content}")
    return md_content, image_summary_audit


def step_5_replace_md_and_save(new_md_content, md_path_obj):
    """
    完成新的md的磁盘备份,并且返回老地址!
    :param new_md_content:新内容
    :param md_path_obj:老地址
    :return:新地址
    """
    # 设置新的地址
    # xx/xx/xxx/xxx/xxx.md  -> xx/xx/xxx/xxx/xxx_new.md
    # os.path.splitext按照扩展名进行分割也就是 .
    new_md_path_str = md_path_obj.with_name(f"{md_path_obj.stem}_new.md") # ai优化版
    # new_md_path_str = os.path.splitext(md_path_obj)[0] + "_new.md" 视频版
    with open(new_md_path_str, "w", encoding="utf-8") as f:
        f.write(new_md_content)
    logger.info(f"已经完成新内容写入,新的md文件地址为:{new_md_path_str}")
    return str(new_md_path_str)


def node_md_img(state: ImportGraphState) -> ImportGraphState:
    """
    节点: 图片处理 (node_md_img)
    为什么叫这个名字: 处理 Markdown 中的图片资源 (Image)。
    未来要实现:
    1. 扫描 Markdown 中的图片链接。
    2. 将图片上传到 MinIO 对象存储。
    3. (可选) 调用多模态模型生成图片描述。
    4. 替换 Markdown 中的图片链接为 MinIO URL。
    """
    function_name = sys._getframe().f_code.co_name
    logger.info(f">>> [{function_name}] 开始执行了!现在状态为: {state}")
    add_running_task(state['task_id'], function_name)
    # 1. 校验并且获取本次操作的数据
    #   参数： state -> md_path md_content
    #   响应： 1. 校验后的md_content  2.md路径对象  3. 获取图片的文件夹 images
    md_content, md_path_obj, images_dir_obj = step_1_get_content(state)
    if not images_dir_obj.exists():
        images_dir_obj.mkdir(parents=True, exist_ok=True)
    # 2. 识别md中使用过的图片，采取做下一步（进行图片总结）
    # [(图片名,图片地址,(上文,下文))]
    targets = step_2_scan_images(md_content, images_dir_obj)
    #   参数： 1. md_content 2. images图片的文件夹地址
    #   响应： [(图片名,图片地址,(上文,下文))]

    if os.getenv("IMPORT_SKIP_IMAGE_SUMMARY", "").strip().lower() in {"1", "true", "yes", "y", "on"}:
        logger.info(f"IMPORT_SKIP_IMAGE_SUMMARY已开启，跳过图片摘要与上传，共跳过{len(targets)}张图片")
        new_md_file_path = step_5_replace_md_and_save(md_content, md_path_obj)
        state["md_path"] = new_md_file_path
        state["md_content"] = md_content
        logger.info(f">>> [{function_name}] 执行结束了!现在状态为: {state}")
        add_done_task(state['task_id'], function_name)
        return state

    # 3. 进行图片内容的总结和处理（视觉模型）
    #    参数： 第二次的响应 [(图片名,图片地址,(上文,下文))]  || md文件的名称（提示词中 md文件名就是存储图片images的文件名）
    #    响应： {图片名:总结,......}
    summaries = step_3_generate_img_summaries(targets, md_path_obj.stem)

    # 4. 上传图片minio以及更新md的内容
    #   参数： minio_client || {图片名:总结,......} || [(图片名,图片地址,(上文,下文)) (minio)] || md_content 旧 || md文件的名称（提示词中 md文件名就是存储图片images的文件名）
    #   响应： new_md_content
    #   state[md_content] = new_md_content
    new_md_content, image_summary_audit = step_4_upload_images_and_update_md(summaries, targets, md_content, md_path_obj.stem)

    # 5. 进行数据的最终处理和备份
    #   参数： new_md_content , 原md地址 -> xx.md -> xx_new.md
    #   响应： 新的md的地址 new_md_path
    #    state[md_path] = new_new_md_path
    new_md_file_path = step_5_replace_md_and_save(new_md_content, md_path_obj)
    state['md_path'] = new_md_file_path
    state['md_content']=new_md_content
    state["image_summary_audit"] = image_summary_audit
    logger.info(f">>> [{function_name}] 执行结束了!现在状态为: {state}")
    add_done_task(state['task_id'], function_name)  # tool中的函数, 用于记录任务状态与前端交互
    return state

if __name__ == "__main__":
    """本地测试入口：单独运行该文件时，执行MD图片处理全流程测试"""
    from app.utils.path_util import PROJECT_ROOT
    logger.info(f"本地测试 - 项目根目录：{PROJECT_ROOT}")

    # 测试MD文件路径（需手动将测试文件放入对应目录）
    test_md_name = os.path.join(r"output/hak180产品安全手册", "hak180产品安全手册.md")
    test_md_path = os.path.join(PROJECT_ROOT, test_md_name)

    # 校验测试文件是否存在
    if not os.path.exists(test_md_path):
        logger.error(f"本地测试 - 测试文件不存在：{test_md_path}")
        logger.info("请检查文件路径，或手动将测试MD文件放入项目根目录的output目录下")
    else:
        # 构造测试状态对象，模拟流程入参
        test_state = {
            "md_path": test_md_path,
            "task_id": "test_task_123456",
            "md_content": ""
        }
        logger.info("开始本地测试 - MD图片处理全流程")
        # 执行核心处理流程
        result_state = node_md_img(test_state)
        logger.info(f"本地测试完成 - 处理结果状态：{result_state}")
