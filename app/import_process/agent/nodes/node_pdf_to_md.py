import os
import shutil
import sys
import time
import zipfile
from pathlib import Path

import requests

from app.conf.mineru_config import mineru_config
from app.core.logger import logger, PROJECT_ROOT
from app.import_process.agent.state import ImportGraphState, create_default_state
from app.utils.task_utils import add_done_task, add_running_task


def _request_direct(method: str, url: str, **kwargs) -> requests.Response:
    """
    强制直连请求（忽略系统代理环境变量）。
    """
    # 创建独立Session，便于精确控制本次请求配置
    session = requests.Session()
    # 关键配置：忽略HTTP_PROXY/HTTPS_PROXY等环境变量，直接出网
    session.trust_env = False
    try:
        # method/url/headers/json/timeout 等参数都通过 kwargs 透传
        return session.request(method=method, url=url, **kwargs)
    finally:
        # 无论成功失败都关闭连接池，避免资源泄漏
        session.close()


def _download_zip_with_ssl_fallback(zip_url: str, timeout: int = 120) -> requests.Response:
    """
    ZIP下载兜底（该函数内部已包含重试）：
    - 先按原URL下载；
    - 若命中 SSL EOF 且URL为 cdn-mineru 的 https 链接，则自动回退到 http 重试一次。
    - 若下载接口返回 5xx（如502），按退避策略重试，避免 CDN 短暂不可用导致流程失败。
    """
    # 最大重试次数：避免CDN瞬时抖动导致一次失败即终止
    max_attempts = 6
    # 退避初始等待时间（秒），后续按指数增长
    backoff_seconds = 2
    # 记录最后一次响应，便于上层输出准确状态码
    last_resp = None

    for attempt in range(1, max_attempts + 1):
        try:
            # 第一优先：按原始URL发起直连请求（不走系统代理）
            resp = _request_direct(method="GET", url=zip_url, timeout=timeout)
        except requests.exceptions.SSLError as ssl_err:
            # 仅对已知问题域名做协议回退，避免扩大影响范围
            if zip_url.startswith("https://cdn-mineru.openxlab.org.cn/"):
                # 将 https 回退为 http，规避本机SSL握手兼容问题
                http_url = zip_url.replace("https://", "http://", 1)
                logger.warning(f"[下载ZIP] 检测到SSL握手异常，回退HTTP重试：{ssl_err}")
                resp = _request_direct(method="GET", url=http_url, timeout=timeout)
            else:
                # 其他域名的SSL错误保持原样抛出，由上层处理
                raise

        # 记录本轮响应，便于循环结束后返回最后状态
        last_resp = resp
        # 成功分支：HTTP 200 直接返回，不再重试
        if resp.status_code == 200:
            return resp

        # CDN临时错误分支：5xx并且还有剩余次数时执行退避重试
        if resp.status_code in (500, 502, 503, 504) and attempt < max_attempts:
            logger.warning(
                f"[下载ZIP] CDN临时错误，状态码：{resp.status_code}，"
                f"第{attempt}/{max_attempts}次重试，{backoff_seconds}s后继续"
            )
            time.sleep(backoff_seconds)
            # 指数退避：2 -> 4 -> 8 -> 16 -> 20(封顶)
            backoff_seconds = min(backoff_seconds * 2, 20)
            continue

        # 非可重试状态码，或已到最后一次，跳出循环由上层统一处理
        break

    # 返回最后一次响应（可能非200），由调用方决定抛错信息
    return last_resp


def step_1_validate_paths(state):
    """
    进行路径校验
    :param state:
    :return:
    """
    # 小细节要低于外面级别,嫌麻烦可以关
    logger.debug(f">>> [step_1_validate_paths]在md转pdf下,开始进行文件格式校验")
    pdf_path = state['pdf_path']
    local_dir = state['local_dir']
    # 常规的非空校验(站在字符串角度)
    if not pdf_path:
        logger.error(f"[step_1_validate_paths] 检查发现没有输入文件!无法进行解析")
        raise ValueError(f"[step_1_validate_paths] 检查发现没有输入文件!无法进行解析")
    if not local_dir:
        local_dir = PROJECT_ROOT / "output"
        logger.info(f"[step_1_validate_paths]检查没有发现输出目录, 默认输出目录为: {local_dir}")

    # 进行文件存在校验
    pdf_path_obj = Path(pdf_path)
    local_dir_obj = Path(local_dir)
    if not pdf_path_obj.exists():
        logger.error(f"[step_1_validate_paths] 检查发现输入文件不存在!请检查输入文件路径: {pdf_path}")
        raise ValueError(f"[step_1_validate_paths] 检查发现输入文件不存在!请检查输入文件路径: {pdf_path}")
    if not local_dir_obj.exists():
        logger.info(f"[step_1_validate_paths]检查没有发现输出目录, 默认输出目录为: {local_dir}")
        local_dir_obj.mkdir(parents=True, exist_ok=True)
    return pdf_path_obj, local_dir_obj


def step_2_upload_and_poll(pdf_path_obj):
    """
        步骤2：上传PDF至MinerU并轮询解析任务状态
        核心流程：配置校验 → 获取上传链接 → 文件上传（含重试） → 任务轮询（直至完成/失败/超时）
        参数：pdf_path_obj-已校验的PDF Path对象；output_dir_obj-输出目录Path对象
        返回：解析结果ZIP包下载链接full_zip_url
        异常：ValueError(配置缺失)、RuntimeError(请求/上传失败)、TimeoutError(任务超时)
        """
    # 前置配置校验，拦截无效配置
    MINERU_BASE_URL = mineru_config.base_url
    MINERU_API_TOKEN = mineru_config.api_key
    if not MINERU_BASE_URL or not MINERU_API_TOKEN:
        raise ValueError("MinerU配置缺失：请在.env中正确配置MINERU_BASE_URL和MINERU_API_TOKEN")
    logger.info(f"[配置校验] MinerU基础配置加载成功，开始处理文件：{pdf_path_obj.name}")

    # 构造请求头（符合HTTP规范，Bearer鉴权）
    request_headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {MINERU_API_TOKEN}"
    }

    # 1. 调用批量接口，获取上传Signed URL和任务batch_id
    url_get_upload = f"{MINERU_BASE_URL}/file-urls/batch"
    req_data = {
        "files": [{"name": pdf_path_obj.name}],
        "model_version": "vlm"  # 官方推荐解析模型
    }
    logger.debug(f"[获取上传链接] 调用接口：{url_get_upload}，请求参数：{req_data}")
    # 强制直连：避免系统代理引入额外不稳定因素（如代理侧502）
    resp = _request_direct(
        method="POST",
        url=url_get_upload,
        headers=request_headers,
        json=req_data,
        timeout=30
    )

    # 响应校验：先验HTTP状态，再验业务返回码
    if resp.status_code != 200:
        raise RuntimeError(f"[获取上传链接] 网络请求失败，状态码：{resp.status_code}，响应内容：{resp.text}")

    resp_data = resp.json()
    if resp_data["code"] != 0:
        raise RuntimeError(f"[获取上传链接] API业务错误，返回数据：{resp_data}")

    # 提取核心数据：上传链接和任务唯一标识
    uploaded_url = resp_data["data"]["file_urls"][0]  # 获取上传链接
    batch_id = resp_data["data"]["batch_id"]  # 处理id,后续根据id获取结果
    logger.info(f"[获取上传链接] 成功，batch_id：{batch_id}，上传链接已生成")

    # 2. 读取PDF二进制数据，准备上传
    # 使用put请求,将pdf_path_obj文件传递到upload_url地址即可!
    # 注意: 不能直接使用put!这块大概率报错!原因:电脑开了各种代理,put的请求头,添加一些额外的参数头!将文件真的转存到第三方的文件存储服务器!
    # 文件存储服务器检查都比较严格! 拒绝存储!报错!get post 宽进宽出, put严进严出!
    logger.info(f"[文件上传] 开始读取PDF文件：{pdf_path_obj.name}")
    with open(pdf_path_obj, "rb") as f:
        file_data = f.read()

    # 创建Session（复用TCP连接，禁用代理避免签名验证失败）
    upload_session = requests.Session()
    upload_session.trust_env = False

    try:
        # 首次上传：自动识别文件类型
        put_resp = upload_session.put(url=uploaded_url, data=file_data, timeout=60)
        # 重试逻辑：首次失败则强制指定PDF的Content-Type
        if put_resp.status_code != 200:
            logger.warning(f"[文件上传] 首次上传失败（状态码：{put_resp.status_code}），强制指定PDF类型重试")
            pdf_headers = {"Content-Type": "application/pdf"}
            put_resp = upload_session.put(url=uploaded_url, data=file_data, headers=pdf_headers, timeout=60)
            # 重试仍失败则抛出异常
            if put_resp.status_code != 200:
                raise RuntimeError(f"[文件上传] 重试后仍失败，状态码：{put_resp.status_code}，响应内容：{put_resp.text}")
        logger.info(f"[文件上传] 成功，文件{pdf_path_obj.name}已存入云存储")
    except Exception as e:
        raise RuntimeError(f"[文件上传] 网络异常导致上传失败，错误信息：{str(e)}")
    finally:
        # 无论成败，关闭Session释放网络连接，避免资源泄漏
        upload_session.close()

    # 3. 根据batch_id轮询任务状态，直至完成/失败/超时
    poll_url = f"{MINERU_BASE_URL}/extract-results/batch/{batch_id}"
    start_time = time.time()
    timeout_seconds = 600  # 最大超时时间10分钟（适配600页内PDF）
    poll_interval = 3  # 轮询间隔3秒（平衡查询频率和服务端压力）
    logger.info(f"[任务轮询] 开始监控任务状态，batch_id：{batch_id}，最大超时：{timeout_seconds}s")

    while True:
        # 超时检查：超过最大时间直接终止轮询
        elapsed_time = time.time() - start_time
        if elapsed_time > timeout_seconds:
            raise TimeoutError(f"[任务轮询] 超时！任务处理超{int(timeout_seconds)}秒，batch_id：{batch_id}")

        # 发起轮询请求，短超时10秒，异常则重试
        try:
            # 强制直连：轮询与上传保持同一路径，减少链路差异
            poll_resp = _request_direct(
                method="GET",
                url=poll_url,
                headers=request_headers,
                timeout=10
            )
        except Exception as e:
            logger.warning(f"[任务轮询] 网络请求异常，{poll_interval}秒后重试：{str(e)}")
            time.sleep(poll_interval)
            continue

        # 处理HTTP响应错误：5xx服务端繁忙则重试，其他错误直接抛出
        if poll_resp.status_code != 200:
            if 500 <= poll_resp.status_code < 600:
                logger.warning(f"[任务轮询] 服务端繁忙（状态码：{poll_resp.status_code}），{poll_interval}秒后重试")
                time.sleep(poll_interval)
                continue
            else:
                raise RuntimeError(f"[任务轮询] HTTP请求失败，状态码：{poll_resp.status_code}，响应内容：{poll_resp.text}")

        # 解析轮询结果，校验业务状态
        poll_data = poll_resp.json()
        if poll_data["code"] != 0:
            raise RuntimeError(f"[任务轮询] API业务错误，返回数据：{poll_data}")

        extract_results = poll_data["data"]["extract_result"]
        # 结果暂空，继续轮询
        if not extract_results:
            logger.debug(f"[任务轮询] 结果暂为空，已耗时{int(elapsed_time)}s，继续等待")
            time.sleep(poll_interval)
            continue

        # 解析任务状态，分支处理
        result_item = extract_results[0]
        state_status = result_item["state"]
        # 状态1：任务完成，提取ZIP下载链接
        if state_status == "done":
            logger.info(f"[任务轮询] 解析任务完成！总耗时：{int(elapsed_time)}s，batch_id：{batch_id}")
            full_zip_url = result_item.get("full_zip_url")
            if not full_zip_url:
                raise RuntimeError("[任务轮询] 任务完成但未返回ZIP包下载链接，batch_id：{batch_id}")
            logger.info(f"[任务轮询] 结果ZIP包下载链接：{full_zip_url}...")
            return full_zip_url
        # 状态2：任务失败，提取错误信息抛出
        elif state_status == "failed":
            err_msg = result_item.get("err_msg", "未知错误，无具体信息")
            raise RuntimeError(f"[任务轮询] 解析任务失败，batch_id：{batch_id}，错误信息：{err_msg}")
        # 状态3：处理中，实时打印进度（覆盖当前行）
        else:
            logger.debug(
                f"[任务轮询] 处理中（已耗时{int(elapsed_time)}s），状态：{state_status} | 刷新间隔{poll_interval}s",
                end="\r"
            )
            time.sleep(poll_interval)


def step_3_download_and_extract(zip_url, local_dir_obj, pdf_stem):
    """
        步骤3：下载MinerU解析结果ZIP包并解压，提取目标MD文件（重命名统一规范）
        核心流程：下载ZIP → 清理旧目录并解压 → 查找MD文件（按优先级） → 重命名统一为PDF同名
        参数：zip_url-ZIP包下载链接；output_dir_obj-输出目录Path；pdf_stem-PDF无后缀纯名称
        返回：最终MD文件的字符串格式绝对路径
        异常：RuntimeError(下载失败)、FileNotFoundError(无MD文件)
        """
    logger.info(f"===== 开始处理[{pdf_stem}]的MinerU解析结果 =====")

    # 1. 下载解析结果ZIP包，120秒超时适配大文件
    logger.info(f"[步骤1/4] 开始下载ZIP包，链接：{zip_url}...")
    # 统一下载入口：内部已处理 SSL->HTTP 回退 + 5xx 退避重试
    resp = _download_zip_with_ssl_fallback(zip_url=zip_url, timeout=120)
    if resp.status_code != 200:
        raise RuntimeError(f"[步骤1/4] ZIP包下载失败，HTTP状态码：{resp.status_code}")

    # 拼接ZIP包保存路径，按PDF名称唯一命名
    zip_save_path = local_dir_obj / f"{pdf_stem}_result.zip"
    with open(zip_save_path, "wb") as f:
        # resp.content响应体中的数据
        f.write(resp.content)
    logger.info(f"[步骤1/4] ZIP包下载成功，保存路径：{zip_save_path}")

    # 2. 清理旧解压目录并解压ZIP包（避免旧文件干扰，为每个PDF创建专属目录）
    logger.info(f"[步骤2/4] 开始解压ZIP包...")
    extract_target_dir = local_dir_obj / pdf_stem

    # 清理旧目录，异常则警告不终止
    if extract_target_dir.exists():
        try:
            # 递归删除整个目录树，包括目录本身及其所有子目录和文件。
            shutil.rmtree(extract_target_dir)
            logger.info(f"[步骤2/4] 已清理旧的解压目录：{extract_target_dir}")
        except Exception as e:
            logger.warning(f"[步骤2/4] 清理旧目录失败，可能不影响新文件解压：{str(e)}")

    # 重新创建解压目录
    # parents = True：自动创建父目录，如果父目录不存在
    # exist_ok = True：如果目录已存在，不会抛出错误。
    extract_target_dir.mkdir(parents=True, exist_ok=True)

    # 核心解压操作，保留原目录结构
    with zipfile.ZipFile(zip_save_path, 'r') as zip_file_obj:
        zip_file_obj.extractall(extract_target_dir)
    logger.info(f"[步骤2/4] ZIP包解压完成，解压目录：{extract_target_dir}")

    # 3. 递归查找解压目录下所有MD文件（适配子目录结构）
    # 解压后的文件可能较 源文件.md 也可能叫 full.md,所以不确定,需要递归查找
    logger.info(f"[步骤3/4] 开始查找解压目录中的MD文件...")
    md_file_list = list(extract_target_dir.rglob("*.md"))
    if not md_file_list:
        raise FileNotFoundError(f"[步骤3/4] 解压目录中未找到任何.md格式文件：{extract_target_dir}")
    logger.info(f"[步骤3/4] 共找到{len(md_file_list)}个MD文件，按优先级匹配目标文件")

    # 4. 按优先级匹配目标MD文件（同名→full.md→第一个，兜底避免流程中断）
    target_md_file = None
    # 优先级1：与PDF纯名称完全同名的MD文件
    for md_file in md_file_list:
        if md_file.stem == pdf_stem:
            target_md_file = md_file
            logger.info(f"[步骤4/4] 匹配到优先级1目标：与PDF同名的MD文件 {target_md_file.name}")
            break
    # 优先级2：MinerU默认生成的full.md（不区分大小写）
    if not target_md_file:
        for md_file in md_file_list:
            if md_file.name.lower() == "full.md":
                target_md_file = md_file
                logger.info(f"[步骤4/4] 匹配到优先级2目标：MinerU默认文件 {target_md_file.name}")
                break
    # 优先级3：兜底取第一个MD文件
    if not target_md_file:
        target_md_file = md_file_list[0]
        logger.info(f"[步骤4/4] 未匹配到前两级目标，兜底取第一个MD文件 {target_md_file.name}")

    # 重命名MD文件：统一为PDF纯名称，便于后续流程处理（仅不同名时执行）
    if target_md_file.stem != pdf_stem:
        logger.info(f"[步骤4/4] 开始重命名MD文件，统一为PDF同名：{pdf_stem}.md")
        new_md_path = target_md_file.with_name(f"{pdf_stem}.md")
        try:
            # 将磁盘上的文件进行重命名
            target_md_file.rename(new_md_path)
            # 更新变量引用
            target_md_file = new_md_path
            logger.info(f"[步骤4/4] MD文件重命名成功：{pdf_stem}.md")
        except OSError as e:
            logger.warning(f"[步骤4/4] MD文件重命名失败，将使用原文件名继续流程：{str(e)}")

    # 转换为字符串绝对路径返回，适配后续仅支持字符串路径的函数
    final_md_path = str(target_md_file.absolute())
    logger.info(f"===== [{pdf_stem}]解析结果处理完成，最终MD文件路径：{final_md_path} =====")
    return final_md_path


def node_pdf_to_md(state: ImportGraphState) -> ImportGraphState:
    """
    节点: PDF转Markdown (node_pdf_to_md)
    为什么叫这个名字: 核心任务是将 PDF 非结构化数据转换为 Markdown 结构化数据。
    未来要实现:
    1. 进入的日志和任务状态的配置
    2. 进入参数校验(load_dir -> 给予默认值 | local_file_path完成字面意思的校验 -> 深入校验校验的文件是否存在)
    3. 调用minerU进行pdf的解析(local_file_path)返回一个下载文件的地址xx.zip url地址
    4. 下载zip包,并且解析和提取(local_dir)
    5. 把md_path地址进行赋值,读取md的文件内容md_content赋值(文本内容)
    6. 结束的日志和任务状态的配置
    因为真正的要调用工具,所以要做容错率处理!try异常处理
    """
    function_name = sys._getframe().f_code.co_name
    logger.info(f">>> [{function_name}] 开始执行了!现在状态为: {state}")
    add_running_task(state['task_id'], function_name)

    try:
        # 2.进入参数校验(load_dir -> 给予默认值 | local_file_path完成字面意思的校验 -> 深入校验校验的文件是否存在)
        pdf_path_obj, local_dir_obj = step_1_validate_paths(state)
        # 3.调用minerU进行pdf的解析(local_file_path)返回一个下载文件的地址xx.zip url地址
        zip_url = step_2_upload_and_poll(pdf_path_obj)
        # 4.下载zip包,并且解析和提取(local_dir)
        md_path = step_3_download_and_extract(zip_url, local_dir_obj, pdf_path_obj.stem)
        # 5.把md_path地址进行赋值,读取md的文件内容md_content赋值(文本内容)
        state['md_path'] = md_path
        state['local_dir'] = str(local_dir_obj)
        with open(md_path, 'r', encoding='utf-8') as f:
            state['md_content'] = f.read()

    except Exception as e:
        logger.error(f">>> [{function_name}] 使用minerU异常,异常信息为: {e}")
        # 关键：不能吞异常继续流转，否则下游会因缺失md_path产生二次错误
        raise
    finally:

        # 结束节点的日志输出[节点+核心参数] ,记录任务状态[哪个任务结束了] -> 给前端推送信息(埋点)
        logger.info(f">>> [{function_name}] 执行结束了!现在状态为: {state}")
        add_done_task(state['task_id'], function_name)  # tool中的函数, 用于记录任务状态与前端交互
    return state

if __name__ == "__main__":

    # 单元测试：验证PDF转MD全流程
    logger.info("===== 开始node_pdf_to_md节点单元测试 =====")

    from app.utils.path_util import PROJECT_ROOT
    logger.info(f"测试获取根地址：{PROJECT_ROOT}")

    test_pdf_name = os.path.join("doc", "hak180产品安全手册.pdf")
    test_pdf_path = os.path.join(PROJECT_ROOT, test_pdf_name)

    # 构造测试状态
    test_state = create_default_state(
        task_id="test_pdf2md_task_001",
        pdf_path=test_pdf_path,
        local_dir=os.path.join(PROJECT_ROOT, "output")
    )

    node_pdf_to_md(test_state)

    logger.info("===== 结束node_pdf_to_md节点单元测试 =====")

