from app.core.logger import logger
from app.prompts.prompt_registry import get_prompt_definition, render_prompt


def load_prompt(name: str, version: str | None = None, **kwargs) -> str:
    """
    统一加载 Prompt 模板并渲染变量。
    默认通过 Prompt Registry 获取模板元数据，保持旧调用方式兼容。
    """
    definition = get_prompt_definition(name, version=version)
    rendered_prompt = render_prompt(name, version=definition.version, **kwargs)
    logger.debug(
        f"提示词渲染成功，prompt={definition.prompt_name}, version={definition.version}, 变量={list(kwargs.keys())}"
    )
    return rendered_prompt



if __name__ == '__main__':
    # 测试：传入参数渲染占位符（和业务代码中实际使用方式一致）
    root_folder = "hl3070使用说明书"  # 要替换的文件名称
    image_content = ("这是图片的上文内容", "这是图片的下文内容")  # 要替换的上下文
    # 调用时传入所有需要渲染的变量（键名必须和.prompt中的占位符完全一致）
    final_prompt = load_prompt(
        name='image_summary',
        root_folder=root_folder,  # 对应{root_folder}
        image_content=image_content  # 对应{image_content[0]}、{image_content[1]}
    )
    print("✅ 渲染后的最终提示词：")
    print(final_prompt)
