# 《RAG 项目优化理解与面试讲解文档》

适合直接保存为：docs/rag_project_interview_guide.md

状态说明：

- 本文基于当前仓库真实代码阅读输出。
- 已实现的能力，我会明确标注“当前已实现”。
- 尚未在代码中落地、但非常适合下一步升级的能力，我会明确标注“建议实现”。
- 不确定或需要进一步核对的点，我会标注“需要结合代码进一步确认”。

------

# 一、项目整体定位

## 1.1 这个项目解决的真实业务问题是什么

这个项目解决的是“企业内部文档知识无法被稳定检索和可靠问答”的问题。

典型业务痛点是：

- 企业有大量 PDF、Markdown、产品手册、操作文档、故障说明。
- 文档格式不统一，尤其 PDF 里存在表格、图片、层级标题、说明图。
- 用户提问时常常不是完整标准问法，而是口语化、多轮上下文、带指代。
- 单纯把文档丢进向量库，往往会出现召回不稳、答非所问、无法解释来源、旧版本污染结果等问题。

这个项目的核心目标是：

- 把复杂企业文档处理成结构化、可检索、可追踪的数据。
- 在问答侧提供一个“多轮对话 + 多路召回 + 精排 + 最终生成”的完整 RAG 系统。
- 在工程侧兼顾导入、状态跟踪、版本管理、评测与后续可维护性。

## 1.2 为什么它不是普通的“向量数据库问答 demo”

它不是一个简单 demo，原因有 6 个：

1. 它不是“上传 txt -> embedding -> 搜索 -> answer”的直线流程。
   当前导入链路已经包含 PDF -> Markdown -> 图片理解 -> 层级切片 -> item_name 识别 -> 混合向量化 -> Milvus，入口在 main_graph.py。
2. 它有真正的工作流编排。
   导入和问答都不是脚本串行，而是通过 LangGraph 编排状态流转，分别在：
   - import_process/agent/main_graph.py
   - query_process/agent/main_graph.py
3. 它处理图片，而不是只处理纯文本。
   Markdown 图片会被上传到 MinIO，并调用多模态模型做图片摘要，逻辑在 node_md_img.py。
4. 它不是单路检索。
   当前问答主链路有：
   - 本地混合检索
   - HyDE 检索
   - Web Search
   - RRF 融合
   - Rerank 精排
     对应节点在 query_process/agent/nodes/。
5. 它已经开始有工程化能力。
   当前已经实现了一版文档版本管理和增量更新，核心在：
   - document_meta_repository.py
   - node_entry.py
   - node_import_milvus.py
6. 它已经有离线评测模块。
   这一点很关键，因为面试里“能证明效果”比“觉得效果不错”更重要。评测模块在：
   - evaluator.py
   - metrics.py
   - report.py

## 1.3 这个项目在简历中应该如何定位

建议定位成：

**企业级多模态 RAG 知识库系统 / LangGraph Agent 工作流项目**

不要写成：

- “做了一个问答 demo”
- “接了个向量数据库”
- “做了个 ChatPDF”

更合适的话术是：

- 基于 LangGraph 编排文档导入与检索问答双工作流的企业级 RAG 系统
- 支持 PDF/Markdown 导入、图片理解、混合检索、RRF、Rerank、多轮对话和离线评测
- 具备文档版本管理、去重、增量更新与导入任务可观测性

## 1.4 这个项目适合体现哪些岗位能力

这个项目适合体现：

1. RAG
   你不只是会调用 LLM，而是理解文档处理、切片、召回、精排、生成、评测全链路。
2. Agent
   虽然它不是那种“多工具自动规划”的重 Agent，但它是典型的工作流式 Agent 系统，LangGraph 就是其调度中枢。
3. LangGraph
   这个点很强，因为你不是简单函数调用，而是在做状态图编排、条件分支和节点路由。
4. 工具调用
   项目里接了 MinerU、MinIO、Milvus、MCP Web Search、多模态模型、Embedding、Reranker。
5. 多轮对话
   当前用历史消息辅助 item_name 识别和 query rewrite，代码在 node_item_name_confirm.py。
6. 向量检索
   不只是 dense，还包括 sparse、hybrid、HyDE、RRF、Rerank 这套体系。
7. 工程化能力
   离线评测、文档元数据管理、增量更新、版本替换、导入状态跟踪。
8. 系统设计能力
   你可以从系统分层、数据流转、状态管理、幂等性、一致性、版本治理、性能权衡这些角度讲。

------

# 二、原始项目流程讲解

------

## 2.1 文档导入流程

### 2.1.1 导入入口在哪里

有两个入口：

1. API 入口
   在 import_server.py 的 POST /upload
2. LangGraph 工作流入口
   在 main_graph.py 的 kb_import_app

前端页面入口是：

- import.html

### 2.1.2 LangGraph 是如何编排导入流程的

导入图由 main_graph.py 编排：

1. 入口节点是 node_entry
2. 之后做条件路由：
   - 如果是重复上传，直接 END
   - 如果是 Markdown，走 node_md_img
   - 如果是 PDF，先走 node_pdf_to_md
3. 主链路是：
   - node_entry
   - node_pdf_to_md 或直接 node_md_img
   - node_document_split
   - node_item_name_recognition
   - node_bge_embedding
   - node_import_milvus

这是一条典型“导入预处理 -> 结构化 -> 表征 -> 入库”的工作流。

### 2.1.3 每个节点的作用是什么

1. node_entry
   文件校验、识别文件类型、计算 file_hash、判断重复上传和新版本。
   文件在 node_entry.py
2. node_pdf_to_md
   把 PDF 提交给 MinerU，轮询解析结果，下载 ZIP，提取 Markdown。
   文件在 node_pdf_to_md.py
3. node_md_img
   扫描 Markdown 中引用的图片，提取图片上下文，调用多模态模型生成图片摘要，上传 MinIO，替换原 Markdown 图片链接。
   文件在 node_md_img.py
4. node_document_split
   按标题做粗切，再对过长 chunk 细切，对过短 chunk 合并，并生成 chunk_hash。
   文件在 node_document_split.py
5. node_item_name_recognition
   从前几个高价值 chunk 中抽上下文，让 LLM 识别文档主体 item_name，再给 item_name 做 embedding 并写入 kb_item_names。
   文件在 node_item_name_recognition.py
6. node_bge_embedding
   给 chunks 生成 BGE-M3 的 dense 和 sparse 向量。
   文件在 node_bge_embedding.py
7. node_import_milvus
   创建或复用 Milvus collection，清理旧版本数据，写入 chunk 向量，并同步写 Mongo 元数据。
   文件在 node_import_milvus.py

### 2.1.4 PDF 为什么要先转 Markdown

因为企业文档不是纯文本，PDF 直接抽文本有 4 个问题：

1. 标题层级容易丢
   后面做结构化切片会很差。
2. 图片与正文关系丢失
   你看不到“某张图前后讲的是什么”。
3. 段落和版式信息差
   故障说明、步骤、参数区往往靠结构表达语义。
4. 后续处理不统一
   项目最终是围绕 Markdown 作为中间标准格式来做图文处理、切片和摘要。

所以项目里先由 node_pdf_to_md.py 把 PDF 交给 MinerU，拿到 Markdown 再进入统一后处理流程。

### 2.1.5 为什么要做图片理解

企业文档里很多关键信息不在正文，而在图里，比如：

- 面板位置
- 接线示意
- 按键布局
- 安装图
- 局部结构说明

如果只保留 ![](url)，模型在检索时并不知道那张图讲了什么。
所以项目在 node_md_img.py 中做了两件事：

1. 取图片在 Markdown 中的上下文
   通过 find_image_in_md_content
2. 调多模态模型生成摘要
   通过 summarize_image

最终 Markdown 中图片变成：

- 有真实 MinIO URL
- 有可检索的图片语义描述

这一步本质是在把“视觉信息文本化”。

### 2.1.6 为什么要做层级标题切片

原因是“切片不仅要短，还要语义完整”。

node_document_split.py 的设计不是简单按字符截断，而是两级处理：

1. 先按标题切
   step_2_split_by_title
   作用是尽量让一个 chunk 对应一个语义主题，比如“产品说明”“注意事项”。
2. 再做精修
   step_3_refine_chunks
   - 太长就二次切分
   - 太短就合并
   - 补齐 parent_title 和 part

这样做的好处：

- 检索召回更准
- Rerank 可分辨度更高
- 最终答案引用的上下文更完整
- 对步骤类、注意事项类问答更友好

### 2.1.7 为什么要识别 item_name

item_name 是这个项目里一个很有业务价值的设计。

它的作用有 3 个：

1. 检索前做实体范围收缩
   如果用户问的是“华为擎云B730 是什么”，那后续检索最好只在这个主体下找，不要在所有文档里搜。
2. 多轮对话中的指代承接
   用户前一句说“HAK180”，后一句问“它怎么开机”，系统需要知道“它”指哪个产品。
3. 导入治理
   早期它也参与去重/覆盖判断；后来版本管理优化后，文档幂等改为主要依赖 file_hash/doc_id/version，这样更安全。

识别逻辑在 node_item_name_recognition.py。

### 2.1.8 为什么使用 BGE-M3

BGE-M3 的价值在于“一个模型同时支持 dense + sparse + multilingual/multi-function 的混合检索能力”。

在这个项目里用它，有两个现实收益：

1. 模型统一
   不需要 dense 一个模型、sparse 再一个模型。
2. 支持混合检索
   可以同时利用：
   - dense 的语义泛化能力
   - sparse 的关键词精确匹配能力

封装在：

- embedding_utils.py

### 2.1.9 dense_vector 和 sparse_vector 分别解决什么问题

1. dense_vector
   解决语义相似问题。
   比如用户问“如何排查供电异常”，文档里写“检查电源线、保险丝和模块”，dense 更容易把它们匹配起来。
2. sparse_vector
   解决关键词、专业术语、型号、参数字面匹配问题。
   比如 HAK180、B730、88W、210-240V 这种信息，sparse 通常更稳。

因此它们不是替代关系，而是互补关系。

### 2.1.10 Milvus 中存了哪些字段

当前至少有两类集合：

1. kb_chunks
   在 node_import_milvus.py 定义了核心 schema：
   - chunk_id
   - file_title
   - item_name
   - content
   - title
   - parent_title
   - part
   - dense_vector
   - sparse_vector

同时通过动态字段还写入了：

- doc_id
- file_hash
- version
- status

1. kb_item_names
   在 node_item_name_recognition.py 定义：
   - pk
   - file_title
   - item_name
   - dense_vector
   - sparse_vector

动态字段还包括：

- doc_id
- file_hash
- version
- status

### 2.1.11 这些字段后续如何服务检索

1. content
   作为最终回答上下文。
2. title / parent_title / part
   用于保留结构信息，提升检索和展示可解释性。
3. item_name
   作为查询过滤条件，缩小搜索范围。
4. dense_vector / sparse_vector
   支撑 hybrid retrieval。
5. doc_id / version / status
   支撑版本管理、删除、旧版本隔离。
6. file_title
   用于来源展示和文档粒度治理。

### 2.1.12 用“用户上传一个 PDF 后发生了什么”的方式讲完整链路

可以这样讲：

用户上传一个 PDF 后，后端先通过 import_server.py 的 /upload 接口接收文件，并为这个文件生成一个 task_id 和本地工作目录。随后 LangGraph 导入工作流从 node_entry 开始，先判断文件类型、生成 file_hash，并检查这个文件是不是重复上传或者同名新版本。如果是重复上传，就直接返回，不再消耗后续解析资源。

如果是新文件或新版本，PDF 会进入 node_pdf_to_md.py，调用 MinerU 进行解析，拿到 Markdown 结果。接着进入 node_md_img.py，扫描 Markdown 中引用的图片，把图片上传到 MinIO，并调用多模态模型给图片生成语义描述，这样图片也能参与后续检索。

然后文档进入 node_document_split.py，先按标题层级切，再对过长内容做二次切片，对过短内容做合并，同时给每个 chunk 生成 chunk_hash。接着在 node_item_name_recognition.py 中，从文档前几个高价值切片中识别文档主体 item_name，并把 item_name 也建成独立向量索引。

之后 node_bge_embedding.py 会为所有 chunks 生成 BGE-M3 的 dense 和 sparse 向量。最后 node_import_milvus.py 把 chunks 写入 Milvus，同时根据版本信息清理旧版本数据，并把 DocumentMeta 和 ChunkMeta 写入 Mongo，形成一条可维护、可版本治理的知识库数据链路。

------

## 2.2 检索问答流程

### 2.2.1 问答入口在哪里

1. API 入口
   在 query_server.py 的 POST /query
2. SSE 入口
   在同文件的 GET /stream/{session_id}
3. LangGraph 入口
   在 main_graph.py 的 query_app

### 2.2.2 用户问题进入系统后经过哪些步骤

当前主链路大致是：

1. node_item_name_confirm
2. node_search_embedding
3. node_search_embedding_hyde
4. node_web_search_mcp
5. node_rrf
6. node_rerank
7. node_answer_output

如果 node_item_name_confirm 已经判断“无法确认商品”或“需要用户澄清”，会直接给 state["answer"]，然后跳到 node_answer_output 返回。

### 2.2.3 为什么要结合历史对话识别 item_name

因为多轮对话里用户经常不会重复完整实体名。

例如：

- 第一轮：“HAK180 是什么？”
- 第二轮：“它怎么操作？”
- 第三轮：“那电源故障怎么排查？”

如果不结合历史，第二轮和第三轮都很难正确过滤到目标文档。
当前逻辑在 node_item_name_confirm.py：

- 先通过 get_recent_messages
- 再让 LLM 同时做 item_names 提取和 rewritten_query

### 2.2.4 为什么要做 query rewrite

目的是把“用户自然问法”转成“更适合检索的表达”。

价值有 4 个：

1. 消除指代
   “它”“这个设备”“那个机器”要还原成实体。
2. 补齐上下文
   多轮对话中的省略要补全。
3. 去掉口语噪声
   “哈哈哈”“那个”“怎么说呢”这类噪声对检索没有帮助。
4. 提升召回率
   改写后的问题更像文档里的表述。

代码在 node_item_name_confirm.py 的 step_3_llm_item_name_and_rewrite_query

### 2.2.5 dense 检索、sparse 检索、hybrid 检索分别适合什么场景

这里要区分“线上主链路”和“评测适配层”。

当前线上主链路主要使用 hybrid，代码在 node_search_embedding.py。

离线评测里拆分了：

- dense_only
- sparse_only
- hybrid
  代码在 evaluator.py

适用场景是：

1. dense
   适合语义改写、近义表达、同义问法、故障描述类问题。
2. sparse
   适合型号、参数、缩写、专业名词、关键词强匹配。
3. hybrid
   适合企业知识库的主流场景，因为业务问题通常既有语义需求，也有术语约束。

### 2.2.6 HyDE 的作用是什么

HyDE 是“先让模型假设写出一段可能的答案，再用这段答案做 embedding 检索”。

它解决的是：

- 用户问题过短
- 文档原文和用户问法差距很大
- 直接 query embedding 表征不足

当前实现：

- 生成假设文档：step_1_create_hyde_doc
- 用 query + hyde_doc 做检索
  都在 node_search_embedding_hyde.py

### 2.2.7 为什么需要多路召回

单一路召回有天然盲区：

1. 纯向量召回
   可能漏掉关键词强相关内容
2. 纯 sparse
   可能漏掉语义改写问题
3. 本地知识库
   可能对补充背景、联网信息不足

所以当前系统做了多路召回：

- 本地 hybrid
- HyDE hybrid
- Web Search

这样可以提升 recall ceiling，也就是上限召回能力。

### 2.2.8 RRF 是如何解决多路结果分数不可比问题的

不同召回源的分数通常不可直接比较：

- dense score 一种分布
- sparse score 一种分布
- HyDE 召回又是一种
- web search 甚至没有同口径分数

所以当前用 RRF，不直接比较原始分数，而比较“排名位置”。
公式思想就是：

1 / (k + rank)

在 node_rrf.py 的 step_3_reciprocal_rank_fusion

它的优点是：

- 对不同来源分数尺度不敏感
- 只要求“哪个结果排得靠前”
- 很适合多路融合第一阶段

### 2.2.9 Rerank 为什么能提升最终准确率

召回阶段的目标是“别漏”，不是“绝对准”。
所以召回后的 topN 里通常会混进一些相关但不最优的 chunk。

Rerank 的作用是：

- 用 cross-encoder 直接看“问题 + 文档片段”配对
- 学更细粒度的语义匹配
- 把真正最适合作答的 chunk 排到最前

当前逻辑在 node_rerank.py

其中还加了一个“断崖截断”逻辑：

- 如果分数出现明显下跌，就截断 topk
- 避免把低质量结果塞进 prompt

这是一个很实用的工程优化点。

### 2.2.10 最终 Prompt 如何约束大模型回答

核心 Prompt 在：

- answer_out.prompt

当前约束重点有：

1. 只回答当前问题
   不要扩写无关背景
2. 只能依据参考内容作答
   不允许补充常识性推测
3. 如果参考内容只支持部分答案，就只答被支持的部分
4. 非视觉类问题禁止输出图片区块
5. 不允许编造图片 URL

另外，在 node_answer_output.py 中还有一个保险逻辑：

- step_0_strip_model_image_block
- 即使模型输出了 【图片】 区块，也会被剥掉，只保留程序真实抽取的图片结果

这属于“Prompt 约束 + 程序后处理”的双保险。

### 2.2.11 SSE 流式输出的作用是什么

SSE 解决的是用户体验问题：

1. 长答案不用等全部生成完
2. 前端可以边生成边展示
3. 过程中的节点进度也能推送

当前实现分两层：

1. 查询接口异步启动工作流
   在 query_server.py
2. 通过 push_to_session 往 SSE 队列发事件
   在 node_answer_output.py 和 task_utils.py

### 2.2.12 用“用户问一个问题后系统如何找到答案”的方式讲完整链路

可以这样讲：

用户通过 /query 提交一个问题后，系统先创建 session_id，然后进入查询 LangGraph 工作流。第一个节点 node_item_name_confirm.py 会结合历史对话，让 LLM 一次性做两件事：识别问题里涉及的 item_name，以及把用户问题改写成更适合检索的 rewritten_query。之后它会去 kb_item_names 里做 item_name 确认，判断是“已明确”“候选多个”还是“完全没匹配到”。

如果主体明确，系统进入多路召回。当前本地召回节点 node_search_embedding.py 会基于改写后的问题生成 BGE-M3 的 dense 和 sparse 向量，在 Milvus 中做 hybrid 检索，并且通过 item_name 过滤只在相关产品文档下搜索。同时，HyDE 节点 node_search_embedding_hyde.py 会先让 LLM 生成一段“假设性答案”，再把“问题 + 假设答案”一起做向量检索，用来补召回一些语义上接近但问法差距较大的内容。第三条支路 node_web_search_mcp.py 则通过 MCP 调用联网搜索，补充外部资料。

三路结果出来以后，系统在 node_rrf.py 用 RRF 做融合，因为多路结果的原始分数不可直接比较，所以这里按排名位置融合。融合后的结果再进入 node_rerank.py，由 BGE-Reranker 对“问题-候选文档”成对打分，选出真正最适合作答的 top 文档。

最后 node_answer_output.py 会把重排后的上下文、历史对话、item_name 和问题一起拼到 Prompt 中，交给 Qwen 生成答案。如果是流式模式，就通过 SSE 持续推送 token 和最终图片结果；如果是非流式模式，就直接返回最终答案。整个过程中，系统还会把对话写入历史记录，用于下一轮对话继续做 item_name 识别和 query rewrite。

------

# 三、优化一：RAG 离线评测体系

状态：

- 当前已实现基础离线评测模块，代码在 app/evaluation/
- 当前已实现检索指标、策略对比、LLM-as-Judge、Markdown 报告
- 当前已实现真实评测数据 schema 和样例集
- 这一部分是你当前仓库里最适合在面试中展示“不是拍脑袋做优化”的模块

------

## 3.1 为什么需要评测体系

### 3.1.1 为什么不能只说“效果不错”

因为“效果不错”没有可复现标准。

面试官会默认追问：

- 你怎么定义效果？
- 是召回变好了，还是生成看起来更顺？
- 你怎么知道 HyDE 真有用？
- 你怎么知道 Rerank 值得那点延迟？

如果你没有评测体系，所有回答都容易变成主观经验。

### 3.1.2 RAG 系统应该如何证明效果

至少要证明 3 件事：

1. 检索找没找对
   靠 Hit@K / Recall@K / MRR / NDCG@K
2. 生成答没答对
   靠 Faithfulness / Answer Relevance
3. 性能是否可接受
   靠 latency / p95 latency

当前这些都已经在：

- metrics.py
- evaluator.py

### 3.1.3 面试官可能会怎么问

典型问题：

1. 你怎么证明 hybrid 比 dense_only 好？
2. 你怎么判断 HyDE 值不值得保留？
3. Rerank 提升了多少，延迟增加了多少？
4. 你怎么定义“回答更可靠”？
5. 你有没有 bad case 复盘机制？

### 3.1.4 没有评测体系会暴露什么问题

1. 无法客观比较策略
2. 优化容易变成“堆技术名词”
3. 线上效果波动时无法定位问题在检索还是生成
4. 无法做回归测试
5. 面试里很难证明你具备系统性优化能力

------

## 3.2 评测数据集如何设计

当前 schema 在 dataset_schema.py

字段解释：

### 3.2.1 question

用户问题。
它定义了这条评测样本的输入。

### 3.2.2 item_names

期望关联的主体名称。
它的作用是：

- 评估 item_name 过滤是否合理
- 评估多轮上下文实体识别是否正确
- 有助于构造按产品维度的评测集

### 3.2.3 golden_chunk_ids

人工标注的正确 chunk_id 列表。
它是检索评测的黄金标准，用于判断：

- 是否召回到了正确证据
- 正确证据排得是否足够靠前

### 3.2.4 golden_answer

参考答案。
它主要服务于：

- answer_relevance
- 部分场景下的人类校验

### 3.2.5 category

问题类型。
它的作用是：

- 分类统计
- 看系统在哪类问题上最弱
- 支持后续专项优化，比如“故障排查”“参数查询”“步骤流程”

### 3.2.6 一条样本如何同时评估检索和生成

例如：

question: “HAK180 的电源故障如何排查？”
golden_chunk_ids: [chunk_a, chunk_b]
golden_answer: “应先检查电源线、保险丝和电源模块。”
category: “故障排查”

评测时：

1. 检索阶段看 topK 里有没有 chunk_a / chunk_b
2. 排序阶段看它们排在第几位
3. 生成阶段看答案是否真的回答了“如何排查”
4. 忠实度阶段看答案是否真的被这些 chunk 支撑

------

## 3.3 检索指标如何理解

### 3.3.1 Hit@K

含义：
topK 里只要出现任意一个正确 chunk，就算命中。

计算方式：
topK 中是否存在 golden_chunk_ids 之一

适合回答什么问题：
“系统能不能把至少一个对的证据捞出来？”

面试中怎么解释：
Hit@K 更像是“有没有找到门”的指标。对 RAG 来说，如果 top5 里连一个正确证据都没有，后面的生成再强也很难答对。

### 3.3.2 Recall@K

含义：
topK 命中的正确 chunk 数 / 全部 golden chunk 数

计算方式：
|topK ∩ golden| / |golden|

和 Hit@K 的区别：

- Hit@K 只关心“有没有”
- Recall@K 关心“找全了没有”

为什么对 RAG 很重要：
很多问题不是一个 chunk 就能完整回答，尤其是：

- 步骤类
- 多条件类
- 注意事项类
  如果 Recall 低，答案容易缺信息。

### 3.3.3 MRR

含义：
第一个正确结果的倒数排名。

为什么关注第一个正确结果的位置：
因为 RAG 通常会把前几个结果塞进 prompt，排得越靠前，被模型优先利用的概率越大。

对用户体验的影响：
如果第一个正确结果总排在第 5 名以后，即使 top5 hit 很高，生成效果也可能不稳定。

### 3.3.4 NDCG@K

含义：
衡量正确结果在排序中的整体质量。

为什么能衡量排序质量：
它不仅看“有没有正确结果”，还看“正确结果排得是不是越靠前越好”。

和 Recall@K 的区别：

- Recall@K 只看找到了多少
- NDCG@K 看找到了多少且排序好不好

### 3.3.5 latency / p95 latency

含义：

- latency：一次请求耗时
- p95 latency：95% 的请求都不会超过的耗时阈值

为什么 P95 比平均延迟更有参考价值：
平均值会掩盖长尾问题。
线上体验通常是被慢的那 5% 请求拖垮的，所以 P95 更能反映真实稳定性。

------

## 3.4 答案质量指标如何理解

### 3.4.1 Faithfulness

它评估：
答案中的核心结论是否被检索上下文支持。

意义：
这是控制幻觉最关键的指标之一。

如何减少幻觉：

1. 提高检索质量
2. 收紧 Prompt
3. 后验校验答案和证据是否一致
4. 在低证据场景下拒答

当前评测 Judge 在：

- eval_faithfulness_judge.prompt

### 3.4.2 Answer Relevance

它评估：
答案是否真的回答了用户问题，而不是只说了一堆相关背景。

这在当前项目里很重要，因为之前 bad case 里就出现过：

- 答案内容基本正确
- 但扩写太多，导致 relevance 分不高

当前 Judge Prompt 在：

- eval_answer_relevance_judge.prompt

### 3.4.3 LLM-as-Judge 的作用和风险

为什么可以用：
因为很多生成质量指标难以用规则算，LLM Judge 能在“是否回答问题”“是否有证据支持”这类任务上提供较强近似。

有什么局限：

1. Judge 自己也不完全稳定
2. Prompt 改一点，分数可能变化
3. Judge 可能偏好某种表达方式
4. 参考答案本身若不够标准，也会影响打分

如何降低不稳定性：

1. 强制 JSON 输出
2. Prompt 规则写清楚
3. 保留 reason
4. Judge 失败不影响整批评测
   当前已经实现了异常兜底，见 evaluator.py

------

## 3.5 策略对比如何讲

当前评测策略在 evaluator.py

### 3.5.1 dense_only

只用 dense 向量搜。
适合语义改写强、关键词约束弱的场景。

### 3.5.2 sparse_only

只用 sparse 向量搜。
适合型号、参数、术语、字面关键词很关键的场景。

### 3.5.3 hybrid

dense + sparse 混合检索。
通常是企业知识库最稳的 baseline。

### 3.5.4 hybrid_rrf

把 dense、sparse、hybrid 多路结果再做 RRF 融合。
适合想进一步榨取 recall 的场景。

### 3.5.5 hybrid_rrf_rerank

RRF 后再用 Reranker 精排。
目标是同时兼顾 recall 和 precision。

### 3.5.6 hyde_hybrid_rrf_rerank

再把 HyDE 路径也加进来。
适合语义 gap 大的问题，但成本和延迟更高。

### 3.5.7 为什么要做策略对比

因为“最复杂的策略”不一定最好。
必须用指标回答：

- 它有没有把 recall 提高
- 有没有把排序质量提高
- 提高多少
- 延迟成本值不值得

### 3.5.8 如何根据评测结果选择默认策略

一般看 3 个维度：

1. 检索效果
   Recall@K、MRR、NDCG@K
2. 答案质量
   Faithfulness、Answer Relevance
3. 性能
   P95 latency

如果一个策略：

- 准确率提升很小
- 延迟上涨很大
  那就不适合做默认策略。

### 3.5.9 如果 HyDE 提升召回但增加延迟，如何权衡

可以这样讲：

1. 先看提升的是不是核心场景
2. 如果只对少数长尾问题有效，可以作为 fallback 策略，而不是默认策略
3. 可以只在低置信度、短 query、语义模糊 query 时启用
4. 可以做异步预取或缓存

### 3.5.10 如果 Rerank 提升准确率但变慢，如何优化

1. 缩小 rerank 候选数
2. 使用断崖截断
3. 批量计算
4. 只对本地文档 rerank，不对所有 web 结果全量 rerank
5. 低成本模型做第一阶段过滤，高成本模型做第二阶段精排

### 3.5.11 面试时如何讲“我不是堆技术，而是用评测驱动优化”

你要强调：

我不是因为“HyDE 听起来高级”就加 HyDE，也不是因为“Rerank 很火”就无脑上 Rerank。我先构建了离线评测集，再把 dense、sparse、hybrid、RRF、Rerank、HyDE 拆成可比较策略，用 Recall@5、MRR、Faithfulness 和 P95 latency 去看真实收益。最后默认策略不是技术最复杂的，而是效果和成本最平衡的。

------

## 3.6 优化后的结果应该如何表述

### 3.6.1 简历版，2-3 行

构建了企业知识库 RAG 离线评测体系，设计真实评测集并对比 dense、sparse、hybrid、RRF、Rerank、HyDE 等检索策略在 Recall@5 / MRR / NDCG@5 / Faithfulness / P95 latency 上的表现。基于评测结果选择默认检索策略，而不是凭经验堆技术，显著提升了系统优化的可解释性和回归验证能力。

### 3.6.2 面试版，1-2 分钟

这个项目里我重点补了一套离线评测体系，因为我不想把 RAG 优化停留在“感觉效果还行”。我先基于 Milvus 里的真实 chunk 构建了评测集，每条样本都包含 question、golden_chunk_ids、golden_answer 和 category。然后在评测模块里把 dense、sparse、hybrid、RRF、Rerank、HyDE 这些策略拆开做对比，分别统计检索层的 Hit@K、Recall@K、MRR、NDCG@K，以及生成层的 Faithfulness、Answer Relevance，再加上 P95 latency 看性能成本。这样做的价值是，我能非常清楚地回答“某个策略到底提升了什么、代价是什么、值不值得保留”，而不是只会说“我加了个 Rerank 效果更好了”。

### 3.6.3 深挖版

我把评测分成三层。第一层是检索层，主要看 Hit@K 和 Recall@K，判断有没有召回正确证据，以及是否召回完整。第二层是排序层，看 MRR 和 NDCG@K，判断正确证据是不是排得足够靠前，因为这直接影响进入 Prompt 的内容质量。第三层是生成层，用 LLM-as-Judge 评估 Faithfulness 和 Answer Relevance，看答案是不是既忠实于上下文，又真正回答了用户问题。最后我会把这些指标和 P95 latency 放在一起看，选择最均衡的默认策略。这样面试里我可以明确说明，我做优化不是堆模型和算法，而是建立了一套可量化、可回归、可解释的优化闭环。

------

# 四、优化二：引用溯源、低置信度拒答、反幻觉校验

状态说明：

- 当前仓库**尚未完整实现**这一整套能力。
- 当前已实现的相关基础只有两点：

1. 更严格的答案 Prompt，见 answer_out.prompt
2. 模型图片区块剥离，见 node_answer_output.py

也就是说：

- supporting_sources：建议实现
- low confidence refusal：建议实现
- hallucination_checker：建议实现
- answer_rewriter：建议实现

下面这一节我会按“应该怎么设计、怎么在面试中讲”的方式展开，并明确标注哪些是建议实现。

------

## 4.1 为什么需要引用溯源

### 4.1.1 企业知识库为什么不能只返回答案

因为企业场景里，用户最关心的不只是“你说了什么”，还关心“你依据什么这么说”。

尤其在这些场景：

- 故障排查
- 安全规范
- 参数口径
- 售后政策
- 流程操作

如果只返回答案，没有来源：

- 用户无法验证
- 运营无法排错
- 面试官也会觉得系统可信度不够

### 4.1.2 supporting_sources 的作用

建议实现的 supporting_sources 应该承担 3 个角色：

1. 给前端展示证据来源
2. 给用户验证答案依据
3. 给开发定位 bad case

### 4.1.3 为什么要返回 chunk_id、file_title、title、content_preview、rerank_score

建议返回这些字段：

1. chunk_id
   便于精确定位 chunk
2. file_title
   告诉用户来自哪个文档
3. title
   告诉用户来自哪个章节
4. content_preview
   给用户一个短摘要，不必展示整段原文
5. rerank_score
   便于调试和前端排序展示，也能支撑低置信度策略

### 4.1.4 前端如何展示来源

建议展示方式：

- 答案下方“引用来源”
- 每条显示：文档名 / 标题 / 摘要 / 分数
- 点击可展开原文或跳转文档定位

### 4.1.5 用户如何验证答案

用户可以看：

- 这条结论来自哪个文档
- 来自哪个章节
- 原文是不是支持这句话

这会显著提升企业知识库的信任度。

------

## 4.2 supporting_sources 生成流程

状态：

- 当前代码中没有 supporting_sources 字段的正式实现
- 建议在 node_answer_output.py 前后扩展

建议设计：

### 4.2.1 它在 Rerank 前还是 Rerank 后生成

建议在 **Rerank 后生成**。

原因：

- Rerank 后的文档才是最终最可信的证据集
- 如果在 Rerank 前做，来源可能掺杂低质量召回结果

### 4.2.2 为什么选 Top-K chunk

因为：

- 太多会噪声大
- 太少可能覆盖不完整

建议：

- 默认取 reranked_docs 前 3 到 5 个

### 4.2.3 content_preview 如何截断

建议：

- 取前 120 到 200 字
- 保留完整句子边界更好
- 去掉过长 markdown 噪声

### 4.2.4 为什么要保留不同分数

建议保留：

- RRF score
- rerank score
  至少保留最终 rerank score

作用：

- 支撑后续置信度判断
- 支撑 bad case 排查
- 帮助分析“答案为什么用了这段”

### 4.2.5 最终 API 返回结构是什么

建议实现的 API 返回结构：

```
{  "answer": "...",  "supporting_sources": [    {      "chunk_id": "4656...",      "file_title": "hak180产品安全手册",      "title": "## 注意事项",      "content_preview": "请在断电状态下维护……",      "rerank_score": 0.93    }  ],  "metadata": {    "strategy": "hybrid_rrf_rerank",    "item_names": ["HAK180"],    "session_id": "..."  } } 
```

------

## 4.3 低置信度拒答

状态：

- 当前代码中没有正式的低置信度拒答模块
- 当前只有“item_name 不匹配时直接返回提示”的逻辑，见 node_item_name_confirm.py
- 基于 rerank_score 的拒答，建议实现

### 4.3.1 为什么 RAG 系统需要会拒答

因为乱答比不答更危险。
尤其是企业知识库场景，如果证据不够却硬答，会导致：

- 错误操作
- 误用参数
- 不可信的系统印象

### 4.3.2 什么情况下应该拒答

建议场景：

1. 无召回结果
2. 有召回，但 top1_rerank_score 很低
3. 平均分也低，说明整体上下文质量差
4. 问题明显超出知识库范围
5. 多个候选 item_name 冲突，用户未澄清

### 4.3.3 top1_rerank_score 阈值怎么理解

建议实现时可这样解释：

- 它代表“最相关证据的匹配强度”
- 如果 top1 都很低，说明最强证据都不够可信

例如：

- \>= 0.8：高可信
- 0.6 - 0.8：谨慎生成
- < 0.6：倾向拒答
  具体阈值需要通过评测数据调优。

### 4.3.4 avg_rerank_score 阈值怎么理解

它代表“整体证据集合质量”。

如果：

- top1 高，但后面都很差
  可能是单点命中，回答要更保守

如果：

- top1 和平均都高
  说明证据整体一致性强

### 4.3.5 如果没有 rerank_score，如何降级判断

可以降级看：

1. 召回数量
2. topK 命中是否集中在同一文档
3. item_name 识别置信度
4. 是否存在足够上下文长度
5. dense/hybrid 原始分数

### 4.3.6 拒答话术为什么要明确、保守

因为模糊拒答会让用户误以为系统坏了。
好的拒答应该明确说明：

- 目前没找到足够依据
- 如果有需要，请补充产品名/场景/错误现象

### 4.3.7 三种场景应该如何处理

1. 无召回结果
   建议返回：
   “当前知识库中未检索到与该问题相关的内容，请补充更具体的产品名称或问题场景。”
2. 有召回但分数很低
   建议返回：
   “已检索到部分相关内容，但证据不足以支持可靠回答，建议补充更具体的问题描述。”
3. 问题超出知识库范围
   建议返回：
   “当前知识库中没有覆盖该问题所需的信息，无法给出可靠回答。”

------

## 4.4 反幻觉校验

状态：

- 当前仓库里没有 hallucination_checker 与 answer_rewriter
- 这是非常值得做的下一步，建议实现

### 4.4.1 为什么生成答案后还要校验

因为即使上下文正确，LLM 也可能：

- 补常识
- 过度扩写
- 把多个 chunk 的信息拼错
- 编造未被支持的细节

所以生成后做一次“答案-证据一致性检查”很有价值。

### 4.4.2 hallucination_checker 做了什么

建议实现：

- 输入：问题、答案、检索上下文
- 输出：
  - score
  - unsupported_claims
  - reason

### 4.4.3 它如何判断答案是否被上下文支持

建议方式：

1. 用 LLM Judge 判定每个核心结论是否能在 context 找到依据
2. 或进一步拆句后逐句判定

### 4.4.4 unsupported_claims 是什么

建议定义为：

- 答案中那些在上下文里找不到支持的关键断言

例如：

- 答案说“需要重启设备”
- 但上下文里根本没提“重启”
  那它就是 unsupported claim

### 4.4.5 score 阈值如何决定

建议实现策略：

1. \>= 0.75
   正常返回
2. 0.5 - 0.75
   重写答案，去掉不被支持的部分
3. < 0.5
   拒答

### 4.4.6 answer_rewriter 如何删除不被支持的内容

建议实现方式：

- 读取 unsupported_claims
- 让 LLM 在“不新增信息”的前提下重写
- 只保留被 context 支撑的内容

### 4.4.7 如果校验器失败，为什么不能影响主流程

因为校验器是 guardrail，不应该成为主流程单点故障。
建议：

- 校验失败时记录日志
- 返回原答案或更保守答案
- 不能让整个问答 500

------

## 4.5 优化后的问答流程

文字版流程图：

用户问题
-> item_name 识别
-> query rewrite
-> 多路召回
-> RRF
-> Rerank
-> 置信度判断
-> 低置信度拒答 / 继续生成
-> LLM 生成答案
-> hallucination_checker
-> 正常返回 / 重写 / 拒答
-> 返回 answer + supporting_sources + metadata

------

## 4.6 面试话术

### 4.6.1 简历版，2-3 行

在现有企业知识库 RAG 系统上规划并推进引用溯源、低置信度拒答与反幻觉校验能力，目标是把系统从“能回答”升级为“可信回答”。设计基于 rerank_score 的置信度策略和答案后验校验链路，提升结果可解释性与企业场景可用性。

### 4.6.2 面试版，1-2 分钟

我认为企业级 RAG 系统不能只追求“能答出来”，还必须解决“答得是否可信”。所以我在这个项目里重点规划了三层护栏。第一层是引用溯源，也就是返回最终答案时把对应的 chunk_id、file_title、title、content_preview、rerank_score 一起返回，让用户知道答案依据来自哪里。第二层是低置信度拒答，我会结合 top1_rerank_score、平均分和召回情况，判断当前证据是否足够，如果不够就明确拒答，而不是让模型硬答。第三层是答案后验校验，也就是生成答案后，再用一个校验器判断答案中的核心结论是否真的被上下文支持。如果支持度高就直接返回，如果中等就重写删除不被支持的部分，如果很低就拒答。这样系统不仅更准，而且更适合企业落地。

### 4.6.3 深挖版：你如何降低 RAG 幻觉？

我会从三层做。第一层是提高检索质量，因为很多幻觉本质是“没找对证据”；这里通过 item_name 过滤、多路召回、RRF 和 Rerank 提高证据质量。第二层是 Prompt 约束，要求模型只能依据参考内容作答，并且不要主动扩写。第三层是答案后验校验，也就是生成后再判断哪些结论没有证据支持，必要时重写或者拒答。我的理解是，幻觉不能靠单一技巧解决，而是要靠“检索、生成、校验”三层共同约束。

### 4.6.4 深挖版：你如何判断什么时候不回答？

核心不是“模型会不会答”，而是“证据够不够”。我会优先看是否有召回结果，其次看 top1 和整体 rerank 分数，如果最相关证据都很弱，或者问题明显超出知识库覆盖范围，我就倾向拒答。另外如果 item_name 都没确认清楚，我也不会直接生成。企业知识库里，错误回答的代价通常比拒答更高，所以拒答能力是可靠性的一部分。

### 4.6.5 深挖版：为什么不能完全相信 LLM 生成的答案？

因为 LLM 的目标是生成高概率自然语言，不是天然做证据约束推理。即使给了上下文，它也可能补常识、做错误归纳，或者把多段内容拼错。所以我不会把 LLM 当作最终真理，而是把它当作“基于证据做语言生成”的组件。真正可靠的系统要有检索、排序、约束和校验机制，而不是只相信生成本身。

------

# 五、优化三：文档增量更新、版本管理、任务状态、失败重试

状态说明：

- file_hash / chunk_hash / DocumentMeta / ChunkMeta / 版本替换 / 文档删除：当前已实现
- 持久化 ImportTask、失败断点恢复、标准化重试策略：当前**尚未完整实现**
- 当前任务状态仍主要是内存态 task_utils.py

------

## 5.1 为什么需要增量更新

### 5.1.1 企业知识库文档会不断变化

产品手册、参数、故障说明、政策文档都会持续更新。
如果系统不能区分“新文档”“重复文档”“同名新版本”，知识库很快就会脏。

### 5.1.2 重复上传会造成什么问题

1. 重复占用存储
2. Milvus 里重复 chunk 污染召回
3. 相同文档不同 doc_id 导致版本混乱
4. 评测和运营无法判断哪条是最新数据

### 5.1.3 同名文件更新会造成什么问题

如果只按文件名覆盖，不看内容：

- 可能误判为同一文档
- 新内容无法形成新版本
- 旧 chunk 仍在向量库中

### 5.1.4 旧版本 chunk 如果不删除会造成什么问题

1. 新旧内容同时被召回
2. 用户可能拿到过期答案
3. Rerank 会在新旧冲突内容里混排
4. 可解释性和可信度下降

### 5.1.5 为什么这体现工程化能力

因为这不是模型能力，而是系统治理能力。
面试里很多人能讲 RAG，但很少有人能讲：

- 文档幂等
- 版本替换
- 元数据治理
- 向量库脏数据清理

------

## 5.2 file_hash 和 chunk_hash

### 5.2.1 file_hash 的作用

当前在 node_entry.py 里导入最前面计算 file_hash。

作用：

1. 判断是否重复上传
2. 作为文档内容级唯一标识
3. 支撑版本管理与元数据治理

### 5.2.2 chunk_hash 的作用

当前在 node_document_split.py 的 step_4_1_fill_chunk_hashes 中生成。

作用：

1. 标识 chunk 核心内容
2. 支撑后续 chunk 级别增量判断
3. 便于后续更细粒度的增量更新

### 5.2.3 如何判断重复上传

当前逻辑：

- 先算 file_hash
- 再查 Mongo document_meta 是否存在 ACTIVE 且相同 file_hash
  代码在 document_meta_repository.py 的 find_active_by_file_hash

如果存在：

- 标记 DUPLICATED
- 直接结束导入图

### 5.2.4 如何判断同名新版本

当前逻辑：

- 查 file_title 的最新版本
- 如果同名但 file_hash 不同
- 则新建 doc_id
- version = old_version + 1

代码在 node_entry.py

### 5.2.5 如何避免重复入库

靠两层：

1. file_hash 判重
2. 重复上传直接在入口结束，不进入后续节点

### 5.2.6 如何避免旧数据污染检索结果

当前靠：

1. 新版本导入前删除旧版本 chunk
2. Mongo 里旧版本标记 REPLACED
3. 删除文档后标记 DELETED
4. Milvus 按 doc_id / file_title / version / item_name 提供删除工具
   在 milvus_utils.py

------

## 5.3 DocumentMeta 和 ChunkMeta

当前实现文件：

- document_meta_repository.py

### 5.3.1 DocumentMeta 字段意义

1. doc_id
   文档全局唯一 ID，也是版本管理和删除的主键。
2. file_title
   原始文件名去后缀后的标题，用于版本族归类。
3. file_hash
   内容哈希，用于去重。
4. file_type
   区分 pdf/md/txt。
5. item_name
   文档主体，用于检索过滤。
6. version
   版本号。
7. status
   当前文档状态：ACTIVE / REPLACED / DELETED / FAILED / DUPLICATED
8. chunk_count
   该文档导入后生成了多少 chunk。
9. created_at / updated_at
   审计与排序用。

当前仓库中 minio_urls/source_path/local_dir 也一并存了，这属于额外扩展字段。

### 5.3.2 ChunkMeta 字段意义

1. chunk_id
   Milvus chunk 主键
2. doc_id
   它属于哪个文档版本
3. file_hash
   来自哪个文件内容版本
4. chunk_hash
   该 chunk 的内容标识
5. title
   该 chunk 的标题
6. item_name
   所属主体
7. milvus_collection
   来自哪个向量集合
8. status
   ACTIVE / DELETED
9. created_at / updated_at
   审计与删除治理

### 5.3.3 它们和 Milvus 数据之间的关系

可以这样理解：

1. Milvus 是“高性能检索存储”
2. Mongo 元数据是“治理与审计存储”

Milvus 负责：

- 搜索
- 召回
- 向量索引

Mongo 负责：

- 文档去重
- 版本管理
- 状态追踪
- 删除审计
- 列表展示

也就是：

- Milvus 管“查”
- Meta Repository 管“管”

------

## 5.4 文档版本管理流程

### 5.4.1 场景一：第一次上传文件

metadata 如何变化：

- 新建 doc_id
- version=1
- status=ACTIVE

Milvus 中 chunk 如何变化：

- 新增一批 chunks
- 新增一条 item_name 记录

检索如何避免脏数据：

- 不涉及旧版本，直接使用当前数据

### 5.4.2 场景二：重复上传完全相同文件

metadata 如何变化：

- 命中相同 file_hash
- 返回 DUPLICATED
- 不创建新版本

Milvus 中 chunk 如何变化：

- 不新增
- 不重复入库

检索如何避免脏数据：

- 因为没有新写入，不会产生重复 chunk

### 5.4.3 场景三：上传同名但内容变化的文件

metadata 如何变化：

- 新建新 doc_id
- version = old_version + 1
- 旧版本状态改为 REPLACED
- 新版本 ACTIVE

Milvus 中 chunk 如何变化：

- 旧版本 chunk 清理
- 新版本 chunk 写入

检索如何避免脏数据：

- 旧 chunk 从 Milvus 删除
- 旧 chunk_meta 标记 DELETED
- 只保留新版本参与检索

### 5.4.4 场景四：删除某个文档

metadata 如何变化：

- DocumentMeta.status = DELETED
- ChunkMeta.status = DELETED

Milvus 中 chunk 如何变化：

- kb_chunks 按 doc_id 删除
- kb_item_names 按 doc_id 删除

检索如何避免脏数据：

- 删除后该文档不再被召回

### 5.4.5 场景五：重新导入失败文档

状态说明：

- 当前 /documents/{doc_id}/reimport API 已加到 import_server.py
- 但“失败节点恢复”和“持久化 ImportTask”还没有完整实现
- 当前更接近“重新发起导入”而不是“从失败节点继续”

metadata 如何变化：

- 失败文档可保留 FAILED
- 重新导入时建议生成新任务

Milvus 中 chunk 如何变化：

- 需要保证失败前半成品不会污染最终检索

检索如何避免脏数据：

- 通过 status 和删除逻辑控制
- 断点恢复能力当前建议实现

------

## 5.5 导入任务状态管理

状态说明：

- 当前项目已经有“任务状态管理”
- 但还是**单进程内存态**
- 文件在 task_utils.py

### 5.5.1 为什么长流程导入需要 ImportTask

因为导入不是瞬时动作，而是长流程：

- 上传
- PDF 解析
- 图片理解
- 切片
- embedding
- 入库

如果没有任务对象：

- 前端不知道进度
- 中途失败没法定位
- 用户体验很差

### 5.5.2 当前已有状态含义

当前已有：

- pending
- processing
- completed
- failed

它们分别在 task_utils.py

你要求的这些状态：

- PENDING
- RUNNING
- SUCCESS
- FAILED
- CANCELED

其中：

- PENDING/RUNNING/SUCCESS/FAILED 可以视为当前的规范化升级版
- CANCELED 当前尚未实现，建议实现

### 5.5.3 current_node 的作用

当前代码中没有正式持久化的 current_node 字段。
但前端能通过：

- running_list
- done_list
  看到当前运行节点。

建议实现：

- 在持久化 ImportTask 中单独保存 current_node

### 5.5.4 error_message 和 error_stack 的作用

当前只在任务结果里保存了 error 字符串。
还没有正式的 error_stack 持久化。

建议实现：

- error_message 给前端展示
- error_stack 给排障和监控系统使用

### 5.5.5 retry_count 和 max_retry_count 的作用

当前尚未实现，建议实现。
作用是：

- 控制自动重试次数
- 防止无限重试
- 支撑重试策略分析

### 5.5.6 前端如何通过 task_id 查询进度

当前已实现：

- /status/{task_id}
  在 import_server.py

返回：

- status
- done_list
- running_list
- result.doc_id
- result.document_status
- result.error

------

## 5.6 失败重试

### 5.6.1 为什么不能简单重新执行

因为导入流程不是纯函数，里面有副作用：

- MinIO 上传
- Milvus 写入
- Mongo 元数据写入

简单重跑可能导致：

- 重复 chunk
- 元数据状态错乱
- 半成品污染

### 5.6.2 如何避免重复写入 Milvus

靠幂等设计：

1. 入口 file_hash 去重
2. 版本替换前删除旧版本
3. 按 doc_id 精确删除
4. chunk 与文档元数据联动

### 5.6.3 从失败节点恢复和从头重试的区别

1. 从失败节点恢复
   更省资源，但实现复杂，需要中间状态持久化。
2. 从头重试
   实现简单，第一版更适合落地，但要保证幂等。

### 5.6.4 第一版为什么可以先实现从头重试

因为它最重要的是先保证正确性，而不是极致效率。
只要：

- 中间副作用可清理
- 写入幂等
  从头重试是完全合理的第一阶段方案。

### 5.6.5 重试前要清理哪些临时数据

建议清理：

1. 本地中间目录
2. MinIO 临时图片
3. 失败的 Milvus 半成品
4. 对应的 FAILED 元数据状态或重试标记

### 5.6.6 如何保证幂等性

可以这样讲：

导入幂等主要靠三个层面。第一层是文档级幂等，使用 file_hash 判断重复上传，如果内容完全相同，就直接返回 DUPLICATED，不会重复解析和入库。第二层是版本级幂等，同名不同内容会生成新的 doc_id 和 version，并在新版本写入前清理旧版本 chunk，避免新旧版本同时参与检索。第三层是元数据级幂等，Mongo 中的 DocumentMeta 和 ChunkMeta 会记录文档和切片状态，保证删除、替换和失败场景下都能有明确状态可追踪。这样即使流程重试，也不会把数据写乱。

------

## 5.7 面试话术

### 5.7.1 简历版，2-3 行

实现了企业知识库文档版本管理与增量更新能力，基于 file_hash/chunk_hash 做重复上传识别、同名新版本替换和旧 chunk 清理，并引入 DocumentMeta/ChunkMeta 元数据仓库治理文档状态。导入侧支持按 doc_id 精确删除和版本审计，显著提升了知识库长期维护能力。

### 5.7.2 面试版，1-2 分钟

我在这个项目里做的一项比较工程化的优化，是把知识库从“能导入”升级成“能长期维护”。核心是引入了 file_hash 和 chunk_hash，并在 Mongo 里增加 DocumentMeta 和 ChunkMeta 两张元数据表。这样同一个文件重复上传时，可以直接识别为 DUPLICATED，跳过整个解析和向量化流程；如果是同名但内容变化的文件，就会生成新的 doc_id 和 version，同时把旧版本状态标记为 REPLACED，并从 Milvus 中清理旧版本 chunk，避免旧数据继续参与检索。这样做的价值是，知识库不再只是“把文档塞进向量库”，而是具备了版本治理、幂等导入和长期维护能力。这类能力在企业场景里很重要，也能很好体现工程化水平。

### 5.7.3 深挖版：如果用户重复上传文档怎么办？

我会先在导入入口计算 file_hash，然后去元数据仓库里查是否已经存在相同 file_hash 且状态为 ACTIVE 的文档。如果存在，就直接判定为重复上传，返回已有 doc_id 和 DUPLICATED 状态，后续解析、切片、向量化和入库都不再执行。这样既避免浪费资源，也避免向量库里出现重复 chunk。

### 5.7.4 深挖版：如果文档更新后旧数据还在向量库里怎么办？

这正是版本管理要解决的问题。对于同名但内容变化的文件，我会创建新版本，同时把旧版本状态改为 REPLACED，并按 doc_id 或 file_title + version 从 Milvus 中删除旧版本 chunk。这样可以保证检索只命中新版本数据。如果旧数据不清掉，就会出现新旧内容混合召回，导致答案不稳定甚至冲突。

### 5.7.5 深挖版：导入流程中某一步失败了你怎么处理？

第一层是把失败状态记录清楚，包括失败文档的 doc_id、file_hash、version 和 FAILED 状态，这样至少可以追踪。第二层是保证失败不会把脏数据留在最终检索链路里，比如半写入的 chunk 要能被后续重试清理。第三层是重试策略，第一版我更倾向于从头重试，但前提是整个流程已经做了幂等设计，这样重跑不会重复污染数据。后续可以再演进到从失败节点恢复。

### 5.7.6 深挖版：如何保证导入流程幂等？

幂等不是靠一个点实现的，而是靠整条链路。入口通过 file_hash 保证重复文件不重复处理；版本替换通过 doc_id/version/status 保证同名更新有明确的新旧关系；Milvus 删除工具通过 doc_id 做精确清理；元数据层记录 ACTIVE/REPLACED/DELETED/FAILED，保证任何时候都能知道一份文档当前处于什么生命周期状态。这样就算重试、多次上传或人工删除，系统状态也不会乱。

------

# 六、整体优化前后对比

| 维度       | 优化前                     | 优化后                                     | 带来的价值         | 面试中怎么讲                                            |
| :--------- | :------------------------- | :----------------------------------------- | :----------------- | :------------------------------------------------------ |
| 检索效果   | 主要靠经验调策略           | 有离线评测集和指标对比                     | 优化变得可量化     | 我不是拍脑袋调参，而是用 Recall、MRR、NDCG 驱动策略选择 |
| 答案可信度 | 主要依赖 Prompt            | 建议引入 supporting_sources + 低置信度拒答 | 用户更信任系统     | 从“能答”升级为“可信答”                                  |
| 幻觉控制   | 主要靠 Prompt 约束         | 建议加入后验 hallucination check           | 降低错误扩写       | 幻觉控制是检索、生成、校验三层联动                      |
| 可观测性   | 有基础任务状态             | 已有导入状态查询，评测报告可追踪           | 更容易排障和回归   | 我补了评测和任务状态，不再是黑盒流程                    |
| 文档维护   | 容易重复入库、同名覆盖风险 | 已实现 file_hash/chunk_hash + 版本管理     | 知识库可长期维护   | 体现工程化，而不是一次性 demo                           |
| 失败恢复   | 失败后主要靠人工重跑       | 当前已有 FAILED 状态，重试能力建议继续实现 | 便于扩展可靠性治理 | 第一版先保证幂等，后续再做断点恢复                      |
| 面试竞争力 | 只能讲 RAG 基础流程        | 能讲评测、可信性、版本治理                 | 显著拉开差距       | 说明我不仅会调模型，也会做系统设计                      |
| 工程化程度 | 偏研发验证                 | 已有评测和版本治理，任务持久化待升级       | 更接近企业落地     | 可以讲“从 demo 到企业级”的演进路径                      |

------

# 七、项目架构图文字版

```
[Frontend]  |- 导入页面: /Users/liwenye/PycharmProjects/study_agent/dataset_rag/app/import_process/page/import.html  |- 问答页面: /Users/liwenye/PycharmProjects/study_agent/dataset_rag/app/query_process/page/chat.html [API Layer]  |- 导入服务: /Users/liwenye/PycharmProjects/study_agent/dataset_rag/app/import_process/api/import_server.py  |- 问答服务: /Users/liwenye/PycharmProjects/study_agent/dataset_rag/app/query_process/api/query_server.py [Import Service]  |- LangGraph Import Workflow  |  |- node_entry  |  |- node_pdf_to_md  |  |- node_md_img  |  |- node_document_split  |  |- node_item_name_recognition  |  |- node_bge_embedding  |  |- node_import_milvus   |- MinerU  |  |- PDF -> Markdown 解析   |- MinIO  |  |- 图片上传  |  |- 图片公网 URL   |- VLM / 多模态模型  |  |- 图片摘要生成   |- Embedding Service  |  |- BGE-M3 dense_vector  |  |- BGE-M3 sparse_vector   |- Milvus  |  |- kb_chunks  |  |- kb_item_names   |- Metadata Repository  |  |- MongoDB document_meta  |  |- MongoDB chunk_meta [Query Service]  |- LangGraph Query Workflow  |  |- node_item_name_confirm  |  |- node_search_embedding  |  |- node_search_embedding_hyde  |  |- node_web_search_mcp  |  |- node_rrf  |  |- node_rerank  |  |- node_answer_output   |- History Store  |  |- Mongo 对话历史   |- Retriever  |  |- item_name 确认  |  |- hybrid retrieval  |  |- HyDE retrieval  |  |- web search   |- Rerank Service  |  |- BGE-Reranker   |- Answer Generation  |  |- Qwen  |  |- answer_out.prompt   |- SSE  |  |- 流式 token 推送  |  |- 进度事件推送 [Evaluation]  |- /Users/liwenye/PycharmProjects/study_agent/dataset_rag/app/evaluation/  |  |- dataset_schema.py  |  |- metrics.py  |  |- evaluator.py  |  |- report.py  |  |- run_eval.py [建议实现的 Guardrails]  |- supporting_sources builder  |- confidence checker  |- hallucination_checker  |- answer_rewriter [建议实现的 Task Persistence]  |- ImportTask repository  |- retry policy  |- current_node / error_stack / retry_count 
```

------

# 八、面试官可能追问的问题

下面给 36 个常见追问和参考回答。

## 1. RAG 基础

### 1. 你这个项目里 RAG 的核心链路是什么？

先做文档结构化导入，再在问答时做 item_name 识别、query rewrite、多路召回、RRF、Rerank 和最终生成。重点不只是“查向量库”，而是让召回、排序和生成形成闭环。

### 2. 你为什么不用直接把全文塞给大模型？

成本高、上下文有限，而且企业文档太长。更关键的是全文塞给模型没有检索能力，也不利于来源追踪和版本治理。

### 3. 你觉得 RAG 系统最难的点是什么？

不是“接一个向量库”，而是三件事一起做好：文档处理质量、检索质量和生成可信度。工程上还要解决版本管理、评测和幂等。

## 2. 文档切片

### 4. 你为什么不直接按固定 500 字切片？

固定长度切片容易把标题和正文拆散，语义完整性很差。这个项目先按标题切，再对长段落二次切，对短片段合并，效果更稳。

### 5. 你怎么确定 chunk 大小？

当前代码里最大长度是 2000，短 chunk 合并阈值是 500，见 node_document_split.py。本质是平衡语义完整性和检索粒度。

### 6. 为什么要保留 parent_title 和 part？

这样 chunk 不是孤立文本，而是带结构信息。后续检索命中后，能知道它来自哪个标题层级、是不是某段的第几部分。

## 3. Embedding

### 7. 为什么选 BGE-M3？

因为它能同时生成 dense 和 sparse 表征，适合企业知识库这种“语义 + 术语”混合场景，模型统一也更利于工程维护。

### 8. dense 向量的优势是什么？

擅长语义泛化，适合用户问法和文档表述不一致的时候。

### 9. sparse 向量的优势是什么？

擅长精确命中型号、参数、缩写和专业词，尤其对产品型号场景很有用。

## 4. 稠密检索与稀疏检索

### 10. 为什么 hybrid 往往比 dense_only 更稳？

因为企业问题通常既有语义信息，也有关键词约束。dense 抓语义，sparse 抓字面，混合起来更稳。

### 11. 什么情况下 sparse_only 可能反而更好？

比如用户问的就是型号、参数、具体术语，这时候关键词匹配非常强，sparse 可能更直接。

## 5. HyDE

### 12. HyDE 的核心原理是什么？

先让 LLM 生成一段假设性答案，再把这段答案做 embedding 去搜，这样检索向量更接近“知识答案空间”。

### 13. HyDE 的风险是什么？

延迟更高，而且如果假设性答案本身偏了，可能把召回带偏。所以它适合做增强路，不一定适合默认路。

## 6. RRF

### 14. 为什么要用 RRF，而不是直接拼分数？

因为不同召回源的分数分布不可比。RRF 只看排名位置，鲁棒性更好。

### 15. RRF 的局限是什么？

它不利用原始分数强弱，只利用排名。如果某一路噪声很多，也可能把低质量结果带进来，所以后面还要接 rerank。

## 7. Rerank

### 16. Rerank 为什么通常放在召回之后？

因为 cross-encoder 计算贵，不适合对全库做。先召回一小批候选，再精排，是效果和成本的平衡。

### 17. 你这个项目里为什么加了“断崖截断”？

因为 rerank 后不是所有 topN 都值得进 prompt。如果分数出现明显断崖，后面的结果大概率噪声更高，截断可以提高答案质量。

## 8. Milvus

### 18. 你在 Milvus 里存了什么？

两类集合：kb_chunks 和 kb_item_names。前者存 chunk 内容和向量，后者存 item_name 及其向量，用于问答前的实体确认。

### 19. 为什么不只存向量，不存 metadata？

因为后续要做过滤、版本管理、删除、来源展示和审计，metadata 很关键。

### 20. 删除文档时为什么还要删 kb_item_names？

因为 item_name 也是索引的一部分。如果删了 chunk 不删 item_name，后续问题仍可能先命中过期实体。

## 9. LangGraph

### 21. 为什么用 LangGraph，而不是普通函数串联？

因为这里不是单条直线流程，而是有状态流转、条件分支、并行召回和不同终止路径。LangGraph 更适合表达这种工作流。

### 22. 这个项目里 LangGraph 的价值最直接体现在哪？

导入图里体现在“PDF / Markdown / 重复上传提前结束”的条件路由；查询图里体现在“item_name 不明确时提前返回”和“多路召回汇合”。

## 10. 多轮对话

### 23. 为什么要结合历史消息做 item_name 识别？

因为用户第二轮、第三轮常常不会重复说完整实体名。历史对话能把“它”“这个设备”还原回真实产品。

### 24. 只做 query rewrite，不做 item_name 确认行不行？

不够。query rewrite 解决问法问题，item_name 确认解决检索范围问题。两者是互补的。

## 11. item_name 识别

### 25. 为什么不直接用文件名当 item_name？

文件名可以兜底，但不稳定。真实产品主体可能和文件名不完全一致，所以当前是让 LLM 从文档内容里识别。

### 26. item_name 识别错了怎么办？

当前会再去 kb_item_names 做向量确认和打分，不是直接信模型原始输出。如果高分不唯一，还会给出候选澄清。

## 12. 反幻觉

### 27. 为什么不能只靠 Prompt 控制幻觉？

Prompt 只能约束趋势，不能保证结果一定被证据支持。企业场景更稳的做法一定是检索、生成、校验三层结合。

### 28. 你现在项目里已经实现反幻觉了吗？

严格说还没有完整实现后验校验链。当前已实现的是更严格的答案 Prompt 和图片区块剥离。完整的 hallucination_checker 还属于建议实现。

## 13. 评测指标

### 29. 为什么 Recall@K 对 RAG 很重要？

因为 RAG 不是只要一个对的 chunk 就够了，很多问题需要多段证据。Recall 低，答案容易缺关键信息。

### 30. 为什么还要看 MRR？

因为第一个正确结果排得越靠前，越有机会进入 prompt 前部，被模型优先利用。

### 31. 为什么 Faithfulness 和 Relevance 要分开看？

一个答案可以很相关但不忠实，也可以很忠实但没真正回答问题。这两个维度不能混为一个分数。

## 14. 性能优化

### 32. 如果线上延迟高，你先优化哪里？

我会先分解耗时，看是 embedding、Milvus 搜索、HyDE 生成、rerank 还是 answer generation。通常 HyDE 和 rerank 是优先怀疑对象。

### 33. 如果 HyDE 太慢但对少量问题有效，你怎么做？

把它变成条件触发策略，比如只在低置信度、短 query、语义模糊 query 场景启用。

## 15. 增量更新

### 34. 为什么要做 file_hash 和 chunk_hash？

file_hash 解决文档级幂等和版本判定，chunk_hash 为后续 chunk 级增量更新打基础。它们是知识库可维护性的关键。

### 35. 如果同名文件内容更新了，你怎么保证旧内容不再被召回？

新版本会生成新的 doc_id/version，旧版本 chunk 会从 Milvus 删除，旧文档状态改成 REPLACED，这样检索只会命中新版本。

## 16. 失败重试与系统设计

### 36. 导入流程失败后你会怎么处理？

第一版我优先保证幂等和状态可追踪，也就是失败要能标记 FAILED，重复重试不重复污染数据。之后再演进到持久化 ImportTask 和从失败节点恢复。

------

# 九、项目讲解稿

------

## 9.1 30 秒版本

这是一个基于 LangGraph 的企业级 RAG 知识库项目，分成导入和问答两条工作流。导入侧支持 PDF/Markdown、图片理解、层级切片、item_name 识别和 BGE-M3 混合向量入库；问答侧支持多轮对话、item_name 识别、query rewrite、多路召回、RRF、Rerank 和流式回答。我后续重点补了离线评测体系，以及文档版本管理和增量更新能力，让它从一个能跑的 RAG 系统升级成更可维护、可评估、可工程落地的知识库系统。

------

## 9.2 1 分钟版本

我做的是一个企业级 RAG 知识库系统，核心难点不在于“接个向量库”，而在于怎么把复杂企业文档稳定处理、可靠检索并可长期维护。项目导入侧基于 LangGraph 编排，支持 PDF 通过 MinerU 转 Markdown、图片上传 MinIO 并调用多模态模型做图片摘要、基于标题层级切片、识别文档主体 item_name，然后用 BGE-M3 生成 dense 和 sparse 向量写入 Milvus。查询侧同样是 LangGraph 工作流，先结合历史对话识别 item_name 并重写 query，再做本地 hybrid 检索、HyDE 检索和 web search，多路结果通过 RRF 融合，再用 BGE-Reranker 精排，最后由 Qwen 生成答案并通过 SSE 流式返回。为了让项目更像企业系统而不是 demo，我重点补了两块能力：一块是离线评测体系，能用 Recall、MRR、Faithfulness 和 P95 latency 比较不同策略；另一块是文档版本管理，基于 file_hash、chunk_hash、DocumentMeta 和 ChunkMeta 解决重复上传、同名覆盖和旧 chunk 污染问题。

------

## 9.3 3 分钟版本

这个项目我会把它定义成一个“企业级多模态 RAG 知识库系统”，而不是普通 ChatPDF。原因是它有两条完整工作流。第一条是文档导入工作流，入口在 import_server.py，核心用 LangGraph 编排。用户上传 PDF 后，系统先在 node_entry 做文件校验、文件类型识别、file_hash 计算和重复上传判断。如果是 PDF，就交给 MinerU 转成 Markdown。转成 Markdown 之后，node_md_img 会扫描图片，把图片上传到 MinIO，并调用多模态模型生成图片摘要，这一步是把文档中的视觉信息也转成可检索文本。接着 node_document_split 会先按标题层级切，再对过长内容二次切分、对过短内容合并，同时生成 chunk_hash。然后 node_item_name_recognition 会从高价值 chunk 中识别文档主体 item_name，这在后续检索里很关键，因为它能把搜索范围限制到具体产品。再之后用 BGE-M3 生成 dense 和 sparse 向量，最后写入 Milvus，并把文档和 chunk 元数据写进 Mongo。

第二条是问答工作流，入口在 query_server.py。用户提问后，node_item_name_confirm 会结合历史对话识别 item_name，同时做 query rewrite，把口语化问题转成更适合检索的表达。然后系统做多路召回：一条是本地 hybrid 检索，一条是 HyDE，也就是先让模型生成假设性答案再去检索，还有一条是 web search。多路结果用 RRF 融合，再交给 BGE-Reranker 重新排序，最后把 top 文档、历史对话和问题一起喂给 Qwen 生成答案，通过 SSE 流式返回。

我在这个项目里重点做了两类优化。第一类是离线评测体系，我不想把优化停留在“感觉效果不错”，所以构建了真实评测集，并对比 dense、sparse、hybrid、RRF、Rerank、HyDE 等策略在 Recall@5、MRR、Faithfulness 和 P95 latency 上的表现。第二类是工程化治理，我实现了基于 file_hash/chunk_hash 的文档版本管理和增量更新，让系统能识别重复上传、同名新版本、旧版本替换和删除，避免向量库里的旧数据污染检索结果。我的理解是，这类能力比单纯把模型接起来更能体现企业级 RAG 的价值。

------

## 9.4 5 分钟版本

这个项目的业务背景很典型：企业内部有大量产品手册、操作说明、故障文档，文档格式复杂，里面既有正文，也有图片、标题层级和步骤说明。用户问问题时，问法又常常是口语化、多轮对话，甚至带指代。一个简单的“上传文本到向量库再问答”的 demo 很难解决这些问题，所以我把项目设计成了两条 LangGraph 工作流：导入工作流和问答工作流。

先说导入侧。用户上传文件后，import_server.py 会生成任务目录和 task_id，然后进入 kb_import_app。在 node_entry，系统会先判断这是 PDF、Markdown 还是 txt，并且在导入一开始就计算 file_hash。这一步非常重要，因为它不仅决定后续是不是要解析文件，也决定了版本治理能力。如果发现是重复上传的同一文件，就直接返回 DUPLICATED，整个解析和向量化都不再执行。如果是新文件或者同名新版本，PDF 会进入 node_pdf_to_md，通过 MinerU 转成 Markdown。之所以先转 Markdown，是因为 Markdown 是一个更适合后续结构化处理的中间格式，标题、图片和文本关系都更容易保留。

接着，node_md_img 会处理文档中的图片。它不是简单把图片 URL 保留下来，而是会找到图片在 Markdown 中的上下文，把图片上传到 MinIO，再调用多模态模型生成图片摘要。这样后面做切片和检索时，图片信息就不是黑盒了，而是变成可检索文本。之后 node_document_split 会先按标题层级做粗切，尽量保证一个 chunk 对应一个完整语义主题；对太长的 chunk 再二次切分，对太短的 chunk 做合并，同时生成 chunk_hash，为后续增量更新打基础。然后 node_item_name_recognition 会从文档前几个高价值 chunk 中识别主体 item_name，这个设计非常有业务价值，因为它后面既可以用于问答时的检索过滤，也可以辅助文档治理。再之后用 BGE-M3 生成 dense 和 sparse 向量，最终写入 Milvus，并把文档级和 chunk 级元数据写入 Mongo。

再说问答侧。用户通过 /query 提问后，系统先进入 node_item_name_confirm。这个节点会把历史消息和当前问题一起给 LLM，让它做两件事：识别 item_name，以及把用户问题改写成更适合检索的 rewritten_query。识别出来的 item_name 不会直接用，而是会先去 kb_item_names 做一次向量确认和打分，判断到底是明确命中、候选多个，还是根本没匹配到。如果主体明确，系统就进入多路召回。当前本地召回主路是 hybrid search，也就是 dense 和 sparse 混合检索；第二条路是 HyDE，它先生成一段假设性答案，再拿“问题+假设答案”一起检索；第三条路是 web search，通过 MCP 补充外部知识。多路结果拿到以后，用 RRF 做融合，因为不同召回源分数分布不可直接比较，用排名融合更稳。融合之后，再用 BGE-Reranker 做 cross-encoder 精排，把真正最能回答问题的 chunk 排到前面。最后，node_answer_output 会把高质量上下文、历史对话和问题一起喂给 Qwen 生成答案，并通过 SSE 流式返回给前端。

我在这个项目里的核心贡献不是单纯把流程接通，而是把它往“可优化、可解释、可维护”的方向推进。第一块是离线评测体系。我基于 Milvus 中的真实 chunk 设计了评测集，每条样本包含 question、golden_chunk_ids、golden_answer 和 category，然后对比 dense、sparse、hybrid、RRF、Rerank、HyDE 等策略在 Hit@K、Recall@K、MRR、NDCG@K、Faithfulness、Answer Relevance、P95 latency 上的表现。这样我可以明确回答面试官：某个策略到底提升了召回、排序还是答案质量，成本是多少，而不是只说“我加了 HyDE 效果更好了”。

第二块是文档版本管理和增量更新。很多 RAG 项目只关注“把文档导进去”，但企业场景里真正难的是“文档不断更新时如何不把知识库搞脏”。所以我实现了 file_hash 和 chunk_hash，并引入了 DocumentMeta 和 ChunkMeta 元数据仓库。这样同一个文件重复上传时会被识别成 DUPLICATED，不会重复入库；同名但内容变化的文件会自动生成新版本，同时清理旧版本 chunk，保证检索结果不会混入过期内容。这个能力很体现工程化，因为它解决的是知识库长期维护问题，而不是一次性实验问题。

如果在面试里总结这个项目，我会说：这是一个以 LangGraph 为工作流编排核心、以 Milvus 和 BGE-M3 为检索基础、以 RRF 和 Rerank 为效果增强、并逐步补齐评测体系和版本治理的企业级 RAG 系统。我在项目里的重点价值，是把它从一个“能问答”的系统，推进成一个“能评估、能维护、能解释”的系统。

------

# 十、补充结论

## 10.1 当前三大优化的真实落地状态

1. RAG 离线评测体系
   当前已实现。
   核心代码：
   - evaluator.py
   - metrics.py
   - report.py
2. 引用溯源、低置信度拒答、答案反幻觉校验
   当前未完整实现。
   当前只有：
   - 更严格的答案 Prompt
   - 模型图片区块剥离
     完整 guardrails 仍属于建议实现。
3. 文档增量更新、版本管理、导入任务状态管理、失败重试
   部分已实现。
   已实现：
   - file_hash
   - chunk_hash
   - DocumentMeta/ChunkMeta
   - 重复上传判定
   - 同名新版本
   - Milvus 旧版本清理
   - 文档删除 API
     未完整实现：
   - 持久化 ImportTask
   - current_node / retry_count / error_stack
   - 失败断点恢复
   - 标准化失败重试策略

## 10.2 面试里最值得主打的亮点

建议主打这 4 个：

1. LangGraph 双工作流设计
   说明你不是只会写脚本。
2. 多模态导入链路
   说明你考虑到了图片和结构化信息。
3. 评测驱动优化
   说明你不是拍脑袋调 RAG。
4. 文档版本治理
   说明你有工程化能力，不只是模型调用能力。

如果你下一步要继续，我建议做两件事之一：

1. 我把这份文档再压缩成“面试速记版”，适合你面试前 10 分钟快速复习。
2. 我继续基于当前仓库，单独给你出一份“技术一面高频追问 + 回答模板”精简版。





# 《RAG 项目面试速记版》

## 一句话介绍

这是一个基于 LangGraph + Milvus + BGE-M3 + BGE-Reranker + Qwen 的企业级 RAG 知识库系统，覆盖“文档导入、混合检索、多轮问答、离线评测、版本治理”全链路。

## 核心流程

### 1. 文档导入

用户上传 PDF/Markdown 后，系统先在 node_entry 做文件校验、类型识别和 file_hash 判重；PDF 先经 MinerU 转 Markdown；再在 node_md_img 中处理图片，上传 MinIO，并调用多模态模型生成图片摘要；node_document_split 按标题层级切片并生成 chunk_hash；node_item_name_recognition 识别文档主体 item_name；node_bge_embedding 生成 dense/sparse 向量；node_import_milvus 写入 Milvus，并同步写 Mongo 元数据。

### 2. 检索问答

用户提问后，node_item_name_confirm 结合历史对话识别 item_name 并做 query rewrite；然后走本地 hybrid 检索、HyDE 检索和 web search 多路召回；node_rrf 做融合，node_rerank 精排；最后 node_answer_output 组装上下文，让 Qwen 生成答案，并通过 SSE 流式返回。

## 两个已完成优化

### 1. 离线评测体系

我补了一套评测模块，支持 dense_only、sparse_only、hybrid、RRF、Rerank、HyDE 等策略对比。检索层看 Hit@K / Recall@K / MRR / NDCG@K，生成层看 Faithfulness / Answer Relevance，性能看 P95 latency。这让我能用数据证明“哪个策略更适合默认上线”，而不是凭感觉堆技术。

### 2. 文档版本管理与增量更新

我实现了 file_hash + chunk_hash + DocumentMeta/ChunkMeta 的治理方案。相同文件重复上传会直接判定为 DUPLICATED，跳过解析和入库；同名但内容变化会生成新版本，旧版本标记 REPLACED，并从 Milvus 清理旧 chunk，避免脏数据继续参与检索。这块很能体现工程化能力。

## 10 个高频追问

1. **为什么不是普通向量问答 demo？**
   因为它有双 LangGraph 工作流、多模态导入、多路召回、评测体系和版本治理。
2. **为什么 PDF 要先转 Markdown？**
   为了保留标题层级、图片位置和结构信息，方便统一后处理。
3. **为什么要做图片理解？**
   很多企业文档关键信息在图里，必须把视觉信息转成可检索文本。
4. **为什么按标题切片？**
   比固定长度切片更保语义完整，召回和答案质量更稳。
5. **为什么识别 item_name？**
   用于按产品范围收缩检索，也支撑多轮对话中的指代承接。
6. **为什么用 BGE-M3？**
   一个模型同时支持 dense 和 sparse，适合企业知识库的语义+术语混合场景。
7. **dense 和 sparse 各解决什么问题？**
   dense 抓语义相似，sparse 抓型号、参数、关键词精确匹配。
8. **为什么要多路召回？**
   单路容易漏召回，多路能提高 recall ceiling，再靠 RRF 和 Rerank 提纯。
9. **RRF 的价值是什么？**
   解决不同召回源分数不可比问题，用排名位置融合更稳。

1. **版本管理的价值是什么？**
   解决重复上传、同名覆盖、旧数据污染检索的问题，让知识库能长期维护。