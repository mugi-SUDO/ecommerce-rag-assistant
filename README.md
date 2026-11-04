# 电商服务平台智能助手（My_ECommerce）

基于 GraphRAG + AgenticRAG + 多 Agent 协作 + LangChain/LangGraph 的电商智能助手。
项目按「路由层 → 策略层 → 统一工具层 → 数据层」分层组织，详见下方目录结构。

## 一、环境准备

```bash
# 1) 创建并激活虚拟环境（Python 3.10+）
python -m venv .venv
.venv\Scripts\activate        # Windows

# 2) 安装依赖（推荐用锁定版本，确保与开发环境一致）
pip install -r requirements.lock.txt
# 或只装版本范围：pip install -r requirements.txt

# 3) 启动 Neo4j（图数据库，GraphRAG 分支需要；不启动则自动降级为纯向量检索）
docker compose up -d neo4j
```

## 二、配置

复制 `.env.example` 为 `.env`，填入 `DEEPSEEK_API_KEY`：

```bash
copy .env.example .env        # Windows
```

`.env` 已被 `.gitignore` 排除，不会提交到仓库。

## 三、运行

```bash
python main.py
```

交互示例问题：
- 简单事实型：`小米14 多少钱？`（普通 RAG / 向量检索）
- 单跳检索型：`推荐几款性价比高的手机`（混合检索 + RRF 重排）
- 多跳推理型：`买过小米14的人还买了什么`（GraphRAG / Neo4j）
- 结构化多步型：`查一下订单 O20241101001 到哪了`（AgenticRAG / 规划+执行）
- 超范围：`我要投诉商家`（转人工）

## 四、测试（离线，无需 LLM/Neo4j/langgraph）

```bash
python tests/test_router.py
```

## 五、目录结构

```
My_ECommerce/
├── config.py                 # 配置
├── main.py                   # 入口
├── app/
│   ├── runtime.py            # 运行时上下文 / 依赖注入
│   ├── router/               # 意图识别 + 复杂度路由
│   ├── agents/               # 规划机 / 执行器 / 反思器(SelfRAG)
│   ├── core_tools/           # 统一核心工具（向量/混合/图谱/订单）
│   ├── strategies/           # naive_rag / hybrid_rag / graph_rag / agentic_rag
│   ├── state_machine/        # 顶层 LangGraph 状态机
│   ├── graph/                # 图谱本体 / 建图 / 检索
│   ├── vectorstore/          # Chroma 向量索引构建
│   └── prompts/              # Prompt 模板
├── data/raw/                 # 商品/订单/评价/售后政策
└── tests/                    # 离线单测
```

## 六、注意事项

- 首次运行会从 HuggingFace 下载 `bge-small-zh-v1.5` 嵌入模型（已配 `HF_ENDPOINT=https://hf-mirror.com` 加速）。
- `langgraph` 必须使用 `0.2.x`（`requirements.txt` 已限定 `<0.3`）；1.x 会强拉 `langchain-core>=1`，与 langchain 0.3 不兼容。
- 向量库用 `langchain_community.vectorstores.Chroma`（避免额外安装 `langchain_chroma`）。
- 向量索引生成在 `vector_index/`，已加入 `.gitignore`，首次运行自动重建。
- 退货 / 订单追踪 / 商品推荐等 Skill+MCP 按需加载为**长期目标**，当前未实现。
