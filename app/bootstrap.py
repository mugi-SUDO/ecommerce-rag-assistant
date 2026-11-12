"""
系统初始化（CLI 和 FastAPI 共用）

main.py 与 app/api/server.py 都从这里拿到同一个初始化好的 ECommerceAssistant，
避免两套启动逻辑漂移。
"""

from pathlib import Path

from dotenv import load_dotenv

from config import DEFAULT_CONFIG
from app.runtime import (
    RuntimeContext,
    set_context,
    build_llm,
    build_embeddings,
    build_memory,
)

# 项目的根目录路径
PROJECT_ROOT = Path(__file__).resolve().parent.parent


# 从项目向上两级目录查找 .env 文件，优先加载项目根目录的 .env
def load_env() -> None:
    """
    三个文件都加载，避免项目 .env 只配了 API Key 时漏掉上级的 JWT_SECRET 等变量。
    """
    for path in (
        PROJECT_ROOT / ".env",
        PROJECT_ROOT.parent / ".env",
        PROJECT_ROOT.parent.parent / ".env",
    ):
        if path.exists():
            load_dotenv(path, override=False)  # override=False 避免覆盖环境变量


def initialize_system():
    """构建 LLM / 嵌入 / 向量索引 / 三层记忆 / 知识图谱 / 状态机"""
    load_env()
    ctx = RuntimeContext(config=DEFAULT_CONFIG)
    set_context(ctx)

    print(">>> 初始化 LLM（DeepSeek）...")
    ctx.llm = build_llm(DEFAULT_CONFIG)

    print(">>> 初始化本地嵌入模型（BGE-small-zh，首次会下载模型）...")
    ctx.embeddings = build_embeddings(DEFAULT_CONFIG)

    print(">>> 加载数据 + 构建向量索引（Chroma）...")
    from app.vectorstore.build_index import build_vector_index

    build_vector_index()  # 内部会加载 raw 数据并写回 ctx.products/orders/reviews/policy_text

    print(">>> 初始化三层记忆（Redis 短期会话/长期历史/用户画像 + Chroma 语义记忆）...")
    ctx.memory = build_memory(DEFAULT_CONFIG, ctx.llm, ctx.embeddings)

    print(">>> 构建知识图谱（Neo4j，未连接则 GraphRAG 分支降级）...")
    from app.graph.build_graph import build_graph

    build_graph(ctx)

    print(">>> 编译状态机（LangGraph）...")
    from app.state_machine.graph import ECommerceAssistant

    return ECommerceAssistant(ctx)
