"""
RuntimeContext是一个全局的唯一的背包，用于依赖注入
集中持有 LLM / 嵌入模型 / 向量库 / Neo4j 驱动 / 原始数据
通过get_context() / set_context() 提供简单的全局访问入口
"""

import json
import os
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Optional

from langchain_openai import ChatOpenAI
from langchain_huggingface import HuggingFaceEmbeddings

from config import Config


@dataclass
class RuntimeContext:
    config: Config = None
    llm: Optional[ChatOpenAI] = None
    embeddings: Optional[HuggingFaceEmbeddings] = None
    vectorstore: object = None  # Chroma 向量库
    chunks: List = field(
        default_factory=list
    )  # 分块文档（供 BM25 / 混合检索），防止实例间共享
    hybrid_engine: object = None  # HybridSearchEngine 实例
    neo4j_driver: object = None  # neo4j.Driver，连接失败时为 None
    graph_retriever: object = None  # GraphRetriever 实例
    products: List = field(default_factory=list)
    orders: List = field(default_factory=list)
    reviews: List = field(default_factory=list)
    policy_text: str = ""
    memory: object = None  # MemoryManager，未启用时为 None


# 模块级全局变量，存储当前运行时的 RuntimeContext 实例
_context: Optional[RuntimeContext] = None


def set_context(ctx: RuntimeContext) -> None:
    global _context  # 修改的是模块级全局变量 _context
    _context = ctx


def get_context() -> Optional[RuntimeContext]:
    return _context


def build_llm(config: Config) -> ChatOpenAI:
    api_key = os.getenv("DEEPSEEK_API_KEY")
    if not api_key:
        raise ValueError("请设置 DEEPSEEK_API_KEY 环境变量")
    return ChatOpenAI(
        model=config.llm_model,
        base_url=config.llm_base_url,
        api_key=api_key,
        temperature=config.llm_temperature,
        max_tokens=config.llm_max_tokens,
    )


def build_embeddings(config: Config) -> HuggingFaceEmbeddings:
    return HuggingFaceEmbeddings(
        model_name=config.embedding_model,
        model_kwargs={"device": config.embedding_device},
        encode_kwargs={"normalize_embeddings": True},
    )


def build_memory(config: Config, llm, embeddings):
    """构建三层记忆；Redis 不可用时自动降级为进程内存储。

    记忆分三层：
    - 短期会话：Redis List，只存当前 session，TTL 到期或会话结束时清空
    - 长期历史：Redis List（逐条）+ Chroma（语义检索），跨会话保留
    - 用户画像：Redis Hash，随对话增量更新
    """
    if not config.memory_enabled:
        print("[memory] 记忆功能已关闭（config.memory_enabled=False）")
        return None

    # 延迟导入，避免 runtime 与 memory 模块循环依赖
    from app.memory.manager import MemoryManager
    from app.memory.store import InMemoryStore, RedisMemoryStore

    try:
        store = RedisMemoryStore(
            config.redis_url,
            session_ttl=config.session_ttl_seconds,
            short_term_max_turns=config.short_term_max_turns,
            history_max=config.long_term_history_max,
        )
        print(f"[memory] Redis 连接成功：{config.redis_url}")
    except Exception as e:
        store = InMemoryStore(
            session_ttl=config.session_ttl_seconds,
            short_term_max_turns=config.short_term_max_turns,
            history_max=config.long_term_history_max,
        )
        print(f"[memory] Redis 不可用，降级为进程内记忆：{e}")

    vector = None
    try:
        from app.memory.vector_memory import MemoryVectorStore

        vector = MemoryVectorStore(config, embeddings)
        print(f"[memory] 向量记忆已启用（collection={config.long_term_collection}）")
    except Exception as e:
        print(f"[memory] 向量记忆不可用，长期语义召回关闭：{e}")

    return MemoryManager(config, llm, store, vector)


# 读原始数据
def load_raw_data(data_dir: str):
    """从 data/raw 加载商品/订单/评价/售后政策"""
    raw = Path(data_dir) / "raw"
    products = _read_json(raw / "products.json")
    orders = _read_json(raw / "orders.json")
    reviews = _read_json(raw / "reviews.json")
    policy_path = raw / "after_sales_policy.md"
    policy_text = (
        policy_path.read_text(encoding="utf-8") if policy_path.exists() else ""
    )
    return products, orders, reviews, policy_text


def parse_json_response(text: str, default):
    """清洗 LLM 返回的 JSON（去掉 markdown 代码块 / 前后杂字），解析失败返回 default"""
    if not text:
        return default
    content = text.strip()
    content = re.sub(r"^```(?:json)?\s*", "", content)
    content = re.sub(r"\s*```$", "", content)
    start = content.find("{")
    end = content.rfind("}")
    if start != -1 and end != -1 and end > start:
        content = content[start : end + 1]
    try:
        return json.loads(content)
    except Exception:
        return default


def _read_json(path: Path):
    if not path.exists():
        return []
    with open(path, encoding="utf-8") as f:
        return json.load(f)


# 格式化检索结果
def format_docs(docs) -> str:
    """把检索到的 Document 列表格式化为一段带来源标注的文本（供工具返回 / 生成 / 反思使用）"""
    if not docs:
        return "（未检索到相关内容）"
    parts = []
    for i, d in enumerate(docs, 1):
        meta = getattr(d, "metadata", {}) or {}
        src = (
            meta.get("name")
            or meta.get("productId")
            or meta.get("orderId")
            or meta.get("reviewId")
            or meta.get("type")
            or "unknown"
        )
        parts.append(f"[{i}] 来源:{src}\n{d.page_content}")
    return "\n\n".join(parts)
