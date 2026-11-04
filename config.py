"""配置文件"""

from dataclasses import dataclass
from pathlib import Path

# 项目根目录（本文件所在目录），用于把相对路径解析成绝对路径，避免受运行时 CWD 影响
PROJECT_ROOT = Path(__file__).resolve().parent


@dataclass
class Config:
    """系统配置类"""

    # LLM
    llm_model: str = "deepseek-chat"
    llm_base_url: str = "https://api.deepseek.com/v1"
    llm_temperature: float = 0.1
    llm_max_tokens: int = 2048

    # 嵌入模型
    embedding_model: str = "BAAI/bge-small-zh-v1.5"
    embedding_device: str = "cpu"

    # Neo4j 图数据库
    neo4j_uri: str = "bolt://localhost:7687"
    neo4j_user: str = "neo4j"
    neo4j_password: str = "all-in-rag"
    neo4j_database: str = "neo4j"

    # Chroma 向量库
    chroma_persist_dir: str = str(PROJECT_ROOT / "vector_index")
    chroma_collection: str = "ecommerce_kb"

    # 数据
    data_dir: str = str(PROJECT_ROOT / "data")

    # 检索相关
    top_k: int = 5
    chunk_size: int = 300
    chunk_overlap: int = 50
    rrf_k: int = 60
    max_iterations: int = 3  # AgenticRAG / SelfRAG 反思最大轮次


DEFAULT_CONFIG = Config()
