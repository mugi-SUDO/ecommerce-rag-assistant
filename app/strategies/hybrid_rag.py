"""
混合检索 + 重排序策略：BM25 + 向量 + RRF，适合单跳检索型问题。
"""
from app.core_tools.hybrid_search import hybrid_search_docs


def hybrid_rag_retrieve(query: str, k: int = 5):
    return hybrid_search_docs(query, k=k)
