"""
普通 RAG 策略：向量语义检索，适合简单事实型问题。
"""
from app.core_tools.vector_search import vector_search_docs


def naive_rag_retrieve(query: str, k: int = 5):
    return vector_search_docs(query, k=k)
