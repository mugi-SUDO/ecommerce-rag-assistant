"""
GraphRAG 策略：知识图谱多跳遍历 / 子图提取，适合多跳推理型问题。
"""
from app.core_tools.kg_query import kg_query_docs


def graph_rag_retrieve(query: str, k: int = 5):
    return kg_query_docs(query, k=k)
