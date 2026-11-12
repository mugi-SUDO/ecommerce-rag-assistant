"""
知识图谱查询工具：多跳遍历 / 子图提取，适合多跳推理型问题。负责实体间的关系
"""

from langchain_core.tools import tool

from app.runtime import get_context, format_docs


def kg_query_docs(query: str, k: int = 5):
    ctx = get_context()
    if ctx.graph_retriever is None:
        from app.graph.kg_retrieval import GraphRetriever

        ctx.graph_retriever = GraphRetriever(ctx)
    return ctx.graph_retriever.search(query, k=k)


@tool
def kg_query(query: str) -> str:
    """知识图谱查询：在电商知识图谱中做多跳/子图推理，适合实体关系、协同购买、替代品等问题（如"买过小米14的人还买了什么"）。"""
    return format_docs(kg_query_docs(query))
