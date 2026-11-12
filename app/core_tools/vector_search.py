"""
向量检索工具：Chroma 语义检索，适合简单事实型问题。
"""

from langchain_core.tools import tool

from app.runtime import get_context, format_docs


# 给内部代码用的函数
def vector_search_docs(query: str, k: int = 5):
    ctx = get_context()
    if ctx is None or ctx.vectorstore is None:
        return []
    return ctx.vectorstore.similarity_search(query, k=k)


# LangChain提供的装饰器，生成一个给LLM专用的函数
# .name : 函数名字
# .description : 函数描述(三引号的文档字符串)
# .invoke : 调用函数，参数是字典
@tool
def vector_search(query: str) -> str:
    """向量检索：在商品/评价/售后政策知识库中做语义检索，适合简单事实型问题（如"小米14 多少钱""退货政策是什么"）。"""
    return format_docs(vector_search_docs(query))
