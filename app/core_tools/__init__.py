"""
统一出口
统一核心工具注册表：Router / Planner / Executor 通过名字查找工具

向量检索 vector_search：简单事实型（向量相似度）
混合检索 hybrid_search：单跳检索型（BM25 + 向量 + RRF）
知识图谱查询 kg_query：多跳推理型（图谱查询）
订单查询 order_query：结构化多步型（订单/物流/库存/退款）
"""

# __init__.py标识目录是一个包
# 作为统一出口，注册所有核心工具，供 Router / Planner / Executor 通过名字查找工具
# 当包被导入时，__init__.py 会被自动执行，从而注册所有工具

from app.core_tools.vector_search import vector_search
from app.core_tools.hybrid_search import hybrid_search
from app.core_tools.kg_query import kg_query
from app.core_tools.order_query import order_query

TOOL_REGISTRY = {
    "vector_search": vector_search,
    "hybrid_search": hybrid_search,
    "kg_query": kg_query,
    "order_query": order_query,
}

TOOL_DESCRIPTIONS = {name: TOOL_REGISTRY[name].description for name in TOOL_REGISTRY}


def run_tool(name: str, query: str) -> str:
    """按名字调用工具，返回文本结果"""
    tool_obj = TOOL_REGISTRY.get(name)
    if tool_obj is None:
        return f"[错误] 未知工具：{name}"
    try:
        return tool_obj.invoke({"query": query})
    except Exception as e:
        return f"[错误] 调用工具 {name} 失败：{e}"


def tools_description_text() -> str:
    lines = []
    for name, desc in TOOL_DESCRIPTIONS.items():
        lines.append(f"- {name}: {desc}")
    return "\n".join(lines)
