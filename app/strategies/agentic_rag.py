"""
AgenticRAG 策略（Plan -> Retrieve 阶段）：适合结构化多步型问题（订单/物流/库存/退款）。

- 规划机（Planner）把问题拆成子任务
- 执行器（Executor）按子任务调用统一核心工具
- 检索结果交给上层状态机的 SelfRAG 反思层做 Critique/Refine
"""
from langchain_core.documents import Document

from app.runtime import get_context


def agentic_rag_retrieve(query: str, k: int = 5):
    from app.agents.planner import PlannerAgent
    from app.agents.executor import ExecutorAgent

    ctx = get_context()
    planner = PlannerAgent(ctx)
    executor = ExecutorAgent(ctx)

    subtasks = planner.plan(query)
    if not subtasks:
        # 规划失败时回退：直接做订单/库存查询
        from app.core_tools.order_query import order_query_docs
        return order_query_docs(query, k=k)

    results = executor.execute(subtasks)
    text = "\n\n".join(
        f"[步骤{r['step']} · 工具 {r['tool']}] 查询「{r['query']}」\n{r['result']}"
        for r in results
    )
    return [Document(page_content=text, metadata={"type": "agentic"})]
