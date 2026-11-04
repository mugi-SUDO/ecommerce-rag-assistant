"""
执行器（Executor）：按规划机的子任务依次调用统一核心工具。
"""
from typing import List

from app.core_tools import run_tool, TOOL_REGISTRY


class ExecutorAgent:
    def __init__(self, ctx):
        self.ctx = ctx

    def execute(self, subtasks: List[dict]) -> List[dict]:
        results = []
        for i, st in enumerate(subtasks or [], 1):
            tool_name = st.get("tool", "")
            query = st.get("query", "")
            step = int(st.get("step", i))
            if tool_name not in TOOL_REGISTRY:
                results.append({
                    "step": step, "tool": tool_name, "query": query,
                    "result": f"[跳过] 未知工具 {tool_name}",
                })
                continue
            results.append({
                "step": step, "tool": tool_name, "query": query,
                "result": run_tool(tool_name, query),
            })
        return results
