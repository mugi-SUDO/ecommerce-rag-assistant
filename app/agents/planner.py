"""
规划机（Planner）：把复杂问题拆解成有序子任务，每个子任务指定工具与查询语句。
"""
from typing import List

from app.core_tools import tools_description_text
from app.prompts.prompts import planner_prompt
from app.runtime import get_context, parse_json_response


class PlannerAgent:
    def __init__(self, ctx):
        self.ctx = ctx

    def plan(self, query: str) -> List[dict]:
        llm = self.ctx.llm
        if llm is None:
            return []
        raw = llm.invoke(planner_prompt(query, tools_description_text()))
        text = getattr(raw, "content", str(raw))
        parsed = parse_json_response(text, {"subtasks": []})
        subtasks = parsed.get("subtasks", [])
        # 过滤非法子任务，确保顺序 step
        cleaned = []
        for i, st in enumerate(subtasks or [], 1):
            if not isinstance(st, dict) or not st.get("tool") or not st.get("query"):
                continue
            cleaned.append({
                "step": int(st.get("step", i)),
                "tool": st["tool"],
                "query": st["query"],
                "depends_on": st.get("depends_on", []),
            })
        return cleaned
