"""
反思器（Reflector，即 SelfRAG 反思层）：评估证据相关性 / 充分性 / 支撑度，
决定是直接生成答案，还是改写查询重新检索。
"""
from dataclasses import dataclass

from app.prompts.prompts import reflector_prompt
from app.runtime import get_context, parse_json_response


@dataclass
class ReflectionResult:
    relevance_score: float
    sufficiency_score: float
    is_supported: bool
    is_sufficient: bool
    failure_reason: str
    suggested_rewrite: str


class ReflectorAgent:
    def __init__(self, ctx):
        self.ctx = ctx

    def reflect(self, query: str, docs_text: str) -> ReflectionResult:
        llm = self.ctx.llm
        default = ReflectionResult(5.0, 5.0, True, True, "", "")
        if llm is None:
            return default
        raw = llm.invoke(reflector_prompt(query, docs_text))
        text = getattr(raw, "content", str(raw))
        parsed = parse_json_response(text, {
            "relevance_score": 5, "sufficiency_score": 5,
            "is_supported": True, "is_sufficient": True,
            "failure_reason": "", "suggested_rewrite": "",
        })

        def _f(x, d=5.0):
            try:
                return float(x)
            except (TypeError, ValueError):
                return d

        def _b(x, d=True):
            if isinstance(x, bool):
                return x
            if isinstance(x, str):
                return x.strip().lower() in ("true", "是", "yes", "1")
            return d

        return ReflectionResult(
            relevance_score=_f(parsed.get("relevance_score")),
            sufficiency_score=_f(parsed.get("sufficiency_score")),
            is_supported=_b(parsed.get("is_supported")),
            is_sufficient=_b(parsed.get("is_sufficient")),
            failure_reason=str(parsed.get("failure_reason", "")),
            suggested_rewrite=str(parsed.get("suggested_rewrite", "")),
        )
