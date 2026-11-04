"""
意图识别 + 复杂度路由

问题分类：
- simple       简单事实型  -> 普通 RAG (vector_search)
- single_hop   单跳检索型  -> 混合检索 + 重排序 (hybrid_search)
- multi_hop    多跳推理型  -> GraphRAG (kg_query)
- structured   结构化多步型 -> AgenticRAG (order_query + 其他工具)
- out_of_scope 超范围      -> 转人工
"""
from dataclasses import dataclass

from app.prompts.prompts import router_prompt
from app.runtime import get_context, parse_json_response

VALID_INTENTS = {"simple", "single_hop", "multi_hop", "structured", "out_of_scope"}


@dataclass
class RoutingResult:
    intent: str
    confidence: float
    explanation: str


class IntentRouter:
    def __init__(self, ctx):
        self.ctx = ctx

    def route(self, query: str) -> RoutingResult:
        llm = self.ctx.llm
        if llm is None:
            return RoutingResult("single_hop", 0.5, "LLM 不可用，回退默认路由")
        raw = llm.invoke(router_prompt(query))
        text = getattr(raw, "content", str(raw))
        parsed = parse_json_response(text, {"intent": "single_hop", "confidence": 0.5, "explanation": ""})
        intent = parsed.get("intent", "single_hop")
        if intent not in VALID_INTENTS:
            intent = "single_hop"
        try:
            confidence = float(parsed.get("confidence", 0.5))
        except (TypeError, ValueError):
            confidence = 0.5
        return RoutingResult(intent, confidence, parsed.get("explanation", ""))
