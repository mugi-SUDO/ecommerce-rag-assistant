"""
顶层 LangGraph 状态机：Router -> 策略分发 -> SelfRAG 反思 -> 生成 / 改写重试 / 转人工

节点：
- route    意图识别 + 复杂度路由
- retrieve 简单/单跳/多跳策略检索
- agentic  结构化多步（AgenticRAG：规划 + 执行）
- reflect  SelfRAG 证据反思（贯穿所有分支）
- generate 生成最终答案
- escalate 转人工

边：
- route --conditional--> retrieve / agentic / escalate
- retrieve / agentic -> reflect
- reflect --conditional--> generate（充分或达上限）/ retrieve / agentic（改写重试）
"""
from typing import List, TypedDict

from langchain_core.documents import Document

from app.runtime import get_context, format_docs
from app.router.intent_router import IntentRouter
from app.agents.planner import PlannerAgent
from app.agents.executor import ExecutorAgent
from app.agents.reflector import ReflectorAgent
from app.prompts.prompts import generate_prompt
from app.strategies.naive_rag import naive_rag_retrieve
from app.strategies.hybrid_rag import hybrid_rag_retrieve
from app.strategies.graph_rag import graph_rag_retrieve
from app.strategies.agentic_rag import agentic_rag_retrieve

try:
    from langgraph.graph import StateGraph, END
except ImportError as e:  # pragma: no cover
    raise RuntimeError("缺少依赖 langgraph，请先执行：pip install langgraph") from e


class GraphState(TypedDict):
    query: str                 # 原始用户问题
    current_query: str         # 当前（可能被反思层改写后的）查询
    intent: str
    routing: dict
    docs: List[Document]
    docs_text: str
    reflection: dict
    iteration: int
    final_answer: str


class ECommerceAssistant:
    def __init__(self, ctx):
        self.ctx = ctx
        self.router = IntentRouter(ctx)
        self.planner = PlannerAgent(ctx)
        self.executor = ExecutorAgent(ctx)
        self.reflector = ReflectorAgent(ctx)
        self.graph = self._build()

    # ---- 节点 ----
    def _route(self, state: GraphState) -> dict:
        r = self.router.route(state["current_query"])
        return {
            "intent": r.intent,
            "routing": {"intent": r.intent, "confidence": r.confidence, "explanation": r.explanation},
        }

    def _retrieve(self, state: GraphState) -> dict:
        q = state["current_query"]
        intent = state.get("intent", "single_hop")
        if intent == "simple":
            docs = naive_rag_retrieve(q)
        elif intent == "single_hop":
            docs = hybrid_rag_retrieve(q)
        elif intent == "multi_hop":
            docs = graph_rag_retrieve(q)
        else:
            docs = naive_rag_retrieve(q)
        return {"docs": docs}

    def _agentic(self, state: GraphState) -> dict:
        docs = agentic_rag_retrieve(state["current_query"])
        return {"docs": docs}

    def _reflect(self, state: GraphState) -> dict:
        docs_text = format_docs(state.get("docs", []))
        refl = self.reflector.reflect(state["current_query"], docs_text)
        iteration = state.get("iteration", 0) + 1
        result = {
            "docs_text": docs_text,
            "iteration": iteration,
            "reflection": {
                "relevance_score": refl.relevance_score,
                "sufficiency_score": refl.sufficiency_score,
                "is_supported": refl.is_supported,
                "is_sufficient": refl.is_sufficient,
                "failure_reason": refl.failure_reason,
                "suggested_rewrite": refl.suggested_rewrite,
            },
        }
        # 证据不足且给出改写建议时，更新当前查询供下一轮重试
        if not refl.is_sufficient and refl.suggested_rewrite:
            result["current_query"] = refl.suggested_rewrite
        return result

    def _generate(self, state: GraphState) -> dict:
        docs_text = state.get("docs_text") or format_docs(state.get("docs", []))
        refl = state.get("reflection") or {}
        note = refl.get("failure_reason", "") if refl and not refl.get("is_sufficient") else ""
        prompt = generate_prompt(state["query"], docs_text, note)
        raw = self.ctx.llm.invoke(prompt)
        answer = getattr(raw, "content", str(raw))
        return {"final_answer": answer}

    def _escalate(self, state: GraphState) -> dict:
        return {"final_answer": "该问题超出服务范围，已为您转接人工客服（在线时间 9:00-22:00，涉及退款仲裁/投诉/发票等请直接联系人工）。"}

    # ---- 条件边 ----
    def _route_next(self, state: GraphState) -> str:
        intent = state.get("intent", "single_hop")
        if intent == "out_of_scope":
            return "escalate"
        if intent == "structured":
            return "agentic"
        return "retrieve"

    def _after_reflect(self, state: GraphState) -> str:
        refl = state.get("reflection") or {}
        iteration = state.get("iteration", 0)
        if refl.get("is_sufficient") or iteration >= self.ctx.config.max_iterations:
            return "generate"
        # 证据不足：改写后回到对应策略重试
        return "agentic" if state.get("intent") == "structured" else "retrieve"

    def _build(self):
        g = StateGraph(GraphState)
        g.add_node("route", self._route)
        g.add_node("retrieve", self._retrieve)
        g.add_node("agentic", self._agentic)
        g.add_node("reflect", self._reflect)
        g.add_node("generate", self._generate)
        g.add_node("escalate", self._escalate)

        g.set_entry_point("route")
        g.add_conditional_edges(
            "route", self._route_next,
            {"retrieve": "retrieve", "agentic": "agentic", "escalate": "escalate"},
        )
        g.add_edge("retrieve", "reflect")
        g.add_edge("agentic", "reflect")
        g.add_conditional_edges(
            "reflect", self._after_reflect,
            {"retrieve": "retrieve", "agentic": "agentic", "generate": "generate"},
        )
        g.add_edge("generate", END)
        g.add_edge("escalate", END)
        return g.compile()

    def run(self, query: str) -> str:
        result = self.graph.invoke({"query": query, "current_query": query, "iteration": 0})
        return result.get("final_answer", "")
