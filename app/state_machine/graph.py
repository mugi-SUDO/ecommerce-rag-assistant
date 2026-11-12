"""
顶层 LangGraph 状态机：Router -> 策略分发 -> SelfRAG 反思 -> 生成 / 改写重试 / 转人工

节点：
    route    意图识别 + 复杂度路由
    retrieve 简单/单跳/多跳策略检索
    agentic  结构化多步（AgenticRAG：规划 + 执行）
    reflect  SelfRAG 证据反思
    generate 生成最终答案
    escalate 转人工

边：
    route --> retrieve / agentic / escalate （条件边，意图识别和路由）
    retrieve / agentic --> reflect （固定边，反思）
    reflect ---> generate / retrieve / agentic（条件边，要么生成，要么改写重试）


    调用了四种策略和ReflectorAgent
"""

# 关于状态机：
# 状态：一个贯穿始终的共享的数据结构，代表了应用在某一时刻的完整快照。
# 结点：一个函数，接收当前状态，返回一个字典，更新状态。
# 边：边决定了节点之间的连接和执行顺序，即“下一步做什么”
# 分为固定边和条件边（一个路由函数，根据当前状态决定下一步节点的名字）

from typing import List, TypedDict

from langchain_core.documents import Document

from app.runtime import get_context, format_docs
from app.router.intent_router import IntentRouter
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


# 状态state
# 继承自 TypedDict，本质就是一个字典
class GraphState(TypedDict):
    query: str  # 原始用户问题
    current_query: str  # 当前（可能被反思层改写后的）查询
    intent: str
    routing: dict
    docs: List[Document]  # 本轮检索结果（供反思层评估）
    all_docs: List[Document]  # 历次重试累计的证据（供最终生成）
    docs_text: str  # all_docs 拼装后的文本
    reflection: dict
    iteration: int
    final_answer: str
    memory_context: str  # 三层记忆拼装出的提示文本（用户画像 + 最近对话 + 长期记忆）


# 空结果 / 报错结果的 metadata.type；只要还有正常证据就不参与最终生成
_EMPTY_DOC_TYPES = {"order_empty", "kg_empty", "kg_error"}


def _merge_docs(old_docs: List[Document], new_docs: List[Document]) -> List[Document]:
    """合并多轮检索证据：按 (type, 内容) 去重，并丢弃空结果/报错占位。

    反思层改写查询后重试时，新的一轮不一定比上一轮更好；
    只保留最后一轮会让前面已经检索到的正确证据凭空消失。
    """
    merged, seen = [], set()
    for doc in list(old_docs or []) + list(new_docs or []):
        meta = doc.metadata or {}
        key = (meta.get("type"), doc.page_content)
        if key in seen:
            continue
        seen.add(key)
        merged.append(doc)

    has_real = any((d.metadata or {}).get("type") not in _EMPTY_DOC_TYPES for d in merged)
    if has_real:
        merged = [d for d in merged if (d.metadata or {}).get("type") not in _EMPTY_DOC_TYPES]
    return merged


class ECommerceAssistant:
    def __init__(self, ctx):
        self.ctx = ctx
        self.router = IntentRouter(ctx)  # 意图识别 + 复杂度路由
        self.reflector = ReflectorAgent(ctx)  # 反思器
        self.graph = self._build()  # 构建状态机

    # 节点node
    # 每个节点都是一个函数，接收当前状态，返回一个字典，更新状态

    # 意图识别 + 复杂度路由
    # 调用IntentRouter的route方法，返回意图和路由信息
    def _route(self, state: GraphState) -> dict:
        r = self.router.route(state["current_query"])
        return {
            "intent": r.intent,
            "routing": {
                "intent": r.intent,
                "confidence": r.confidence,
                "explanation": r.explanation,
            },
        }

    # 检索
    # 按类型调用不同的检索策略
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

    # agentic
    def _agentic(self, state: GraphState) -> dict:
        docs = agentic_rag_retrieve(state["current_query"])
        return {"docs": docs}

    # 反思
    def _reflect(self, state: GraphState) -> dict:
        current_docs = state.get("docs", [])
        docs_text = format_docs(current_docs)  # 本轮证据：只用于反思评估
        refl = self.reflector.reflect(state["current_query"], docs_text)  # 调用反思器
        iteration = state.get("iteration", 0) + 1
        # 累计历次证据：改写后重试变差时，不丢失前面检索到的正确信息
        all_docs = _merge_docs(state.get("all_docs", []), current_docs)
        result = {
            "all_docs": all_docs,
            "docs_text": format_docs(all_docs),  # 生成阶段看到的是累计证据
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

    # 生成最终答案
    def _generate(self, state: GraphState) -> dict:
        docs_text = state.get("docs_text") or format_docs(state.get("docs", []))
        refl = state.get("reflection") or {}
        note = (
            refl.get("failure_reason", "")
            if refl and not refl.get("is_sufficient")
            else ""
        )
        # 用原始问题生成（反射层的 suggested_rewrite 只服务检索），
        # 指代由 memory_context 中的最近对话补全
        prompt = generate_prompt(
            state["query"], docs_text, note, state.get("memory_context", "")
        )
        raw = self.ctx.llm.invoke(prompt)
        answer = getattr(raw, "content", str(raw))
        return {"final_answer": answer}

    # 转人工（简单实现）
    def _escalate(self, state: GraphState) -> dict:
        return {
            "final_answer": "该问题超出服务范围，已为您转接人工客服（在线时间 9:00-22:00，涉及退款仲裁/投诉/发票等请直接联系人工）。"
        }

    # 条件边
    # 一个函数根据当前状态决定下一步节点的名字

    # 决定route下一步节点
    def _route_next(self, state: GraphState) -> str:
        intent = state.get("intent", "single_hop")
        if intent == "out_of_scope":
            return "escalate"
        if intent == "structured":
            return "agentic"
        return "retrieve"

    # 决定reflect下一步节点
    def _after_reflect(self, state: GraphState) -> str:
        refl = state.get("reflection") or {}
        iteration = state.get("iteration", 0)
        if refl.get("is_sufficient") or iteration >= self.ctx.config.max_iterations:
            return "generate"
        # 证据不足：改写后回到对应策略重试
        return "agentic" if state.get("intent") == "structured" else "retrieve"

    # 构建状态机
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
            "route",
            self._route_next,
            {"retrieve": "retrieve", "agentic": "agentic", "escalate": "escalate"},
        )
        g.add_edge("retrieve", "reflect")
        g.add_edge("agentic", "reflect")
        g.add_conditional_edges(
            "reflect",
            self._after_reflect,
            {"retrieve": "retrieve", "agentic": "agentic", "generate": "generate"},
        )
        g.add_edge("generate", END)
        g.add_edge("escalate", END)
        return g.compile()

    def run(self, query: str, session_id: str = "", user_id: str = "U001") -> str:
        """执行一轮问答。

        - session_id 非空且记忆已启用时：先读三层记忆（可能改写 query），再写入本轮对话
        - 记忆任何一层出错都不影响主流程
        """

        # 读取记忆
        memory = getattr(self.ctx, "memory", None)
        memory_context = ""
        current_query = query

        if memory is not None and session_id:
            # 会话归属校验：同一 session_id 只能被首次绑定的用户使用
            if not memory.claim_session(session_id, user_id):
                raise PermissionError("session_id 不属于当前用户")
            try:
                mem = memory.prepare(query, session_id, user_id)
                current_query = mem.contextual_query or query
                memory_context = mem.to_prompt_text()
            except Exception as e:
                print(f"[memory] 记忆读取失败，按无记忆模式继续：{e}")

        # 执行状态机，同步阻塞等待
        result = self.graph.invoke(
            {
                "query": query,
                "current_query": current_query,
                "iteration": 0,
                "all_docs": [],
                "memory_context": memory_context,
            }
        )
        answer = result.get("final_answer", "")

        # 写入记忆
        if memory is not None and session_id:
            try:
                memory.save_turn(
                    session_id,
                    user_id,
                    query,
                    answer,
                    {"intent": result.get("intent", "")},
                )
            except Exception as e:
                print(f"[memory] 记忆写入失败：{e}")

        return answer

    def end_session(self, session_id: str, user_id: str = "U001") -> None:
        """会话结束：长期记忆归档 + 短期会话清空。"""
        memory = getattr(self.ctx, "memory", None)
        if memory is not None and session_id:
            memory.end_session(session_id, user_id)
