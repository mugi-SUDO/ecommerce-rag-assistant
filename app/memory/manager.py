"""
三层记忆管理器

读路径（每轮问答前）：
  短期会话（Redis List） + 用户画像（Redis Hash） + 长期语义记忆（Chroma，按 user_id 过滤）
  -> 必要时用最近对话把当前问题改写成独立问题
  -> 组装成 memory_context 注入生成 prompt

写路径（每轮问答后）：
  短期会话追加一轮 + 长期历史追加一条 + 向量库写入一条 + 用户画像增量更新

会话结束：
  生成会话摘要写入向量库（type=summary），然后删除短期会话 key。
"""
import json
import time
from dataclasses import dataclass, field
from typing import Dict, List, Optional

from app.prompts.prompts import (
    contextualize_prompt,
    profile_update_prompt,
    session_summary_prompt,
)
from app.runtime import parse_json_response

# 出现这些词时，当前问题很可能依赖上文，需要做指代消解
_REFERENCE_WORDS = (
    "这个", "那个", "它", "它们", "上面", "刚才", "之前", "再", "还有",
    "那款", "这款", "同款", "该商品", "那个订单", "这台", "那台",
)


@dataclass
class MemoryContext:
    contextual_query: str
    recent_turns: List[dict] = field(default_factory=list)
    long_term: List[str] = field(default_factory=list)
    profile: Dict[str, str] = field(default_factory=dict)

    def to_prompt_text(self, max_turns: int = 6, max_chars: int = 200) -> str:
        parts = []
        if self.profile:
            items = [f"- {k}: {v}" for k, v in self.profile.items() if str(v).strip()]
            if items:
                parts.append("【用户画像】\n" + "\n".join(items))
        if self.recent_turns:
            lines = []
            for t in self.recent_turns[-max_turns:]:
                u = str(t.get("user", ""))[:max_chars]
                a = str(t.get("assistant", ""))[:max_chars]
                lines.append(f"用户: {u}\n助手: {a}")
            parts.append("【最近对话】\n" + "\n".join(lines))
        if self.long_term:
            parts.append(
                "【长期记忆（与当前问题相关）】\n"
                + "\n".join(f"- {str(x)[:max_chars]}" for x in self.long_term)
            )
        return "\n\n".join(parts)


class MemoryManager:
    def __init__(self, config, llm, store, vector_memory=None):
        self.config = config
        self.llm = llm
        self.store = store
        self.vector = vector_memory

    # ---------------- 会话归属 ----------------
    def claim_session(self, session_id: str, user_id: str) -> bool:
        """把 session_id 绑定到 user_id。

        第一次使用时绑定；之后只有同一 user_id 能用这个 session_id。
        返回 False 表示该会话属于其他用户（调用方应拒绝请求）。
        """
        return self.store.claim_session(session_id, user_id)

    def get_session_owner(self, session_id: str) -> Optional[str]:
        return self.store.get_session_owner(session_id)

    def list_sessions(self, user_id: str) -> List[str]:
        return self.store.list_sessions(user_id)

    # ---------------- 读 ----------------
    def prepare(self, query: str, session_id: str, user_id: str) -> MemoryContext:
        recent = self.store.get_session_turns(
            session_id, self.config.short_term_max_turns)
        profile = self.store.get_profile(user_id)
        contextual = self._contextualize(query, recent)

        long_term: List[str] = []
        if self.vector is not None:
            try:
                long_term = self.vector.search(
                    user_id, contextual, self.config.long_term_top_k)
            except Exception as e:
                print(f"[memory] 长期记忆检索失败：{e}")

        return MemoryContext(contextual, recent, long_term, profile)

    def _needs_context(self, query: str) -> bool:
        return any(w in query for w in _REFERENCE_WORDS)

    def _contextualize(self, query: str, recent: List[dict]) -> str:
        """把"这个/那个/再便宜点"改写成独立问题；无需改写时直接返回原问题。"""
        if not recent or not self._needs_context(query) or self.llm is None:
            return query
        try:
            raw = self.llm.invoke(
                contextualize_prompt(query, self._format_turns(recent)))
            text = getattr(raw, "content", str(raw)).strip()
            return text or query
        except Exception:
            return query

    @staticmethod
    def _format_turns(turns: List[dict], max_turns: int = 6,
                      max_chars: int = 200) -> str:
        lines = []
        for t in turns[-max_turns:]:
            lines.append(f"用户: {str(t.get('user', ''))[:max_chars]}")
            lines.append(f"助手: {str(t.get('assistant', ''))[:max_chars]}")
        return "\n".join(lines)

    # ---------------- 写 ----------------
    def save_turn(self, session_id: str, user_id: str, query: str,
                  answer: str, metadata: Optional[dict] = None) -> None:
        try:
            self.store.append_turn(session_id, user_id, query, answer, metadata)
        except Exception as e:
            print(f"[memory] 短期记忆写入失败：{e}")

        try:
            self.store.append_history(user_id, {
                "user": query,
                "assistant": answer,
                "session_id": session_id,
                "ts": time.time(),
            })
        except Exception as e:
            print(f"[memory] 长期历史写入失败：{e}")

        if self.vector is not None:
            try:
                self.vector.add(
                    user_id, session_id,
                    f"用户: {query}\n助手: {answer}", "turn")
            except Exception as e:
                print(f"[memory] 向量记忆写入失败：{e}")

        if self.config.profile_update_enabled and self.llm is not None:
            try:
                # 「读画像 -> 合并 -> 写画像」是读改写，必须串行化
                with self.store.user_lock(user_id, timeout=10):
                    self._update_profile(user_id, query, answer)
            except TimeoutError:
                print(f"[memory] 跳过本轮画像更新（用户 {user_id} 正忙）")
            except Exception as e:
                print(f"[memory] 用户画像更新失败：{e}")

    def _update_profile(self, user_id: str, query: str, answer: str) -> None:
        current = self.store.get_profile(user_id)
        raw = self.llm.invoke(profile_update_prompt(
            json.dumps(current, ensure_ascii=False) if current else "{}",
            query, answer))
        text = getattr(raw, "content", str(raw))
        patch = parse_json_response(text, {})
        if not isinstance(patch, dict) or not patch:
            return

        merged = dict(current)
        for k, v in patch.items():
            if isinstance(v, (str, int, float)) and str(v).strip():
                merged[k] = str(v).strip()
        if merged:
            self.store.save_profile(user_id, merged)

    # ---------------- 会话结束 ----------------
    def end_session(self, session_id: str, user_id: str) -> None:
        owner = self.store.get_session_owner(session_id)
        if owner and owner != user_id:
            raise PermissionError("session_id 不属于当前用户")

        try:
            turns = self.store.get_session_turns(session_id, limit=100)
        except Exception:
            turns = []

        if turns and self.vector is not None:
            try:
                summary = self._summarize(turns)
                self.vector.add(user_id, session_id, summary, "summary")
            except Exception as e:
                print(f"[memory] 会话归档失败：{e}")

        try:
            self.store.clear_session(session_id)
        except Exception as e:
            print(f"[memory] 会话清理失败：{e}")

    def _summarize(self, turns: List[dict]) -> str:
        text = self._format_turns(turns, max_turns=100, max_chars=500)
        if self.llm is None:
            return "会话摘要：\n" + text
        raw = self.llm.invoke(session_summary_prompt(text))
        summary = getattr(raw, "content", str(raw)).strip()
        return summary or ("会话摘要：\n" + text)

    # ---------------- 删除用户全部记忆 ----------------
    def clear_user(self, user_id: str) -> dict:
        """删除某用户的短期会话、长期历史、画像和向量记忆。"""
        sessions = self.store.clear_user(user_id)
        vector_deleted = False
        if self.vector is not None:
            vector_deleted = self.vector.delete_user(user_id)
        return {
            "user_id": user_id,
            "cleared_sessions": sessions,
            "vector_deleted": vector_deleted,
        }
