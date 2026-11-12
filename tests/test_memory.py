"""
离线单元测试：三层记忆（不依赖 Redis / Chroma / LLM）

运行：
    python tests/test_memory.py
    或  pytest tests/
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from config import Config
from app.memory.store import InMemoryStore
from app.memory.manager import MemoryContext, MemoryManager


class FakeLLM:
    """按顺序返回固定内容的假 LLM，避免真实 API 调用。"""

    def __init__(self, contents):
        self._contents = list(contents)

    def invoke(self, prompt):
        content = self._contents.pop(0) if self._contents else ""

        class _Msg:
            pass

        msg = _Msg()
        msg.content = content
        return msg


def _manager(llm=None, vector=None, short_term=3, history=5):
    cfg = Config()
    cfg.profile_update_enabled = True
    store = InMemoryStore(session_ttl=60, short_term_max_turns=short_term,
                          history_max=history)
    return MemoryManager(cfg, llm=llm, store=store, vector_memory=vector)


def test_short_term_trim():
    m = _manager()
    for i in range(5):
        m.save_turn("s1", "u1", f"q{i}", f"a{i}")
    turns = m.store.get_session_turns("s1")
    assert len(turns) == 3, turns
    assert turns[-1]["user"] == "q4"
    assert turns[0]["user"] == "q2"


def test_long_term_history_trim():
    m = _manager(history=2)
    for i in range(4):
        m.save_turn("s1", "u1", f"q{i}", f"a{i}")
    history = m.store.get_history("u1")
    assert len(history) == 2
    assert history[-1]["user"] == "q3"


def test_profile_roundtrip():
    store = InMemoryStore()
    store.save_profile("u1", {"budget": "4000", "brands": "小米"})
    assert store.get_profile("u1")["budget"] == "4000"


def test_memory_context_text():
    ctx = MemoryContext(
        contextual_query="小米14 多少钱",
        recent_turns=[{"user": "推荐手机", "assistant": "推荐小米14"}],
        long_term=["用户上次咨询过小米14的价格"],
        profile={"budget": "4000"},
    )
    text = ctx.to_prompt_text()
    assert "用户画像" in text
    assert "最近对话" in text
    assert "长期记忆" in text
    assert "4000" in text


def test_end_session_clears_short_term_keeps_long_term():
    m = _manager()
    m.save_turn("s1", "u1", "q", "a")
    m.end_session("s1", "u1")
    assert m.store.get_session_turns("s1") == []
    assert len(m.store.get_history("u1")) == 1


def test_contextualize_skips_without_reference_words():
    m = _manager(llm=FakeLLM(["不应被调用"]))
    assert m._contextualize("小米14多少钱", [{"user": "x", "assistant": "y"}]) == "小米14多少钱"


def test_contextualize_rewrites_reference_query():
    m = _manager(llm=FakeLLM(["小米14 多少钱"]))
    out = m._contextualize("这个多少钱", [{"user": "小米14怎么样", "assistant": "不错"}])
    assert out == "小米14 多少钱"


def test_prepare_uses_store_and_vector():
    class FakeVector:
        def add(self, user_id, session_id, text, kind="turn"):
            pass

        def search(self, user_id, query, k):
            assert user_id == "u1"
            return ["历史记忆"]

    m = _manager(vector=FakeVector())
    m.save_turn("s1", "u1", "上次问题", "上次回答")
    mem = m.prepare("它多少钱", "s1", "u1")
    assert mem.recent_turns[-1]["user"] == "上次问题"
    assert mem.long_term == ["历史记忆"]


def test_claim_session_isolates_users():
    store = InMemoryStore()
    assert store.claim_session("s1", "u1") is True
    assert store.claim_session("s1", "u1") is True       # 同一用户可重复使用
    assert store.claim_session("s1", "u2") is False      # 其他用户被拒绝
    assert store.get_session_owner("s1") == "u1"


def test_end_session_rejects_other_user():
    m = _manager()
    m.save_turn("s1", "u1", "q", "a")
    try:
        m.end_session("s1", "u2")
    except PermissionError:
        pass
    else:
        raise AssertionError("其他用户不应能结束该会话")
    # 原会话未被清空
    assert len(m.store.get_session_turns("s1")) == 1


def test_session_owner_survives_end_session():
    """会话结束后归属标记仍在：原用户可以续用，其他用户不能冒用。"""
    m = _manager()
    m.save_turn("s2", "u1", "q", "a")
    m.end_session("s2", "u1")
    # 对话内容已清空，但归属保留
    assert m.store.get_session_turns("s2") == []
    assert m.claim_session("s2", "u1") is True
    assert m.claim_session("s2", "u2") is False


def test_clear_user_removes_all_memory():
    m = _manager()
    m.save_turn("s1", "u1", "q1", "a1")
    m.save_turn("s2", "u1", "q2", "a2")
    m.save_turn("s3", "u2", "q3", "a3")
    result = m.clear_user("u1")
    assert result["cleared_sessions"] == 2
    assert m.store.get_session_turns("s1") == []
    assert m.store.get_history("u1") == []
    assert m.store.get_profile("u1") == {}
    # 其他用户不受影响
    assert len(m.store.get_history("u2")) == 1


def test_list_sessions():
    m = _manager()
    m.save_turn("s1", "u1", "q1", "a1")
    m.save_turn("s2", "u1", "q2", "a2")
    assert sorted(m.list_sessions("u1")) == ["s1", "s2"]
    assert m.list_sessions("u2") == []


def test_profile_update_merges_json_patch():
    llm = FakeLLM(['{"preferences": "注重性价比", "budget": "4000", "brands": "", '
                   '"category_interest": "手机", "notes": ""}'])
    m = _manager(llm=llm)
    m.save_turn("s1", "u1", "预算4000，想要性价比手机", "推荐小米14")
    profile = m.store.get_profile("u1")
    assert profile["budget"] == "4000"
    assert profile["preferences"] == "注重性价比"
    # 空字段不应覆盖已有值
    assert "brands" not in profile


if __name__ == "__main__":
    import traceback

    tests = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    passed = 0
    for fn in tests:
        try:
            fn()
            print(f"[PASS] {fn.__name__}")
            passed += 1
        except Exception:
            print(f"[FAIL] {fn.__name__}")
            traceback.print_exc()
    print(f"\n{passed}/{len(tests)} passed")
    sys.exit(0 if passed == len(tests) else 1)
