"""
记忆存储层：Redis 实现 + 进程内兜底实现

三层记忆的存储分工：
- 短期会话：mem:session:{session_id}:turns  List，TTL 到期自动清空，按轮数截断
- 长期历史：mem:user:{user_id}:history      List，按条数截断，默认 90 天过期
- 用户画像：mem:user:{user_id}:profile      Hash，field=data 存 JSON

并发控制：
- user_lock(user_id)    ：保护「读画像 -> 合并 -> 写画像」这段读改写
- session_lock(session) ：同一会话的问答串行执行，避免上下文互相覆盖

Redis 不可用时由 build_memory 自动回退到 InMemoryStore，主流程不受影响；
但 API 模式（FastAPI + JWT）要求 Redis，因为账号数据不能降级到进程内存。
"""
import json
import threading
import time
from collections import defaultdict
from contextlib import contextmanager
from typing import Dict, Iterator, List, Optional

try:
    import redis
except ImportError:  # pragma: no cover
    redis = None


def session_turns_key(session_id: str) -> str:
    return f"mem:session:{session_id}:turns"


def session_meta_key(session_id: str) -> str:
    return f"mem:session:{session_id}:meta"


def session_owner_key(session_id: str) -> str:
    """会话归属单独存一份，生命周期长于会话本身。

    结束会话只清空对话内容（turns/meta），归属标记保留一段时间，
    这样别人拿到一个已结束的 session_id 也无法冒用。
    """
    return f"mem:session:{session_id}:owner"


def user_history_key(user_id: str) -> str:
    return f"mem:user:{user_id}:history"


def user_profile_key(user_id: str) -> str:
    return f"mem:user:{user_id}:profile"


def user_sessions_key(user_id: str) -> str:
    return f"mem:user:{user_id}:sessions"


class InMemoryStore:
    """无 Redis 时的兜底实现，同时用于离线单元测试。"""

    def __init__(self, session_ttl: int = 1800, short_term_max_turns: int = 10,
                 history_max: int = 200):
        self.session_ttl = session_ttl
        self.short_term_max_turns = short_term_max_turns
        self.history_max = history_max
        self._turns: Dict[str, List[dict]] = defaultdict(list)
        self._history: Dict[str, List[dict]] = defaultdict(list)
        self._profiles: Dict[str, dict] = {}
        self._session_owner: Dict[str, str] = {}
        self._user_sessions: Dict[str, set] = defaultdict(set)
        self._locks: Dict[str, threading.Lock] = {}
        self._locks_guard = threading.Lock()

    def ping(self) -> bool:
        return True

    # ---- 锁 ----
    def _get_lock(self, key: str) -> threading.Lock:
        with self._locks_guard:
            return self._locks.setdefault(key, threading.Lock())

    @contextmanager
    def user_lock(self, user_id: str, timeout: float = 10) -> Iterator[None]:
        lock = self._get_lock(f"user:{user_id}")
        if not lock.acquire(timeout=timeout):
            raise TimeoutError(f"获取用户锁超时：{user_id}")
        try:
            yield
        finally:
            lock.release()

    @contextmanager
    def session_lock(self, session_id: str, timeout: float = 60) -> Iterator[None]:
        lock = self._get_lock(f"session:{session_id}")
        if not lock.acquire(timeout=timeout):
            raise TimeoutError(f"获取会话锁超时：{session_id}")
        try:
            yield
        finally:
            lock.release()

    # ---- 会话归属 ----
    def claim_session(self, session_id: str, user_id: str) -> bool:
        with self._locks_guard:
            owner = self._session_owner.get(session_id)
            if owner is None:
                self._session_owner[session_id] = user_id
                return True
            return owner == user_id

    def get_session_owner(self, session_id: str) -> Optional[str]:
        return self._session_owner.get(session_id)

    def list_sessions(self, user_id: str) -> List[str]:
        return sorted(self._user_sessions.get(user_id, set()))

    # ---- 短期会话 ----
    def append_turn(self, session_id: str, user_id: str, query: str,
                    answer: str, metadata: Optional[dict] = None) -> None:
        item = {"user": query, "assistant": answer, "ts": time.time(),
                "user_id": user_id, **(metadata or {})}
        turns = self._turns[session_id]
        turns.append(item)
        if len(turns) > self.short_term_max_turns:
            del turns[:-self.short_term_max_turns]
        self._session_owner.setdefault(session_id, user_id)
        self._user_sessions[user_id].add(session_id)

    def get_session_turns(self, session_id: str, limit: Optional[int] = None) -> List[dict]:
        turns = self._turns.get(session_id, [])
        return list(turns[-limit:]) if limit else list(turns)

    def clear_session(self, session_id: str) -> None:
        self._turns.pop(session_id, None)
        # 只清对话内容，保留归属：会话结束后其他用户仍不能冒用该 session_id
        owner = self._session_owner.get(session_id)
        if owner:
            self._user_sessions.get(owner, set()).discard(session_id)

    # ---- 长期历史 ----
    def append_history(self, user_id: str, item: dict) -> None:
        history = self._history[user_id]
        history.append(item)
        if len(history) > self.history_max:
            del history[:-self.history_max]

    def get_history(self, user_id: str, limit: Optional[int] = None) -> List[dict]:
        history = self._history.get(user_id, [])
        return list(history[-limit:]) if limit else list(history)

    # ---- 用户画像 ----
    def get_profile(self, user_id: str) -> dict:
        return dict(self._profiles.get(user_id, {}))

    def save_profile(self, user_id: str, profile: dict) -> None:
        self._profiles[user_id] = dict(profile)

    # ---- 删除用户全部记忆 ----
    def clear_user(self, user_id: str) -> int:
        sessions = list(self._user_sessions.get(user_id, set()))
        # 已结束会话不在 sessions 集合里，但归属标记仍指向该用户，一并清理
        owned = [sid for sid, owner in self._session_owner.items() if owner == user_id]
        for sid in set(sessions) | set(owned):
            self._turns.pop(sid, None)
            self._session_owner.pop(sid, None)
        self._user_sessions.pop(user_id, None)
        self._history.pop(user_id, None)
        self._profiles.pop(user_id, None)
        return len(sessions)


class RedisMemoryStore:
    """Redis 实现。连接失败时由 build_memory 回退到 InMemoryStore。"""

    def __init__(self, url: str, session_ttl: int = 1800,
                 short_term_max_turns: int = 10, history_max: int = 200,
                 history_ttl: int = 90 * 24 * 3600,
                 owner_ttl: int = 7 * 24 * 3600):
        if redis is None:
            raise RuntimeError("未安装 redis 客户端，请执行 pip install redis")
        self.client = redis.Redis.from_url(url, decode_responses=True)
        self.client.ping()
        self.session_ttl = session_ttl
        self.short_term_max_turns = short_term_max_turns
        self.history_max = history_max
        self.history_ttl = history_ttl
        self.owner_ttl = owner_ttl

    def ping(self) -> bool:
        return bool(self.client.ping())

    # ---- 锁（跨进程生效）----
    @contextmanager
    def user_lock(self, user_id: str, timeout: float = 10) -> Iterator[None]:
        lock = self.client.lock(
            f"mem:lock:user:{user_id}",
            timeout=max(timeout * 6, 60),   # 覆盖一次画像 LLM 调用的耗时
            blocking_timeout=timeout,
        )
        if not lock.acquire():
            raise TimeoutError(f"获取用户锁超时：{user_id}")
        try:
            yield
        finally:
            try:
                lock.release()
            except Exception:
                pass

    @contextmanager
    def session_lock(self, session_id: str, timeout: float = 60) -> Iterator[None]:
        lock = self.client.lock(
            f"mem:lock:session:{session_id}",
            timeout=max(timeout * 3, 180),
            blocking_timeout=timeout,
        )
        if not lock.acquire():
            raise TimeoutError(f"获取会话锁超时：{session_id}")
        try:
            yield
        finally:
            try:
                lock.release()
            except Exception:
                pass

    # ---- 会话归属 ----
    def claim_session(self, session_id: str, user_id: str) -> bool:
        key = session_owner_key(session_id)
        if self.client.set(key, user_id, nx=True, ex=self.owner_ttl):
            return True
        owner = self.client.get(key)
        if owner is None:
            # 兼容改动前创建的会话：当时归属只记录在 meta 里
            owner = self.client.hget(session_meta_key(session_id), "user_id")
        return owner == user_id

    def get_session_owner(self, session_id: str) -> Optional[str]:
        return (self.client.get(session_owner_key(session_id))
                or self.client.hget(session_meta_key(session_id), "user_id"))

    def list_sessions(self, user_id: str) -> List[str]:
        return sorted(self.client.smembers(user_sessions_key(user_id)))

    # ---- 短期会话 ----
    def append_turn(self, session_id: str, user_id: str, query: str,
                    answer: str, metadata: Optional[dict] = None) -> None:
        item = {"user": query, "assistant": answer, "ts": time.time(),
                "user_id": user_id, **(metadata or {})}
        key = session_turns_key(session_id)
        meta = session_meta_key(session_id)
        sessions = user_sessions_key(user_id)
        pipe = self.client.pipeline()
        pipe.rpush(key, json.dumps(item, ensure_ascii=False))
        pipe.ltrim(key, -self.short_term_max_turns, -1)
        pipe.expire(key, self.session_ttl)
        pipe.hset(meta, mapping={"user_id": user_id, "last_active": str(item["ts"])})
        pipe.expire(meta, self.session_ttl)
        pipe.sadd(sessions, session_id)
        pipe.expire(sessions, self.session_ttl)
        pipe.execute()

    def get_session_turns(self, session_id: str, limit: Optional[int] = None) -> List[dict]:
        n = limit or self.short_term_max_turns
        raw = self.client.lrange(session_turns_key(session_id), -n, -1)
        return [json.loads(x) for x in raw]

    def clear_session(self, session_id: str) -> None:
        owner = self.get_session_owner(session_id)
        pipe = self.client.pipeline()
        pipe.delete(session_turns_key(session_id), session_meta_key(session_id))
        if owner:
            pipe.srem(user_sessions_key(owner), session_id)
        pipe.execute()

    # ---- 长期历史 ----
    def append_history(self, user_id: str, item: dict) -> None:
        key = user_history_key(user_id)
        pipe = self.client.pipeline()
        pipe.rpush(key, json.dumps(item, ensure_ascii=False))
        pipe.ltrim(key, -self.history_max, -1)
        pipe.expire(key, self.history_ttl)
        pipe.execute()

    def get_history(self, user_id: str, limit: Optional[int] = None) -> List[dict]:
        n = limit or self.history_max
        raw = self.client.lrange(user_history_key(user_id), -n, -1)
        return [json.loads(x) for x in raw]

    # ---- 用户画像 ----
    def get_profile(self, user_id: str) -> dict:
        raw = self.client.hget(user_profile_key(user_id), "data")
        if not raw:
            return {}
        try:
            return json.loads(raw)
        except Exception:
            return {}

    def save_profile(self, user_id: str, profile: dict) -> None:
        key = user_profile_key(user_id)
        self.client.hset(key, mapping={
            "data": json.dumps(profile, ensure_ascii=False),
            "updated_at": str(time.time()),
        })
        self.client.expire(key, self.history_ttl)

    # ---- 删除用户全部记忆 ----
    def clear_user(self, user_id: str) -> int:
        sessions = list(self.client.smembers(user_sessions_key(user_id)))
        pipe = self.client.pipeline()
        for sid in sessions:
            pipe.delete(session_turns_key(sid), session_meta_key(sid),
                        session_owner_key(sid))
        pipe.delete(
            user_history_key(user_id),
            user_profile_key(user_id),
            user_sessions_key(user_id),
        )
        pipe.execute()
        return len(sessions)
