"""
用户账号存储（Redis）

Key 设计：
- auth:username:{lower_username}  -> user_id（String，SETNX 保证唯一）
- auth:user:{user_id}             -> Hash(username, password_hash, created_at)

说明：账号数据必须持久化，所以 API 模式要求 Redis 可用（与可降级的记忆层不同）。
"""
import time
import uuid
from typing import Optional

import redis

from app.auth.security import hash_password, verify_password


class UsernameTaken(Exception):
    """用户名已存在"""


def _username_key(username: str) -> str:
    return f"auth:username:{username.strip().lower()}"


def _user_key(user_id: str) -> str:
    return f"auth:user:{user_id}"


class UserAuthStore:
    def __init__(self, url: str):
        self.client = redis.Redis.from_url(url, decode_responses=True)
        self.client.ping()

    def register(self, username: str, password: str) -> dict:
        username = username.strip()
        user_id = "U" + uuid.uuid4().hex[:12]
        # SETNX：并发注册同名用户时只有一个能成功
        if not self.client.set(_username_key(username), user_id, nx=True):
            raise UsernameTaken(username)

        created_at = str(time.time())
        try:
            self.client.hset(_user_key(user_id), mapping={
                "user_id": user_id,
                "username": username,
                "password_hash": hash_password(password),
                "created_at": created_at,
            })
        except Exception:
            # 回滚用户名占位，避免留下“抢到名字但无法登录”的死账号
            self.client.delete(_username_key(username))
            raise
        return {"user_id": user_id, "username": username, "created_at": created_at}

    def authenticate(self, username: str, password: str) -> Optional[dict]:
        user_id = self.client.get(_username_key(username))
        if not user_id:
            return None
        data = self.client.hgetall(_user_key(user_id))
        if not data or not verify_password(password, data.get("password_hash", "")):
            return None
        return {"user_id": user_id, "username": data.get("username", "")}

    def get(self, user_id: str) -> Optional[dict]:
        data = self.client.hgetall(_user_key(user_id))
        if not data:
            return None
        return {
            "user_id": user_id,
            "username": data.get("username", ""),
            "created_at": data.get("created_at", ""),
        }
