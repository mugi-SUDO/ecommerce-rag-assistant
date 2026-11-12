"""
密码哈希与 JWT

- 密码：PBKDF2-HMAC-SHA256（标准库，避免额外依赖），每个密码独立随机盐
- Token：HS256 JWT，sub 为用户 ID；服务端只信任自己签发的 token
"""
import base64
import hashlib
import hmac
import secrets
from datetime import datetime, timedelta, timezone

import jwt

PBKDF2_ITERATIONS = 200_000
HASH_PREFIX = "pbkdf2_sha256"


class AuthError(Exception):
    """认证失败（token 无效 / 过期 / 缺少用户标识）"""


def _b64e(raw: bytes) -> str:
    return base64.b64encode(raw).decode("ascii")


def _b64d(text: str) -> bytes:
    return base64.b64decode(text.encode("ascii"))


def hash_password(password: str, iterations: int = PBKDF2_ITERATIONS) -> str:
    """返回 pbkdf2_sha256$iterations$salt$hash，可直接存库。"""
    salt = secrets.token_bytes(16)
    digest = hashlib.pbkdf2_hmac("sha256", password.encode("utf-8"), salt, iterations)
    return f"{HASH_PREFIX}${iterations}${_b64e(salt)}${_b64e(digest)}"


def verify_password(password: str, encoded: str) -> bool:
    """校验密码；编码格式非法时返回 False，不抛异常。"""
    try:
        prefix, iter_text, salt_text, digest_text = encoded.split("$", 3)
        if prefix != HASH_PREFIX:
            return False
        iterations = int(iter_text)
        salt = _b64d(salt_text)
        expected = _b64d(digest_text)
    except (ValueError, TypeError):
        return False

    actual = hashlib.pbkdf2_hmac("sha256", password.encode("utf-8"), salt, iterations)
    return hmac.compare_digest(actual, expected)


def create_access_token(user_id: str, secret: str, expires_minutes: int = 120,
                        now: datetime = None) -> str:
    issued = now or datetime.now(timezone.utc)
    payload = {
        "sub": user_id,
        "iat": issued,
        "exp": issued + timedelta(minutes=expires_minutes),
    }
    return jwt.encode(payload, secret, algorithm="HS256")


def decode_access_token(token: str, secret: str) -> str:
    """校验 token 并返回 user_id；失败统一抛 AuthError。"""
    try:
        payload = jwt.decode(token, secret, algorithms=["HS256"])
    except jwt.ExpiredSignatureError as e:
        raise AuthError("token 已过期") from e
    except jwt.PyJWTError as e:
        raise AuthError("token 无效") from e

    user_id = payload.get("sub")
    if not isinstance(user_id, str) or not user_id:
        raise AuthError("token 缺少用户标识")
    return user_id
