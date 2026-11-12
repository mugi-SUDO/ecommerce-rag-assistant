"""
离线单元测试：密码哈希与 JWT（不依赖 Redis / 网络）

运行：
    python tests/test_auth.py
"""
import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import jwt

from app.auth.security import (
    AuthError,
    create_access_token,
    decode_access_token,
    hash_password,
    verify_password,
)

SECRET = "test-secret-key-at-least-32-characters-long"


def test_hash_and_verify():
    encoded = hash_password("correct horse battery staple")
    assert encoded.startswith("pbkdf2_sha256$")
    assert verify_password("correct horse battery staple", encoded)
    assert not verify_password("wrong password", encoded)


def test_hash_is_salted():
    a = hash_password("same-password")
    b = hash_password("same-password")
    assert a != b, "相同密码每次哈希结果应不同（随机盐）"
    assert verify_password("same-password", a)
    assert verify_password("same-password", b)


def test_verify_rejects_malformed():
    assert not verify_password("x", "")
    assert not verify_password("x", "not$enough$parts")
    assert not verify_password("x", "md5$1$abc$def")


def test_token_roundtrip():
    token = create_access_token("U123", SECRET, 30)
    assert decode_access_token(token, SECRET) == "U123"


def test_token_wrong_secret():
    token = create_access_token("U123", SECRET, 30)
    try:
        decode_access_token(token, "another-secret-also-long-enough-123")
    except AuthError:
        pass
    else:
        raise AssertionError("错误密钥应校验失败")


def test_token_expired():
    past = datetime.now(timezone.utc) - timedelta(hours=1)
    token = create_access_token("U123", SECRET, 1, now=past)
    try:
        decode_access_token(token, SECRET)
    except AuthError as e:
        assert "过期" in str(e)
    else:
        raise AssertionError("过期 token 应校验失败")


def test_token_tampered():
    token = create_access_token("U123", SECRET, 30)
    tampered = token[:-2] + ("aa" if token[-2:] != "aa" else "bb")
    try:
        decode_access_token(tampered, SECRET)
    except AuthError:
        pass
    else:
        raise AssertionError("被篡改的 token 应校验失败")


def test_token_without_subject():
    token = jwt.encode({"foo": "bar"}, SECRET, algorithm="HS256")
    try:
        decode_access_token(token, SECRET)
    except AuthError as e:
        assert "用户标识" in str(e)
    else:
        raise AssertionError("缺少 sub 的 token 应校验失败")


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
