"""
FastAPI 服务：多用户认证 + 会话问答

安全模型（重要）：
- 用户身份只来自 `Authorization: Bearer <JWT>`，`sub` 即 user_id
- 请求体不接受 user_id（ChatRequest 设了 extra="forbid"），防止越权读写他人记忆
- session_id 首次使用时绑定到当前 user_id，其他用户无法复用

启动：
    python api_server.py
    或 uvicorn app.api.server:app --host 0.0.0.0 --port 8000 --workers 1
"""
import os
from contextlib import asynccontextmanager

from fastapi import Depends, FastAPI, HTTPException, Query, Request, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer

from app.api.schemas import (
    ChatRequest,
    ChatResponse,
    EndSessionRequest,
    HistoryResponse,
    LoginRequest,
    MemoryDeleteResponse,
    ProfileResponse,
    RegisterRequest,
    SessionsResponse,
    StatusResponse,
    TokenResponse,
    UserResponse,
)
from app.auth.security import AuthError, create_access_token, decode_access_token
from app.auth.store import UserAuthStore, UsernameTaken
from app.bootstrap import initialize_system
from config import DEFAULT_CONFIG

bearer_scheme = HTTPBearer(auto_error=False)


def _allow_registration() -> bool:
    return os.getenv("ALLOW_REGISTRATION", "true").lower() not in ("0", "false", "no")


@asynccontextmanager
async def lifespan(app: FastAPI):
    # 初始化 LLM / 嵌入 / 向量索引 / 三层记忆 / 图谱 / 状态机
    assistant = initialize_system()

    # 账号数据必须持久化：API 模式要求 Redis 可用（连不上会直接启动失败）
    auth = UserAuthStore(DEFAULT_CONFIG.redis_url)

    jwt_secret = os.getenv("JWT_SECRET") or DEFAULT_CONFIG.jwt_secret
    placeholder = "change-me-to-a-random-string-at-least-32-chars"
    if jwt_secret in (DEFAULT_CONFIG.jwt_secret, placeholder):
        print("[auth] 警告：正在使用默认/示例 JWT_SECRET，仅限本地开发；"
              "生产请设置至少 32 位的 JWT_SECRET")
    elif len(jwt_secret) < 32:
        print("[auth] 警告：JWT_SECRET 长度不足 32 位，建议使用更长的随机密钥")

    app.state.assistant = assistant
    app.state.auth = auth
    app.state.jwt_secret = jwt_secret
    print("[api] 服务初始化完成")
    yield


app = FastAPI(
    title="电商服务平台智能助手 API",
    version="1.0.0",
    description="JWT 认证 + 多用户会话 + 三层记忆",
    lifespan=lifespan,
)

_cors_origins = [o.strip() for o in os.getenv("CORS_ORIGINS", "").split(",") if o.strip()]
if _cors_origins:
    app.add_middleware(
        CORSMiddleware,
        allow_origins=_cors_origins,
        allow_methods=["GET", "POST", "DELETE"],
        allow_headers=["Authorization", "Content-Type"],
        allow_credentials=False,
    )


# ---------------- 依赖 ----------------
def get_current_user(
    request: Request,
    creds: HTTPAuthorizationCredentials = Depends(bearer_scheme),
) -> str:
    """从 Bearer token 解析 user_id；token 无效一律 401。"""
    if creds is None:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="缺少 Bearer token",
            headers={"WWW-Authenticate": "Bearer"},
        )
    try:
        return decode_access_token(creds.credentials, request.app.state.jwt_secret)
    except AuthError as e:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail=str(e),
            headers={"WWW-Authenticate": "Bearer"},
        ) from e


def _memory(request: Request):
    memory = getattr(request.app.state.assistant.ctx, "memory", None)
    if memory is None:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="记忆服务未启用，无法提供多用户会话",
        )
    return memory


# ---------------- 健康检查 ----------------
@app.get("/health", response_model=StatusResponse)
def health():
    return {"status": "ok"}


# ---------------- 认证 ----------------
@app.post("/auth/register", response_model=UserResponse,
          status_code=status.HTTP_201_CREATED)
def register(req: RegisterRequest, request: Request):
    if not _allow_registration():
        raise HTTPException(status_code=403, detail="注册已关闭")
    try:
        return request.app.state.auth.register(req.username, req.password)
    except UsernameTaken:
        raise HTTPException(status_code=409, detail="用户名已存在")


@app.post("/auth/login", response_model=TokenResponse)
def login(req: LoginRequest, request: Request):
    user = request.app.state.auth.authenticate(req.username, req.password)
    if user is None:
        # 不区分“用户不存在”和“密码错误”，避免用户名枚举
        raise HTTPException(status_code=401, detail="用户名或密码错误")

    minutes = int(os.getenv("JWT_EXPIRE_MINUTES", DEFAULT_CONFIG.jwt_expire_minutes))
    token = create_access_token(user["user_id"], request.app.state.jwt_secret, minutes)
    return {
        "access_token": token,
        "token_type": "bearer",
        "expires_in": minutes * 60,
        "user_id": user["user_id"],
        "username": user["username"],
    }


@app.get("/me", response_model=UserResponse)
def me(request: Request, user_id: str = Depends(get_current_user)):
    user = request.app.state.auth.get(user_id)
    if user is None:
        raise HTTPException(status_code=401, detail="用户不存在或已被删除")
    return user


# ---------------- 对话 ----------------
@app.post("/chat", response_model=ChatResponse)
def chat(req: ChatRequest, request: Request, user_id: str = Depends(get_current_user)):
    assistant = request.app.state.assistant
    memory = _memory(request)

    session_id = req.session_id or os.urandom(16).hex()

    # 会话归属：首次使用绑定到当前用户；被别人占用则拒绝
    if not memory.claim_session(session_id, user_id):
        raise HTTPException(status_code=403, detail="session_id 不属于当前用户")

    try:
        # 同一会话串行处理，避免两条消息同时读到相同上下文
        with memory.store.session_lock(session_id, timeout=60):
            answer = assistant.run(req.message, session_id=session_id, user_id=user_id)
    except PermissionError as e:
        raise HTTPException(status_code=403, detail=str(e))
    except TimeoutError:
        raise HTTPException(status_code=409, detail="该会话正在处理中，请稍后重试")
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"问答失败：{type(e).__name__}") from e

    return {"answer": answer, "session_id": session_id, "user_id": user_id}


@app.post("/chat/end", response_model=StatusResponse)
def end_session(req: EndSessionRequest, request: Request,
                user_id: str = Depends(get_current_user)):
    assistant = request.app.state.assistant
    _memory(request)
    try:
        assistant.end_session(req.session_id, user_id)
    except PermissionError as e:
        raise HTTPException(status_code=403, detail=str(e))
    return {"status": "archived", "detail": "会话摘要已归档，短期记忆已清空"}


# ---------------- 记忆管理 ----------------
@app.get("/me/memory/profile", response_model=ProfileResponse)
def get_profile(request: Request, user_id: str = Depends(get_current_user)):
    memory = _memory(request)
    return {"user_id": user_id, "profile": memory.store.get_profile(user_id)}


@app.get("/me/memory/history", response_model=HistoryResponse)
def get_history(
    request: Request,
    limit: int = Query(default=20, ge=1, le=200),
    user_id: str = Depends(get_current_user),
):
    memory = _memory(request)
    return {"user_id": user_id, "items": memory.store.get_history(user_id, limit)}


@app.get("/me/sessions", response_model=SessionsResponse)
def list_sessions(request: Request, user_id: str = Depends(get_current_user)):
    memory = _memory(request)
    return {"user_id": user_id, "sessions": memory.list_sessions(user_id)}


@app.delete("/me/memory", response_model=MemoryDeleteResponse)
def delete_memory(request: Request, user_id: str = Depends(get_current_user)):
    """删除当前用户的全部记忆（短期会话 + 长期历史 + 画像 + 向量记忆）。"""
    memory = _memory(request)
    result = memory.clear_user(user_id)
    return result
