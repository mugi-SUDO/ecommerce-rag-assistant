"""API 请求 / 响应模型"""
from typing import List, Optional

from pydantic import BaseModel, ConfigDict, Field

USERNAME_PATTERN = r"^[A-Za-z0-9_一-龥]{3,32}$"
SESSION_PATTERN = r"^[0-9a-fA-F]{8,64}$"


class RegisterRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")
    username: str = Field(min_length=3, max_length=32, pattern=USERNAME_PATTERN)
    password: str = Field(min_length=8, max_length=128)


class LoginRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")
    username: str = Field(min_length=1, max_length=32)
    password: str = Field(min_length=1, max_length=128)


class UserResponse(BaseModel):
    user_id: str
    username: str
    created_at: Optional[str] = None


class TokenResponse(BaseModel):
    access_token: str
    token_type: str = "bearer"
    expires_in: int
    user_id: str
    username: str


class ChatRequest(BaseModel):
    # extra="forbid"：客户端若传 user_id / 其他字段会直接 422，
    # 用户身份只能来自 Authorization: Bearer <token>
    model_config = ConfigDict(extra="forbid")
    message: str = Field(min_length=1, max_length=2000)
    session_id: Optional[str] = Field(
        default=None, pattern=SESSION_PATTERN,
        description="续接已有会话时传入；不传则服务端生成新会话",
    )


class ChatResponse(BaseModel):
    answer: str
    session_id: str
    user_id: str


class EndSessionRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")
    session_id: str = Field(pattern=SESSION_PATTERN)


class ProfileResponse(BaseModel):
    user_id: str
    profile: dict


class HistoryResponse(BaseModel):
    user_id: str
    items: List[dict]


class SessionsResponse(BaseModel):
    user_id: str
    sessions: List[str]


class MemoryDeleteResponse(BaseModel):
    user_id: str
    cleared_sessions: int
    vector_deleted: bool


class StatusResponse(BaseModel):
    status: str
    detail: Optional[str] = None
