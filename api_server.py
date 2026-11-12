"""
FastAPI 启动入口
API 服务器模式，支持多用户并发访问
user_id 由服务器端发放，支持跨会话的用户画像和长期记忆

运行：
    python api_server.py

环境变量（可选）：
    JWT_SECRET           JWT 签名密钥（生产必填，至少 32 位随机字符）
    JWT_EXPIRE_MINUTES   token 有效期，默认 120 分钟
    API_HOST / API_PORT  监听地址，默认 0.0.0.0:8000
    API_WORKERS          worker 数，默认 1（本地 Chroma 不建议多进程写）
    CORS_ORIGINS         允许的前端域名，逗号分隔；留空则不启用 CORS
    ALLOW_REGISTRATION   是否开放注册，默认 true；生产可设 false
"""

import os

import uvicorn

from app.bootstrap import load_env
from config import DEFAULT_CONFIG

if __name__ == "__main__":
    load_env()
    uvicorn.run(
        "app.api.server:app",
        host=os.getenv("API_HOST", DEFAULT_CONFIG.api_host),
        port=int(os.getenv("API_PORT", DEFAULT_CONFIG.api_port)),
        workers=int(os.getenv("API_WORKERS", "1")),
    )
