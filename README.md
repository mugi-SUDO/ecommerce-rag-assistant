# 电商服务平台智能助手（My_ECommerce）

AI辅助生成的仅供学习的项目。前置学习课程是https://github.com/datawhalechina/all-in-rag
基于 GraphRAG + AgenticRAG + 多 Agent 协作 + LangChain/LangGraph 的电商智能助手，
带三层记忆（Redis + Chroma）和 JWT 多用户 HTTP API。
项目按「接口/认证层 → 路由层 → 策略层 → 统一工具层 → 数据层」分层组织，详见下方目录结构。

## 一、环境准备

```bash
# 1) 创建并激活虚拟环境（Python 3.10+）
python -m venv .venv
.venv\Scripts\activate        # Windows

# 2) 安装依赖（推荐用锁定版本，确保与开发环境一致）
pip install -r requirements.lock.txt
# 或只装版本范围：pip install -r requirements.txt

# 3) 启动 Neo4j 和 Redis
#    Neo4j：GraphRAG 分支需要，未启动则自动降级为纯向量检索
#    Redis：三层记忆需要，未启动则自动降级为进程内记忆
docker compose up -d neo4j redis
```

## 二、配置

复制 `.env.example` 为 `.env`，填入 `DEEPSEEK_API_KEY`：

```bash
copy .env.example .env        # Windows
```

## 三、运行

### 命令行模式（离线）

```bash
python main.py
```

启动时会打印本次会话 ID 和用户 ID。用户 ID 默认 `U001`，可通过环境变量固定，用于跨会话积累画像：

```bash
set MEMORY_USER_ID=U001        # Windows
export MEMORY_USER_ID=U001     # Git Bash / Linux
```

交互示例问题：

- 简单事实型：`小米14 多少钱？`（普通 RAG / 向量检索）
- 单跳检索型：`推荐几款性价比高的手机`（混合检索 + RRF 重排）
- 多跳推理型：`买过小米14的人还买了什么`（GraphRAG / Neo4j）
- 结构化多步型：`查一下订单 O20241101001 到哪了`（AgenticRAG / 规划+执行）
- 超范围：`我要投诉商家`（转人工）

### HTTP API 模式（多用户）

```bash
# 先设置 JWT_SECRET（否则使用仅限开发的默认密钥）
set JWT_SECRET=请替换为至少32位的随机字符串
python api_server.py
# 文档：http://localhost:8000/docs
```

## 四、测试（离线，无需 LLM/Neo4j/Redis/langgraph）

```bash
python tests/test_router.py
python tests/test_memory.py
python tests/test_auth.py
```

## 五、目录结构

```
My_ECommerce/
├── config.py                 # 配置
├── main.py                   # 命令行入口
├── api_server.py             # FastAPI 启动入口
├── app/
│   ├── bootstrap.py          # CLI / API 共用的初始化流程
│   ├── runtime.py            # 运行时上下文 / 依赖注入
│   ├── api/                  # FastAPI 路由 + 请求/响应模型
│   ├── auth/                 # 密码哈希 / JWT / 用户账号存储
│   ├── router/               # 意图识别 + 复杂度路由
│   ├── agents/               # 规划机 / 执行器 / 反思器(SelfRAG)
│   ├── core_tools/           # 统一核心工具（向量/混合/图谱/订单）
│   ├── strategies/           # naive_rag / hybrid_rag / graph_rag / agentic_rag
│   ├── state_machine/        # 顶层 LangGraph 状态机
│   ├── graph/                # 图谱本体 / 建图 / 检索
│   ├── vectorstore/          # Chroma 向量索引构建
│   ├── memory/               # 三层记忆（Redis 存储 / Chroma 长期记忆 / 管理器）
│   └── prompts/              # Prompt 模板
├── data/raw/                 # 商品/订单/评价/售后政策
└── tests/                    # 离线单测
```

## 六、三层记忆（Redis + Chroma）

多轮对话依赖三层记忆，读写都由 `app/memory/manager.py` 的 `MemoryManager` 统一编排：

| 层           | 作用                        | 存储                | Key / Collection                                        | 生命周期                                          |
| ------------ | --------------------------- | ------------------- | ------------------------------------------------------- | ------------------------------------------------- |
| 短期会话记忆 | 当前会话的上下文            | Redis List          | `mem:session:{session_id}:turns`                        | 最多 10 轮，30 分钟无活动自动过期；会话结束时清空 |
| 长期对话历史 | 跨会话的问答记录 + 语义召回 | Redis List + Chroma | `mem:user:{user_id}:history` + collection `user_memory` | 保留最近 200 条 / 90 天；向量记忆长期保留         |
| 用户画像     | 偏好、预算、品牌、类目兴趣  | Redis Hash          | `mem:user:{user_id}:profile`                            | 跨会话累积，随对话增量更新                        |

**一轮问答的完整流程**

1. 读：取短期会话最近 10 轮 + 用户画像；若问题含“这个/那个/它/再便宜点”等指代词，用 LLM 结合最近对话改写成独立问题。
2. 召：用改写后的问题到 `user_memory` 做向量检索（`filter={"user_id": ...}`，只召回本人记忆）。
3. 拼：画像 + 最近对话 + 长期记忆拼成 `memory_context`，注入生成 Prompt；改写后的问题同时用于路由和检索。
4. 写：本轮问答追加到短期会话、长期历史、向量库，并用 LLM 抽取/合并用户画像。
5. 归档：退出时把本次会话压缩成摘要写入向量库（`type=summary`），再删除短期会话 key。

**降级策略**

- Redis 连不上：自动切换为 `InMemoryStore`（进程内），对话照常进行，重启即失效。
- Chroma 长期记忆不可用：只关闭语义召回，短期会话和画像仍工作。
- 任意一层写入失败只打印告警，不影响回答。
- **API 模式例外**：账号与认证数据不能降级到进程内存，所以 FastAPI 启动时 Redis 不可用会直接失败；命令行模式仍可降级运行。

相关配置在 [config.py](config.py) 的「三层记忆」段：`redis_url`、`session_ttl_seconds`、`short_term_max_turns`、`long_term_top_k`、`long_term_history_max`、`profile_update_enabled`；设 `memory_enabled=False` 可整体关闭。

## 七、HTTP API 与多用户认证（FastAPI）

### 身份从哪里来

- 用户身份**只来自 JWT**：`Authorization: Bearer <token>`，服务端校验签名后取 `sub` 作为 `user_id`
- `POST /chat` 的请求体**不接受 `user_id`**（Pydantic `extra="forbid"`，传了直接 422），杜绝前端伪造身份
- 密码用 PBKDF2-HMAC-SHA256 + 随机盐存储，登录成功才签发 token（HS256，默认 2 小时有效）
- 账号数据存 Redis 的 `auth:username:*` / `auth:user:*`，**API 模式要求 Redis 可用**（账号不能降级到进程内存）
- `session_id` 第一次使用时会绑定到当前 `user_id`；其他用户拿同一个 session_id 请求会得到 403

### 接口

| 方法   | 路径                          | 认证 | 说明                                  |
| ------ | ----------------------------- | ---- | ------------------------------------- |
| GET    | `/health`                     | 否   | 健康检查                              |
| POST   | `/auth/register`              | 否   | 注册，返回 user_id                    |
| POST   | `/auth/login`                 | 否   | 登录，返回 JWT                        |
| GET    | `/me`                         | 是   | 当前用户信息                          |
| POST   | `/chat`                       | 是   | 提问；`session_id` 可选，用于续接会话 |
| POST   | `/chat/end`                   | 是   | 归档并清空该会话的短期记忆            |
| GET    | `/me/memory/profile`          | 是   | 用户画像                              |
| GET    | `/me/memory/history?limit=20` | 是   | 长期历史                              |
| GET    | `/me/sessions`                | 是   | 当前活跃会话列表                      |
| DELETE | `/me/memory`                  | 是   | 删除本人全部记忆（Redis + Chroma）    |

### 调用示例

```bash
# 注册
curl -X POST http://localhost:8000/auth/register \
  -H "Content-Type: application/json" \
  -d '{"username":"alice","password":"password123"}'

# 登录拿 token
curl -X POST http://localhost:8000/auth/login \
  -H "Content-Type: application/json" \
  -d '{"username":"alice","password":"password123"}'

# 提问（user_id 不在请求体里，只认 token）
curl -X POST http://localhost:8000/chat \
  -H "Authorization: Bearer <TOKEN>" \
  -H "Content-Type: application/json" \
  -d '{"message":"小米14多少钱？"}'

# 续接同一会话
curl -X POST http://localhost:8000/chat \
  -H "Authorization: Bearer <TOKEN>" \
  -H "Content-Type: application/json" \
  -d '{"message":"这个有现货吗？","session_id":"<SESSION_ID>"}'
```

### 生产环境注意事项

- **必须设置 `JWT_SECRET`**（至少 32 位随机字符），不要用默认值；泄露后所有人都能伪造任意用户
- 不需要开放注册时设 `ALLOW_REGISTRATION=false`
- 前面加 HTTPS 反向代理；按需设置 `CORS_ORIGINS`（留空不启用 CORS）
- 登录接口建议加限流 / 验证码，防止撞库
- JWT 是无状态的，当前没有主动吊销机制；需要“立即登出/封禁”时，在 Redis 维护 token 黑名单或改用短期 token + refresh token
- 删除账号时要同时清理 `auth:*` 与 `/me/memory` 两处数据

## 八、注意事项和优化方向

- 首次运行会从 HuggingFace 下载 `bge-small-zh-v1.5` 嵌入模型（已配 `HF_ENDPOINT=https://hf-mirror.com` 加速）。
- `langgraph` 必须使用 `0.2.x`（`requirements.txt` 已限定 `<0.3`）；1.x 会强拉 `langchain-core>=1`，与 langchain 0.3 不兼容。
- 向量库用 `langchain_community.vectorstores.Chroma`（避免额外安装 `langchain_chroma`）。
- 知识库索引和长期语义记忆都存放在 `vector_index/`（两个独立 collection），已加入 `.gitignore`。知识库索引会自动重建；**删除该目录会清空长期语义记忆**（Redis 中的长期历史与画像不受影响）。
- 退货 / 订单追踪 / 商品推荐等 Skill+MCP 按需加载为**长期目标**，当前未实现。
- 状态机中Reflect是固定边。对于简单问题其实不需要SelfRAG的介入。
- GraphRAG的降级只实现了当Neo4j不可用时报错，没自动回退到混合检索。
- Executor是列表执行，如果让它能够正确处理依赖关系。这是因为Planner一开始就固定好了每个任务的提示词。优化的策略是直接加一个回环变成Plan-Execute循环，或者把任务列表变成真正的DAG有向无环图：让Planner输出带depends_on和query_template的任务节点，而不是线性列表；维护一个共享的结果池（context），反复找出依赖已满足的节点去执行，执行时用context里的上游输出渲染query_template。
