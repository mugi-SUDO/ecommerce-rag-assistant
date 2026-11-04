"""
电商服务平台智能助手入口

运行：
    python main.py

流程：
    加载配置 -> 构建 LLM/嵌入模型 -> 加载数据 -> 构建向量索引(Chroma)
    -> 构建知识图谱(Neo4j) -> 编译 LangGraph 状态机 -> 交互问答
"""

import os
import sys
from pathlib import Path

from dotenv import load_dotenv

# 让 config / app 可被导入
sys.path.insert(0, str(Path(__file__).resolve().parent))

from config import DEFAULT_CONFIG
from app.runtime import RuntimeContext, set_context, build_llm, build_embeddings


def load_env():
    """加载环境依次尝试：项目本地 .env -> code/.env -> 仓库根 .env（回退不覆盖已有值）"""

    root = Path(__file__).resolve().parent
    load_dotenv(root / ".env")
    if os.getenv("DEEPSEEK_API_KEY"):
        return
    load_dotenv(root.parent / ".env", override=False)
    if os.getenv("DEEPSEEK_API_KEY"):
        return
    load_dotenv(root.parent.parent / ".env", override=False)


def initialize_system():
    load_env()
    ctx = RuntimeContext(config=DEFAULT_CONFIG)
    set_context(ctx)

    print(">>> 初始化 LLM（DeepSeek）...")
    ctx.llm = build_llm(DEFAULT_CONFIG)

    print(">>> 初始化本地嵌入模型（BGE-small-zh，首次会下载模型）...")
    ctx.embeddings = build_embeddings(DEFAULT_CONFIG)

    print(">>> 加载数据 + 构建向量索引（Chroma）...")
    from app.vectorstore.build_index import build_vector_index

    build_vector_index()  # 内部会加载 raw 数据并写回 ctx.products/orders/reviews/policy_text

    print(">>> 构建知识图谱（Neo4j，未连接则 GraphRAG 分支降级）...")
    from app.graph.build_graph import build_graph

    build_graph(ctx)

    print(">>> 编译状态机（LangGraph）...")
    from app.state_machine.graph import ECommerceAssistant

    assistant = ECommerceAssistant(ctx)
    return assistant


def run_interactive():
    try:
        assistant = initialize_system()
    except RuntimeError as e:
        print(f"\n[初始化失败] {e}")
        return
    except Exception as e:
        print(f"\n[初始化失败] {type(e).__name__}: {e}")
        return

    print("\n===== 电商智能助手就绪，输入问题开始，输入 exit/退出 结束 =====\n")
    while True:
        try:
            query = input("你：").strip()
        except (EOFError, KeyboardInterrupt):
            print("\n再见！")
            break
        if not query:
            continue
        if query.lower() in ("exit", "quit", "q", "退出"):
            print("再见！")
            break
        print("助手思考中...")
        try:
            answer = assistant.run(query)
        except Exception as e:
            answer = f"[运行出错] {type(e).__name__}: {e}"
        print(f"\n助手：{answer}\n")


if __name__ == "__main__":
    run_interactive()
