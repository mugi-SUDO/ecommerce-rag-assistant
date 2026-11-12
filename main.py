"""
支持CLI和HTTP API两种模式

这里是CLI模式的电商服务平台智能助手入口
本地单用户模式，user_id从环境变量里加载

运行：
    python main.py

流程：
    加载配置 -> 构建 LLM/嵌入模型 -> 加载数据 -> 构建向量索引(Chroma)
    -> 构建三层记忆(Redis + Chroma) -> 构建知识图谱(Neo4j)
    -> 编译 LangGraph 状态机 -> 交互问答 -> 会话结束归档记忆

多用户 HTTP 服务请用：
    python api_server.py
"""

import os
import sys
import uuid
from pathlib import Path

# 把当前文件所在目录插入到模块搜索路径 sys.path 的最前面
# 从而让 Python 能优先从该目录导入模块。
sys.path.insert(0, str(Path(__file__).resolve().parent))

from app.bootstrap import initialize_system


def run_interactive():
    try:
        assistant = initialize_system()
    except RuntimeError as e:
        print(f"\n[初始化失败] {e}")
        return
    except Exception as e:
        print(f"\n[初始化失败] {type(e).__name__}: {e}")
        return

    session_id = uuid.uuid4().hex[:8]  # 随机生成一个 8 位十六进制的会话 ID
    user_id = os.getenv(
        "MEMORY_USER_ID", "U001"
    )  # 从环境变量获取用户 ID，默认值为 "U001"
    print(f"\n本次会话 ID：{session_id}    用户 ID：{user_id}")
    print("===== 电商智能助手就绪，输入问题开始，输入 exit/退出 结束 =====\n")
    print(
        "提示：会话结束时短期记忆会归档为摘要并清空；用户画像与长期历史跨会话保留。\n"
    )

    try:
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
                answer = assistant.run(query, session_id=session_id, user_id=user_id)
            except Exception as e:
                answer = f"[运行出错] {type(e).__name__}: {e}"
            print(f"\n助手：{answer}\n")
    finally:
        try:
            assistant.end_session(session_id, user_id)
            print("[memory] 本次会话已归档：摘要写入长期记忆，短期会话已清空。")
        except Exception as e:
            print(f"[memory] 会话归档失败：{type(e).__name__}: {e}")


if __name__ == "__main__":
    run_interactive()
