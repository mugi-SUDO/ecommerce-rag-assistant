"""
长期对话的向量记忆：使用独立 Chroma collection（默认 user_memory）。

- 每轮对话写入一条 Document，metadata 带 user_id / session_id / type / ts
- 检索时按 user_id 过滤，只召回当前用户自己的长期记忆
- 写操作加进程内锁；多进程部署请改用 Chroma server（见 README）
"""
import threading
import time
import uuid
from typing import List

from langchain_core.documents import Document

from config import Config


class MemoryVectorStore:
    def __init__(self, config: Config, embeddings, collection_name: str = None):
        from langchain_community.vectorstores import Chroma
        self.config = config
        self.store = Chroma(
            collection_name=collection_name or config.long_term_collection,
            embedding_function=embeddings,
            persist_directory=config.chroma_persist_dir,
        )
        self._lock = threading.Lock()

    def add(self, user_id: str, session_id: str, text: str, kind: str = "turn") -> None:
        doc = Document(
            page_content=text,
            metadata={
                "user_id": user_id,
                "session_id": session_id,
                "type": kind,
                "ts": time.time(),
            },
        )
        with self._lock:
            self.store.add_documents([doc], ids=[uuid.uuid4().hex])

    def search(self, user_id: str, query: str, k: int = 3) -> List[str]:
        with self._lock:
            try:
                docs = self.store.similarity_search(
                    query, k=k, filter={"user_id": user_id})
            except Exception:
                # 兼容不同版本的 Chroma / LangChain 参数差异：
                # 先多取一些再在本地按 user_id 过滤，避免把其他用户的记忆返回出去
                docs = self.store.similarity_search(query, k=k * 4)
                docs = [d for d in docs
                        if (d.metadata or {}).get("user_id") == user_id][:k]
        return [d.page_content for d in docs]

    def delete_user(self, user_id: str) -> bool:
        """删除某个用户的全部长期记忆（按 metadata 过滤）。"""
        with self._lock:
            try:
                self.store.delete(where={"user_id": user_id})
                return True
            except Exception as e:
                print(f"[memory] 删除用户向量记忆失败：{e}")
                return False
