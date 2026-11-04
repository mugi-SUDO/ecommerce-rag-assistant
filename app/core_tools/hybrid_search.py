"""
混合检索工具：BM25（关键词）+ 向量（语义）+ RRF 融合重排，适合单跳检索型问题。
"""
import hashlib
from collections import defaultdict
from typing import List

import jieba
from langchain_core.documents import Document
from langchain_core.tools import tool
from rank_bm25 import BM25Okapi

from app.runtime import get_context, format_docs

_STOPWORDS = set(
    """的 了 和 是 在 我 有 就 不 也 都 还 这 那 一 个 与 及 等 上 下 中 为 以 于
    从 把 被 让 使 又 而 但 或 什么 怎么 如何 哪些 哪个 哪里 谁 多少 几 你 他 她 它 我们 他们
    请问 请 想 要 需要 能 可以 应该 会 啊 呢 吧 嘛 吗 哦 呀 哈 之 其 此 该 即 各 每 些 种 类
    时 后 前 里 外 内 间 已经 正在 一些 一下""".split()
)


def _tokenize(text: str) -> List[str]:
    return [t for t in jieba.lcut(text) if t.strip() and t not in _STOPWORDS]


class HybridSearchEngine:
    """BM25 + 向量 + RRF 的混合检索引擎（复用 C9 的思路，见 C9/rag_modules/hybrid_retrieval.py）"""

    def __init__(self, chunks):
        self.chunks = list(chunks)
        self.bm25 = None
        self._built = False

    def _ensure_bm25(self):
        if self._built:
            return
        if self.chunks:
            self.bm25 = BM25Okapi([_tokenize(d.page_content) for d in self.chunks])
        self._built = True

    def _bm25_search(self, query, k):
        self._ensure_bm25()
        if self.bm25 is None:
            return []
        tokens = _tokenize(query)
        if not tokens:
            return []
        scores = self.bm25.get_scores(tokens)
        ranked = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)
        out = []
        for i in ranked:
            if len(out) >= k:
                break
            if scores[i] <= 0:
                continue
            d = self.chunks[i]
            out.append(Document(
                page_content=d.page_content,
                metadata={**d.metadata, "search_method": "bm25", "score": float(scores[i])},
            ))
        return out

    def _vector_search(self, query, k):
        ctx = get_context()
        if ctx is None or ctx.vectorstore is None:
            return []
        out = []
        for d, dist in ctx.vectorstore.similarity_search_with_score(query, k=k):
            out.append(Document(
                page_content=d.page_content,
                metadata={**d.metadata, "search_method": "vector", "score": float(dist)},
            ))
        return out

    def _rrf_merge(self, ranked_lists, k):
        scores = defaultdict(float)
        objs = {}
        for lst in ranked_lists:
            for rank, d in enumerate(lst, 1):
                doc_id = str(
                    d.metadata.get("productId")
                    or d.metadata.get("reviewId")
                    or hashlib.md5(d.page_content[:200].encode("utf-8")).hexdigest()
                )
                scores[doc_id] += 1.0 / (60 + rank)
                objs[doc_id] = d
        sorted_ids = sorted(scores, key=lambda i: scores[i], reverse=True)[:k]
        out = []
        for doc_id in sorted_ids:
            d = objs[doc_id]
            md = dict(d.metadata)
            md["rrf_score"] = round(scores[doc_id], 6)
            out.append(Document(page_content=d.page_content, metadata=md))
        return out

    def search(self, query, k=5):
        bm25 = self._bm25_search(query, max(k * 2, 10))
        vec = self._vector_search(query, max(k * 2, 10))
        return self._rrf_merge([bm25, vec], k)


def _get_engine():
    ctx = get_context()
    if ctx.hybrid_engine is None:
        ctx.hybrid_engine = HybridSearchEngine(ctx.chunks or [])
    return ctx.hybrid_engine


def hybrid_search_docs(query: str, k: int = 5):
    return _get_engine().search(query, k=k)


@tool
def hybrid_search(query: str) -> str:
    """混合检索：BM25 关键词 + 向量语义 + RRF 融合重排，适合单跳检索型问题（如"推荐几款性价比高的手机"）。"""
    return format_docs(hybrid_search_docs(query))
