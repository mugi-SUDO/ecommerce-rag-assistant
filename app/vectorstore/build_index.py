"""
向量索引构建：把商品 / 评价 / 售后政策转为文本 → 分块 → 嵌入 → 存入 Chroma

注意：使用 langchain_community.vectorstores.Chroma。
"""

# 本项目中有两个chroma向量库
# 一个是知识库collection：ecommerce_kb
# 另一个是记忆库collection：memory_vectorstore

# 互不影响， 这里是构建知识库collection的向量索引
# 记忆库的向量索引在MemoryManager中构建。

from typing import List

from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma

from app.runtime import get_context, load_raw_data


def _product_to_text(p: dict) -> str:
    attrs = "; ".join(f"{k}={v}" for k, v in p.get("attrs", {}).items())
    return (
        f"商品ID: {p['productId']}\n"
        f"名称: {p['name']}\n"
        f"品牌: {p['brand']}\n"
        f"类目: {p['category']}\n"
        f"价格: {p['price']}元\n"
        f"评分: {p['rating']}\n"
        f"销量: {p['sales']}\n"
        f"库存: {p['stock']}件\n"
        f"属性: {attrs}"
    )


def _review_to_text(r: dict) -> str:
    return (
        f"评价ID: {r['reviewId']}\n"
        f"商品ID: {r['productId']}\n"
        f"用户: {r['userId']}\n"
        f"评分: {r['rating']}\n"
        f"评价内容: {r['content']}"
    )


def build_documents(products, reviews, policy_text) -> List[Document]:
    docs = []
    for p in products:
        docs.append(
            Document(
                page_content=_product_to_text(p),
                metadata={
                    "type": "product",
                    "productId": p["productId"],
                    "name": p["name"],
                },
            )
        )
    for r in reviews:
        docs.append(
            Document(
                page_content=_review_to_text(r),
                metadata={
                    "type": "review",
                    "reviewId": r["reviewId"],
                    "productId": r["productId"],
                },
            )
        )
    if policy_text.strip():
        docs.append(Document(page_content=policy_text, metadata={"type": "policy"}))
    return docs


def build_vector_index() -> None:
    """构建 Chroma 向量库并写回 ctx.vectorstore / ctx.chunks"""
    ctx = get_context()  # 从 RuntimeContext 获取当前上下文
    config = ctx.config
    products, orders, reviews, policy_text = load_raw_data(config.data_dir)
    ctx.products, ctx.orders, ctx.reviews, ctx.policy_text = (
        products,
        orders,
        reviews,
        policy_text,
    )

    docs = build_documents(products, reviews, policy_text)

    # 分块文本
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=config.chunk_size,
        chunk_overlap=config.chunk_overlap,
        separators=["\n\n", "\n", "。", "！", "？", "，", " ", ""],
    )
    chunks = splitter.split_documents(docs)
    ctx.chunks = chunks

    # 复用已有 collection：Chroma.from_documents 会不断追加新 ID，
    # 每次启动都重建会造成重复文档，也会拖慢 API 启动。
    ctx.vectorstore = Chroma(
        collection_name=config.chroma_collection,
        embedding_function=ctx.embeddings,
        persist_directory=config.chroma_persist_dir,
    )
    try:
        existing = ctx.vectorstore._collection.count()
    except Exception:
        existing = 0

    if existing == 0:
        ctx.vectorstore.add_documents(chunks)
        print(
            f"[build_index] 已构建向量库，文档 {len(docs)} 篇 → 分块 {len(chunks)} 个"
        )
    elif existing == len(chunks):
        print(f"[build_index] 复用已有向量索引（{existing} 个分块）")
    else:
        # 数据或分块参数变了：重建知识库 collection
        ctx.vectorstore.delete_collection()
        ctx.vectorstore = Chroma.from_documents(
            documents=chunks,
            embedding=ctx.embeddings,
            collection_name=config.chroma_collection,
            persist_directory=config.chroma_persist_dir,
        )
        print(f"[build_index] 索引数量不一致（{existing} → {len(chunks)}），已重建")
