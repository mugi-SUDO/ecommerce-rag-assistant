"""
向量索引构建：把商品 / 评价 / 售后政策转为文本 → 分块 → 嵌入 → 存入 Chroma

注意：使用 langchain_community.vectorstores.Chroma（langchain_chroma 未安装）。
"""
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
        docs.append(Document(
            page_content=_product_to_text(p),
            metadata={"type": "product", "productId": p["productId"], "name": p["name"]},
        ))
    for r in reviews:
        docs.append(Document(
            page_content=_review_to_text(r),
            metadata={"type": "review", "reviewId": r["reviewId"], "productId": r["productId"]},
        ))
    if policy_text.strip():
        docs.append(Document(page_content=policy_text, metadata={"type": "policy"}))
    return docs


def build_vector_index() -> None:
    """构建 Chroma 向量库并写回 ctx.vectorstore / ctx.chunks"""
    ctx = get_context()
    config = ctx.config
    products, orders, reviews, policy_text = load_raw_data(config.data_dir)
    ctx.products, ctx.orders, ctx.reviews, ctx.policy_text = products, orders, reviews, policy_text

    docs = build_documents(products, reviews, policy_text)

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=config.chunk_size,
        chunk_overlap=config.chunk_overlap,
        separators=["\n\n", "\n", "。", "！", "？", "，", " ", ""],
    )
    chunks = splitter.split_documents(docs)
    ctx.chunks = chunks

    ctx.vectorstore = Chroma.from_documents(
        documents=chunks,
        embedding=ctx.embeddings,
        collection_name=config.chroma_collection,
        persist_directory=config.chroma_persist_dir,
    )
    print(f"[build_index] 已构建向量库，文档 {len(docs)} 篇 → 分块 {len(chunks)} 个")
