"""
订单查询工具：查订单状态 / 物流轨迹 / 商品库存 / 退款，适合结构化多步型问题。
"""

# 规则匹配，不涉及向量库，直接在JSON中查找

from langchain_core.documents import Document
from langchain_core.tools import tool

from app.runtime import get_context, format_docs


def order_query_docs(query: str, k: int = 5):
    ctx = get_context()
    products = {p["productId"]: p for p in ctx.products}
    orders = ctx.orders

    # 含库关键词则走库存查询逻辑
    is_inventory = any(
        w in query
        for w in ["库存", "有货", "没货", "还有没有", "现货", "有没有货", "缺货"]
    )
    if is_inventory:
        docs = []
        for p in ctx.products:
            if p["name"] and p["name"] in query:
                docs.append(
                    Document(
                        page_content=f"商品 {p['name']}（{p['productId']}）当前库存 {p['stock']} 件",
                        metadata={"type": "inventory", "productId": p["productId"]},
                    )
                )
        if docs:
            return docs[:k]
        return [
            Document(
                page_content="（请指明要查询库存的具体商品名称）",
                metadata={"type": "order_empty"},
            )
        ]

    # 订单查询逻辑
    matched = []
    # 1) 直接订单号匹配
    for o in orders:
        if o["orderId"] in query:
            matched.append(("order", o))
    # 2) 用户 ID 匹配
    if not matched:
        for o in orders:
            if o["userId"] in query:
                matched.append(("order", o))
    # 3) 商品名匹配订单
    if not matched:
        for o in orders:
            names = [
                products[i["productId"]]["name"]
                for i in o.get("items", [])
                if i["productId"] in products
            ]
            if any(n and n in query for n in names):
                matched.append(("order", o))

    if not matched:
        return [
            Document(
                page_content="（未查询到相关订单，请核对订单号、用户ID或商品名称）",
                metadata={"type": "order_empty"},
            )
        ]

    docs = []
    for kind, obj in matched[:k]:
        items = "、".join(
            f"{products[i['productId']]['name']} x{i.get('quantity', 1)}"
            for i in obj.get("items", [])
            if i["productId"] in products
        )
        if obj.get("courier"):
            logistics = f"；物流：{obj['courier']} {obj['trackingNo']}"
        else:
            logistics = "；尚未发货，暂无物流单号"
        docs.append(
            Document(
                page_content=(
                    f"订单 {obj['orderId']}（用户 {obj['userId']}）：{items}；"
                    f"金额 {obj['totalAmount']}元；状态 {obj['status']}；下单日期 {obj['orderDate']}{logistics}"
                ),
                metadata={
                    "type": "order",
                    "orderId": obj["orderId"],
                    "userId": obj["userId"],
                },
            )
        )
    return docs


# 给LLM专用的函数
@tool
def order_query(query: str) -> str:
    """订单/物流/库存/退款查询：查订单状态、物流轨迹、商品库存，适合结构化多步型问题（如"我的订单到哪了""这个还有没有货"）。"""
    return format_docs(order_query_docs(query))
