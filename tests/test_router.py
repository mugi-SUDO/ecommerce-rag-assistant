"""
离线单元测试：不依赖 LLM / langgraph / Neo4j，只测路由与订单查询的确定性逻辑。

运行：
    python tests/test_router.py
    或  pytest tests/
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from app.runtime import RuntimeContext, set_context, parse_json_response
from app.router.intent_router import IntentRouter


class FakeLLM:
    """返回固定 content 的假 LLM，避免真实 API 调用。"""
    def __init__(self, content: str):
        self._content = content

    def invoke(self, prompt):
        class _Msg:
            content = self._content
        return _Msg()


def _ctx_with_llm(json_text: str) -> RuntimeContext:
    ctx = RuntimeContext()
    ctx.llm = FakeLLM("```json\n" + json_text + "\n```")
    return ctx


def test_route_simple():
    ctx = _ctx_with_llm('{"intent":"simple","confidence":0.9,"explanation":"单个事实"}')
    set_context(ctx)
    r = IntentRouter(ctx).route("小米14 多少钱")
    assert r.intent == "simple"
    assert r.confidence == 0.9


def test_route_out_of_scope():
    ctx = _ctx_with_llm('{"intent":"out_of_scope","confidence":0.8,"explanation":"投诉"}')
    set_context(ctx)
    r = IntentRouter(ctx).route("我要投诉商家")
    assert r.intent == "out_of_scope"


def test_route_invalid_intent_falls_back():
    ctx = _ctx_with_llm('{"intent":"weird","confidence":0.5,"explanation":""}')
    set_context(ctx)
    r = IntentRouter(ctx).route("随便")
    assert r.intent == "single_hop"


def test_parse_json_response():
    assert parse_json_response("```json\n{\"a\":1}\n```", {}) == {"a": 1}
    assert parse_json_response("不是 json", {"a": 0}) == {"a": 0}


def test_order_query_by_order_id():
    from app.core_tools.order_query import order_query_docs
    ctx = RuntimeContext()
    ctx.products = [{"productId": "P001", "name": "小米14", "stock": 320}]
    ctx.orders = [{
        "orderId": "O20241101001", "userId": "U001",
        "items": [{"productId": "P001", "quantity": 1}],
        "totalAmount": 3999, "status": "已签收",
        "orderDate": "2024-11-01", "courier": "顺丰", "trackingNo": "SF111",
    }]
    set_context(ctx)
    docs = order_query_docs("O20241101001 到哪了")
    assert any("O20241101001" in d.page_content for d in docs)


def test_order_query_inventory():
    from app.core_tools.order_query import order_query_docs
    ctx = RuntimeContext()
    ctx.products = [{"productId": "P001", "name": "小米14", "stock": 320}]
    # 即便该商品也出现在订单里，库存意图也必须优先返回库存而非订单
    ctx.orders = [{
        "orderId": "O20241101001", "userId": "U001",
        "items": [{"productId": "P001", "quantity": 1}],
        "totalAmount": 3999, "status": "已签收",
        "orderDate": "2024-11-01", "courier": "顺丰", "trackingNo": "SF111",
    }]
    set_context(ctx)
    docs = order_query_docs("小米14 还有没有货")
    assert docs and docs[0].metadata["type"] == "inventory"
    assert "320" in docs[0].page_content


if __name__ == "__main__":
    passed = 0
    for name, fn in sorted(globals().items()):
        if name.startswith("test_") and callable(fn):
            try:
                fn()
                print(f"PASS {name}")
                passed += 1
            except AssertionError as e:
                print(f"FAIL {name}: {e}")
    print(f"\n{passed} 个测试通过")
