"""
知识图谱构建：连接 Neo4j，导入商品 / 用户 / 订单 / 评价，
并推导 CO_BOUGHT_WITH（协同购买）与 SUBSTITUTE_OF（替代品）边。
"""
from neo4j import GraphDatabase

from app.graph import schema


def connect_neo4j(config):
    """连接 Neo4j；失败返回 None（GraphRAG 分支降级）。"""
    try:
        driver = GraphDatabase.driver(
            config.neo4j_uri,
            auth=(config.neo4j_user, config.neo4j_password),
        )
        driver.verify_connectivity()
        return driver
    except Exception as e:
        print(f"[build_graph] ⚠️ Neo4j 连接失败（GraphRAG 分支将降级）：{e}")
        return None


def _run_constraints(driver):
    with driver.session() as session:
        for stmt in schema.INDEX_STATEMENTS:
            session.run(stmt)


def _load_products(driver, products):
    with driver.session() as session:
        for p in products:
            session.run(
                """
                MERGE (p:Product {productId: $productId})
                SET p.name = $name, p.price = $price, p.rating = $rating,
                    p.sales = $sales, p.stock = $stock
                MERGE (b:Brand {name: $brand})
                MERGE (c:Category {name: $category})
                MERGE (p)-[:MADE_BY]->(b)
                MERGE (p)-[:BELONGS_TO]->(c)
                """,
                productId=p["productId"], name=p["name"], price=p["price"],
                rating=p["rating"], sales=p["sales"], stock=p["stock"],
                brand=p["brand"], category=p["category"],
            )
            for key, value in p.get("attrs", {}).items():
                session.run(
                    """
                    MATCH (p:Product {productId: $productId})
                    MERGE (a:Attribute {productId: $productId, name: $key})
                    SET a.value = $value
                    MERGE (p)-[:HAS_ATTRIBUTE]->(a)
                    """,
                    productId=p["productId"], key=key, value=str(value),
                )


def _load_orders(driver, orders):
    with driver.session() as session:
        for o in orders:
            session.run(
                """
                MERGE (u:User {userId: $userId})
                MERGE (o:Order {orderId: $orderId})
                SET o.orderDate = $orderDate, o.status = $status,
                    o.totalAmount = $totalAmount, o.courier = $courier,
                    o.trackingNo = $trackingNo, o.userId = $userId
                MERGE (u)-[:PLACED]->(o)
                """,
                userId=o["userId"], orderId=o["orderId"], orderDate=o["orderDate"],
                status=o["status"], totalAmount=o["totalAmount"],
                courier=o.get("courier", ""), trackingNo=o.get("trackingNo", ""),
            )
            for item in o.get("items", []):
                session.run(
                    """
                    MATCH (o:Order {orderId: $orderId})
                    MATCH (p:Product {productId: $productId})
                    MERGE (o)-[:CONTAINS]->(p)
                    """,
                    orderId=o["orderId"], productId=item["productId"],
                )


def _load_reviews(driver, reviews):
    with driver.session() as session:
        for r in reviews:
            session.run(
                """
                MERGE (u:User {userId: $userId})
                MERGE (r:Review {reviewId: $reviewId})
                SET r.rating = $rating, r.content = $content, r.productId = $productId
                MERGE (u)-[:WROTE]->(r)
                WITH r
                MATCH (p:Product {productId: $productId})
                MERGE (p)-[:HAS_REVIEW]->(r)
                """,
                userId=r["userId"], reviewId=r["reviewId"], rating=r["rating"],
                content=r["content"], productId=r["productId"],
            )


def _derive_relations(driver):
    """由订单共现推导协同购买，由同类目同价位推导替代品。"""
    with driver.session() as session:
        session.run(
            """
            MATCH (o:Order)-[:CONTAINS]->(p1:Product), (o)-[:CONTAINS]->(p2:Product)
            WHERE p1.productId < p2.productId
            MERGE (p1)-[:CO_BOUGHT_WITH]->(p2)
            """
        )
        session.run(
            """
            MATCH (p1:Product)-[:BELONGS_TO]->(c:Category)<-[:BELONGS_TO]-(p2:Product)
            WHERE p1.productId < p2.productId
              AND abs(p1.price - p2.price) / toFloat(p1.price) < 0.25
            MERGE (p1)-[:SUBSTITUTE_OF]->(p2)
            """
        )


def build_graph(ctx) -> object:
    """构建知识图谱，返回 neo4j.Driver（失败返回 None）。"""
    driver = connect_neo4j(ctx.config)
    if driver is None:
        ctx.neo4j_driver = None
        return None
    _run_constraints(driver)
    _load_products(driver, ctx.products)
    _load_orders(driver, ctx.orders)
    _load_reviews(driver, ctx.reviews)
    _derive_relations(driver)
    ctx.neo4j_driver = driver
    print("[build_graph] 知识图谱构建完成（Neo4j）")
    return driver
