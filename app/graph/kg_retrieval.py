"""
知识图谱检索（GraphRAG 核心）：多跳遍历 / 子图提取

- 实体识别：优先确定性匹配（商品名 / 品牌 / 类目），失败时用 LLM 兜底。
- 查询类型：关键词启发式判断 multi_hop（协同购买 / 替代品）或 subgraph / entity_relation。
- 返回 List[Document]，供 GraphRAG 策略与 kg_query 工具使用。
"""
from typing import List

from langchain_core.documents import Document

from app.prompts.prompts import graph_query_prompt
from app.runtime import get_context, parse_json_response


class GraphRetriever:
    def __init__(self, ctx):
        self.ctx = ctx

    @property
    def _brands(self):
        return sorted({p["brand"] for p in self.ctx.products})

    @property
    def _categories(self):
        return sorted({p["category"] for p in self.ctx.products})

    def _extract_entities(self, query: str) -> List[str]:
        entities = []
        for p in self.ctx.products:
            if p["name"] and p["name"] in query:
                entities.append(("product", p["name"]))
        for b in self._brands:
            if b in query:
                entities.append(("brand", b))
        for c in self._categories:
            if c in query:
                entities.append(("category", c))
        # 去重（保持顺序）
        seen, out = set(), []
        for item in entities:
            if item not in seen:
                seen.add(item)
                out.append(item)
        return out

    def _detect_query_type(self, query: str) -> str:
        if any(k in query for k in ["还买了", "也买了", "一起买", "搭配", "配套", "组合", "连带"]):
            return "co_bought"
        if any(k in query for k in ["替代", "类似", "同价位", "换一个", "差不多", "媲美", "相近"]):
            return "substitute"
        return "subgraph"

    def _llm_understand(self, query: str):
        """LLM 兜底：理解图查询类型与实体。"""
        llm = self.ctx.llm
        if llm is None:
            return {"query_type": "subgraph", "entities": []}
        known = ", ".join([p["name"] for p in self.ctx.products][:30])
        raw = llm.invoke(graph_query_prompt(query, known))
        parsed = parse_json_response(getattr(raw, "content", str(raw)), {"query_type": "subgraph", "entities": []})
        return parsed

    def _run(self, query_type: str, entities: List) -> List[Document]:
        driver = self.ctx.neo4j_driver
        if driver is None:
            return [Document(page_content="[知识图谱不可用：Neo4j 未连接]", metadata={"type": "kg_error"})]

        results = []
        with driver.session() as session:
            for kind, name in entities:
                if query_type == "co_bought" and kind == "product":
                    rec = session.run(
                        """
                        MATCH (p:Product {name: $name})-[:CO_BOUGHT_WITH]-(q:Product)
                        RETURN p.name AS source, q.name AS name, q.price AS price, q.rating AS rating
                        """,
                        name=name,
                    )
                    for r in rec:
                        results.append(f"买了「{r['source']}」的用户还买了「{r['name']}」（{r['price']}元，评分 {r['rating']}）")
                elif query_type == "substitute" and kind == "product":
                    rec = session.run(
                        """
                        MATCH (p:Product {name: $name})-[:SUBSTITUTE_OF]-(q:Product)
                        RETURN p.name AS source, q.name AS name, q.price AS price, q.rating AS rating
                        """,
                        name=name,
                    )
                    for r in rec:
                        results.append(f"「{r['source']}」的替代品/相似款：{r['name']}（{r['price']}元，评分 {r['rating']}）")
                elif kind == "product":
                    rec = session.run(
                        """
                        MATCH (p:Product {name: $name})
                        OPTIONAL MATCH (p)-[:MADE_BY]->(b:Brand)
                        OPTIONAL MATCH (p)-[:BELONGS_TO]->(c:Category)
                        OPTIONAL MATCH (p)-[:HAS_ATTRIBUTE]->(a:Attribute)
                        RETURN p, b.name AS brand, c.name AS category, collect(a.name + '=' + a.value) AS attrs
                        """,
                        name=name,
                    )
                    for r in rec:
                        p = r["p"]
                        attrs = ", ".join(r["attrs"]) if r["attrs"] else "无"
                        results.append(
                            f"商品 {p['name']}：品牌 {r['brand']}，类目 {r['category']}，"
                            f"价格 {p['price']}元，评分 {p['rating']}，库存 {p['stock']}件；属性：{attrs}"
                        )
                elif kind == "category":
                    rec = session.run(
                        """
                        MATCH (c:Category {name: $name})<-[:BELONGS_TO]-(p:Product)
                        RETURN p.name AS name, p.price AS price, p.rating AS rating
                        ORDER BY p.sales DESC
                        """,
                        name=name,
                    )
                    for r in rec:
                        results.append(f"{name}类目下的商品：{r['name']}（{r['price']}元，评分 {r['rating']}）")
                elif kind == "brand":
                    rec = session.run(
                        """
                        MATCH (b:Brand {name: $name})<-[:MADE_BY]-(p:Product)
                        RETURN p.name AS name, p.price AS price, p.rating AS rating
                        """,
                        name=name,
                    )
                    for r in rec:
                        results.append(f"{name}品牌商品：{r['name']}（{r['price']}元，评分 {r['rating']}）")

        if not results:
            return [Document(page_content=f"[知识图谱未检索到与「{query_type}」相关的结果]", metadata={"type": "kg_empty"})]
        return [Document(page_content="\n".join(results), metadata={"type": "kg", "query_type": query_type})]

    def search(self, query: str, k: int = 5) -> List[Document]:
        entities = self._extract_entities(query)
        query_type = self._detect_query_type(query)
        if not entities:
            # LLM 兜底理解
            understood = self._llm_understand(query)
            query_type = understood.get("query_type", "subgraph")
            mapping = {"co_bought": "co_bought", "substitute": "substitute", "multi_hop": "co_bought",
                       "entity_relation": "subgraph", "subgraph": "subgraph"}
            query_type = mapping.get(query_type, "subgraph")
            ent_names = understood.get("entities", [])
            entities = [("product", n) for n in ent_names if any(p["name"] == n for p in self.ctx.products)]
        docs = self._run(query_type, entities)
        return docs[:k]
