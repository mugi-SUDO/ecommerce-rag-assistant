"""
电商知识图谱本体定义：节点标签 / 关系类型常量，以及索引与约束语句
"""

# 定义图里有什么

# 节点标签
NODE_PRODUCT = "Product"
NODE_CATEGORY = "Category"
NODE_BRAND = "Brand"
NODE_ATTRIBUTE = "Attribute"
NODE_USER = "User"
NODE_ORDER = "Order"
NODE_REVIEW = "Review"

# 关系类型
REL_BELONGS_TO = "BELONGS_TO"  # Product -> Category
REL_MADE_BY = "MADE_BY"  # Product -> Brand
REL_HAS_ATTRIBUTE = "HAS_ATTRIBUTE"  # Product -> Attribute
REL_CO_BOUGHT_WITH = "CO_BOUGHT_WITH"  # Product -> Product
REL_SUBSTITUTE_OF = "SUBSTITUTE_OF"  # Product -> Product
REL_PURCHASED = "PURCHASED"  # User -> Product
REL_PLACED = "PLACED"  # User -> Order
REL_CONTAINS = "CONTAINS"  # Order -> Product
REL_WROTE = "WROTE"  # User -> Review
REL_HAS_REVIEW = "HAS_REVIEW"  # Product -> Review

# 索引 / 约束（建图时执行）
INDEX_STATEMENTS = [
    "CREATE CONSTRAINT product_id_unique IF NOT EXISTS FOR (p:Product) REQUIRE p.productId IS UNIQUE",
    "CREATE CONSTRAINT order_id_unique IF NOT EXISTS FOR (o:Order) REQUIRE o.orderId IS UNIQUE",
    "CREATE CONSTRAINT review_id_unique IF NOT EXISTS FOR (r:Review) REQUIRE r.reviewId IS UNIQUE",
    "CREATE CONSTRAINT user_id_unique IF NOT EXISTS FOR (u:User) REQUIRE u.userId IS UNIQUE",
    "CREATE INDEX product_name_index IF NOT EXISTS FOR (p:Product) ON (p.name)",
    "CREATE INDEX category_name_index IF NOT EXISTS FOR (c:Category) ON (c.name)",
    "CREATE INDEX brand_name_index IF NOT EXISTS FOR (b:Brand) ON (b.name)",
    "CREATE INDEX order_user_index IF NOT EXISTS FOR (o:Order) ON (o.userId)",
]
