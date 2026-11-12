"""
集中管理 Prompt 模板

约定：模板使用 .format() 填充，占位符为 {query} / {docs} / {tools} 等；
JSON 示例里的花括号需要写成 {{ }}。
"""


def router_prompt(query: str) -> str:
    return f"""你是电商服务平台的智能路由助手。分析用户问题，判断它属于哪一类并给出置信度。

用户问题：{query}

类别定义：
- simple：简单事实型，单个事实、知识库直接有答案、无需推理（如"小米14 多少钱""退货政策是什么"）
- single_hop：单跳检索型，需要检索但实体关系简单（如"推荐几款性价比高的手机""有哪些降噪耳机"）
- multi_hop：多跳推理型，涉及实体间关系、需要连边推理（如"买过小米14的人还买了什么""华为Mate60的替代品"）
- structured：结构化多步型，查订单/物流/库存等（如"我的订单到哪了""这个还有没有货"）
- out_of_scope：超出范围，知识库与工具都覆盖不了、或需人工介入（如"帮我申请退款仲裁""我要投诉商家"）

严格返回 JSON（不要 markdown 代码块）：
{{"intent": "simple|single_hop|multi_hop|structured|out_of_scope", "confidence": 0.0-1.0, "explanation": "一句话说明"}}
"""


def planner_prompt(query: str, tools_desc: str) -> str:
    return f"""你是任务规划助手。把用户问题拆解成若干个可执行的子任务，每个子任务指定一个工具和一条检索/查询语句。

可用工具：
{tools_desc}

用户问题：{query}

规则：
1. 子任务按依赖顺序排列（step 从 1 开始递增）
2. 每个子任务的 tool 必须来自上面的工具列表
3. depends_on 表示该子任务依赖哪些前面步骤的结果（没有就写 []）
4. 尽量用最少的子任务完成

严格返回 JSON（不要 markdown 代码块）：
{{"subtasks": [{{"step": 1, "tool": "order_query", "query": "查询订单...", "depends_on": []}}]}}
"""


def reflector_prompt(query: str, docs_text: str) -> str:
    return f"""你是严格的证据反思助手（SelfRAG 反思）。评估检索到的证据是否足以回答用户问题。

用户问题：{query}

检索到的证据：
{docs_text}

请逐项评估：
1. relevance_score (0-10)：证据与问题的相关性
2. sufficiency_score (0-10)：证据是否充分到能回答
3. is_supported (bool)：证据是否支持回答（而非相互矛盾）
4. is_sufficient (bool)：是否足够生成高质量答案（严格判断）
5. failure_reason：若不足，原因（信息缺失 / 查询角度偏差 / 信息源不对 / 超出知识范围）
6. suggested_rewrite：若不足，建议的改写查询

严格返回 JSON（不要 markdown 代码块）：
{{"relevance_score": 8, "sufficiency_score": 7, "is_supported": true, "is_sufficient": true, "failure_reason": "", "suggested_rewrite": ""}}
"""


def generate_prompt(query: str, docs_text: str, reflection_note: str = "",
                    memory_text: str = "") -> str:
    note = f"\n\n（反思层提示：{reflection_note}）" if reflection_note else ""
    memory = f"\n\n以下是历史对话与用户画像，供理解指代和个性化回答：\n{memory_text}" if memory_text else ""
    return f"""你是电商服务平台客服助手。请基于以下检索到的信息回答用户问题。

用户问题：{query}
{memory}

检索到的信息：
{docs_text}
{note}

回答要求：
1. 准确、完整，严格基于给定信息
2. 信息不足时诚实说明，不要编造
3. 适当标注信息来源
4. 历史对话只用于理解上下文（如"这个""那款"指什么），事实仍以检索信息为准

回答：
"""


def graph_query_prompt(query: str, known_entities: str) -> str:
    return f"""你是电商知识图谱查询助手。分析问题，判断图查询类型并提取实体。

图结构（本体）：
- 节点：Product(商品)、Category(类目)、Brand(品牌)、Attribute(属性)、User(用户)、Order(订单)、Review(评价)
- 关系：BELONGS_TO、MADE_BY、HAS_ATTRIBUTE、CO_BOUGHT_WITH、SUBSTITUTE_OF、PURCHASED、PLACED、CONTAINS、WROTE、HAS_REVIEW

已知图中可能出现的实体名：{known_entities}

用户问题：{query}

请识别：
1. query_type：
   - entity_relation：查单个商品的基础信息（价格/品牌/类目/属性）
   - multi_hop：多跳推理（协同购买、替代品等）
   - subgraph：围绕某商品/类目的完整信息
2. entities：问题中出现、且图中可能存在的具体商品/品牌/类目名称列表

严格返回 JSON（不要 markdown 代码块）：
{{"query_type": "subgraph", "entities": ["小米14"]}}
"""


def contextualize_prompt(query: str, history_text: str) -> str:
    return f"""你是对话理解助手。根据最近对话，把用户当前问题改写成一个不依赖上下文、可独立理解的完整问题。

最近对话：
{history_text}

当前问题：{query}

要求：
1. 只补全指代和省略（如"这个""那个""它""再便宜点""那款"），不要回答问题，不要新增事实。
2. 如果当前问题本身已经完整，原样返回。
3. 只输出改写后的问题，不要解释、不要加引号。

改写后的问题："""


def profile_update_prompt(current_profile: str, query: str, answer: str) -> str:
    return f"""你是用户画像抽取助手。根据"当前画像"和"最新对话"，输出更新后的完整用户画像。

当前画像（JSON）：
{current_profile}

最新对话：
用户：{query}
助手：{answer}

要求：
1. 只提取用户明确表达或可稳妥推断的信息，不要编造。
2. 字段固定为：preferences（偏好）、budget（预算）、brands（品牌偏好）、category_interest（类目兴趣）、notes（其他备注）。
3. 保留当前画像中仍然有效的信息，合并新信息；没有信息的字段写空字符串。
4. 严格返回 JSON，不要 markdown 代码块：
{{"preferences": "", "budget": "", "brands": "", "category_interest": "", "notes": ""}}
"""


def session_summary_prompt(turns_text: str) -> str:
    return f"""把以下对话压缩成一段 150 字以内的会话摘要，保留用户关注点、已确认事实和未解决问题。

{turns_text}

摘要："""


def order_query_prompt(query: str) -> str:
    return f"""你是订单查询助手。从用户问题中提取结构化查询参数（用于查询订单/物流/库存/退款）。

用户问题：{query}

请提取（没有就写空字符串）：
- order_id：订单号（如 O20241101001）
- user_id：用户ID（如 U001）
- product_name：商品名称
- query_type：order_status（查状态）/ logistics（查物流）/ inventory（查库存）/ refund（查退款）

严格返回 JSON（不要 markdown 代码块）：
{{"order_id": "", "user_id": "", "product_name": "", "query_type": "logistics"}}
"""
