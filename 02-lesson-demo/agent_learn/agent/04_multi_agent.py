"""
思想四：Multi-Agent（多智能体协作）
场景：企业智能客服路由系统

Multi-Agent 架构：
  Supervisor：意图分类 + 路由决策
  TechAgent：技术问题专家（有工单系统工具）
  BillingAgent：账单问题专家（有财务系统工具）
  PolicyAgent：政策咨询专家（有知识库检索工具）
  Escalator：无法解决时上报人工

企业价值：
  - 专业化：每个 Agent 只处理自己擅长的领域，降低幻觉
  - 可扩展：新增业务线只需新增 Agent，Supervisor 路由规则微调
  - 可追溯：每个 Agent 的输入输出独立记录，便于质检和改进
  - 成本优化：简单问题用小模型，复杂问题升级大模型
"""

import os
from dataclasses import dataclass
from typing import Optional
from dotenv import load_dotenv, find_dotenv
from pydantic import BaseModel, Field
from langchain_core.tools import tool
from langchain_core.messages import HumanMessage, SystemMessage, AIMessage
from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI
from langchain.agents import create_tool_calling_agent, AgentExecutor

load_dotenv(find_dotenv())


# ══════════════════════════════════════════════════════════════════════════════
# Supervisor 路由模型
# ══════════════════════════════════════════════════════════════════════════════

class RoutingDecision(BaseModel):
    """Supervisor 的路由决策"""
    intent: str = Field(description="识别的用户意图（简短描述）")
    agent: str = Field(
        description="路由到的 Agent 名称",
        pattern="^(tech|billing|policy|escalate)$"
    )
    confidence: float = Field(description="路由置信度 0-1", ge=0, le=1)
    extracted_context: str = Field(description="从用户消息中提取的关键上下文，传给目标 Agent")
    reason: str = Field(description="路由原因（一句话）")


# ══════════════════════════════════════════════════════════════════════════════
# 各 Agent 专属工具
# ══════════════════════════════════════════════════════════════════════════════

# ── TechAgent 工具 ─────────────────────────────────────────────────────────

@tool
def query_system_status(service_name: str) -> str:
    """
    查询指定服务的当前运行状态。
    输入服务名称，返回健康状态、响应时间、错误率。
    """
    # 模拟系统状态数据
    status_db = {
        "api": {"status": "正常", "response_time_ms": 45, "error_rate": "0.02%", "uptime": "99.98%"},
        "database": {"status": "正常", "response_time_ms": 12, "error_rate": "0%", "uptime": "99.99%"},
        "payment": {"status": "降级", "response_time_ms": 850, "error_rate": "2.1%", "uptime": "98.5%",
                    "note": "支付服务响应偏慢，技术团队正在处理"},
        "auth": {"status": "正常", "response_time_ms": 28, "error_rate": "0.01%", "uptime": "99.97%"},
        "storage": {"status": "维护中", "response_time_ms": None, "error_rate": None,
                    "note": "计划维护窗口：02:00-04:00，预计 03:20 恢复"},
    }
    key = service_name.lower().replace("-", "").replace("_", "").replace("服务", "").replace("system", "")
    for k, v in status_db.items():
        if k in key or key in k:
            return f"服务状态查询结果：\n" + "\n".join(f"  {kk}: {vv}" for kk, vv in v.items())
    return f"未找到服务 '{service_name}'，可查询：api/database/payment/auth/storage"


@tool
def create_tech_ticket(problem_description: str, severity: str = "medium") -> str:
    """
    创建技术支持工单。
    severity 可选：critical/high/medium/low
    返回工单号和预计响应时间。
    """
    import random, datetime
    ticket_id = f"TK-{random.randint(100000, 999999)}"
    response_times = {"critical": "15分钟内", "high": "1小时内", "medium": "4小时内", "low": "24小时内"}
    response_time = response_times.get(severity.lower(), "4小时内")

    return (f"工单已创建：\n"
            f"  工单号：{ticket_id}\n"
            f"  严重程度：{severity}\n"
            f"  问题描述：{problem_description[:100]}\n"
            f"  预计响应：{response_time}\n"
            f"  创建时间：{datetime.datetime.now().strftime('%Y-%m-%d %H:%M')}\n"
            f"  您将收到邮件确认，技术团队将在{response_time}与您联系")


@tool
def search_tech_knowledge_base(query: str) -> str:
    """
    搜索技术知识库，获取常见问题解决方案。
    输入问题描述，返回匹配的解决方案。
    """
    kb = {
        "登录": "登录问题排查步骤：\n1. 确认账号密码是否正确\n2. 清除浏览器缓存和Cookie\n3. 尝试无痕模式\n4. 检查邮箱是否收到异常登录提醒\n5. 尝试重置密码",
        "速度慢": "性能问题排查：\n1. 检查当前网络连接\n2. 清除浏览器缓存\n3. 尝试切换不同的服务区域\n4. 查看 status.example.com 确认是否有服务异常\n5. 提供具体慢的页面和操作，便于技术排查",
        "崩溃": "崩溃/闪退处理：\n1. 记录崩溃时间和操作步骤\n2. 截图错误信息\n3. 尝试清理缓存重启\n4. 如问题持续，请提交崩溃日志（帮助→反馈→附加日志）",
        "集成": "API集成帮助：\n1. 参考开发者文档：docs.example.com\n2. 确认使用正确的环境（测试/生产）\n3. 检查API密钥权限\n4. 联系技术支持获取集成协助",
    }

    for keyword, solution in kb.items():
        if keyword in query:
            return f"找到相关解决方案：\n{solution}"
    return f"知识库中未找到 '{query}' 的直接解决方案，建议创建技术工单由专家处理"


# ── BillingAgent 工具 ──────────────────────────────────────────────────────

@tool
def query_invoice(order_id: str) -> str:
    """
    查询指定订单的发票信息。
    输入订单号，返回发票状态和下载链接。
    """
    import random
    if not order_id.strip():
        return "请提供有效的订单号"

    # 模拟发票查询
    status = random.choice(["已开具", "处理中", "未开具"])
    if status == "已开具":
        return (f"订单 {order_id} 发票信息：\n"
                f"  状态：{status}\n"
                f"  发票号：INV-{random.randint(10000, 99999)}\n"
                f"  金额：¥{random.randint(100, 5000)}.00\n"
                f"  下载：invoice.example.com/{order_id}\n"
                f"  有效期至：2025-12-31")
    elif status == "处理中":
        return f"订单 {order_id} 发票正在处理中，预计1-3个工作日内开具完成，届时将发送到您的注册邮箱"
    else:
        return f"订单 {order_id} 尚未申请发票，请登录控制台→账单→发票管理中申请"


@tool
def process_refund_request(order_id: str, reason: str) -> str:
    """
    提交退款申请。
    输入订单号和退款原因，返回退款申请结果。
    """
    if not order_id.strip():
        return "退款申请失败：请提供有效的订单号"

    import random, datetime
    refund_id = f"RF-{random.randint(100000, 999999)}"
    return (f"退款申请已提交：\n"
            f"  退款申请号：{refund_id}\n"
            f"  关联订单：{order_id}\n"
            f"  退款原因：{reason[:80]}\n"
            f"  预计到账：3-7个工作日\n"
            f"  处理状态：审核中\n"
            f"  提交时间：{datetime.datetime.now().strftime('%Y-%m-%d %H:%M')}\n"
            f"  退款将返回至原付款方式，您将收到短信和邮件通知")


@tool
def query_billing_detail(month: str) -> str:
    """
    查询指定月份的账单明细。
    月份格式：YYYY-MM，如 2024-03
    """
    import random
    if not month.strip():
        return "请提供有效的月份（格式：YYYY-MM）"

    items = [
        ("基础套餐费用", f"¥{random.randint(200, 500)}.00"),
        ("API调用超额", f"¥{random.randint(0, 200)}.00"),
        ("存储费用", f"¥{random.randint(10, 100)}.00"),
        ("优惠折扣", f"-¥{random.randint(10, 50)}.00"),
    ]
    total = sum(int(v[1].replace("¥","").replace(".00","").replace("-","")) * (-1 if v[1].startswith("-") else 1)
                for v in items)
    detail = "\n".join(f"  {name}: {amount}" for name, amount in items)
    return f"{month} 账单明细：\n{detail}\n  总计：¥{total}.00\n\n完整账单下载：billing.example.com/download/{month}"


# ── PolicyAgent 工具 ───────────────────────────────────────────────────────

@tool
def search_policy_docs(topic: str) -> str:
    """
    检索政策和条款文档。
    输入查询主题（如：数据隐私、服务协议、退款政策），返回相关条款。
    """
    policies = {
        "退款": """退款政策（节选）：
• 服务未使用部分可申请退款，扣除10%手续费
• 年付套餐在购买30天内可全额退款
• 因平台故障造成的损失，按SLA赔偿标准处理
• 退款处理周期：3-7个工作日
• 详细条款：legal.example.com/refund-policy""",
        "隐私": """数据隐私政策（节选）：
• 用户数据仅用于提供服务，不向第三方出售
• 用户有权申请导出或删除个人数据
• 数据存储在中国大陆，符合《数据安全法》要求
• 数据保留期限：账户注销后90天删除
• 详细政策：legal.example.com/privacy""",
        "SLA": """服务级别协议（节선）：
• 可用性承诺：月度99.9%（约8.7小时/年故障容忍）
• 赔偿标准：
  - 可用性 99.0-99.9%：赔偿10%月费
  - 可用性 95.0-99.0%：赔偿25%月费
  - 可用性 < 95%：赔偿50%月费
• 赔偿形式：账户抵扣券（不退现金）
• 申请窗口：故障结束后30天内
• 详细条款：legal.example.com/sla""",
        "账号": """账号与安全政策（节选）：
• 账号不可转让，禁止共享给第三方使用
• 密码修改后其他设备会话将在24小时内失效
• 发现异常登录后，立即冻结并发送安全邮件
• 账号注销需提前15天申请，数据不可恢复
• 详细条款：legal.example.com/account-policy""",
    }

    for key, policy in policies.items():
        if key in topic:
            return policy
    return f"未找到关于 '{topic}' 的具体政策，请访问 legal.example.com 查看完整条款，或联系客服获取帮助"


# ══════════════════════════════════════════════════════════════════════════════
# 专家 Agent 构建器
# ══════════════════════════════════════════════════════════════════════════════

def _make_llm(temperature: float = 0) -> ChatOpenAI:
    return ChatOpenAI(
        model="qwen-plus",
        api_key=os.getenv("DASHSCOPE_API_KEY"),
        base_url="https://dashscope.aliyuncs.com/compatible-mode/v1",
        temperature=temperature,
    )


def _build_specialist_agent(
    system_prompt: str,
    tools: list,
    agent_name: str,
) -> AgentExecutor:
    """通用专家 Agent 构建器"""
    prompt = ChatPromptTemplate.from_messages([
        ("system", system_prompt),
        ("human", "{input}"),
        ("placeholder", "{agent_scratchpad}"),
    ])
    return AgentExecutor(
        agent=create_tool_calling_agent(_make_llm(), tools, prompt),
        tools=tools,
        verbose=False,
        max_iterations=5,
        handle_parsing_errors=True,
    )


TECH_AGENT = _build_specialist_agent(
    system_prompt="""你是一名专业的技术支持工程师，擅长解决软件故障、系统异常、API集成问题。

工作原则：
1. 先查系统状态，确认是否为已知故障
2. 提供具体的排查步骤，而非泛泛的建议
3. 如无法自助解决，创建优先级合适的工单
4. 回答简洁专业，避免不必要的道歉和寒暄""",
    tools=[query_system_status, create_tech_ticket, search_tech_knowledge_base],
    agent_name="TechAgent",
)

BILLING_AGENT = _build_specialist_agent(
    system_prompt="""你是一名专业的账单服务专员，处理发票、退款、账单查询等财务相关问题。

工作原则：
1. 先确认用户的订单号，再查询具体信息
2. 退款申请需获取明确的退款原因
3. 所有操作结果给出清晰的后续预期（如：几天到账）
4. 超出权限的操作（如大额退款）说明需人工审核""",
    tools=[query_invoice, process_refund_request, query_billing_detail],
    agent_name="BillingAgent",
)

POLICY_AGENT = _build_specialist_agent(
    system_prompt="""你是一名政策法规顾问，专门解答服务条款、数据政策、SLA等合规问题。

工作原则：
1. 援引具体条款，不做模糊承诺
2. 区分"政策规定"和"技术实现"，后者转给技术支持
3. 涉及法律争议时，建议用户联系法务或监管机构
4. 提供官方文档链接，让用户可以自行核实""",
    tools=[search_policy_docs],
    agent_name="PolicyAgent",
)


# ══════════════════════════════════════════════════════════════════════════════
# Supervisor
# ══════════════════════════════════════════════════════════════════════════════

class Supervisor:
    """
    多 Agent 系统的路由协调者。
    负责意图识别、路由决策、结果后处理。
    """

    # ── [旧] 规则引导版：枚举关键词→Agent 的映射规则 ──
    # 缺点：关键词列表需要手动维护，新增 Agent 时需同步更新规则；
    #       遇到关键词不典型的问题（如"我的操作一直转圈"）可能路由错误
    # ROUTING_RULES = """
    # 路由规则：
    # - tech：软件故障、系统错误、API/SDK问题、功能异常、性能问题
    # - billing：发票、退款、账单查询、付款问题、定价疑问
    # - policy：服务条款、隐私政策、SLA/赔偿、合规要求、数据权限
    # - escalate：涉及法律纠纷、多次无法解决、客户明确要求人工、情绪激烈投诉
    # """

    # ── [新] 动态推理版：描述各 Agent 的工具能力，让 LLM 推断用户需求与能力的最佳匹配 ──
    # 核心改变：从"关键词规则"改为"能力描述"，LLM 通过理解用户意图与 Agent 能力的匹配关系做路由；
    # 新增 Agent 只需在此追加能力描述，无需维护关键词列表
    AGENT_CAPABILITIES = """
可用专家及其核心能力：

tech（技术支持工程师）：
  具备工具：查询服务运行状态 / 创建技术支持工单 / 搜索技术知识库
  能解决：系统状态异常排查、软件故障诊断、API/SDK 集成问题、技术操作指导

billing（账单服务专员）：
  具备工具：查询发票 / 处理退款申请 / 查询账单明细
  能解决：财务相关查询和操作、订单支付事项、费用异议

policy（政策法规顾问）：
  具备工具：检索政策和条款文档
  能解决：服务条款解读、数据隐私政策、SLA 赔偿标准、合规性问题

escalate（人工升级）：
  适用于：自动化专家无法解决、需要特殊授权处理、或客户明确需要人工介入的情况
"""

    def __init__(self):
        self.llm = _make_llm(temperature=0).with_structured_output(RoutingDecision)
        self.agents = {
            "tech": TECH_AGENT,
            "billing": BILLING_AGENT,
            "policy": POLICY_AGENT,
        }
        self.conversation_history: list[dict] = []

    def route(self, user_message: str) -> RoutingDecision:
        """分析用户意图并决定路由目标"""
        messages = [
            SystemMessage(content=f"""你是一个智能客服路由系统，负责将用户问题分配给最合适的专家处理。

{self.AGENT_CAPABILITIES}

根据用户表达的实际需求，判断哪位专家最有能力解决，并给出路由理由。

对话历史（如有）：
{self._format_history()}"""),
            HumanMessage(content=f"用户消息：{user_message}"),
        ]
        return self.llm.invoke(messages)

    def dispatch(self, decision: RoutingDecision, user_message: str) -> str:
        """将用户请求派发给对应的专家 Agent"""
        if decision.agent == "escalate":
            return self._escalate(user_message, decision.reason)

        agent = self.agents.get(decision.agent)
        if not agent:
            return f"系统错误：未知路由目标 '{decision.agent}'"

        # 注入路由上下文（让专家 Agent 了解路由原因）
        enriched_input = f"""用户问题：{user_message}

路由上下文：{decision.extracted_context}
"""
        result = agent.invoke({"input": enriched_input})
        return result["output"]

    def _escalate(self, user_message: str, reason: str) -> str:
        """上报人工处理"""
        import random
        ticket_id = f"HMN-{random.randint(10000, 99999)}"
        return (f"您的问题已升级至人工专员处理。\n\n"
                f"工单号：{ticket_id}\n"
                f"预计等待：5-10分钟\n"
                f"您也可以通过以下方式联系我们：\n"
                f"  • 电话：400-888-8888（工作日9:00-18:00）\n"
                f"  • 邮件：support@example.com\n\n"
                f"感谢您的耐心等待。")

    def _format_history(self) -> str:
        if not self.conversation_history:
            return "（无历史对话）"
        return "\n".join(
            f"  用户: {h['user']}\n  客服: {h['response'][:100]}..."
            for h in self.conversation_history[-3:]  # 只取最近3轮
        )

    def chat(self, user_message: str) -> str:
        """处理一条用户消息的完整链路"""
        print(f"\n{'─'*50}")
        print(f"用户：{user_message}")

        # Step 1: 路由决策
        decision = self.route(user_message)
        print(f"[Supervisor] 意图：{decision.intent}")
        print(f"[Supervisor] 路由 → {decision.agent.upper()}（置信度：{decision.confidence:.0%}）")
        print(f"[Supervisor] 原因：{decision.reason}")

        # Step 2: 专家处理
        print(f"[{decision.agent.upper()}] 处理中...")
        response = self.dispatch(decision, user_message)

        # Step 3: 记录对话历史（供后续路由参考）
        self.conversation_history.append({"user": user_message, "response": response})

        print(f"\n客服回复：")
        print(response)
        return response


# ══════════════════════════════════════════════════════════════════════════════
# 演示入口
# ══════════════════════════════════════════════════════════════════════════════

DEMO_QUERIES = [
    "我的API调用一直返回503错误，已经持续了半小时了",
    "我需要上个月的发票，订单号是ORD-2024031501",
    "你们的SLA是多少？如果服务中断我能得到什么赔偿？",
    "支付服务怎么了，我下单一直支付失败",
    "我要退款，订单号 ORD-2024030801，产品根本没有宣传的那些功能",
]


def main():
    print("=" * 60)
    print("  Multi-Agent 系统：企业智能客服路由")
    print("  架构：Supervisor → Tech/Billing/Policy Agent")
    print("=" * 60)

    supervisor = Supervisor()

    for query in DEMO_QUERIES:
        supervisor.chat(query)

    print("\n" + "=" * 60)
    print(f"演示完成，共处理 {len(DEMO_QUERIES)} 个用户请求")
    print("=" * 60)


if __name__ == "__main__":
    main()
