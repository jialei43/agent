"""
思想一：ReAct（Reasoning + Acting）- 优化版
场景：企业代码审查助手

优化内容：
  1. 统一使用 create_tool_calling_agent（结构化工具调用，更可靠）
  2. 完整思考-执行流程输出（Thought → Action → Observation 循环可见）
  3. 执行过程实时存储到 MySQL，支持结果溯源和审计

流程：
  think 工具 → 检查工具 → think 工具 → 检查工具 → ... → 最终报告
  每一步均打印 Thought / Action / Observation，并实时写入 MySQL
"""

import ast
import os
import re
import json
import textwrap
import uuid
from datetime import datetime
from dotenv import load_dotenv, find_dotenv
from langchain_core.tools import tool
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.callbacks import BaseCallbackHandler
from langchain_openai import ChatOpenAI
from langchain.agents import create_tool_calling_agent, AgentExecutor
import mysql.connector

load_dotenv(find_dotenv())


# ══════════════════════════════════════════════════════════════════════════════
# MySQL 存储层
# ══════════════════════════════════════════════════════════════════════════════

MYSQL_CONFIG = {
    "host": os.getenv("MYSQL_HOST", "localhost"),
    "user": os.getenv("MYSQL_USER", "root"),
    "password": os.getenv("MYSQL_PASSWORD", "123456"),
    "database": os.getenv("MYSQL_DATABASE", "agent"),
}


def _get_conn():
    return mysql.connector.connect(**MYSQL_CONFIG)


def init_db():
    """初始化审计表，幂等操作"""
    conn = _get_conn()
    cur = conn.cursor()
    try:
        cur.execute("""
            CREATE TABLE IF NOT EXISTS review_sessions (
                id           VARCHAR(36)   PRIMARY KEY,
                code_snippet MEDIUMTEXT,
                mode         VARCHAR(50),
                final_report MEDIUMTEXT,
                total_steps  INT           DEFAULT 0,
                created_at   DATETIME      DEFAULT CURRENT_TIMESTAMP,
                completed_at DATETIME
            ) CHARACTER SET utf8mb4
        """)
        cur.execute("""
            CREATE TABLE IF NOT EXISTS review_steps (
                id          INT AUTO_INCREMENT PRIMARY KEY,
                session_id  VARCHAR(36),
                step_order  INT,
                step_type   VARCHAR(20),
                tool_name   VARCHAR(100),
                content     MEDIUMTEXT,
                created_at  DATETIME DEFAULT CURRENT_TIMESTAMP,
                FOREIGN KEY (session_id) REFERENCES review_sessions(id)
            ) CHARACTER SET utf8mb4
        """)
        conn.commit()
        print("[DB] 数据库表初始化完成")
    finally:
        cur.close()
        conn.close()


class AuditLogger:
    """实时将 Agent 每个推理步骤写入 MySQL，支持事后溯源"""

    def __init__(self, session_id: str, code: str, mode: str):
        self.session_id = session_id
        self._step_order = 0
        conn = _get_conn()
        cur = conn.cursor()
        try:
            cur.execute(
                "INSERT INTO review_sessions (id, code_snippet, mode) VALUES (%s, %s, %s)",
                (session_id, code, mode),
            )
            conn.commit()
        finally:
            cur.close()
            conn.close()

    def log_step(self, step_type: str, content: str, tool_name: str = ""):
        """记录单步：thought / action / observation / final"""
        self._step_order += 1
        conn = _get_conn()
        cur = conn.cursor()
        try:
            cur.execute(
                """INSERT INTO review_steps
                   (session_id, step_order, step_type, tool_name, content)
                   VALUES (%s, %s, %s, %s, %s)""",
                (self.session_id, self._step_order, step_type, tool_name, content),
            )
            conn.commit()
        finally:
            cur.close()
            conn.close()

    def complete(self, final_report: str):
        conn = _get_conn()
        cur = conn.cursor()
        try:
            cur.execute(
                """UPDATE review_sessions
                   SET final_report = %s, total_steps = %s, completed_at = NOW()
                   WHERE id = %s""",
                (final_report, self._step_order, self.session_id),
            )
            conn.commit()
        finally:
            cur.close()
            conn.close()


# ══════════════════════════════════════════════════════════════════════════════
# 工具定义（单一职责，描述精确是 ReAct 正确选工具的关键）
# ══════════════════════════════════════════════════════════════════════════════

@tool
def analyze_complexity(code: str) -> str:
    """
    分析代码圈复杂度（Cyclomatic Complexity）。
    返回函数级别的复杂度评分和风险等级。
    适用场景：判断代码是否过于复杂、难以测试和维护。
    """
    lines = code.split("\n")
    func_pattern = re.compile(r"^\s*def\s+(\w+)\s*\(")
    branch_keywords = {"if", "elif", "else", "for", "while", "try", "except", "with", "and", "or"}

    results = []
    current_func = None
    complexity = 1
    func_start = 0

    for i, line in enumerate(lines):
        m = func_pattern.match(line)
        if m:
            if current_func:
                level = "低" if complexity <= 5 else "中" if complexity <= 10 else "高"
                results.append({
                    "function": current_func,
                    "complexity": complexity,
                    "risk": level,
                    "lines": i - func_start,
                })
            current_func = m.group(1)
            complexity = 1
            func_start = i
        elif current_func:
            tokens = set(line.split())
            complexity += len(tokens & branch_keywords)

    if current_func:
        level = "低" if complexity <= 5 else "中" if complexity <= 10 else "高"
        results.append({
            "function": current_func,
            "complexity": complexity,
            "risk": level,
            "lines": len(lines) - func_start,
        })

    if not results:
        return "未检测到函数定义"

    summary = f"共检测到 {len(results)} 个函数：\n"
    for r in results:
        summary += f"  [{r['risk']}] {r['function']}()  复杂度={r['complexity']}  行数={r['lines']}\n"

    high_risk = [r for r in results if r["risk"] == "高"]
    if high_risk:
        summary += f"\n⚠️  高风险函数（复杂度>10）：{[r['function'] for r in high_risk]}，建议拆分重构"
    return summary


@tool
def check_security_issues(code: str) -> str:
    """
    扫描代码中的安全漏洞模式。
    检测：SQL注入风险、硬编码凭证、不安全的反序列化、路径穿越等。
    返回安全问题列表和修复建议。
    """
    issues = []
    patterns = [
        (r'eval\s*\(', "CRITICAL", "eval() 执行任意代码，存在代码注入风险", "改用 ast.literal_eval() 或 json.loads()"),
        (r'exec\s*\(', "CRITICAL", "exec() 执行任意代码，存在代码注入风险", "避免使用 exec()，用具体逻辑替代"),
        (r'(?i)(password|secret|api_key|token)\s*=\s*["\'][^"\']{4,}["\']', "HIGH",
         "疑似硬编码凭证", "改用环境变量或 Secret Manager"),
        (r'pickle\.loads?\s*\(', "HIGH", "pickle 反序列化可执行恶意代码", "改用 json 或加签名校验"),
        (r'subprocess\.(?:call|run|Popen)\s*\([^)]*shell\s*=\s*True', "HIGH",
         "shell=True 存在命令注入风险", "改用列表参数并设置 shell=False"),
        (r'os\.system\s*\(', "MEDIUM", "os.system() 存在命令注入风险", "改用 subprocess.run() 并传列表参数"),
        (r'\.\./', "MEDIUM", "疑似路径穿越字符串", "使用 pathlib.Path 并校验路径合法性"),
        (r'(?i)select.*\+\s*(?:f["\']|str\()', "HIGH",
         "疑似 SQL 拼接，存在注入风险", "改用参数化查询（? 或 %s 占位符）"),
        (r'random\.\w+\(\)', "LOW", "random 模块不适用于密码学场景", "密码学场景改用 secrets 模块"),
        (r'assert\s+', "LOW", "assert 语句在优化模式(-O)下被跳过", "替换为显式 if 校验"),
    ]

    lines = code.split("\n")
    for i, line in enumerate(lines, 1):
        for pattern, severity, desc, fix in patterns:
            if re.search(pattern, line):
                issues.append({
                    "line": i, "severity": severity,
                    "issue": desc, "fix": fix,
                    "code": line.strip()[:80],
                })

    if not issues:
        return "✅ 未发现安全问题"

    result = f"发现 {len(issues)} 个安全问题：\n"
    for issue in sorted(issues, key=lambda x: ["CRITICAL", "HIGH", "MEDIUM", "LOW"].index(x["severity"])):
        result += f"\n  [{issue['severity']}] 第{issue['line']}行\n"
        result += f"    问题：{issue['issue']}\n"
        result += f"    建议：{issue['fix']}\n"
        result += f"    代码：{issue['code']}\n"
    return result


@tool
def check_code_style(code: str) -> str:
    """
    检查代码规范问题（基于 PEP8 和企业编码规范）。
    检测：命名规范、函数长度、注释缺失、魔法数字等。
    返回规范违反列表。
    """
    issues = []
    lines = code.split("\n")
    func_pattern = re.compile(r"^\s*def\s+(\w+)\s*\(")
    class_pattern = re.compile(r"^\s*class\s+(\w+)")
    magic_num_pattern = re.compile(r"\b(?<![\w.])\d{2,}\b(?![\w.])")

    in_func = False
    func_line_count = 0
    func_name = ""
    func_start_line = 0

    for i, line in enumerate(lines, 1):
        stripped = line.rstrip()
        m = func_pattern.match(line)
        if m:
            if in_func and func_line_count > 50:
                issues.append(f"第{func_start_line}行：函数 {func_name}() 长度 {func_line_count} 行，超过50行建议拆分")
            in_func = True
            func_name = m.group(1)
            func_start_line = i
            func_line_count = 0
            if not re.match(r'^[a-z_][a-z0-9_]*$', func_name) and func_name != "__init__":
                issues.append(f"第{i}行：函数名 '{func_name}' 不符合 snake_case 规范")
        elif in_func:
            func_line_count += 1

        mc = class_pattern.match(line)
        if mc:
            class_name = mc.group(1)
            if not re.match(r'^[A-Z][a-zA-Z0-9]*$', class_name):
                issues.append(f"第{i}行：类名 '{class_name}' 不符合 PascalCase 规范")

        if len(stripped) > 120:
            issues.append(f"第{i}行：行长度 {len(stripped)} 字符，超过120字符建议换行")

        if not stripped.lstrip().startswith('#') and magic_num_pattern.search(stripped):
            if 'def ' not in stripped and 'class ' not in stripped:
                issues.append(f"第{i}行：疑似魔法数字，建议提取为命名常量")

        if re.search(r'\b(TODO|FIXME|HACK|XXX)\b', stripped):
            tag = re.search(r'\b(TODO|FIXME|HACK|XXX)\b', stripped).group(1)
            issues.append(f"第{i}行：存在未处理的 {tag} 标记，上线前需解决")

    if in_func and func_line_count > 50:
        issues.append(f"第{func_start_line}行：函数 {func_name}() 长度 {func_line_count} 行，超过50行建议拆分")

    if not issues:
        return "✅ 代码规范检查通过"

    return f"发现 {len(issues)} 个规范问题：\n" + "\n".join(f"  • {issue}" for issue in issues)


@tool
def check_test_coverage_hints(code: str) -> str:
    """
    分析代码中的可测试性问题，给出测试覆盖建议。
    检测：无测试的公共函数、异常处理边界、边界值场景。
    返回建议补充的测试用例列表。
    """
    suggestions = []
    lines = code.split("\n")
    public_funcs = []
    has_exception_handling = False
    has_conditional = False

    for i, line in enumerate(lines, 1):
        m = re.match(r'^\s*def\s+([a-zA-Z][a-zA-Z0-9_]*)\s*\(([^)]*)\)', line)
        if m and not m.group(1).startswith('_'):
            public_funcs.append((i, m.group(1), m.group(2)))
        if re.search(r'\b(try|except|raise)\b', line):
            has_exception_handling = True
        if re.search(r'\b(if|elif)\b', line):
            has_conditional = True

    for line_no, func_name, params in public_funcs:
        param_list = [p.strip().split(':')[0].strip() for p in params.split(',') if p.strip() and p.strip() != 'self']
        suggestions.append(f"函数 {func_name}()（第{line_no}行）：")
        suggestions.append(f"  ✓ 正常路径：传入合法参数 {param_list} 验证返回值")
        if has_conditional:
            suggestions.append(f"  ✓ 边界路径：测试所有条件分支（if/elif）")
        if has_exception_handling:
            suggestions.append(f"  ✓ 异常路径：传入非法参数验证异常抛出是否符合预期")
        if any('str' in p or 'path' in p.lower() for p in param_list):
            suggestions.append(f"  ✓ 边界值：空字符串、超长字符串、特殊字符")
        if any('int' in p or 'num' in p.lower() for p in param_list):
            suggestions.append(f"  ✓ 边界值：0、负数、最大整数")

    if not suggestions:
        return "未检测到需要测试的公共函数"

    return "测试覆盖建议：\n" + "\n".join(suggestions)


# ══════════════════════════════════════════════════════════════════════════════
# think 工具：强制模型在每次工具调用前显式写下推理
# ══════════════════════════════════════════════════════════════════════════════

@tool
def think(reasoning: str) -> str:
    """
    调用任何检查工具前，先调用此工具记录本步的判断依据。
    参数 reasoning：只说明【当前这一步】为什么选下一个工具。
    调用此工具后必须紧接着调用一个实际检查工具，不可连续两次调用 think。
    """
    print(f"\nThought: {reasoning}")
    return "推理已记录。现在必须立即调用一个实际检查工具（analyze_complexity / check_security_issues / check_code_style / check_test_coverage_hints）。"


REVIEW_TOOLS = [analyze_complexity, check_security_issues, check_code_style, check_test_coverage_hints]
ALL_TOOLS = [think] + REVIEW_TOOLS


# ══════════════════════════════════════════════════════════════════════════════
# Callback Handler：Thought/Action/Observation 输出 + MySQL 实时写入
# ══════════════════════════════════════════════════════════════════════════════

class ThinkExecuteCallbackHandler(BaseCallbackHandler):
    """
    拦截 Tool Calling Agent 每步行为：
    - think 工具：工具函数打印 Thought:，此处负责写 DB
    - 检查工具：打印 Action: + Observation:，同时写 DB

    事件顺序：on_tool_start → [工具执行] → on_tool_end → on_agent_finish
    """

    def __init__(self, logger: AuditLogger):
        super().__init__()
        self._logger = logger
        self._current_tool: str | None = None

    def on_tool_start(self, serialized: dict, input_str: str, **kwargs):
        tool_name = serialized.get("name", "unknown")
        self._current_tool = tool_name

        if tool_name == "think":
            # 提取 reasoning 写入 DB（终端由工具函数打印）
            # 优先用 inputs kwarg（新版 LangChain 传原始 dict），
            # 降级用 ast.literal_eval（input_str 是 Python repr，单引号，json.loads 会失败）
            raw_inputs = kwargs.get("inputs")
            if isinstance(raw_inputs, dict):
                reasoning = raw_inputs.get("reasoning", input_str)
            else:
                try:
                    parsed = ast.literal_eval(input_str)
                    reasoning = parsed.get("reasoning", input_str) if isinstance(parsed, dict) else input_str
                except Exception:
                    reasoning = input_str
            self._logger.log_step("thought", reasoning, "think")
        else:
            print(f"\nAction: {tool_name}")
            display = input_str[:300] + ("..." if len(input_str) > 300 else "")
            print(f"Action Input: {display}")
            self._logger.log_step("action", input_str, tool_name)

    def on_tool_end(self, output: str, **_):
        if self._current_tool == "think":
            return  # think 的返回值是给 LLM 的控制提示，不展示给用户
        output_str = str(output)
        display = output_str[:300] + ("..." if len(output_str) > 300 else "")
        print(f"Observation: {display}")
        self._logger.log_step("observation", output_str, self._current_tool or "")

    def on_agent_finish(self, finish, **_):
        report = finish.return_values.get("output", "")
        print("\nThought: 已收集完所有必要信息，生成最终报告")
        self._logger.log_step("final", report)
        self._logger.complete(report)
        print(f"\n[DB] 审计日志已写入 MySQL，session_id: {self._logger.session_id}")


# ══════════════════════════════════════════════════════════════════════════════
# Agent 构建（统一使用 create_tool_calling_agent）
# ══════════════════════════════════════════════════════════════════════════════

AGENT_PROMPT = ChatPromptTemplate.from_messages([
    ("system", """你是一个专业的企业级代码审查助手。

【强制执行，每一轮必须严格遵守，违反则本次审查无效】：
第一步：必须先调用 think 工具，写下你选择下一个检查工具的理由
第二步：紧接着调用一个检查工具（analyze_complexity / check_security_issues / check_code_style / check_test_coverage_hints）
第三步：重复"think → 检查工具"，直到完成所有必要检查

绝对禁止：
- 连续两次 think
- think 后直接输出结论而不调用检查工具
- 不调用 think 直接调用检查工具
- 一次调用多个工具（每轮只允许一个工具调用）

根据代码特征按需选用检查工具，不必强制全部调用。"""),
    ("human", "{input}"),
    MessagesPlaceholder("agent_scratchpad"),
])


def build_review_agent(session_id: str, code: str) -> tuple[AgentExecutor, AuditLogger]:
    """
    构建审查 Agent，返回 (executor, logger)。

    parallel_tool_calls=False：禁止并行工具调用，确保 think → 检查工具 严格交替。
    """
    llm = ChatOpenAI(
        model="qwen-plus",
        api_key=os.getenv("DASHSCOPE_API_KEY"),
        base_url="https://dashscope.aliyuncs.com/compatible-mode/v1",
        temperature=0,
        model_kwargs={"parallel_tool_calls": False},
    )
    logger = AuditLogger(session_id, code, "tool_calling+think+mysql")
    agent = create_tool_calling_agent(llm, ALL_TOOLS, AGENT_PROMPT)
    executor = AgentExecutor(
        agent=agent,
        tools=ALL_TOOLS,
        verbose=False,
        max_iterations=12,
        max_execution_time=180,
        return_intermediate_steps=True,
    )
    return executor, logger


# ══════════════════════════════════════════════════════════════════════════════
# 演示入口
# ══════════════════════════════════════════════════════════════════════════════

SAMPLE_CODE = textwrap.dedent("""
    import pickle
    import os

    SECRET_KEY = "hardcoded_secret_123"

    class userManager:  # 命名不规范
        def process_user_data(self, user_input, db_conn, cache, logger, config, retry=3):
            # TODO: 添加更多校验
            query = "SELECT * FROM users WHERE id = " + str(user_input)
            if user_input:
                if isinstance(user_input, int):
                    if user_input > 0:
                        if user_input < 999999:
                            result = db_conn.execute(query)
                            if result:
                                data = pickle.loads(result.fetchone()[0])
                                if data:
                                    processed = eval(data.get('transform', '{}'))
                                    if processed:
                                        for key in processed:
                                            if key in cache:
                                                cache[key] = processed[key] * 1.15
                                                if cache[key] > 10000:
                                                    os.system(f"notify {cache[key]}")
                                                    logger.info(f"Updated {key}")
                            return result
            return None

    def calculate_discount(price, user_type):
        if user_type == "vip":
            return price * 0.8
        elif user_type == "member":
            return price * 0.9
        else:
            return price
""")


def main():
    # 初始化 MySQL 表
    init_db()

    session_id = str(uuid.uuid4())
    executor, logger = build_review_agent(session_id, SAMPLE_CODE)
    handler = ThinkExecuteCallbackHandler(logger)

    print("=" * 60)
    print("  代码审查助手  [Tool Calling + Think Tool + MySQL 审计]")
    print("=" * 60)
    print(f"\n[会话ID] {session_id}")
    print("\n待审查代码：")
    print(SAMPLE_CODE)
    print("\n" + "=" * 60)
    print("开始审查（Thought → Action → Observation 实时输出）...")
    print("=" * 60 + "\n")

    result = executor.invoke(
        {"input": f"请对以下代码进行全面的企业级代码审查：\n```python\n{SAMPLE_CODE}\n```"},
        config={"callbacks": [handler]},
    )

    print("\n" + "=" * 60)
    print("最终审查报告：")
    print("=" * 60)
    print(result["output"])
    print(f"\n共执行了 {len(result['intermediate_steps'])} 次工具调用")
    print(f"\n溯源查询：SELECT * FROM review_steps WHERE session_id = '{session_id}' ORDER BY step_order;")


if __name__ == "__main__":
    main()
