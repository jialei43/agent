"""
思想一：ReAct（Reasoning + Acting）
场景：企业代码审查助手

ReAct 核心：Thought → Action → Observation 循环
每一步推理都可见，工具按需调用，直到问题解决。

企业价值：
  - 可解释性：每步 Thought 记录审查推理链，可审计
  - 动态调度：根据代码特征动态选择检查维度（复杂度/安全/规范）
  - 结构化输出：最终生成标准化审查报告，可直接入 CI/CD 流水线
"""

import os
import re
import json
import uuid
import sqlite3
import asyncio
import textwrap
from dotenv import load_dotenv, find_dotenv
from langchain_core.tools import tool
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.callbacks import BaseCallbackHandler, StdOutCallbackHandler
from langchain_openai import ChatOpenAI
from langchain.agents import create_react_agent, create_tool_calling_agent, AgentExecutor

load_dotenv(find_dotenv())


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
                    "line": i,
                    "severity": severity,
                    "issue": desc,
                    "fix": fix,
                    "code": line.strip()[:80],
                })

    if not issues:
        return "✅ 未发现安全问题"

    result = f"发现 {len(issues)} 个安全问题：\n"
    for issue in sorted(issues, key=lambda x: ["CRITICAL","HIGH","MEDIUM","LOW"].index(x["severity"])):
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
    has_docstring = False

    for i, line in enumerate(lines, 1):
        stripped = line.rstrip()

        # 函数长度检查
        m = func_pattern.match(line)
        if m:
            if in_func and func_line_count > 50:
                issues.append(f"第{func_start_line}行：函数 {func_name}() 长度 {func_line_count} 行，超过50行建议拆分")
            in_func = True
            func_name = m.group(1)
            func_start_line = i
            func_line_count = 0
            has_docstring = False

            # 命名规范：函数应为 snake_case
            if not re.match(r'^[a-z_][a-z0-9_]*$', func_name) and func_name != "__init__":
                issues.append(f"第{i}行：函数名 '{func_name}' 不符合 snake_case 规范")

        elif in_func:
            func_line_count += 1
            if func_line_count == 1 and '"""' in line:
                has_docstring = True

        # 类命名规范：应为 PascalCase
        mc = class_pattern.match(line)
        if mc:
            class_name = mc.group(1)
            if not re.match(r'^[A-Z][a-zA-Z0-9]*$', class_name):
                issues.append(f"第{i}行：类名 '{class_name}' 不符合 PascalCase 规范")

        # 行长度检查
        if len(stripped) > 120:
            issues.append(f"第{i}行：行长度 {len(stripped)} 字符，超过120字符建议换行")

        # 魔法数字检查（跳过注释行和文档字符串）
        if not stripped.lstrip().startswith('#') and magic_num_pattern.search(stripped):
            if 'def ' not in stripped and 'class ' not in stripped:
                issues.append(f"第{i}行：疑似魔法数字，建议提取为命名常量")

        # TODO/FIXME 追踪
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
        # 找公共函数（不以 _ 开头）
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
# Agent 构建（两种方式对比）
# ══════════════════════════════════════════════════════════════════════════════

REVIEW_TOOLS = [analyze_complexity, check_security_issues, check_code_style, check_test_coverage_hints]


# ══════════════════════════════════════════════════════════════════════════════
# 让 Tool Calling 推理过程可见的两种方式
# ══════════════════════════════════════════════════════════════════════════════

# ─────────────────────────────────────────────────────────────────────────────
# 可见推理方式一：自定义 Callback Handler（推荐，无侵入）
# ─────────────────────────────────────────────────────────────────────────────
#
# 原理：LangChain 在 Agent 每次调用工具前后都会触发回调事件，
#       通过实现 BaseCallbackHandler 的钩子方法，可以拦截并打印推理过程。
# 优点：完全不改变 Agent 逻辑，生产环境可接入日志系统/监控平台（如 LangSmith）

class ReasoningCallbackHandler(BaseCallbackHandler):
    """拦截 Tool Calling Agent 的每一步行为，将推理过程打印出来。

    react_format=True 时输出与 build_react_agent(verbose=True) 完全一致的
    Thought/Action/Action Input/Observation 格式，方便两种 Agent 对比。
    """

    # ── 为什么用 on_tool_start 而不是 on_agent_action ──────────────────────────
    # create_tool_calling_agent 返回的是 LCEL Runnable，不是老版 Agent 类。
    # 在 LCEL 执行链路中，on_agent_action 不一定可靠触发；
    # on_tool_start 由工具自身的 Runnable 发出，无论哪种 Agent 类型都会触发。
    #
    # 事件顺序：on_tool_start → 工具执行 → on_tool_end → on_agent_finish

    def __init__(self, react_format: bool = False):
        super().__init__()
        self._react_format = react_format
        self._current_tool: str | None = None  # 追踪当前工具名，用于 on_tool_end 判断

    def on_tool_start(self, serialized: dict, input_str: str, **_):
        tool_name = serialized.get("name", "unknown")
        self._current_tool = tool_name
        if tool_name == "think":
            return  # think 工具由其自身函数体打印 Thought:，此处跳过
        if self._react_format:
            print(f"\nAction: {tool_name}")
            print(f"Action Input: {input_str[:200]}{'...' if len(input_str) > 200 else ''}")
        else:
            print(f"\n  ┌─[推理] 调用工具: {tool_name}")
            print(f"  │  输入: {input_str[:120]}{'...' if len(input_str) > 120 else ''}")

    def on_tool_end(self, output: str, **_):
        if self._current_tool == "think":
            return  # think 工具的返回值只是提示语，不需要打印
        output_str = str(output)
        if self._react_format:
            print(f"Observation: {output_str[:200]}{'...' if len(output_str) > 200 else ''}")
        else:
            print(f"  └─[观察] {output_str[:200]}{'...' if len(output_str) > 200 else ''}")

    def on_agent_finish(self, _finish, **_kw):
        if self._react_format:
            print("\nThought: 已收集完所有必要信息，生成最终报告")
        else:
            print(f"\n  [完成] 推理结束，生成最终报告")


# ─────────────────────────────────────────────────────────────────────────────
# 可见推理方式二：think 工具（强制模型在每次调用前显式记录推理）
# ─────────────────────────────────────────────────────────────────────────────
#
# 原理：注册一个特殊的 think 工具，tool description 要求模型在调用任何实际工具前
#       先调用 think() 写下判断依据。think 工具本身什么都不做，只把推理内容打印出来。
# 优点：推理内容结构化、可存储到数据库，形成可审计的决策日志
# 缺点：增加一次额外的工具调用（多一次 LLM → 工具交互），有轻微延迟

@tool
def think(reasoning: str) -> str:
    """
    调用任何检查工具前，先调用此工具记录本步的判断依据。
    参数 reasoning：只说明【当前这一步】为什么选下一个工具，不要一次列出所有计划。
    调用此工具后必须紧接着调用一个实际检查工具，不可连续两次调用 think，也不可在 think 后直接结束。
    """
    print(f"\nThought: {reasoning}")
    return "推理已记录。现在必须立即调用一个实际检查工具（analyze_complexity / check_security_issues / check_code_style / check_test_coverage_hints）。"


REVIEW_TOOLS_WITH_THINK = [think] + REVIEW_TOOLS


# ─────────────────────────────────────────────────────────────────────────────
# 方式一：[旧] ReAct —— 工具列表写在 Prompt 里，LLM 输出文本格式推理链
# ─────────────────────────────────────────────────────────────────────────────
#
# 原理：
#   1. {tools} 占位符把所有工具的名称+描述拼成一大段文本注入 prompt
#   2. LLM 输出 "Thought/Action/Action Input/Observation" 格式的纯文本
#   3. 框架用正则表达式从文本中解析出工具名和参数，再去调用工具
#
# 缺点：
#   - 工具描述占用大量 token（工具越多 prompt 越长）
#   - 依赖文本格式解析，格式错误时需要 handle_parsing_errors=True 重试
#   - 模型必须"记住"并精确输出特定格式，较小模型容易格式混乱
#
# 适用：模型不支持 function calling 时的唯一选择（如早期开源模型）

# [旧-更早] 规则引导版：强制必须使用所有工具、硬编码报告格式
# REACT_PROMPT_OLD = ChatPromptTemplate.from_template("""...
# 审查要求：
# 1. 必须使用所有四个工具进行检查（复杂度、安全、规范、测试覆盖）
# ...
# """)

# [旧] 动态 ReAct 版：工具仍在 prompt 里，但不强制调用所有工具
REACT_PROMPT = ChatPromptTemplate.from_template("""你是一个专业的企业级代码审查助手。
你必须通过调用工具来收集信息，禁止在调用工具之前直接输出 Final Answer。

可用工具：
{tools}

工具名称：{tool_names}

严格按照以下格式循环执行，每一行格式不能省略：
Thought: 分析当前情况，说明为什么选下一个工具
Action: 工具名称（只写名称，不加其他内容）
Action Input: 传给工具的完整代码或参数
Observation: （此行由系统填入工具返回结果，你不需要填写）

重复上述循环，直到掌握足够信息，然后输出：
Thought: 已收集完所有必要信息
Final Answer: 完整的审查报告

开始：
{input}

{agent_scratchpad}""")


def build_react_agent(verbose: bool = True) -> AgentExecutor:
    llm = ChatOpenAI(
        model="qwen-plus",
        api_key=os.getenv("DASHSCOPE_API_KEY"),
        base_url="https://dashscope.aliyuncs.com/compatible-mode/v1",
        temperature=0,
    )
    agent = create_react_agent(llm, REVIEW_TOOLS, REACT_PROMPT)
    return AgentExecutor(
        agent=agent,
        tools=REVIEW_TOOLS,
        verbose=verbose,
        max_iterations=10,
        max_execution_time=120,
        handle_parsing_errors=True,
        return_intermediate_steps=True,
        callbacks=[StdOutCallbackHandler()],  # 显式挂载，保证 Thought/Action/Observation 一定输出
    )


# ─────────────────────────────────────────────────────────────────────────────
# 方式二：[新] Tool Calling —— 工具通过 API 绑定给模型，Prompt 里不再列举工具
# ─────────────────────────────────────────────────────────────────────────────
#
# 原理：
#   1. create_tool_calling_agent 内部调用 llm.bind_tools(tools)，
#      把工具的 JSON Schema 通过 API 的 tools 参数直接传给模型
#   2. 模型原生返回结构化的工具调用请求（JSON），无需文本解析
#   3. Prompt 里不再需要 {tools}/{tool_names} 占位符，模型通过 API 感知工具
#
# 优点：
#   - Prompt 更简洁，不占用工具描述 token
#   - 工具调用可靠：结构化 JSON，不会因格式混乱解析失败
#   - 模型能更精确地构造工具参数（有 Schema 约束）
#   - 支持并行工具调用（一次 LLM 调用同时触发多个工具）
#
# 适用：所有支持 function calling 的现代模型（GPT-4/Claude/Qwen-plus 等）

TOOL_CALLING_PROMPT = ChatPromptTemplate.from_messages([
    ("system", """你是一个专业的企业级代码审查助手。
根据提交代码的实际特征，自主选择需要调用的检查工具，给出有针对性的专业审查意见。
重点突出实际发现的问题，无问题的维度简略带过，不必强制调用所有工具。"""),
    ("human", "{input}"),
    MessagesPlaceholder("agent_scratchpad"),   # Tool Calling 的中间步骤占位符
])
# 注意：Prompt 里没有 {tools} 和 {tool_names}，工具通过 llm.bind_tools() 在 API 层传递


def build_tool_calling_agent(
    verbose: bool = False,
    reasoning_mode: str = "callback",   # "callback" | "think_tool" | "none"
) -> AgentExecutor:
    """
    reasoning_mode 控制推理过程的可见方式：
      "callback"   — 通过 Callback Handler 拦截每步行为（无侵入，推荐）
      "think_tool" — 注入 think 工具，强制模型在每次调用前显式写下推理
      "none"       — 不显示推理过程（verbose=True 时仍显示 LangChain 内置日志）
    """
    if reasoning_mode == "think_tool":
        # parallel_tool_calls=False：禁止模型一次并行调用多个工具。
        # 只有禁止并行，模型每轮才只调用一个工具，think → 检查工具 的交替模式才能成立。
        llm = ChatOpenAI(
            model="qwen-plus",
            api_key=os.getenv("DASHSCOPE_API_KEY"),
            base_url="https://dashscope.aliyuncs.com/compatible-mode/v1",
            temperature=0,
            model_kwargs={"parallel_tool_calls": False},
        )
        tools = REVIEW_TOOLS_WITH_THINK
        prompt = ChatPromptTemplate.from_messages([
            ("system", """你是一个专业的企业级代码审查助手。
你必须通过调用工具来收集信息，禁止在调用工具之前直接输出 Final Answer。


严格按照以下格式循环执行，每一行格式不能省略：
Thought: 分析当前情况，说明为什么选下一个工具
Action: 工具名称（只写名称，不加其他内容）
Action Input: 传给工具的完整代码或参数
Observation: （此行由系统填入工具返回结果，你不需要填写）

重复上述循环，直到掌握足够信息，然后输出：
Thought: 已收集完所有必要信息
Final Answer: 完整的审查报告

开始：
{input}

{agent_scratchpad}。"""),
            ("human", "{input}"),
            MessagesPlaceholder("agent_scratchpad"),
        ])
        callbacks = []
    else:
        llm = ChatOpenAI(
            model="qwen-plus",
            api_key=os.getenv("DASHSCOPE_API_KEY"),
            base_url="https://dashscope.aliyuncs.com/compatible-mode/v1",
            temperature=0,
        )
        tools = REVIEW_TOOLS
        prompt = TOOL_CALLING_PROMPT
        callbacks = [ReasoningCallbackHandler()] if reasoning_mode == "callback" else []

    agent = create_tool_calling_agent(llm, tools, prompt)
    return AgentExecutor(
        agent=agent,
        tools=tools,
        verbose=verbose,
        max_iterations=10,
        max_execution_time=120,
        return_intermediate_steps=True,
        callbacks=callbacks,
    )


def _format_react_output(intermediate_steps: list) -> None:
    """
    将 intermediate_steps 以 Thought/Action/Action Input/Observation 格式打印。

    为什么用后处理而非 callback：
      LCEL 版 AgentExecutor 的 callbacks 参数只对 executor 层事件可靠（如 on_agent_finish），
      on_tool_start / on_tool_end 在新版 LangChain 中不一定能传播到工具子 run。
      intermediate_steps 由框架保证完整记录，与 LangChain 版本无关。
    """
    for action, observation in intermediate_steps:
        tool_input = action.tool_input
        input_str = (
            str(next(iter(tool_input.values()), ""))
            if isinstance(tool_input, dict)
            else str(tool_input)
        )
        if action.tool == "think":
            print(f"\nThought: {input_str}")
        else:
            print(f"\nAction: {action.tool}")
            print(f"Action Input: {input_str[:300]}{'...' if len(input_str) > 300 else ''}")
            obs_str = str(observation)
            print(f"Observation: {obs_str[:300]}{'...' if len(obs_str) > 300 else ''}")
    print("\nThought: 已收集完所有必要信息，生成最终报告")


# ══════════════════════════════════════════════════════════════════════════════
# 企业级落地：流式输出 + 执行记录存储
# ══════════════════════════════════════════════════════════════════════════════

TOOL_LABELS = {
    "think":                     "思考中...",
    "check_security_issues":     "正在扫描安全漏洞...",
    "analyze_complexity":        "正在分析代码复杂度...",
    "check_code_style":          "正在检查代码规范...",
    "check_test_coverage_hints": "正在分析测试覆盖...",
}


def init_db(db_path: str = "agent_executions.db") -> sqlite3.Connection:
    conn = sqlite3.connect(db_path)
    conn.execute("""
        CREATE TABLE IF NOT EXISTS tasks (
            task_id    TEXT PRIMARY KEY,
            final_answer TEXT,
            step_count INTEGER,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )
    """)
    conn.execute("""
        CREATE TABLE IF NOT EXISTS execution_steps (
            id         INTEGER PRIMARY KEY AUTOINCREMENT,
            task_id    TEXT,
            step_index INTEGER,
            type       TEXT,       -- 'thought' | 'action'
            tool       TEXT,
            input      TEXT,       -- JSON 字符串
            output     TEXT,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )
    """)
    conn.commit()
    return conn


async def stream_and_save(
    agent_executor: AgentExecutor,
    code: str,
    conn: sqlite3.Connection,
    task_id: str,
) -> None:
    """
    单次执行：通过 astream_events 同时完成
      - 实时流式输出（用户侧体验）
      - 收集每步结构化数据并存入 SQLite（持久化侧）

    astream_events v2 关键事件：
      on_tool_start  → 工具即将执行，显示友好进度提示
      on_tool_end    → 工具执行完毕，收集结构化结果
      on_chat_model_stream → Final Answer 逐字流式推送
    """
    steps: list[dict] = []
    current_tool: dict | None = None
    final_answer_parts: list[str] = []
    step_index = 0
    in_final_answer = False  # 区分推理阶段的 stream 和 Final Answer 的 stream

    print("\n" + "─" * 60)

    async for event in agent_executor.astream_events(
        {"input": f"请对以下代码进行全面的企业级代码审查：\n```python\n{code}\n```"},
        version="v2",
    ):
        kind = event["event"]

        if kind == "on_tool_start":
            in_final_answer = False
            tool_name = event["name"]
            tool_input = event["data"].get("input", {})
            current_tool = {"tool": tool_name, "input": tool_input, "index": step_index}
            print(f"\n⚙️  {TOOL_LABELS.get(tool_name, tool_name)}", flush=True)

        elif kind == "on_tool_end":
            in_final_answer = False
            if current_tool:
                output = str(event["data"].get("output", ""))
                steps.append({
                    "task_id":    task_id,
                    "step_index": current_tool["index"],
                    "type":       "thought" if current_tool["tool"] == "think" else "action",
                    "tool":       current_tool["tool"],
                    "input":      json.dumps(current_tool["input"], ensure_ascii=False),
                    "output":     output,
                })
                step_index += 1
                if current_tool["tool"] != "think":
                    print("  ✅ 完成", flush=True)
                current_tool = None

        elif kind == "on_chain_start" and event.get("name") == "AgentExecutor":
            pass  # AgentExecutor 开始，忽略

        elif kind == "on_chat_model_stream":
            chunk_content = event["data"]["chunk"].content
            if not chunk_content:
                continue
            # think_tool 模式下推理阶段也会触发 stream；
            # Final Answer 阶段工具调用已结束，step_index > 0 且 current_tool 为 None
            if current_tool is None and step_index > 0:
                if not in_final_answer:
                    in_final_answer = True
                    print("\n\n" + "─" * 60)
                    print("最终审查报告：")
                    print("─" * 60)
                print(chunk_content, end="", flush=True)
                final_answer_parts.append(chunk_content)

    final_answer = "".join(final_answer_parts)

    # 持久化到 SQLite
    if steps:
        conn.executemany(
            """INSERT INTO execution_steps
               (task_id, step_index, type, tool, input, output)
               VALUES (:task_id, :step_index, :type, :tool, :input, :output)""",
            steps,
        )
    conn.execute(
        "INSERT INTO tasks (task_id, final_answer, step_count) VALUES (?, ?, ?)",
        (task_id, final_answer, len(steps)),
    )
    conn.commit()
    print(f"\n\n✅ 执行记录已存库  task_id={task_id}  共 {len(steps)} 步")


# 默认使用 Tool Calling 方式（更可靠）
build_review_agent = build_tool_calling_agent


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
    """同步演示：invoke + Callback 推理可见（调试/学习用）"""
    agent_executor = build_tool_calling_agent(reasoning_mode="think_tool")
    mode_label = "Tool Calling + Think Tool（ReAct 格式推理输出）"

    print("=" * 60)
    print(f"  代码审查助手  [{mode_label}]")
    print("=" * 60)
    print("\n待审查代码：")
    print(SAMPLE_CODE)
    print("\n" + "=" * 60)
    print("开始审查（推理过程实时输出）...")
    print("=" * 60 + "\n")

    handler = ReasoningCallbackHandler(react_format=True)
    result = agent_executor.invoke(
        {"input": f"请对以下代码进行全面的企业级代码审查：\n```python\n{SAMPLE_CODE}\n```"},
        config={"callbacks": [handler]},
    )

    print("\n" + "=" * 60)
    print("最终审查报告：")
    print("=" * 60)
    print(result["output"])
    print(f"\n共执行了 {len(result['intermediate_steps'])} 次工具调用")


async def main_stream():
    """
    异步演示：astream_events 流式输出 + SQLite 执行记录存储

    特点：
      - 每步工具执行完立刻推送友好提示，不等全部完成
      - Final Answer 逐字流式输出
      - 同一次执行同时完成展示和存库，不重复运行 Agent
    """
    agent_executor = build_tool_calling_agent(reasoning_mode="think_tool", verbose=False)

    print("=" * 60)
    print("  代码审查助手  [流式输出 + 执行记录存储]")
    print("=" * 60)
    print("\n待审查代码：")
    print(SAMPLE_CODE)
    print("\n" + "=" * 60)
    print("开始审查（实时流式输出，结果同步存库）...")
    print("=" * 60)

    conn = init_db()
    task_id = str(uuid.uuid4())
    print(f"task_id: {task_id}")

    await stream_and_save(agent_executor, SAMPLE_CODE, conn, task_id)
    conn.close()


if __name__ == "__main__":
    # 切换注释选择演示模式
    # main()                        # 同步模式：Callback 推理可见
    asyncio.run(main_stream())    # 异步模式：流式输出 + 存库
