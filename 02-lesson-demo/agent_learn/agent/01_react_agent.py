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
import textwrap
from typing import Any
from dotenv import load_dotenv, find_dotenv
from pydantic import BaseModel, Field
from langchain_core.tools import tool
from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI
from langchain.agents import create_react_agent, AgentExecutor

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
# ReAct Agent 构建
# ══════════════════════════════════════════════════════════════════════════════

REVIEW_TOOLS = [analyze_complexity, check_security_issues, check_code_style, check_test_coverage_hints]

# ReAct 的 Prompt 决定 Agent 的推理风格和输出格式
REACT_PROMPT = ChatPromptTemplate.from_template("""你是一个专业的企业级代码审查助手。
使用工具对提交的代码进行全面审查，最终输出一份结构化的审查报告。

审查要求：
1. 必须使用所有四个工具进行检查（复杂度、安全、规范、测试覆盖）
2. 综合所有工具结果，给出整体质量评分（0-100）
3. 最终报告格式：
   - 整体评分与结论（通过/需修改/拒绝）
   - 关键问题列表（按优先级排序）
   - 修复建议

可用工具：
{tools}

工具名称：{tool_names}

使用格式（严格遵守）：
Thought: 分析当前情况，决定下一步
Action: 工具名称
Action Input: 工具的输入参数
Observation: 工具返回的结果
... （重复 Thought/Action/Observation）
Thought: 已有足够信息，可以给出最终答案
Final Answer: 完整的审查报告

开始审查：
{input}

{agent_scratchpad}""")


def build_review_agent(verbose: bool = True) -> AgentExecutor:
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
        max_iterations=10,          # 最多10轮工具调用，防止死循环
        max_execution_time=120,     # 最长120秒
        handle_parsing_errors=True, # 格式解析失败时让 Agent 自我修正
        return_intermediate_steps=True,  # 返回中间推理步骤，可用于审计
    )


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
    print("=" * 60)
    print("  ReAct Agent：企业代码审查助手")
    print("=" * 60)
    print("\n待审查代码：")
    print(SAMPLE_CODE)
    print("\n" + "=" * 60)
    print("开始 ReAct 推理链...")
    print("=" * 60 + "\n")

    agent_executor = build_review_agent(verbose=True)
    result = agent_executor.invoke({
        "input": f"请对以下代码进行全面的企业级代码审查：\n```python\n{SAMPLE_CODE}\n```"
    })

    print("\n" + "=" * 60)
    print("最终审查报告：")
    print("=" * 60)
    print(result["output"])

    # 中间推理步骤（可用于审计日志）
    print(f"\n共执行了 {len(result['intermediate_steps'])} 次工具调用")


if __name__ == "__main__":
    main()
