"""
思想二：Plan-and-Execute（规划-执行分离）
场景：企业软件项目风险评估系统

Plan-and-Execute 核心：
  阶段一 Planner LLM → 生成结构化执行计划（步骤列表）
  阶段二 Executor LLM → 按计划逐步执行，每步可调工具
  阶段三 可选 Re-planning → 执行失败或发现新信息时重新规划

企业价值：
  - 全局视野：先规划再执行，避免 ReAct 短视导致的无效工具调用
  - 可审计：计划本身是可检查的制品，适合需要人工审批的场景
  - 结构化：计划用 Pydantic 约束，保证每步格式一致
  - 断点续跑：计划存档后可以从失败步骤重新执行
"""

import os
import json
import time
from typing import Optional
from dataclasses import dataclass, field
from dotenv import load_dotenv, find_dotenv
from pydantic import BaseModel, Field
from langchain_core.tools import tool
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_openai import ChatOpenAI

load_dotenv(find_dotenv())


# ══════════════════════════════════════════════════════════════════════════════
# 计划数据模型（Pydantic 约束计划格式，防止 LLM 生成格式混乱的计划）
# ══════════════════════════════════════════════════════════════════════════════

class PlanStep(BaseModel):
    """单个执行步骤"""
    step_id: int = Field(description="步骤编号，从1开始")
    title: str = Field(description="步骤标题")
    objective: str = Field(description="本步骤的目标")
    tool_to_use: str = Field(description="需要调用的工具名称")
    tool_input: str = Field(description="传给工具的输入内容")
    depends_on: list[int] = Field(default=[], description="依赖的前置步骤编号列表")
    expected_output: str = Field(description="预期产出")


class ExecutionPlan(BaseModel):
    """完整执行计划"""
    task_summary: str = Field(description="任务总结")
    total_steps: int = Field(description="总步骤数")
    steps: list[PlanStep] = Field(description="执行步骤列表")
    success_criteria: str = Field(description="成功判定标准")


@dataclass
class StepResult:
    """步骤执行结果"""
    step_id: int
    status: str           # success / failed / skipped
    output: str
    error: Optional[str] = None
    retry_count: int = 0
    duration_ms: int = 0


# ══════════════════════════════════════════════════════════════════════════════
# 风险评估工具集
# ══════════════════════════════════════════════════════════════════════════════

@tool
def analyze_tech_stack(description: str) -> str:
    """
    分析项目技术栈的成熟度和社区支持情况。
    输入项目描述，返回技术栈风险评估。
    """
    risks = []
    desc_lower = description.lower()

    tech_risks = {
        "latest": ("版本过新", "HIGH", "使用最新版本可能缺乏生产验证，建议使用 LTS 版本"),
        "beta": ("测试版依赖", "CRITICAL", "生产环境不应使用 beta 版本"),
        "deprecated": ("废弃技术", "HIGH", "使用已废弃的技术，存在安全和维护风险"),
        "monolith": ("单体架构", "MEDIUM", "单体架构在高并发场景下扩展性受限"),
        "microservice": ("微服务复杂度", "MEDIUM", "微服务架构增加运维复杂度，需评估团队能力"),
        "nosql": ("NoSQL 一致性", "LOW", "NoSQL 通常为最终一致性，需确认业务是否可接受"),
        "redis": ("缓存穿透风险", "LOW", "Redis 缓存需考虑穿透、击穿、雪崩场景"),
        "elasticsearch": ("ES 资源消耗", "MEDIUM", "Elasticsearch 内存消耗大，需规划资源"),
    }

    for keyword, (risk_name, level, advice) in tech_risks.items():
        if keyword in desc_lower:
            risks.append({"risk": risk_name, "level": level, "advice": advice})

    if not risks:
        return "技术栈风险：未发现明显风险点，建议进一步深入评估"

    result = f"技术栈风险分析（共 {len(risks)} 项）：\n"
    for r in sorted(risks, key=lambda x: ["CRITICAL","HIGH","MEDIUM","LOW"].index(x["level"])):
        result += f"  [{r['level']}] {r['risk']}：{r['advice']}\n"
    return result


@tool
def evaluate_team_capacity(team_info: str) -> str:
    """
    评估团队能力与项目需求的匹配度。
    输入团队规模、技能栈、经验描述，返回能力风险评估。
    """
    risks = []
    info_lower = team_info.lower()

    # 团队规模评估
    import re
    size_match = re.search(r'(\d+)\s*(?:人|members?|developers?|engineers?)', team_info)
    team_size = int(size_match.group(1)) if size_match else 0

    if team_size == 1:
        risks.append(("CRITICAL", "单人团队", "关键路径存在单点风险，无法 Code Review，建议至少2人"))
    elif 1 < team_size < 3:
        risks.append(("HIGH", "团队规模不足", "小团队面对复杂项目有超负荷风险"))

    if "junior" in info_lower or "初级" in info_lower:
        risks.append(("HIGH", "经验不足", "初级工程师主导复杂系统存在设计决策风险"))
    if "兼职" in info_lower or "part-time" in info_lower:
        risks.append(("MEDIUM", "兼职资源", "兼职团队沟通成本高，进度难以保障"))
    if "分布" in info_lower or "remote" in info_lower or "distributed" in info_lower:
        risks.append(("LOW", "远程协作", "跨时区团队需要异步工作流和明确的文档规范"))

    if not risks:
        return "团队能力评估：团队配置合理，未发现明显风险"

    result = "团队能力风险：\n"
    for level, name, advice in sorted(risks, key=lambda x: ["CRITICAL","HIGH","MEDIUM","LOW"].index(x[0])):
        result += f"  [{level}] {name}：{advice}\n"
    return result


@tool
def assess_timeline_risk(timeline_info: str) -> str:
    """
    评估项目时间线的合理性和风险。
    输入时间计划描述，返回进度风险评估和缓冲建议。
    """
    risks = []
    info_lower = timeline_info.lower()

    import re
    # 提取工期
    month_match = re.search(r'(\d+)\s*(?:个?月|months?)', timeline_info)
    week_match = re.search(r'(\d+)\s*(?:周|weeks?)', timeline_info)

    duration_weeks = 0
    if month_match:
        duration_weeks = int(month_match.group(1)) * 4
    elif week_match:
        duration_weeks = int(week_match.group(1))

    if duration_weeks > 0:
        if duration_weeks < 4:
            risks.append(("HIGH", "工期过短", f"{duration_weeks}周对多数功能开发不够，建议评估MVP范围"))
        elif duration_weeks > 52:
            risks.append(("MEDIUM", "超长项目", "超过一年的项目需求变化风险极高，建议分阶段交付"))

    if "固定" in timeline_info or "hard deadline" in info_lower or "不可延期" in timeline_info:
        risks.append(("HIGH", "硬性截止日期", "没有缓冲的固定截止日期会在后期产生极大质量压力"))
    if "同时" in timeline_info or "parallel" in info_lower or "并行" in timeline_info:
        risks.append(("MEDIUM", "并行开发风险", "多模块并行开发增加集成风险，需要明确接口契约"))
    if "节假日" in timeline_info or "holiday" in info_lower:
        risks.append(("LOW", "节假日影响", "时间计划中需扣除节假日，并预留 15% 缓冲时间"))

    buffer = "建议在总工期基础上预留 20% 缓冲时间用于联调、测试和 Bug 修复"
    if not risks:
        return f"时间线评估：计划基本合理\n{buffer}"

    result = "时间线风险：\n"
    for level, name, advice in sorted(risks, key=lambda x: ["CRITICAL","HIGH","MEDIUM","LOW"].index(x[0])):
        result += f"  [{level}] {name}：{advice}\n"
    result += f"\n建议：{buffer}"
    return result


@tool
def calculate_risk_score(risk_factors: str) -> str:
    """
    综合各维度风险因子，计算项目整体风险评分。
    输入格式：逗号分隔的风险列表，如 "CRITICAL:2,HIGH:3,MEDIUM:1"
    返回综合评分（0-100，越低越危险）和风险等级。
    """
    try:
        weights = {"CRITICAL": 25, "HIGH": 10, "MEDIUM": 4, "LOW": 1}
        total_penalty = 0

        for item in risk_factors.split(","):
            item = item.strip()
            if ":" in item:
                level, count_str = item.split(":", 1)
                level = level.strip().upper()
                count = int(count_str.strip())
                total_penalty += weights.get(level, 0) * count

        score = max(0, 100 - total_penalty)

        if score >= 80:
            grade, recommendation = "A（低风险）", "项目风险可控，建议正常推进"
        elif score >= 60:
            grade, recommendation = "B（中等风险）", "存在中等风险，建议制定缓解措施后推进"
        elif score >= 40:
            grade, recommendation = "C（高风险）", "风险较高，建议解决关键风险点再启动"
        else:
            grade, recommendation = "D（极高风险）", "风险极高，强烈建议重新评估项目范围或可行性"

        return f"综合风险评分：{score}/100  等级：{grade}\n建议：{recommendation}"
    except Exception as e:
        return f"评分计算失败：{e}，请按格式传入 'LEVEL:count' 列表"


# ══════════════════════════════════════════════════════════════════════════════
# Plan-and-Execute 核心逻辑
# ══════════════════════════════════════════════════════════════════════════════

TOOLS_MAP = {
    "analyze_tech_stack": analyze_tech_stack,
    "evaluate_team_capacity": evaluate_team_capacity,
    "assess_timeline_risk": assess_timeline_risk,
    "calculate_risk_score": calculate_risk_score,
}


def _tools_desc(tools: dict) -> str:
    return "\n".join(
        f"- {name}：{fn.description.splitlines()[0].strip()}"
        for name, fn in tools.items()
    )


class PlanAndExecuteAgent:
    """
    规划-执行分离的 Agent。
    Planner 用结构化输出生成计划，Executor 按计划调用工具逐步执行。
    """

    def __init__(self):
        self.llm = ChatOpenAI(
            model="qwen-plus",
            api_key=os.getenv("DASHSCOPE_API_KEY"),
            base_url="https://dashscope.aliyuncs.com/compatible-mode/v1",
            temperature=0,
        )
        self.planner_llm = self.llm.with_structured_output(ExecutionPlan)

    # ── 阶段一：Planner ───────────────────────────────────────────────────────

    def plan(self, task: str) -> ExecutionPlan:
        """调用 Planner LLM，生成结构化执行计划"""
        print("\n【阶段一：规划】Planner 正在生成执行计划...")

        messages = [
            SystemMessage(content=f"""你是一个专业的项目风险评估规划师。
根据用户描述的项目，生成一个详细的风险评估执行计划。

可用工具：
{_tools_desc(TOOLS_MAP)}

规划原则：
1. calculate_risk_score 必须在其他评估完成后执行（依赖前三步结果）
2. 前三个工具可以并行执行（depends_on 为空）
3. tool_input 必须是从原始需求中提取的具体内容，不能是模板占位符"""),
            HumanMessage(content=f"请为以下项目制定风险评估计划：\n\n{task}"),
        ]

        plan = self.planner_llm.invoke(messages)
        print(f"  ✓ 计划生成完成，共 {plan.total_steps} 个步骤")
        for step in plan.steps:
            deps = f"（依赖步骤 {step.depends_on}）" if step.depends_on else "（可立即执行）"
            print(f"  Step {step.step_id}: {step.title} {deps}")
        return plan

    # ── 阶段二：Executor ──────────────────────────────────────────────────────

    def execute_step(self, step: PlanStep, context: dict[int, StepResult]) -> StepResult:
        """执行单个步骤，支持依赖注入和错误重试"""
        start = time.time()
        print(f"\n  【执行 Step {step.step_id}】{step.title}")
        print(f"    目标：{step.objective}")

        # 检查依赖是否已完成
        for dep_id in step.depends_on:
            dep_result = context.get(dep_id)
            if not dep_result or dep_result.status != "success":
                return StepResult(
                    step_id=step.step_id,
                    status="skipped",
                    output="",
                    error=f"依赖步骤 {dep_id} 未成功完成，跳过本步骤",
                )

        # 如果 tool_input 需要引用前序步骤结果，用 Executor LLM 组装输入
        actual_input = step.tool_input
        if step.depends_on:
            dep_outputs = {dep_id: context[dep_id].output for dep_id in step.depends_on}
            actual_input = self._compose_input_from_deps(step, dep_outputs)

        # 调用工具（最多重试2次）
        tool_fn = TOOLS_MAP.get(step.tool_to_use)
        if not tool_fn:
            return StepResult(
                step_id=step.step_id, status="failed", output="",
                error=f"工具 '{step.tool_to_use}' 不存在",
            )

        for attempt in range(3):
            try:
                output = tool_fn.invoke(actual_input)
                duration = int((time.time() - start) * 1000)
                print(f"    ✓ 执行成功（{duration}ms）")
                print(f"    输出：{output[:200]}{'...' if len(output) > 200 else ''}")
                return StepResult(
                    step_id=step.step_id, status="success",
                    output=output, retry_count=attempt, duration_ms=duration,
                )
            except Exception as e:
                if attempt == 2:
                    return StepResult(
                        step_id=step.step_id, status="failed",
                        output="", error=str(e), retry_count=attempt,
                    )
                print(f"    ⚠️  第{attempt+1}次尝试失败：{e}，正在重试...")
                time.sleep(0.5)

    def _compose_input_from_deps(self, step: PlanStep, dep_outputs: dict) -> str:
        """用 Executor LLM 根据依赖步骤的输出组装当前步骤的输入"""
        dep_summary = "\n".join(
            f"步骤{dep_id}结果：{output[:300]}"
            for dep_id, output in dep_outputs.items()
        )
        messages = [
            SystemMessage(content="根据前序步骤的结果，提取关键风险数量，组装为格式 'LEVEL:count,LEVEL:count' 的字符串"),
            HumanMessage(content=f"前序结果：\n{dep_summary}\n\n请统计各级别风险数量并输出格式字符串"),
        ]
        response = self.llm.invoke(messages)
        return response.content.strip()

    def execute_plan(self, plan: ExecutionPlan) -> list[StepResult]:
        """按计划顺序执行所有步骤"""
        print(f"\n【阶段二：执行】开始执行 {plan.total_steps} 个步骤...")
        context: dict[int, StepResult] = {}

        for step in sorted(plan.steps, key=lambda s: s.step_id):
            result = self.execute_step(step, context)
            context[step.step_id] = result

            if result.status == "failed":
                print(f"    ✗ Step {step.step_id} 失败：{result.error}")
                # 判断是否需要 Re-planning（此处简化为跳过后续依赖步骤）

        return list(context.values())

    # ── 阶段三：汇总 ─────────────────────────────────────────────────────────

    def summarize(self, task: str, plan: ExecutionPlan, results: list[StepResult]) -> str:
        """用 LLM 综合所有步骤结果，生成最终评估报告"""
        print("\n【阶段三：汇总】生成最终风险评估报告...")

        successful = [r for r in results if r.status == "success"]
        step_outputs = "\n\n".join(
            f"Step {r.step_id} 结果：\n{r.output}"
            for r in successful
        )

        messages = [
            SystemMessage(content="""你是一个专业的项目风险评估师，负责撰写最终评估报告。
报告格式：
1. 项目概览（1-2句）
2. 风险总览（评分 + 等级）
3. 关键风险点（按 CRITICAL/HIGH/MEDIUM/LOW 分组，每条带修复建议）
4. 总体建议（是否推进 + 前提条件）"""),
            HumanMessage(content=f"项目：{task}\n\n各维度评估结果：\n{step_outputs}"),
        ]

        response = self.llm.invoke(messages)
        return response.content

    # ── 主入口 ────────────────────────────────────────────────────────────────

    def run(self, task: str) -> str:
        """完整的 Plan-Execute 流程"""
        plan = self.plan(task)
        results = self.execute_plan(plan)
        return self.summarize(task, plan, results)


# ══════════════════════════════════════════════════════════════════════════════
# 演示入口
# ══════════════════════════════════════════════════════════════════════════════

SAMPLE_PROJECT = """
项目名称：电商平台重构
项目描述：将现有的 Python 2.7 + MySQL 单体电商系统迁移到微服务架构，
技术栈选型：FastAPI + React + PostgreSQL + Redis + Elasticsearch，
使用最新版本的所有框架。
团队规模：3人团队，1名高级工程师（团队负责人）+ 2名初级工程师，
均为全栈兼职状态（同时承接其他项目）。
时间计划：3个月完成所有功能迁移，有固定截止日期不可延期。
"""


def main():
    print("=" * 60)
    print("  Plan-and-Execute Agent：项目风险评估系统")
    print("=" * 60)
    print(f"\n项目信息：{SAMPLE_PROJECT}")

    agent = PlanAndExecuteAgent()
    report = agent.run(SAMPLE_PROJECT)

    print("\n" + "=" * 60)
    print("最终风险评估报告：")
    print("=" * 60)
    print(report)


if __name__ == "__main__":
    main()
