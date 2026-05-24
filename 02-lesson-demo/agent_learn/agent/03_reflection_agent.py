"""
思想三：Reflection（反思-自我优化）
场景：企业技术事故复盘报告自动生成系统

Reflection 核心：
  Generator 生成初稿 → Critic 按维度评分 → 分数未达标则迭代修订
  循环直到质量达标或达到最大迭代次数

企业价值：
  - 质量保证：每份报告都经过自动质检，减少人工审核成本
  - 可定制标准：不同报告类型使用不同评分维度（合规性、完整性、可操作性）
  - 可解释改进：Critic 给出结构化修改意见，Generator 针对性修订
  - 审计友好：每轮迭代记录保留，便于复查生成过程
"""

import os
from dataclasses import dataclass, field
from typing import Optional
from dotenv import load_dotenv, find_dotenv
from pydantic import BaseModel, Field
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_openai import ChatOpenAI

load_dotenv(find_dotenv())


# ══════════════════════════════════════════════════════════════════════════════
# 评分模型（Critic 输出严格结构化，Generator 才能针对性修订）
# ══════════════════════════════════════════════════════════════════════════════

class DimensionScore(BaseModel):
    """单个维度评分"""
    dimension: str = Field(description="评分维度名称")
    score: float = Field(description="得分 0-10", ge=0, le=10)
    issues: list[str] = Field(description="发现的问题列表")
    suggestions: list[str] = Field(description="具体改进建议")


class CriticResult(BaseModel):
    """Critic 完整评估结果"""
    overall_score: float = Field(description="综合评分 0-100", ge=0, le=100)
    dimensions: list[DimensionScore] = Field(description="各维度评分明细")
    critical_issues: list[str] = Field(description="必须修复的关键问题（影响报告可用性）")
    revision_priority: str = Field(description="修订重点（简明描述最重要的改进方向）")
    approved: bool = Field(description="是否通过质检（overall_score >= 75 且无 critical_issues）")


# ══════════════════════════════════════════════════════════════════════════════
# 迭代状态记录
# ══════════════════════════════════════════════════════════════════════════════

@dataclass
class IterationRecord:
    """单次迭代记录"""
    iteration: int
    draft: str
    critic_result: CriticResult
    revision_guidance: str = ""


# ══════════════════════════════════════════════════════════════════════════════
# Reflection Agent
# ══════════════════════════════════════════════════════════════════════════════

class ReflectionAgent:
    """
    基于反思循环的高质量文档生成 Agent。
    Generator 和 Critic 使用不同 temperature，保证评估客观性。
    """

    QUALITY_THRESHOLD = 75.0    # 通过阈值：综合评分 >= 75
    MAX_ITERATIONS = 4          # 最大迭代次数（防止无限循环）

    # 事故复盘报告的评分维度定义
    RUBRIC = """
评分维度（每项满分10分）：

1. 时间线完整性（Timeline Completeness）
   - 事件时间线是否清晰、完整
   - 关键节点是否都有记录（发现时间/告警时间/响应时间/恢复时间）
   - 时间戳是否精确到分钟

2. 根因分析深度（Root Cause Analysis）
   - 是否识别了直接原因
   - 是否追溯到根本原因（技术/流程/组织层面）
   - 是否避免了"人为错误"这种浅层结论

3. 影响范围量化（Impact Quantification）
   - 受影响用户数是否有数据
   - 业务损失是否可量化（GMV损失/SLA损失）
   - 是否区分了直接影响和间接影响

4. 行动项可执行性（Action Item Actionability）
   - 每个行动项是否有明确负责人
   - 是否有截止日期
   - 行动项是否可验证（完成标准是否明确）

5. 预防措施系统性（Prevention Systematicness）
   - 措施是否能防止同类事故（而非只修复本次故障）
   - 是否涵盖检测、响应、恢复三个层面
   - 是否有监控指标可以验证预防效果
"""

    def __init__(self):
        # Generator 使用较高 temperature，激发多样性
        self.generator = ChatOpenAI(
            model="qwen-plus",
            api_key=os.getenv("DASHSCOPE_API_KEY"),
            base_url="https://dashscope.aliyuncs.com/compatible-mode/v1",
            temperature=0.3,
        )
        # Critic 使用低 temperature + 结构化输出，保证评估一致性
        base_llm = ChatOpenAI(
            model="qwen-plus",
            api_key=os.getenv("DASHSCOPE_API_KEY"),
            base_url="https://dashscope.aliyuncs.com/compatible-mode/v1",
            temperature=0,
        )
        self.critic = base_llm.with_structured_output(CriticResult)

    # ── Generator ─────────────────────────────────────────────────────────────

    def generate(self, incident_info: str, previous_draft: str = "", revision_guidance: str = "") -> str:
        """生成或修订事故复盘报告"""
        if not previous_draft:
            # 首次生成
            messages = [
                SystemMessage(content="""你是一个资深的 SRE 工程师，负责撰写专业的事故复盘报告（Post-Mortem）。
报告结构：
1. 事故摘要（影响范围 + 持续时间 + 严重程度）
2. 详细时间线（精确到分钟，每个关键事件一行）
3. 根因分析（直接原因 → 根本原因 → 系统性原因）
4. 影响量化（受影响用户数、业务损失、SLA 影响）
5. 响应过程评估（什么做得好、什么可以改进）
6. 行动项（负责人 + 截止日期 + 验证方式）
7. 预防措施（检测增强 + 响应改进 + 架构改进）"""),
                HumanMessage(content=f"请根据以下事故信息撰写复盘报告：\n\n{incident_info}"),
            ]
        else:
            # 基于反馈修订
            messages = [
                SystemMessage(content="""你是一个资深的 SRE 工程师，正在修订事故复盘报告。
根据质检反馈，有针对性地改进报告质量，不要改变正确的内容，只修复被指出的问题。"""),
                HumanMessage(content=f"""原始事故信息：
{incident_info}

当前报告草稿：
{previous_draft}

质检反馈（请严格按此修订）：
{revision_guidance}

请输出修订后的完整报告（不要只输出修改部分）："""),
            ]

        return self.generator.invoke(messages).content

    # ── Critic ────────────────────────────────────────────────────────────────

    def critique(self, draft: str) -> CriticResult:
        """对报告草稿进行多维度质量评估"""
        messages = [
            SystemMessage(content=f"""你是一个严格的事故复盘报告质检专家。
按以下评分维度对报告进行评估：

{self.RUBRIC}

评分规则：
- overall_score = 各维度分数的加权平均 * 10
- approved = overall_score >= 75 且 critical_issues 为空
- critical_issues：严重到影响报告可用性的问题（如：缺少根因分析、时间线完全缺失）
- 评分要严格，不要给满分"""),
            HumanMessage(content=f"请评估以下事故复盘报告：\n\n{draft}"),
        ]
        return self.critic.invoke(messages)

    # ── 生成修订指引 ──────────────────────────────────────────────────────────

    def _build_revision_guidance(self, critic_result: CriticResult) -> str:
        """将 Critic 的结构化评估转为 Generator 可理解的修订指引"""
        lines = ["请根据以下质检反馈修订报告：\n"]

        if critic_result.critical_issues:
            lines.append("【必须修复的关键问题】")
            for issue in critic_result.critical_issues:
                lines.append(f"  ✗ {issue}")
            lines.append("")

        lines.append("【各维度改进建议】")
        for dim in sorted(critic_result.dimensions, key=lambda d: d.score):
            if dim.score < 8:  # 只列出不够好的维度
                lines.append(f"\n{dim.dimension}（当前评分：{dim.score}/10）")
                if dim.issues:
                    lines.append("  问题：")
                    for issue in dim.issues:
                        lines.append(f"    - {issue}")
                if dim.suggestions:
                    lines.append("  改进：")
                    for sug in dim.suggestions:
                        lines.append(f"    + {sug}")

        lines.append(f"\n【修订重点】{critic_result.revision_priority}")
        return "\n".join(lines)

    # ── 主循环 ────────────────────────────────────────────────────────────────

    def run(self, incident_info: str) -> tuple[str, list[IterationRecord]]:
        """
        执行反思循环。
        返回：(最终报告, 迭代记录列表)
        """
        history: list[IterationRecord] = []
        current_draft = ""
        revision_guidance = ""

        for i in range(1, self.MAX_ITERATIONS + 1):
            print(f"\n{'═'*50}")
            print(f"  迭代 {i}/{self.MAX_ITERATIONS}")
            print(f"{'═'*50}")

            # Step 1: Generate
            print(f"  [Generator] {'生成初稿...' if i == 1 else '根据反馈修订...'}")
            current_draft = self.generate(incident_info, current_draft, revision_guidance)
            print(f"  [Generator] 生成完成（{len(current_draft)}字）")

            # Step 2: Critique
            print(f"  [Critic] 正在质检...")
            critic_result = self.critique(current_draft)
            print(f"  [Critic] 综合评分：{critic_result.overall_score:.1f}/100  "
                  f"{'✅ 通过' if critic_result.approved else '❌ 未通过'}")
            for dim in critic_result.dimensions:
                print(f"    {dim.dimension}: {dim.score:.1f}/10")

            # 生成修订指引
            revision_guidance = self._build_revision_guidance(critic_result)
            history.append(IterationRecord(
                iteration=i,
                draft=current_draft,
                critic_result=critic_result,
                revision_guidance=revision_guidance,
            ))

            # Step 3: 判断是否通过
            if critic_result.approved:
                print(f"\n  ✅ 质检通过（迭代{i}次），报告生成完毕")
                break

            if i == self.MAX_ITERATIONS:
                print(f"\n  ⚠️  达到最大迭代次数，输出当前最佳版本（评分：{critic_result.overall_score:.1f}）")

        return current_draft, history


# ══════════════════════════════════════════════════════════════════════════════
# 演示入口
# ══════════════════════════════════════════════════════════════════════════════

SAMPLE_INCIDENT = """
事故基本信息：
- 发生时间：2024年3月15日 凌晨2点左右
- 系统：订单服务
- 问题：数据库挂了，然后用户下不了单
- 影响：持续大约3个小时
- 解决方式：重启数据库

背景：
- 是 MySQL 主库
- 运维半夜发现监控报警
- 后来发现是某个慢查询把连接池打满了
- 慢查询是新上的促销活动功能引入的
"""


def main():
    print("=" * 60)
    print("  Reflection Agent：事故复盘报告自动生成系统")
    print("=" * 60)
    print(f"\n原始事故信息（质量较低）：")
    print(SAMPLE_INCIDENT)
    print(f"\n质检阈值：{ReflectionAgent.QUALITY_THRESHOLD}/100")
    print(f"最大迭代：{ReflectionAgent.MAX_ITERATIONS}轮")

    agent = ReflectionAgent()
    final_report, history = agent.run(SAMPLE_INCIDENT)

    print("\n" + "=" * 60)
    print("迭代历史：")
    for record in history:
        print(f"  第{record.iteration}轮：评分={record.critic_result.overall_score:.1f}  "
              f"通过={'是' if record.critic_result.approved else '否'}")
    print("\n" + "=" * 60)
    print("最终事故复盘报告：")
    print("=" * 60)
    print(final_report)


if __name__ == "__main__":
    main()
