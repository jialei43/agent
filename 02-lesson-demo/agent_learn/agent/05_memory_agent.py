"""
思想五：Memory-Augmented（记忆增强）
场景：企业知识助手 —— 能从历史对话中学习的个人技术顾问

Memory 四层架构：
  短期记忆（会话缓冲）：当前会话的消息历史
  情节记忆（经验存档）：历史 Q&A 对，按语义相似度检索
  语义记忆（领域知识）：固定知识库（运维手册、架构文档等）
  用户画像（个性化）：用户身份、技术水平、偏好、常见问题模式

企业价值：
  - 连续性：跨会话记住用户情况，不需要重复介绍背景
  - 经验复用：相似问题自动检索历史成功案例
  - 个性化：根据用户技术水平调整回答深度
  - 知识积累：团队共享知识库，避免重复踩坑
"""

import os
import json
import time
import hashlib
from dataclasses import dataclass, field
from datetime import datetime
from typing import Optional
from dotenv import load_dotenv, find_dotenv
from langchain_core.messages import HumanMessage, SystemMessage, AIMessage
from langchain_openai import ChatOpenAI, OpenAIEmbeddings

load_dotenv(find_dotenv())


# ══════════════════════════════════════════════════════════════════════════════
# 记忆数据模型
# ══════════════════════════════════════════════════════════════════════════════

@dataclass
class EpisodicMemory:
    """情节记忆：一次完整的 Q&A 记录"""
    memory_id: str
    question: str
    answer: str
    tags: list[str]           # 主题标签，用于快速过滤
    quality_score: float      # 答案质量分（0-1），低质量记忆不检索
    created_at: str
    user_id: str
    embedding: Optional[list[float]] = None  # 语义向量


@dataclass
class UserProfile:
    """用户画像：跨会话持久化"""
    user_id: str
    name: str
    role: str                         # 角色：SRE/Backend/Frontend/Manager 等
    tech_level: str                   # 技术水平：junior/mid/senior/principal
    preferred_language: str = "zh"    # 偏好语言
    known_stack: list[str] = field(default_factory=list)    # 熟悉的技术栈
    pain_points: list[str] = field(default_factory=list)    # 常见问题点
    interaction_count: int = 0
    last_seen: str = ""


@dataclass
class ConversationTurn:
    """会话单轮"""
    role: str        # user / assistant
    content: str
    timestamp: str


# ══════════════════════════════════════════════════════════════════════════════
# 情节记忆存储（内存实现，生产中替换为向量数据库）
# ══════════════════════════════════════════════════════════════════════════════

class EpisodicMemoryStore:
    """
    情节记忆存储。
    生产实现：使用 Pinecone/Weaviate/pgvector 替代内存存储。
    此处用余弦相似度 + 简单 TF-IDF 模拟语义检索。
    """

    def __init__(self, max_memories: int = 1000):
        self.memories: list[EpisodicMemory] = []
        self.max_memories = max_memories

    def add(self, memory: EpisodicMemory):
        """添加记忆，超过容量时淘汰最旧且质量最低的"""
        self.memories.append(memory)
        if len(self.memories) > self.max_memories:
            # LRU + quality 混合淘汰策略
            self.memories.sort(key=lambda m: (m.quality_score, m.created_at))
            self.memories = self.memories[len(self.memories) // 5:]  # 淘汰 20%

    def search(self, query: str, top_k: int = 3, min_quality: float = 0.5) -> list[EpisodicMemory]:
        """
        基于词袋相似度检索相关记忆。
        生产中替换为 embedding 余弦相似度检索。
        """
        if not self.memories:
            return []

        query_tokens = set(query.lower().split())

        def relevance_score(mem: EpisodicMemory) -> float:
            if mem.quality_score < min_quality:
                return 0.0
            mem_tokens = set((mem.question + " " + " ".join(mem.tags)).lower().split())
            overlap = len(query_tokens & mem_tokens)
            if overlap == 0:
                return 0.0
            # TF-IDF 近似：重叠词数 / 总词数 * 质量分
            jaccard = overlap / len(query_tokens | mem_tokens)
            # 时间衰减：越新的记忆权重越高
            age_days = max(0, (datetime.now() - datetime.fromisoformat(mem.created_at)).days)
            time_weight = 1.0 / (1 + age_days * 0.01)  # 100天后权重降一半
            return jaccard * mem.quality_score * time_weight

        scored = [(mem, relevance_score(mem)) for mem in self.memories]
        scored.sort(key=lambda x: x[1], reverse=True)
        return [mem for mem, score in scored[:top_k] if score > 0]

    def __len__(self):
        return len(self.memories)


# ══════════════════════════════════════════════════════════════════════════════
# 语义记忆（固定知识库）
# ══════════════════════════════════════════════════════════════════════════════

SEMANTIC_KNOWLEDGE_BASE = {
    "数据库连接池": """连接池最佳实践（运维手册 v3.2）：
- 连接池大小公式：pool_size = (core_count * 2) + disk_spindles
- PostgreSQL：max_connections 建议不超过 200，超过用 PgBouncer
- MySQL：innodb_thread_concurrency = CPU核数*2，wait_timeout=28800
- 连接泄漏检测：设置 connection_timeout + 定期检测空闲连接
- 监控指标：active_connections/max_connections，超过 80% 告警""",

    "Redis 缓存": """Redis 运维手册：
- 缓存穿透：空值缓存（TTL 5分钟）+ Bloom Filter
- 缓存击穿：分布式锁（Redlock）+ 热点 key 永不过期+后台更新
- 缓存雪崩：TTL 加随机抖动（±10%）+ 熔断降级
- 内存优化：使用压缩编码（hash-max-ziplist-entries=512）
- 集群规划：单节点不超过 10GB，超过用 Cluster 模式""",

    "Kubernetes": """K8s 运维经验：
- Pod 资源：requests/limits 务必设置，防止单 Pod 吃光节点资源
- HPA：基于 CPU+自定义指标（如队列长度）双维度伸缩
- 滚动更新：maxSurge=25%、maxUnavailable=0 保证零停机
- 故障排查：kubectl describe pod + kubectl logs --previous
- 镜像：使用多阶段构建，禁止 latest 标签，强制 digest 固定版本""",

    "分布式链路": """分布式系统排查：
- 链路追踪：使用 OpenTelemetry 标准，接入 Jaeger/Zipkin
- 关键指标：P99 延迟、错误率、QPS（USE + RED 方法论）
- 熔断配置：错误率>50% 触发，half-open 状态探测恢复
- 超时设置：客户端超时 < 服务端超时（防止重试风暴）
- 重试策略：指数退避 + jitter，最多重试3次""",
}


def search_semantic_knowledge(query: str) -> str:
    """从语义知识库检索相关条目"""
    query_lower = query.lower()
    matches = []
    for topic, content in SEMANTIC_KNOWLEDGE_BASE.items():
        topic_words = set(topic.lower().split())
        if any(word in query_lower for word in topic_words) or topic.lower() in query_lower:
            matches.append((topic, content))
    return matches


# ══════════════════════════════════════════════════════════════════════════════
# Memory-Augmented Agent
# ══════════════════════════════════════════════════════════════════════════════

class MemoryAugmentedAgent:
    """
    记忆增强的企业知识助手。
    每次回答都会检索历史经验，并将本次 Q&A 存入记忆库。
    """

    WINDOW_SIZE = 8  # 短期记忆保留的最近轮次数

    def __init__(self, user_id: str = "default"):
        self.llm = ChatOpenAI(
            model="qwen-plus",
            api_key=os.getenv("DASHSCOPE_API_KEY"),
            base_url="https://dashscope.aliyuncs.com/compatible-mode/v1",
            temperature=0.1,
        )
        self.user_id = user_id
        self.episodic_store = EpisodicMemoryStore()
        self.short_term: list[ConversationTurn] = []  # 短期记忆
        self.user_profile = self._init_profile(user_id)

    def _init_profile(self, user_id: str) -> UserProfile:
        """初始化或加载用户画像（生产中从数据库加载）"""
        # 模拟预置的用户画像
        profiles = {
            "zhang_wei": UserProfile(
                user_id="zhang_wei", name="张威", role="SRE",
                tech_level="senior",
                known_stack=["Python", "Kubernetes", "PostgreSQL", "Redis"],
                pain_points=["数据库性能调优", "容器化部署"],
                interaction_count=47,
                last_seen="2024-03-10",
            ),
            "li_na": UserProfile(
                user_id="li_na", name="李娜", role="Backend",
                tech_level="mid",
                known_stack=["Java", "Spring Boot", "MySQL"],
                pain_points=["并发问题", "分布式事务"],
                interaction_count=12,
                last_seen="2024-03-14",
            ),
        }
        return profiles.get(user_id, UserProfile(
            user_id=user_id, name="用户", role="Unknown", tech_level="mid",
        ))

    # ── 记忆检索 ──────────────────────────────────────────────────────────────

    def _retrieve_relevant_memories(self, query: str) -> list[EpisodicMemory]:
        """检索与当前问题相关的历史记忆"""
        return self.episodic_store.search(query, top_k=3, min_quality=0.5)

    def _retrieve_knowledge(self, query: str) -> list[tuple[str, str]]:
        """从语义知识库检索相关知识"""
        return search_semantic_knowledge(query)

    # ── 上下文组装 ────────────────────────────────────────────────────────────

    def _build_system_prompt(
        self,
        relevant_memories: list[EpisodicMemory],
        knowledge_items: list[tuple[str, str]],
    ) -> str:
        """组装完整的系统 Prompt（用户画像 + 历史经验 + 领域知识）"""
        profile = self.user_profile

        # ── [旧] 规则引导版：4档固定风格表，技术水平→回答风格一一对应 ──
        # 缺点：同一等级的用户背景差异很大（如 senior Java 工程师 vs senior ML 研究员），
        #       固定映射无法体现用户具体技术栈和痛点对沟通方式的影响
        # style_map = {
        #     "junior": "用简单易懂的语言，避免缩写，多举例子，提供完整的操作步骤",
        #     "mid":    "保持技术准确性，适当提供背景原理，给出实用的代码示例",
        #     "senior": "直接给出核心结论，聚焦边界情况和性能影响，无需解释基础概念",
        #     "principal": "提供架构层面的思考，权衡利弊，讨论规模化方案",
        # }
        # answer_style = style_map.get(profile.tech_level, style_map["mid"])

        # ── [新] 动态推理版：将完整用户画像注入，让 LLM 自主判断合适的沟通深度 ──
        # 核心改变：不再用 tech_level 查表，而是把用户的角色、等级、技术栈、痛点一起告诉 LLM，
        # 让 LLM 综合这些信息自主决定回答深度、是否需要解释概念、示例复杂程度等
        answer_style = (
            f"用户是 {profile.tech_level} 级别的 {profile.role}，"
            f"熟悉 {', '.join(profile.known_stack) if profile.known_stack else '技术背景未知'}，"
            f"常见痛点为 {', '.join(profile.pain_points) if profile.pain_points else '未记录'}。"
            "请根据其完整背景自主判断：回答的技术深度、是否需要解释基础概念、示例的复杂程度。"
        )

        system_parts = [
            f"""你是一位企业技术顾问，专门服务 {profile.role} 工程师。

用户档案：
- 姓名：{profile.name}
- 角色：{profile.role}（{profile.tech_level} 水平）
- 熟悉技术：{', '.join(profile.known_stack) if profile.known_stack else '未知'}
- 历史交互：{profile.interaction_count}次
- 常见痛点：{', '.join(profile.pain_points) if profile.pain_points else '未记录'}

回答风格要求：{answer_style}""",
        ]

        # 注入相关历史经验
        if relevant_memories:
            memory_section = ["【相关历史经验（来自记忆库）】"]
            for i, mem in enumerate(relevant_memories, 1):
                memory_section.append(
                    f"\n经验{i}（{mem.created_at[:10]}，质量:{mem.quality_score:.1f}）："
                    f"\n  问题：{mem.question}"
                    f"\n  解决方案摘要：{mem.answer[:200]}..."
                )
            memory_section.append("\n注意：以上为历史经验，仅供参考，请结合当前具体情况判断适用性")
            system_parts.append("\n".join(memory_section))

        # 注入领域知识
        if knowledge_items:
            knowledge_section = ["【相关专业知识（来自知识库）】"]
            for topic, content in knowledge_items:
                knowledge_section.append(f"\n{topic}：\n{content}")
            system_parts.append("\n".join(knowledge_section))

        return "\n\n".join(system_parts)

    # ── 回答生成 ──────────────────────────────────────────────────────────────

    def _generate_answer(self, user_question: str, system_prompt: str) -> str:
        """生成回答，注入短期记忆（最近N轮对话）"""
        messages = [SystemMessage(content=system_prompt)]

        # 短期记忆：最近 WINDOW_SIZE 轮
        for turn in self.short_term[-self.WINDOW_SIZE:]:
            if turn.role == "user":
                messages.append(HumanMessage(content=turn.content))
            else:
                messages.append(AIMessage(content=turn.content))

        messages.append(HumanMessage(content=user_question))
        return self.llm.invoke(messages).content

    # ── 记忆存储 ──────────────────────────────────────────────────────────────

    def _extract_tags(self, question: str, answer: str) -> list[str]:
        """从 Q&A 中提取主题标签"""
        # ── [旧] 规则引导版：预设关键词白名单，只能匹配列表中的技术词 ──
        # 缺点：列表需手动维护；新技术词（如 Rust、eBPF）不在列表中就无法打标；
        #       无法理解语义，无法区分"Redis 锁"和"Redis 缓存"
        # tech_keywords = [
        #     "MySQL", "PostgreSQL", "Redis", "Kafka", "Kubernetes", "Docker",
        #     "Python", "Java", "Go", "API", "HTTP", "gRPC", "SQL", "索引",
        #     "缓存", "并发", "锁", "事务", "超时", "熔断", "监控", "告警",
        #     "部署", "容器", "集群", "网络", "安全", "认证", "性能", "内存",
        # ]
        # text = (question + " " + answer).lower()
        # return [kw for kw in tech_keywords if kw.lower() in text][:5]

        # ── [新] 动态推理版：使用 LLM 从问答内容中语义提取核心技术标签 ──
        # 核心改变：不依赖预设词表，LLM 自主理解问答主题并提取最有代表性的技术标签，
        # 可识别任意技术词汇，且能区分同一技术的不同使用场景（如"Redis 缓存穿透" vs "Redis 集群"）
        try:
            response = self.llm.invoke([
                SystemMessage(content=(
                    "从以下技术问答中提取3-5个核心技术主题标签，用英文逗号分隔，"
                    "只输出标签本身，不要编号，不要解释。"
                    "标签应尽量具体（如'Redis 缓存击穿'优于'Redis'）。"
                )),
                HumanMessage(content=f"问题：{question}\n\n回答摘要：{answer[:300]}"),
            ])
            tags = [t.strip() for t in response.content.split(",") if t.strip()]
            return tags[:5]
        except Exception:
            return []

    def _score_answer_quality(self, question: str, answer: str) -> float:
        """评估答案质量（简化版，生产中可用 LLM 评估）"""
        score = 0.5  # 基础分
        if len(answer) > 100:
            score += 0.1
        if len(answer) > 300:
            score += 0.1
        if any(kw in answer for kw in ["步骤", "建议", "注意", "原因", "解决"]):
            score += 0.1
        if "```" in answer or "示例" in answer:
            score += 0.1
        if "不确定" in answer or "可能" in answer[:50]:
            score -= 0.1
        return min(1.0, max(0.0, score))

    def _store_to_episodic_memory(self, question: str, answer: str, tags: list[str] | None = None):
        """将本次 Q&A 存入情节记忆"""
        memory_id = hashlib.md5(f"{question}{time.time()}".encode()).hexdigest()[:12]
        memory = EpisodicMemory(
            memory_id=memory_id,
            question=question,
            answer=answer,
            # [新] 接受外部传入的 tags（由 chat() 统一提取），避免重复调用 LLM
            tags=tags if tags is not None else self._extract_tags(question, answer),
            quality_score=self._score_answer_quality(question, answer),
            created_at=datetime.now().isoformat(),
            user_id=self.user_id,
        )
        self.episodic_store.add(memory)

    # ── 用户画像更新 ──────────────────────────────────────────────────────────

    def _update_user_profile(self, question: str, learned_techs: list[str] | None = None):
        """根据对话更新用户画像（技术栈、痛点）"""
        profile = self.user_profile
        profile.interaction_count += 1
        profile.last_seen = datetime.now().strftime("%Y-%m-%d")

        # ── [旧] 规则引导版：硬编码技术词集合，只识别列表中的技术 ──
        # 缺点：覆盖范围有限，无法学习列表之外的技术词
        # tech_keywords = {"Python", "Java", "Go", "Kubernetes", "MySQL", "Redis", "Kafka"}
        # for tech in tech_keywords:
        #     if tech.lower() in question.lower() and tech not in profile.known_stack:
        #         profile.known_stack.append(tech)

        # ── [新] 动态推理版：复用 LLM 提取的标签，从中学习用户技术栈 ──
        # 核心改变：不依赖预设词表，直接将 _extract_tags 已提取的标签用于画像更新，
        # 复用同一次 LLM 调用的结果，不产生额外 API 开销
        if learned_techs:
            for tech in learned_techs:
                if tech not in profile.known_stack:
                    profile.known_stack.append(tech)

    # ── 主对话接口 ────────────────────────────────────────────────────────────

    def chat(self, user_question: str) -> str:
        """处理一条用户消息，返回增强记忆后的回答"""
        print(f"\n{'─'*50}")
        print(f"[用户/{self.user_profile.name}] {user_question}")

        # Step 1: 检索记忆
        relevant_memories = self._retrieve_relevant_memories(user_question)
        knowledge_items = self._retrieve_knowledge(user_question)

        memory_summary = ""
        if relevant_memories:
            memory_summary = f"检索到 {len(relevant_memories)} 条历史经验"
        if knowledge_items:
            memory_summary += f"，{len(knowledge_items)} 条知识库条目"
        if memory_summary:
            print(f"[Memory] {memory_summary}")

        # Step 2: 构建上下文
        system_prompt = self._build_system_prompt(relevant_memories, knowledge_items)

        # Step 3: 生成回答
        answer = self._generate_answer(user_question, system_prompt)

        # Step 4: 更新记忆
        # [新] 提取一次标签，同时供情节记忆存储和用户画像更新使用，避免重复 LLM 调用
        tags = self._extract_tags(user_question, answer)
        self._store_to_episodic_memory(user_question, answer, tags=tags)
        self.short_term.append(ConversationTurn("user", user_question, datetime.now().isoformat()))
        self.short_term.append(ConversationTurn("assistant", answer, datetime.now().isoformat()))
        self._update_user_profile(user_question, learned_techs=tags)

        print(f"[Assistant] {answer}\n")
        print(f"[Memory] 情节记忆库：{len(self.episodic_store)} 条  短期记忆：{len(self.short_term)//2} 轮")
        return answer

    def memory_stats(self) -> dict:
        """返回记忆系统统计信息"""
        return {
            "user": self.user_profile.name,
            "episodic_memories": len(self.episodic_store),
            "short_term_turns": len(self.short_term) // 2,
            "known_stack": self.user_profile.known_stack,
            "interaction_count": self.user_profile.interaction_count,
        }


# ══════════════════════════════════════════════════════════════════════════════
# 演示入口
# ══════════════════════════════════════════════════════════════════════════════

def main():
    print("=" * 60)
    print("  Memory-Augmented Agent：企业知识助手")
    print("  记忆层：短期对话 + 情节经验 + 语义知识 + 用户画像")
    print("=" * 60)

    # 创建有历史记录的 SRE 用户
    agent = MemoryAugmentedAgent(user_id="zhang_wei")
    print(f"\n用户：{agent.user_profile.name}（{agent.user_profile.role} / {agent.user_profile.tech_level}）")
    print(f"熟悉技术：{agent.user_profile.known_stack}")

    # 模拟多轮对话（包含跨轮上下文引用）
    conversations = [
        # 第一轮：技术问题，会检索语义知识库
        "我们 PostgreSQL 主库连接数经常打满，现在是 200 个连接，QPS 大概在 3000，应该怎么优化？",

        # 第二轮：引用上一轮（短期记忆）
        "你刚才提到 PgBouncer，它部署复杂吗？和我们现有的 K8s 集群怎么集成？",

        # 第三轮：新话题，会存入情节记忆
        "Redis 集群有一个节点的内存使用率到了 90%，现在怎么处理比较好？",

        # 第四轮：与第一轮相似的问题，应该会检索到第一轮的记忆
        "MySQL 也遇到了连接数问题，跟 PostgreSQL 的处理思路一样吗？",
    ]

    for question in conversations:
        agent.chat(question)

    print("\n" + "=" * 60)
    print("记忆系统统计：")
    stats = agent.memory_stats()
    for k, v in stats.items():
        print(f"  {k}: {v}")
    print("=" * 60)


if __name__ == "__main__":
    main()
