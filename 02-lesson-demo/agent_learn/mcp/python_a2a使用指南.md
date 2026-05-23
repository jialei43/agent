# python-a2a 使用指南

> **python-a2a** 是 Google A2A（Agent-to-Agent）协议的 Python 实现库。
> A2A 与 MCP 是互补关系：MCP 解决"Agent 连接工具"，A2A 解决"Agent 连接 Agent"。

---

## 一、A2A 与 MCP 的关系

```
                    ┌─────────────────────────┐
                    │       AI 应用            │
                    └────────────┬────────────┘
                                 │
              ┌──────────────────┼──────────────────┐
              │                  │                  │
              ▼                  ▼                  ▼
       ┌─────────────┐   ┌─────────────┐   ┌─────────────┐
       │  MCP Server │   │  A2A Agent  │   │  A2A Agent  │
       │  （工具层） │   │  （子 Agent）│   │  （子 Agent）│
       │             │   │             │   │             │
       │ 文件/数据库  │   │ 研究 Agent  │   │ 执行 Agent  │
       └─────────────┘   └──────┬──────┘   └─────────────┘
                                │
                         ┌──────▼──────┐
                         │  MCP Server │
                         │  （工具层） │
                         └─────────────┘

MCP：纵向 —— Agent 调用工具（函数、数据库、API）
A2A：横向 —— Agent 调用 Agent（多智能体协作）
```

---

## 二、安装

```bash
# 基础安装
pip install python-a2a

# 含 LangChain 集成（推荐）
pip install "python-a2a[langchain]"

# 含服务端支持
pip install "python-a2a[server]"

# 全部依赖
pip install "python-a2a[all]"
```

---

## 三、A2A 核心概念

### 3.1 Agent Card（智能体名片）

每个 A2A Agent 必须在 `/.well-known/agent.json` 暴露自己的能力描述，相当于 Agent 的"自我介绍"：

```json
{
  "name": "研究助手 Agent",
  "description": "专门负责网络搜索和信息整理的智能体",
  "url": "http://research-agent.internal:8001",
  "version": "1.0.0",
  "capabilities": {
    "streaming": true,
    "pushNotifications": false
  },
  "skills": [
    {
      "id": "web_research",
      "name": "网络调研",
      "description": "搜索互联网获取最新信息",
      "tags": ["搜索", "调研", "信息收集"],
      "examples": ["帮我调研一下 MCP 协议的最新进展"]
    }
  ]
}
```

### 3.2 Task（任务）

A2A 的工作单元，包含完整的请求-响应生命周期：

```
Task 状态流转：
submitted → working → (input-required) → working → completed
                                                  → failed
                                                  → canceled
```

```python
from python_a2a import Task, TaskState, Message, TextPart

# Task 结构示意
task = {
    "id": "task_001",                    # 唯一任务 ID
    "status": {"state": "working"},      # 当前状态
    "history": [...],                    # 完整消息历史
    "artifacts": [...]                   # 任务产出物
}
```

### 3.3 Message 与 Part

```python
from python_a2a import Message, MessageRole, TextPart, FilePart, DataPart

# 文本消息
text_msg = Message(
    role=MessageRole.USER,              # user 或 agent
    parts=[TextPart(text="帮我调研 MCP 协议")]  # 消息内容
)

# 文件消息
file_msg = Message(
    role=MessageRole.AGENT,
    parts=[
        FilePart(
            file={
                "name": "report.pdf",
                "mimeType": "application/pdf",
                "bytes": "<base64编码内容>"
            }
        )
    ]
)

# 结构化数据消息
data_msg = Message(
    role=MessageRole.AGENT,
    parts=[DataPart(data={"result": "调研完成", "sources": [...]})]
)
```

---

## 四、创建 A2A Agent（服务端）

### 4.1 最简单的 Agent

```python
# simple_agent.py
from python_a2a import A2AServer, Task, Message, MessageRole, TextPart, TaskState
import uvicorn

class SimpleResearchAgent(A2AServer):
    """最简单的 A2A Agent：接收任务，处理，返回结果"""

    def handle_task(self, task: Task) -> Task:
        """
        A2AServer 的核心方法，必须实现。
        接收 Task 对象，处理后返回更新后的 Task。
        """
        # 取出用户的最后一条消息
        last_message = task.history[-1] if task.history else None
        if not last_message:
            task.status.state = TaskState.FAILED
            return task

        # 提取文本内容
        user_text = ""
        for part in last_message.parts:
            if hasattr(part, "text"):      # TextPart 有 text 属性
                user_text += part.text

        # 处理任务（这里模拟调研逻辑）
        result_text = f"已完成调研：{user_text}\n\n调研结果：这是一份关于该主题的详细报告..."

        # 构造 Agent 的回复消息
        reply = Message(
            role=MessageRole.AGENT,
            parts=[TextPart(text=result_text)]
        )

        # 把回复加入 Task 历史，并标记完成
        task.history.append(reply)
        task.status.state = TaskState.COMPLETED
        return task


if __name__ == "__main__":
    agent = SimpleResearchAgent()
    # 启动 HTTP 服务，自动注册 /.well-known/agent.json 和 /tasks/send 等端点
    uvicorn.run(agent.app, host="0.0.0.0", port=8001)
```

### 4.2 带 Agent Card 的完整 Agent

```python
# research_agent.py
from python_a2a import (
    A2AServer, AgentCard, AgentCapabilities, AgentSkill,
    Task, Message, MessageRole, TextPart, TaskState
)
from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage
import os

class ResearchAgent(A2AServer):
    """带完整元数据的研究 Agent，使用 LLM 处理任务"""

    def __init__(self):
        # 定义 Agent Card（/.well-known/agent.json 的内容）
        agent_card = AgentCard(
            name="研究助手 Agent",                       # Agent 名称
            description="专注于信息检索和内容整理的智能体",  # 描述
            url="http://localhost:8001",               # 本服务的访问地址
            version="1.0.0",                           # 版本号
            capabilities=AgentCapabilities(
                streaming=False,          # 是否支持流式输出
                pushNotifications=False,  # 是否支持主动推送
            ),
            skills=[
                AgentSkill(
                    id="research",
                    name="信息调研",
                    description="搜索和整理特定主题的信息",
                    tags=["搜索", "调研", "信息"],
                    examples=["调研 MCP 协议的三种通信方式"]
                ),
                AgentSkill(
                    id="summarize",
                    name="内容摘要",
                    description="对长文本进行摘要提炼",
                    tags=["摘要", "总结"],
                    examples=["总结这篇文章的核心观点"]
                )
            ]
        )
        super().__init__(agent_card=agent_card)  # 传给父类

        # 初始化 LLM（用于真实的任务处理）
        self.llm = ChatOpenAI(
            model="qwen-plus",
            api_key=os.getenv("DASHSCOPE_API_KEY"),
            base_url="https://dashscope.aliyuncs.com/compatible-mode/v1",
        )

    def handle_task(self, task: Task) -> Task:
        """处理传入的 Task，调用 LLM 完成任务"""
        # 把 A2A Task 的消息历史转换成 LLM 可用的消息格式
        user_input = self._extract_user_input(task)
        if not user_input:
            task.status.state = TaskState.FAILED
            return task

        # 调用 LLM 处理
        response = self.llm.invoke([HumanMessage(content=user_input)])
        result = response.content

        # 把 LLM 结果写回 Task
        task.history.append(Message(
            role=MessageRole.AGENT,
            parts=[TextPart(text=result)]
        ))
        task.status.state = TaskState.COMPLETED
        return task

    def _extract_user_input(self, task: Task) -> str:
        """从 Task 的历史消息里提取用户输入文本"""
        texts = []
        for msg in task.history:
            if msg.role == MessageRole.USER:  # 只取用户消息
                for part in msg.parts:
                    if hasattr(part, "text"):  # TextPart
                        texts.append(part.text)
        return "\n".join(texts)  # 合并多条消息


if __name__ == "__main__":
    import uvicorn
    agent = ResearchAgent()
    uvicorn.run(agent.app, host="0.0.0.0", port=8001)
```

---

## 五、调用 A2A Agent（客户端）

### 5.1 直接调用单个 Agent

```python
# client_basic.py
from python_a2a import A2AClient, Message, MessageRole, TextPart

async def call_research_agent():
    """直接调用一个 A2A Agent"""

    # 创建客户端，指向 Agent 的地址
    client = A2AClient(url="http://localhost:8001")

    # 构造消息
    message = Message(
        role=MessageRole.USER,
        parts=[TextPart(text="帮我调研 MCP 协议的三种通信方式，整理成简报")]
    )

    # 发送任务并等待结果
    task = await client.send_task(
        message=message,
        task_id="my_task_001"   # 可选，不传则自动生成 UUID
    )

    # 处理结果
    if task.status.state.value == "completed":
        # 取最后一条 Agent 消息（即结果）
        last_agent_msg = next(
            (m for m in reversed(task.history) if m.role == MessageRole.AGENT),
            None
        )
        if last_agent_msg:
            for part in last_agent_msg.parts:
                if hasattr(part, "text"):
                    print(f"Agent 回复：\n{part.text}")
    else:
        print(f"任务失败，状态：{task.status.state}")


if __name__ == "__main__":
    import asyncio
    asyncio.run(call_research_agent())
```

### 5.2 发现并自动调用 Agent（通过 Agent Card）

```python
# client_discovery.py
from python_a2a import A2AClient

async def discover_and_call():
    """先获取 Agent Card，了解能力，再决定是否调用"""

    client = A2AClient(url="http://localhost:8001")

    # 获取 Agent Card（自动请求 /.well-known/agent.json）
    card = await client.get_agent_card()
    print(f"Agent 名称：{card.name}")
    print(f"支持的技能：")
    for skill in card.skills:
        print(f"  - {skill.name}：{skill.description}")
        print(f"    示例：{skill.examples[0] if skill.examples else '无'}")

    # 根据 Agent Card 判断是否适合当前任务，再发送
    from python_a2a import Message, MessageRole, TextPart
    task = await client.send_task(
        message=Message(
            role=MessageRole.USER,
            parts=[TextPart(text="调研 python-a2a 包的核心功能")]
        )
    )
    print(f"\n任务状态：{task.status.state}")
```

---

## 六、多 Agent 协作（Orchestrator 模式）

这是 A2A 最核心的使用场景：**一个主 Agent 调度多个专业子 Agent 协作完成复杂任务**。

```python
# orchestrator.py
from python_a2a import A2AClient, A2AServer, Task, Message, MessageRole, TextPart, TaskState, AgentCard
import asyncio

class OrchestratorAgent(A2AServer):
    """
    编排 Agent：把复杂任务拆分，分发给专业子 Agent，汇总结果返回。

    协作流程：
        用户 ──► Orchestrator ──► 研究 Agent（调研信息）
                             └── 写作 Agent（生成报告）
                             └── 翻译 Agent（中英互译）
    """

    def __init__(self):
        card = AgentCard(
            name="编排 Agent",
            description="协调多个专业 Agent 完成复杂任务",
            url="http://localhost:8000",
            version="1.0.0"
        )
        super().__init__(agent_card=card)

        # 注册子 Agent 的客户端
        self.research_client = A2AClient(url="http://localhost:8001")  # 研究 Agent
        self.writer_client = A2AClient(url="http://localhost:8002")    # 写作 Agent

    def handle_task(self, task: Task) -> Task:
        """同步包装，内部跑异步逻辑"""
        result = asyncio.run(self._handle_async(task))
        return result

    async def _handle_async(self, task: Task) -> Task:
        """真正的异步处理逻辑"""
        user_input = self._extract_text(task)

        # ── Step 1：并发调用研究 Agent ──────────────────────────
        print(f"[Orchestrator] 调度研究 Agent 处理：{user_input[:30]}...")
        research_task = await self.research_client.send_task(
            message=Message(
                role=MessageRole.USER,
                parts=[TextPart(text=f"请调研以下主题：{user_input}")]
            )
        )

        # 提取研究结果
        research_result = self._extract_agent_reply(research_task)
        print(f"[Orchestrator] 研究 Agent 完成，结果长度：{len(research_result)} 字符")

        # ── Step 2：把研究结果交给写作 Agent 生成报告 ───────────
        print(f"[Orchestrator] 调度写作 Agent 生成报告...")
        writer_task = await self.writer_client.send_task(
            message=Message(
                role=MessageRole.USER,
                parts=[TextPart(text=f"根据以下调研结果生成一份专业报告：\n\n{research_result}")]
            )
        )

        final_report = self._extract_agent_reply(writer_task)
        print(f"[Orchestrator] 写作 Agent 完成，报告生成成功")

        # ── Step 3：汇总结果写回 Task ───────────────────────────
        task.history.append(Message(
            role=MessageRole.AGENT,
            parts=[TextPart(text=final_report)]
        ))
        task.status.state = TaskState.COMPLETED
        return task

    def _extract_text(self, task: Task) -> str:
        """提取 Task 中用户的文本输入"""
        for msg in reversed(task.history):
            if msg.role == MessageRole.USER:
                return " ".join(
                    p.text for p in msg.parts if hasattr(p, "text")
                )
        return ""

    def _extract_agent_reply(self, task: Task) -> str:
        """提取 Task 中 Agent 最后一条回复的文本"""
        for msg in reversed(task.history):
            if msg.role == MessageRole.AGENT:
                return " ".join(
                    p.text for p in msg.parts if hasattr(p, "text")
                )
        return "（无结果）"


if __name__ == "__main__":
    import uvicorn
    orchestrator = OrchestratorAgent()
    uvicorn.run(orchestrator.app, host="0.0.0.0", port=8000)
```

---

## 七、与 LangChain 集成

### 7.1 把 A2A Agent 包装成 LangChain Tool

```python
# a2a_as_langchain_tool.py
from python_a2a.langchain import A2ATool
from langchain_openai import ChatOpenAI
from langchain.agents import create_tool_calling_agent, AgentExecutor
from langchain_core.prompts import ChatPromptTemplate
import os

# 把远程 A2A Agent 包装成 LangChain Tool
research_tool = A2ATool(
    url="http://localhost:8001",         # A2A Agent 地址
    name="research_agent",              # Tool 名称（供 LLM 识别）
    description="专业的信息调研智能体，输入调研主题，返回详细的调研报告"  # LLM 根据此描述决定是否调用
)

writer_tool = A2ATool(
    url="http://localhost:8002",
    name="writer_agent",
    description="专业的写作智能体，输入素材，输出格式规范的文章或报告"
)

# 正常使用 LangChain Agent，A2A Agent 作为普通 Tool 参与
llm = ChatOpenAI(
    model="qwen-plus",
    api_key=os.getenv("DASHSCOPE_API_KEY"),
    base_url="https://dashscope.aliyuncs.com/compatible-mode/v1"
)

prompt = ChatPromptTemplate.from_messages([
    ("system", "你是一个任务编排助手，合理调用各专业 Agent 完成用户需求。"),
    ("human", "{input}"),
    ("placeholder", "{agent_scratchpad}"),
])

tools = [research_tool, writer_tool]
agent = create_tool_calling_agent(llm, tools, prompt)
executor = AgentExecutor(agent=agent, tools=tools, verbose=True)

# 运行
result = executor.invoke({
    "input": "帮我写一篇关于 MCP 协议的技术博客，需要先调研再写作"
})
print(result["output"])
```

### 7.2 把 LangChain Agent 暴露为 A2A Server

```python
# langchain_as_a2a.py
from python_a2a.langchain import LangChainAgent
from langchain_openai import ChatOpenAI
from langchain.agents import create_tool_calling_agent, AgentExecutor
from langchain_core.tools import tool
from langchain_core.prompts import ChatPromptTemplate
import os, uvicorn

# 正常创建 LangChain Agent
@tool
def search_web(query: str) -> str:
    """搜索网络获取信息"""
    return f"搜索结果：关于「{query}」的信息..."

llm = ChatOpenAI(
    model="qwen-plus",
    api_key=os.getenv("DASHSCOPE_API_KEY"),
    base_url="https://dashscope.aliyuncs.com/compatible-mode/v1"
)
prompt = ChatPromptTemplate.from_messages([
    ("system", "你是一个研究助手，善用工具完成调研任务。"),
    ("human", "{input}"),
    ("placeholder", "{agent_scratchpad}"),
])
lc_agent = AgentExecutor(
    agent=create_tool_calling_agent(llm, [search_web], prompt),
    tools=[search_web]
)

# 用 LangChainAgent 包装，使其暴露为 A2A Server
a2a_server = LangChainAgent(
    agent=lc_agent,                     # 传入 LangChain AgentExecutor
    name="LangChain 研究 Agent",        # A2A Agent 名称
    description="基于 LangChain 构建的研究智能体",
    url="http://localhost:8001"
)

if __name__ == "__main__":
    uvicorn.run(a2a_server.app, host="0.0.0.0", port=8001)
```

---

## 八、错误处理与超时

```python
from python_a2a import A2AClient, Message, MessageRole, TextPart
from python_a2a.exceptions import A2AError, TaskFailedError, ConnectionError
import asyncio

async def robust_call(url: str, text: str, timeout: int = 30) -> str:
    """带完整错误处理的 A2A 调用"""
    client = A2AClient(url=url, timeout=timeout)   # 设置超时

    try:
        task = await client.send_task(
            message=Message(
                role=MessageRole.USER,
                parts=[TextPart(text=text)]
            )
        )

        if task.status.state.value == "completed":
            # 提取结果
            for msg in reversed(task.history):
                if msg.role == MessageRole.AGENT:
                    return " ".join(p.text for p in msg.parts if hasattr(p, "text"))
            return ""

        elif task.status.state.value == "failed":
            error_info = task.status.message or "未知错误"
            raise TaskFailedError(f"任务执行失败：{error_info}")

        else:
            raise A2AError(f"意外的任务状态：{task.status.state}")

    except ConnectionError as e:
        print(f"[错误] 无法连接到 Agent：{url}，原因：{e}")
        raise

    except asyncio.TimeoutError:
        print(f"[错误] Agent 调用超时（{timeout}秒）：{url}")
        raise

    except A2AError as e:
        print(f"[错误] A2A 协议错误：{e}")
        raise
```

---

## 九、完整项目结构参考

```
my_a2a_project/
│
├── agents/
│   ├── orchestrator.py     # 编排 Agent（主入口）
│   ├── research_agent.py   # 研究 Agent
│   └── writer_agent.py     # 写作 Agent
│
├── client/
│   └── main.py             # 调用端示例
│
├── docker-compose.yml      # 多 Agent 容器编排
│
└── requirements.txt
```

```yaml
# docker-compose.yml（多 Agent 部署）
version: "3.8"
services:
  orchestrator:
    build: .
    command: python agents/orchestrator.py
    ports:
      - "8000:8000"
    environment:
      - RESEARCH_AGENT_URL=http://research:8001
      - WRITER_AGENT_URL=http://writer:8002

  research:
    build: .
    command: python agents/research_agent.py
    ports:
      - "8001:8001"
    environment:
      - DASHSCOPE_API_KEY=${DASHSCOPE_API_KEY}

  writer:
    build: .
    command: python agents/writer_agent.py
    ports:
      - "8002:8002"
    environment:
      - DASHSCOPE_API_KEY=${DASHSCOPE_API_KEY}
```

---

## 十、A2A vs MCP 对比速查

| 维度 | MCP | A2A |
|---|---|---|
| 发布方 | Anthropic（2024.11） | Google（2025.04） |
| 解决问题 | Agent ↔ 工具/数据 | Agent ↔ Agent |
| 通信方向 | 单向（Host 调用 Server） | 双向（Agent 互调） |
| 服务发现 | Host 配置文件 | /.well-known/agent.json |
| 消息格式 | JSON-RPC 2.0 | Task / Message / Part |
| 传输协议 | stdio / SSE / Streamable HTTP | HTTP（JSON） |
| 状态管理 | Session ID（Streamable HTTP） | Task ID + 状态机 |
| Python 库 | `mcp`（官方） | `python-a2a` |
| 典型场景 | 调用数据库、文件、API | 多 Agent 分工协作 |

---

## 十一、快速入门检查清单

```
□ pip install "python-a2a[all]"
□ 实现 A2AServer 子类，重写 handle_task() 方法
□ 定义 AgentCard（name、description、url、skills）
□ 用 uvicorn 启动 agent.app
□ 用 A2AClient 测试调用，检查 /.well-known/agent.json 端点
□ 集成 LangChain：用 A2ATool 把远程 Agent 当 Tool 用
□ 多 Agent：Orchestrator 持有多个子 Agent 的 A2AClient
□ 生产部署：docker-compose 编排各 Agent 服务
```
