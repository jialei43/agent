"""
MCP + LangChain 集成实战：把 MCP Server 的工具接入 LangChain Agent
场景：LangChain Agent 通过 MCP 协议调用外部工具服务，实现工具与模型的解耦

核心模式：
  MCP Server（工具提供方）→ MCP Client（工具代理）→ LangChain Tool → LangChain Agent → LLM

运行前提：先启动 MCP Server（任意一种 transport）
  python 01_stdio_mcp_server.py  （stdio，不需要提前启动）
  或 python 03_streamable_http_server.py  （Streamable HTTP）

依赖：pip install "mcp[cli]" langchain langchain-openai python-dotenv
"""

import asyncio  # 异步
import os  # 环境变量
import json  # JSON
from typing import Any  # 类型
from pathlib import Path  # 路径
import sys  # 系统
from dotenv import load_dotenv, find_dotenv  # 环境变量

# MCP 客户端相关
from mcp import ClientSession, types  # MCP 会话和类型
from mcp.client.stdio import stdio_client  # stdio 客户端
from mcp.client.streamable_http import streamablehttp_client  # Streamable HTTP 客户端

# LangChain 相关
from langchain_core.tools import tool, BaseTool  # LangChain 工具基类
from langchain_core.messages import HumanMessage  # 消息类型
from langchain_openai import ChatOpenAI  # LLM

load_dotenv(find_dotenv())  # 加载 .env


# ══════════════════════════════════════════════════════════════════════════════
# 核心：把 MCP Session 的工具动态转换为 LangChain Tool
# ══════════════════════════════════════════════════════════════════════════════

class MCPToolWrapper(BaseTool):
    """
    MCP 工具适配器：把单个 MCP Tool 包装成 LangChain BaseTool。
    LangChain Agent 会像使用普通 Tool 一样调用它，内部通过 MCP 协议转发。
    """
    name: str  # LangChain Tool 名称（来自 MCP Tool）
    description: str  # LangChain Tool 描述（来自 MCP Tool，LLM 靠这个决定是否调用）
    session: Any  # MCP ClientSession（用于转发调用，Any 绕过 Pydantic 序列化限制）
    mcp_tool_name: str  # 实际的 MCP 工具名（可能和 LangChain name 不同）

    class Config:
        arbitrary_types_allowed = True  # 允许 session 等非 Pydantic 类型字段

    def _run(self, **kwargs) -> str:
        """同步调用（LangChain 默认接口），内部转成异步执行"""
        return asyncio.get_event_loop().run_until_complete(self._arun(**kwargs))

    async def _arun(self, **kwargs) -> str:
        """
        异步调用：通过 MCP Session 调用远程工具，返回文本结果。
        kwargs 就是 LLM 填好的参数，直接转发给 MCP Server。
        """
        result = await self.session.call_tool(self.mcp_tool_name, kwargs)  # 转发 MCP 调用
        # 把 MCP 结果（TextContent 列表）拼接成字符串返回给 LangChain
        texts = []
        for content in result.content:
            if isinstance(content, types.TextContent):
                texts.append(content.text)  # 提取文本内容
        return "\n".join(texts) if texts else "（无结果）"


async def load_mcp_tools_as_langchain(session: ClientSession) -> list[MCPToolWrapper]:
    """
    从 MCP Session 加载所有工具，转换为 LangChain Tool 列表。
    调用一次即可，之后 LangChain Agent 直接使用返回的 tool 列表。
    """
    await session.initialize()  # 先完成 MCP 握手
    tools_result = await session.list_tools()  # 获取工具列表

    lc_tools = []
    for mcp_tool in tools_result.tools:
        # 从 JSON Schema 生成 LangChain 可读的参数描述
        props = mcp_tool.inputSchema.get("properties", {})
        required = mcp_tool.inputSchema.get("required", [])
        param_desc = []
        for param_name, param_schema in props.items():
            is_required = "（必填）" if param_name in required else "（可选）"
            desc = param_schema.get("description", "")
            param_desc.append(f"{param_name}{is_required}：{desc}")

        # 构造完整的工具描述（LLM 靠这个决定是否调用以及如何填参数）
        full_description = mcp_tool.description
        if param_desc:
            full_description += f"\n参数：{'; '.join(param_desc)}"

        wrapper = MCPToolWrapper(
            name=mcp_tool.name,  # 工具名
            description=full_description,  # 增强后的描述
            session=session,  # MCP 会话（工具调用时使用）
            mcp_tool_name=mcp_tool.name,  # MCP 端的工具名
        )
        lc_tools.append(wrapper)
        print(f"  ✅ 已加载 MCP 工具 → LangChain Tool：{mcp_tool.name}")

    return lc_tools


# ══════════════════════════════════════════════════════════════════════════════
# 演示一：stdio MCP Server + LangChain Agent
# ══════════════════════════════════════════════════════════════════════════════

async def demo_stdio_with_langchain():
    """使用 stdio transport 连接本地 MCP Server，把工具接入 LangChain Agent"""
    print("\n" + "═" * 60)
    print("  演示一：stdio MCP → LangChain Agent")
    print("═" * 60)

    server_script = Path(__file__).parent / "01_stdio_mcp_server.py"  # stdio Server 路径

    async with stdio_client(sys.executable, [str(server_script)]) as (read, write):
        async with ClientSession(read, write) as session:
            # ── Step 1：加载 MCP 工具为 LangChain Tools ──────────
            print("\n正在从 MCP Server 加载工具...")
            lc_tools = await load_mcp_tools_as_langchain(session)
            print(f"共加载 {len(lc_tools)} 个工具")

            # ── Step 2：创建绑定工具的 LLM ───────────────────────
            llm = ChatOpenAI(
                model="qwen-plus",
                api_key=os.getenv("DASHSCOPE_API_KEY"),
                base_url="https://dashscope.aliyuncs.com/compatible-mode/v1",
            )
            llm_with_tools = llm.bind_tools(lc_tools)  # 把 MCP 工具绑定到 LLM

            # ── Step 3：运行 Agent 对话 ──────────────────────────
            questions = [
                "现在几点了？",
                "帮我查看当前目录下有哪些文件？",
                "帮我把「MCP测试」写入 /tmp/langchain_mcp.txt，然后再读出来确认一下",
            ]

            for question in questions:
                print(f"\n{'─'*40}")
                print(f"用户：{question}")
                messages = [HumanMessage(content=question)]

                # 简单 Agent 循环（详细版见 function_call 目录）
                for _ in range(5):  # 最多 5 轮
                    response = llm_with_tools.invoke(messages)
                    messages.append(response)

                    if not response.tool_calls:  # 无工具调用，给出最终答案
                        print(f"Agent：{response.content}")
                        break

                    # 有工具调用，执行并把结果加入历史
                    from langchain_core.messages import ToolMessage
                    for tc in response.tool_calls:
                        tool_obj = next((t for t in lc_tools if t.name == tc["name"]), None)
                        if tool_obj:
                            result = await tool_obj._arun(**tc["args"])  # 异步调用 MCP 工具
                            print(f"  [工具] {tc['name']}({tc['args']}) → {result[:100]}...")
                            messages.append(ToolMessage(content=result, tool_call_id=tc["id"]))


# ══════════════════════════════════════════════════════════════════════════════
# 演示二：Streamable HTTP MCP Server + LangChain Agent
# ══════════════════════════════════════════════════════════════════════════════

async def demo_streamable_http_with_langchain():
    """使用 Streamable HTTP 连接远程 MCP Server，接入 LangChain Agent"""
    print("\n" + "═" * 60)
    print("  演示二：Streamable HTTP MCP → LangChain Agent")
    print("═" * 60)
    print("前提：先运行 python 03_streamable_http_server.py")

    try:
        async with streamablehttp_client("http://localhost:8002/mcp") as (read, write, _):
            async with ClientSession(read, write) as session:
                print("\n正在从 Streamable HTTP MCP Server 加载工具...")
                lc_tools = await load_mcp_tools_as_langchain(session)
                print(f"共加载 {len(lc_tools)} 个工具")

                llm = ChatOpenAI(
                    model="qwen-plus",
                    api_key=os.getenv("DASHSCOPE_API_KEY"),
                    base_url="https://dashscope.aliyuncs.com/compatible-mode/v1",
                )
                llm_with_tools = llm.bind_tools(lc_tools)

                # 测试企业级工具
                print(f"\n{'─'*40}")
                question = "帮我查一下今年第一季度华东地区的营收，并计算一下 KPI 完成率（目标是150万）"
                print(f"用户：{question}")
                messages = [HumanMessage(content=question)]

                from langchain_core.messages import ToolMessage
                for _ in range(5):
                    response = llm_with_tools.invoke(messages)
                    messages.append(response)
                    if not response.tool_calls:
                        print(f"Agent：{response.content}")
                        break
                    for tc in response.tool_calls:
                        tool_obj = next((t for t in lc_tools if t.name == tc["name"]), None)
                        if tool_obj:
                            result = await tool_obj._arun(**tc["args"])
                            print(f"  [工具] {tc['name']}({tc['args']}) →")
                            print(f"         {result[:200]}")
                            messages.append(ToolMessage(content=result, tool_call_id=tc["id"]))

    except ConnectionRefusedError:
        print("⚠️  Server 未启动，跳过演示二")


# ══════════════════════════════════════════════════════════════════════════════
# 演示三：动态切换 MCP Server（展示解耦价值）
# ══════════════════════════════════════════════════════════════════════════════

async def demo_hot_switch():
    """
    展示 MCP 解耦价值：同一个 LangChain Agent，通过修改 MCP Server 配置，
    无需改业务代码即可切换工具集（本地工具 vs 企业工具）。
    """
    print("\n" + "═" * 60)
    print("  演示三：动态切换 MCP Server（解耦价值展示）")
    print("═" * 60)

    llm = ChatOpenAI(
        model="qwen-plus",
        api_key=os.getenv("DASHSCOPE_API_KEY"),
        base_url="https://dashscope.aliyuncs.com/compatible-mode/v1",
    )

    async def run_with_server(session: ClientSession, question: str, server_label: str):
        """通用的 Agent 运行逻辑，与具体 Server 无关"""
        tools = await load_mcp_tools_as_langchain(session)
        llm_with_tools = llm.bind_tools(tools)
        messages = [HumanMessage(content=question)]
        from langchain_core.messages import ToolMessage

        print(f"\n[{server_label}] 用户：{question}")
        for _ in range(3):
            response = llm_with_tools.invoke(messages)
            messages.append(response)
            if not response.tool_calls:
                print(f"[{server_label}] Agent：{response.content}")
                break
            for tc in response.tool_calls:
                tool_obj = next((t for t in tools if t.name == tc["name"]), None)
                if tool_obj:
                    result = await tool_obj._arun(**tc["args"])
                    messages.append(ToolMessage(content=result[:200], tool_call_id=tc["id"]))

    server_script = Path(__file__).parent / "01_stdio_mcp_server.py"

    # 同一套业务逻辑，连接不同的 MCP Server
    print("\n→ 连接到本地工具 Server（stdio）")
    async with stdio_client(sys.executable, [str(server_script)]) as (r, w):
        async with ClientSession(r, w) as session:
            await run_with_server(session, "当前目录有哪些文件？", "本地 Server")

    print("\n→ 连接到企业工具 Server（Streamable HTTP，如已启动）")
    try:
        async with streamablehttp_client("http://localhost:8002/mcp") as (r, w, _):
            async with ClientSession(r, w) as session:
                await run_with_server(session, "查一下 Server 当前的运行指标", "企业 Server")
    except ConnectionRefusedError:
        print("  [企业 Server] 未启动，跳过")

    print("\n结论：同一个 Agent 框架，只需切换 MCP Server 地址，工具集自动更新")


# ══════════════════════════════════════════════════════════════════════════════
# 主入口
# ══════════════════════════════════════════════════════════════════════════════

async def main():
    print("=" * 60)
    print("  MCP + LangChain 集成实战演示")
    print("=" * 60)

    await demo_stdio_with_langchain()          # 演示一：stdio + LangChain
    await demo_streamable_http_with_langchain() # 演示二：Streamable HTTP + LangChain
    await demo_hot_switch()                     # 演示三：动态切换 Server

    print("\n" + "=" * 60)
    print("  全部演示完成")
    print("=" * 60)


if __name__ == "__main__":
    asyncio.run(main())
