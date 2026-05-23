"""
实战 MCP Client - 覆盖全部四种调用方式

方式一 stdio          ：自动启动 server 子进程，无需提前启动
方式二 SSE            ：先运行 python server.py sse
方式三 Streamable HTTP：先运行 python server.py http
方式四 MultiServer    ：同时连接多个 MCP Server，工具自动合并

核心链路：
  transport → ClientSession → load_mcp_tools
    → create_tool_calling_agent → AgentExecutor → 交互式对话
"""

import asyncio
import os
import sys
from pathlib import Path

from dotenv import load_dotenv, find_dotenv
from mcp import ClientSession
from mcp.client.stdio import stdio_client, StdioServerParameters
from mcp.client.sse import sse_client
from mcp.client.streamable_http import streamablehttp_client
from langchain_mcp_adapters.tools import load_mcp_tools
from langchain_mcp_adapters.client import MultiServerMCPClient
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain.agents import create_tool_calling_agent, AgentExecutor

load_dotenv(find_dotenv())

SERVER_PATH = Path(__file__).parent / "server.py"
SSE_URL     = "http://localhost:8001/sse"
HTTP_URL    = "http://localhost:8002/mcp"


# ══════════════════════════════════════════════════════════════════════════════
# 公共工厂
# ══════════════════════════════════════════════════════════════════════════════

def _make_llm() -> ChatOpenAI:
    return ChatOpenAI(
        model="qwen-plus",
        api_key=os.getenv("DASHSCOPE_API_KEY"),
        base_url="https://dashscope.aliyuncs.com/compatible-mode/v1",
        temperature=0,
    )


def _make_agent(tools) -> AgentExecutor:
    prompt = ChatPromptTemplate.from_messages([
        ("system", "你是一个乐于助人的助手，优先使用工具回答问题。"),
        ("human", "{input}"),
        ("placeholder", "{agent_scratchpad}"),
    ])
    return AgentExecutor(
        agent=create_tool_calling_agent(_make_llm(), tools, prompt),
        tools=tools,
        verbose=True,
    )


async def _chat_loop(agent: AgentExecutor, label: str):
    """交互式对话循环，输入 quit 退出"""
    print(f"\n[{label}] 已就绪，输入 quit 退出")
    while True:
        query = input("Query: ").strip()
        if not query:
            continue
        if query.lower() == "quit":
            break
        result = await agent.ainvoke({"input": query})
        print(f"Answer: {result['output']}\n")


# ══════════════════════════════════════════════════════════════════════════════
# 方式一：stdio（Client 自动启动 server 子进程）
# ══════════════════════════════════════════════════════════════════════════════

async def run_stdio():
    """stdio transport：无需手动启动 server，Client 自动管理子进程生命周期"""
    params = StdioServerParameters(command=sys.executable, args=[str(SERVER_PATH)])
    async with stdio_client(params) as (read, write):
        async with ClientSession(read, write) as session:
            await session.initialize()
            tools = await load_mcp_tools(session)
            print(f"已加载 {len(tools)} 个工具：{[t.name for t in tools]}")
            await _chat_loop(_make_agent(tools), "stdio")


# ══════════════════════════════════════════════════════════════════════════════
# 方式二：SSE（连接已启动的 SSE server）
# ══════════════════════════════════════════════════════════════════════════════

async def run_sse():
    """SSE transport：前提是已运行 python server.py sse"""
    try:
        async with sse_client(SSE_URL) as (read, write):
            async with ClientSession(read, write) as session:
                await session.initialize()
                tools = await load_mcp_tools(session)
                print(f"已加载 {len(tools)} 个工具：{[t.name for t in tools]}")
                await _chat_loop(_make_agent(tools), "SSE")
    except Exception as e:
        print(f"SSE 连接失败：{e}")
        print("请先运行：python server.py sse")


# ══════════════════════════════════════════════════════════════════════════════
# 方式三：Streamable HTTP（连接已启动的 HTTP server）
# ══════════════════════════════════════════════════════════════════════════════

async def run_http():
    """Streamable HTTP transport：前提是已运行 python server.py http"""
    try:
        async with streamablehttp_client(HTTP_URL) as (read, write, _):
            async with ClientSession(read, write) as session:
                await session.initialize()
                tools = await load_mcp_tools(session)
                print(f"已加载 {len(tools)} 个工具：{[t.name for t in tools]}")
                await _chat_loop(_make_agent(tools), "Streamable HTTP")
    except Exception as e:
        print(f"HTTP 连接失败：{e}")
        print("请先运行：python server.py http")


# ══════════════════════════════════════════════════════════════════════════════
# 方式四：MultiServerMCPClient（同时连接多个 Server，工具自动合并）
# ══════════════════════════════════════════════════════════════════════════════

async def run_multi():
    """
    MultiServerMCPClient：声明式配置多个 Server，工具自动聚合。
    每次工具调用时自动建立新 session，无需手动管理连接生命周期。
    可同时混用 stdio / SSE / Streamable HTTP 三种 transport。
    """
    client = MultiServerMCPClient({
        "local-tools": {
            "command": sys.executable,
            "args": [str(SERVER_PATH)],
            "transport": "stdio",
        },
        # 如需同时连接其他 Server，在此继续添加：
        # "remote-tools": {
        #     "url": "http://other-server/mcp",
        #     "transport": "streamable_http",
        # },
    })
    tools = await client.get_tools()
    print(f"已从所有 Server 加载 {len(tools)} 个工具：{[t.name for t in tools]}")
    await _chat_loop(_make_agent(tools), "MultiServer")


# ══════════════════════════════════════════════════════════════════════════════
# 主入口
# ══════════════════════════════════════════════════════════════════════════════

_MODES = {
    "1": ("stdio          （自动启动 server，无需提前准备）", run_stdio),
    "2": ("SSE            （需提前运行 python server.py sse）", run_sse),
    "3": ("Streamable HTTP（需提前运行 python server.py http）", run_http),
    "4": ("MultiServer    （多 Server 聚合，声明式配置）", run_multi),
}


async def main():
    print("=" * 56)
    print("  实战 MCP Client")
    print("=" * 56)
    for k, (name, _) in _MODES.items():
        print(f"  {k}.  {name}")
    print()
    choice = input("选择调用方式（1-4）：").strip()
    if choice in _MODES:
        await _MODES[choice][1]()
    else:
        print("无效选择，退出")


if __name__ == "__main__":
    asyncio.run(main())
