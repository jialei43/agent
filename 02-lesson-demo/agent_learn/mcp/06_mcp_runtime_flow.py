"""
MCP 完整运行时流程：从 Host 启动到 function_call 执行
展示 MCP 和普通 function_call 的本质相同点与区别

完整链路：
  阶段一【启动】  Host 启动 → Client 连接 Server → 拉取工具列表
  阶段二【绑定】  MCP Tool → LLM Schema → llm.bind_tools()
  阶段三【请求】  用户提问 → LLM 返回 tool_calls → Client 通过 MCP 执行 → ToolMessage → 最终回答

对比普通 function_call（03_pydantic_tool.py）：
  区别仅在"执行工具"这一步：
    function_call：直接 tool_obj.invoke(args)（本地 Python 函数）
    MCP：          session.call_tool(name, args)（通过协议转发到 Server 进程/远程服务）
"""

import asyncio
import json
import os
import sys
from pathlib import Path
from datetime import datetime

from dotenv import load_dotenv, find_dotenv
from mcp import ClientSession, types
from mcp.client.stdio import stdio_client
from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage, ToolMessage, AIMessage, SystemMessage
from langchain_core.tools import BaseTool

load_dotenv(find_dotenv())


# ══════════════════════════════════════════════════════════════════════════════
# 阶段一：MCP Tool → LLM Schema 转换器
# 这是 MCP 和 function_call 的衔接层
# ══════════════════════════════════════════════════════════════════════════════

def mcp_tool_to_llm_schema(mcp_tool: types.Tool) -> dict:
    """
    把 MCP Server 返回的 Tool 对象转换成 LLM bind_tools 能识别的 JSON Schema。
    这一步让 LLM 知道"有哪些工具可用、每个工具叫什么、参数怎么填"。

    MCP Tool 结构：                    LLM Schema 结构（OpenAI function calling 格式）：
    {                                  {
      name: "get_current_time",          "type": "function",
      description: "获取时间",           "function": {
      inputSchema: {                       "name": "get_current_time",
        type: "object",                    "description": "获取时间",
        properties: {...},                 "parameters": {
        required: [...]                      "type": "object",
      }                                      "properties": {...},
    }                                        "required": [...]
                                           }
                                         }
                                       }
    """
    return {
        "type": "function",                      # LLM function call 固定写法
        "function": {
            "name": mcp_tool.name,               # 工具名（LLM 调用时返回这个名字）
            "description": mcp_tool.description, # 工具描述（LLM 根据此决定是否调用）
            "parameters": mcp_tool.inputSchema,  # 参数 Schema（MCP 和 LLM 格式一致，直接复用）
        }
    }


# ══════════════════════════════════════════════════════════════════════════════
# 核心：MCPHost 类 —— 封装完整的启动 + 运行时流程
# ══════════════════════════════════════════════════════════════════════════════

class MCPHost:
    """
    模拟真实的 MCP Host（如 Claude Desktop）行为：
      1. 启动时连接所有配置的 MCP Server
      2. 从每个 Server 拉取工具列表，转换为 LLM Schema
      3. 把所有工具绑定到 LLM
      4. 请求时走标准 function_call 流程，工具执行转发给对应 MCP Server
    """

    def __init__(self):
        self.llm = ChatOpenAI(
            model="qwen-plus",
            api_key=os.getenv("DASHSCOPE_API_KEY"),
            base_url="https://dashscope.aliyuncs.com/compatible-mode/v1",
        )
        # MCP 运行时状态（Host 生命周期内保持）
        self._sessions: dict[str, ClientSession] = {}   # server_name → session
        self._tool_schemas: list[dict] = []             # LLM 可用的工具 Schema 列表
        self._tool_server_map: dict[str, str] = {}      # tool_name → server_name（路由表）
        self._llm_with_tools = None                     # 绑定工具后的 LLM

    # ── 阶段一：启动 ──────────────────────────────────────────────────────────

    async def startup(self, server_configs: list[dict]) -> None:
        """
        Host 启动流程：
          对每个配置的 Server：
            1. 建立连接（Client → Server）
            2. 完成 MCP 握手（initialize）
            3. 拉取工具列表（tools/list）
            4. 转换为 LLM Schema，建立路由映射
          最后把所有工具 Schema 绑定到 LLM。
        """
        print("\n" + "═"*60)
        print("  阶段一：Host 启动，连接 MCP Server 并拉取工具")
        print("═"*60)

        for config in server_configs:  # 遍历所有配置的 Server
            server_name = config["name"]
            print(f"\n[{server_name}] 正在连接...")
            await self._connect_server(server_name, config)

        # ── 阶段二：工具绑定 ──────────────────────────────────────
        print("\n" + "═"*60)
        print("  阶段二：把所有 MCP 工具 Schema 绑定到 LLM")
        print("═"*60)
        self._llm_with_tools = self.llm.bind_tools(self._tool_schemas)  # 关键：绑定工具
        print(f"\n✅ LLM 已绑定 {len(self._tool_schemas)} 个工具")
        print(f"   工具列表：{[s['function']['name'] for s in self._tool_schemas]}")
        print("\n  → 此后每次请求，LLM 都携带这份工具清单，可随时决定调用")

    async def _connect_server(self, server_name: str, config: dict) -> None:
        """
        连接单个 MCP Server：握手 + 拉取工具。
        注意：真实 Host 会保持 session 不关闭，整个应用生命周期复用。
        这里为简化演示，session 在调用时临时建立（见 _call_mcp_tool）。
        """
        # 根据 transport 类型选择对应的连接方式
        if config["transport"] == "stdio":
            cmd = config["command"]
            args = config.get("args", [])
            # 临时连接只用于启动时获取工具列表
            async with stdio_client(cmd, args) as (read, write):
                async with ClientSession(read, write) as session:
                    # ── MCP 握手（必须第一步）──────────────────────
                    init = await session.initialize()
                    print(f"[{server_name}] ✅ 握手成功，Server: {init.serverInfo.name}")

                    # ── 拉取工具列表（tools/list）──────────────────
                    tools_result = await session.list_tools()
                    print(f"[{server_name}] 工具拉取完成，共 {len(tools_result.tools)} 个：")

                    for mcp_tool in tools_result.tools:
                        # 转换为 LLM Schema
                        schema = mcp_tool_to_llm_schema(mcp_tool)
                        self._tool_schemas.append(schema)               # 加入工具清单
                        self._tool_server_map[mcp_tool.name] = server_name  # 记录路由

                        params = list(mcp_tool.inputSchema.get("properties", {}).keys())
                        print(f"  • {mcp_tool.name}（参数：{params}）")
                        print(f"    描述：{mcp_tool.description[:60]}...")

        # 可在此扩展 sse / streamable_http 分支
        else:
            print(f"[{server_name}] ⚠️  transport '{config['transport']}' 暂未在此演示中实现")

    # ── 阶段三：请求处理（function_call 流程）────────────────────────────────

    async def chat(self, user_question: str, server_config: dict) -> None:
        """
        运行时请求处理，完全复用 function_call 模式：
          1. 用户问题 → LLM（带工具 Schema）
          2. LLM 返回 tool_calls
          3. 解析 tool_calls，通过 MCP Session 执行
          4. 结果包装成 ToolMessage，加入消息历史
          5. 再次调用 LLM，生成最终回答
        """
        print("\n" + "─"*60)
        print(f"  阶段三：处理请求")
        print("─"*60)
        print(f"用户：{user_question}")

        messages = [HumanMessage(content=user_question)]  # 初始化消息历史

        # 用 stdio 建立连接（真实场景：启动时建立并复用，这里每次请求新建仅为演示清晰）
        async with stdio_client(server_config["command"], server_config.get("args", [])) as (r, w):
            async with ClientSession(r, w) as session:
                await session.initialize()  # 握手（真实场景已在启动时完成）

                # ── 多轮 function_call 循环（与 03_pydantic_tool.py 完全相同）──
                for round_num in range(5):  # 最多 5 轮，防死循环
                    print(f"\n[第{round_num+1}轮] 调用 LLM...")

                    # ── Step 1：LLM 推理 ──────────────────────────
                    response = self._llm_with_tools.invoke(messages)  # 携带工具 Schema 的 LLM
                    messages.append(response)  # 记录 LLM 回复

                    finish_reason = response.response_metadata.get("finish_reason", "stop")
                    print(f"  finish_reason: {finish_reason}")
                    print(f"  tool_calls: {[tc['name'] for tc in response.tool_calls]}")

                    # ── Step 2：判断是否需要工具调用 ──────────────
                    if not response.tool_calls:  # 无工具调用 = 最终回答
                        print(f"\nLLM 最终回答：\n{response.content}")
                        return

                    # ── Step 3：执行工具调用（MCP 版 vs function_call 版的唯一区别）──
                    print(f"\n  LLM 决定调用工具，开始执行...")
                    for tc in response.tool_calls:
                        tool_name = tc["name"]   # 工具名
                        tool_args = tc["args"]   # 参数（LLM 填好的）
                        tool_id   = tc["id"]     # 调用 ID（ToolMessage 必须对应）

                        print(f"\n  [工具调用] {tool_name}")
                        print(f"  [参数]     {json.dumps(tool_args, ensure_ascii=False)}")

                        # ╔══════════════════════════════════════════════════╗
                        # ║  MCP 执行 vs 普通 function_call 执行             ║
                        # ║                                                  ║
                        # ║  普通 function_call（本地）：                    ║
                        # ║    func = TOOL_MAP[tool_name]                   ║
                        # ║    result = func(**tool_args)                   ║
                        # ║    或 tool_obj.invoke(tool_args)                 ║
                        # ║                                                  ║
                        # ║  MCP（通过协议转发到 Server）：                  ║
                        # ║    result = await session.call_tool(            ║
                        # ║        tool_name, tool_args)                    ║
                        # ║                                                  ║
                        # ║  对 LLM 来说：两者完全等价，都是 ToolMessage    ║
                        # ╚══════════════════════════════════════════════════╝

                        mcp_result = await session.call_tool(tool_name, tool_args)  # 通过 MCP 协议执行

                        # 提取结果文本（MCP 返回 TextContent 列表）
                        result_text = "\n".join(
                            c.text for c in mcp_result.content
                            if isinstance(c, types.TextContent)
                        )
                        print(f"  [工具结果] {result_text[:200]}")

                        # ── Step 4：包装成 ToolMessage，加入历史 ──────
                        # 这步和普通 function_call 完全相同
                        messages.append(ToolMessage(
                            content=result_text,   # 工具执行结果
                            tool_call_id=tool_id,  # 必须和 LLM 返回的 id 对应
                        ))

                    # ── 继续下一轮，让 LLM 消化工具结果，生成最终回答 ──

    # ── 完整演示 ─────────────────────────────────────────────────────────────

    async def run_demo(self):
        """把三个阶段串起来，展示完整的 MCP Host 生命周期"""
        server_script = Path(__file__).parent / "01_stdio_mcp_server.py"

        # 模拟 Host 的 Server 配置（类似 claude_desktop_config.json）
        server_configs = [
            {
                "name": "local-tools",           # Server 别名
                "transport": "stdio",            # 传输方式
                "command": sys.executable,       # 启动命令（Python 解释器）
                "args": [str(server_script)],    # 参数
            }
        ]

        # ── 阶段一 + 二：启动 Host，拉取工具，绑定 LLM ──
        await self.startup(server_configs)

        # ── 阶段三：处理用户请求（function_call 流程）────
        print("\n" + "═"*60)
        print("  开始处理用户请求（function_call 流程）")
        print("═"*60)

        questions = [
            "现在是几点？",
            "列一下当前目录的文件，然后把文件数量写入 /tmp/mcp_count.txt",
        ]

        for q in questions:
            await self.chat(q, server_configs[0])
            print()


# ══════════════════════════════════════════════════════════════════════════════
# 流程对比图：MCP vs 普通 function_call
# ══════════════════════════════════════════════════════════════════════════════

def print_flow_comparison():
    """打印 MCP 和普通 function_call 的流程对比，帮助理解差异"""
    print("""
╔══════════════════════════════════════════════════════════════════╗
║              MCP  vs  普通 function_call  流程对比               ║
╠══════════════════════════════════════════════════════════════════╣
║                                                                  ║
║  普通 function_call（03_pydantic_tool.py）：                    ║
║                                                                  ║
║  [启动] 代码里硬编码工具定义（Pydantic Schema）                  ║
║          ↓                                                       ║
║  [绑定] llm.bind_tools([tool1, tool2, ...])                     ║
║          ↓                                                       ║
║  [请求] LLM 返回 tool_calls                                      ║
║          ↓                                                       ║
║  [执行] TOOL_MAP[name].invoke(args)  ← 本地 Python 函数         ║
║          ↓                                                       ║
║  [回答] LLM 消化结果，给出最终回答                               ║
║                                                                  ║
╠══════════════════════════════════════════════════════════════════╣
║                                                                  ║
║  MCP（本文件）：                                                  ║
║                                                                  ║
║  [启动] Client 连接 Server → session.list_tools()               ║
║          ↓  ← ★ 唯一新增的步骤：动态发现工具                    ║
║  [转换] MCP Tool → LLM Schema（mcp_tool_to_llm_schema）         ║
║          ↓                                                       ║
║  [绑定] llm.bind_tools(schemas)  ← 与 function_call 完全相同   ║
║          ↓                                                       ║
║  [请求] LLM 返回 tool_calls      ← 与 function_call 完全相同   ║
║          ↓                                                       ║
║  [执行] session.call_tool(name, args)  ← 唯一不同点：协议转发   ║
║          ↓                                                       ║
║  [回答] LLM 消化结果，给出最终回答  ← 与 function_call 完全相同 ║
║                                                                  ║
╠══════════════════════════════════════════════════════════════════╣
║                                                                  ║
║  核心差异只有两点：                                               ║
║    1. 工具从哪来：硬编码 vs MCP Server 动态拉取                  ║
║    2. 工具怎么执行：本地函数调用 vs MCP 协议转发                 ║
║                                                                  ║
║  LLM 视角：完全无感知，收到的 Schema 一模一样                    ║
║                                                                  ║
╚══════════════════════════════════════════════════════════════════╝
""")


# ══════════════════════════════════════════════════════════════════════════════
# 主入口
# ══════════════════════════════════════════════════════════════════════════════

async def main():
    print_flow_comparison()  # 先打印流程对比，建立认知框架

    host = MCPHost()          # 创建 Host 实例
    await host.run_demo()     # 运行完整三阶段演示


if __name__ == "__main__":
    asyncio.run(main())
