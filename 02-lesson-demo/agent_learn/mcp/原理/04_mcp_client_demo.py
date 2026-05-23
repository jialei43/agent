"""
MCP 客户端 × LangChain × StructuredTool × Qwen 完整调用演示

完整链路：
  MCP Server（工具服务）
      ↓ session.list_tools()            ← 动态拉取工具 Schema
  build_args_schema()                   ← JSON Schema → Pydantic Model
      ↓
  StructuredTool(args_schema, coroutine) ← 把 MCP 工具包装成 LangChain Tool
      ↓
  llm.bind_tools(structured_tools)       ← 工具 Schema 注入 LLM
      ↓
  LLM 推理 → tool_calls                 ← LLM 决定调用哪个工具及参数
      ↓
  await tool.ainvoke(args)              ← StructuredTool 执行 → MCP session.call_tool()
      ↓
  ToolMessage → LLM 二次推理            ← 最终回答

运行前提：
  演示一/四：无需提前启动（stdio 自动启动子进程）
  演示二：先运行 python 02_sse_mcp_server.py
  演示三：先运行 python 03_streamable_http_server.py

依赖：pip install "mcp[cli]" langchain langchain-openai python-dotenv
"""

import asyncio
import json
import os
import sys
from pathlib import Path
from typing import Optional, Any

from dotenv import load_dotenv, find_dotenv
from pydantic import BaseModel, Field, create_model  # 动态 Schema 构建
from mcp import ClientSession, types
from mcp.client.stdio import stdio_client, StdioServerParameters
from mcp.client.sse import sse_client
from mcp.client.streamable_http import streamablehttp_client
from langchain_openai import ChatOpenAI
from langchain_core.tools import StructuredTool  # 核心：LangChain 工具包装器
from langchain_core.messages import BaseMessage, HumanMessage, ToolMessage
from langchain_core.utils.function_calling import convert_to_openai_tool  # LangChain 0.2+ 标准转换接口

load_dotenv(find_dotenv())


# ══════════════════════════════════════════════════════════════════════════════
# LLM 初始化
# 注意：用户指定 qwen-plus（原文 qwen3.6-plus，DashScope 标准模型名为 qwen-plus）
# ══════════════════════════════════════════════════════════════════════════════

def make_llm() -> ChatOpenAI:
    """创建 Qwen LLM 实例"""
    return ChatOpenAI(
        model="qwen-plus",               # DashScope Qwen Plus 模型
        api_key=os.getenv("DASHSCOPE_API_KEY"),
        base_url="https://dashscope.aliyuncs.com/compatible-mode/v1",
        temperature=0,                   # 工具调用场景建议关闭随机性
    )


# ══════════════════════════════════════════════════════════════════════════════
# 核心一：MCP Tool JSON Schema → Pydantic BaseModel（动态构建）
# StructuredTool 需要 args_schema 是一个 Pydantic BaseModel 类
# ══════════════════════════════════════════════════════════════════════════════

# JSON Schema type → Python type 映射
_JSON_TYPE_MAP: dict[str, Any] = {
    "string":  str,
    "integer": int,
    "number":  float,
    "boolean": bool,
    "object":  dict,
    "array":   list,
}


def _extract_py_type(field_def: dict) -> type:
    """
    从单个 JSON Schema 字段定义中提取 Python 类型。

    两种结构：
      普通字段：{"type": "string", ...}
      Optional 字段：{"anyOf": [{"type": "string"}, {"type": "null"}], ...}
                    （Pydantic v2 对 Optional[str] 生成的 Schema）
    """
    if "type" in field_def:
        return _JSON_TYPE_MAP.get(field_def["type"], str)

    if "anyOf" in field_def:
        # 过滤掉 {"type": "null"}，取第一个非 null 类型
        non_null = [s for s in field_def["anyOf"] if s.get("type") != "null"]
        if non_null:
            return _JSON_TYPE_MAP.get(non_null[0].get("type", "string"), str)

    return str  # 兜底


def build_args_schema(tool_name: str, input_schema: dict) -> type[BaseModel]:
    """
    从 MCP Tool 的 inputSchema（JSON Schema dict）动态生成 Pydantic BaseModel。
    用于 StructuredTool(args_schema=...) 参数，使 LangChain 能读取参数描述和类型。

    转换规则：
      JSON Schema properties → Pydantic 字段
      required 列表 → 必填字段（无默认值）
      default 值 → Optional 字段（有默认值）
      description → Field(description=...)
    """
    props = input_schema.get("properties", {})      # 参数属性
    required_fields = set(input_schema.get("required", []))  # 必填参数集合

    field_definitions: dict[str, tuple] = {}
    for field_name, field_def in props.items():
        # 1. 确定 Python 类型（兼容直接 type 和 anyOf 两种结构）
        py_type = _extract_py_type(field_def)

        # 2. 构建 Field 注解
        description = field_def.get("description", "")

        if field_name in required_fields:
            # 必填字段：类型保持原样，default 为 ... (无默认值)
            field_definitions[field_name] = (py_type, Field(..., description=description))
        else:
            # 可选字段：包装为 Optional，default 取 Schema 默认值或 None
            actual_default = field_def.get("default", None)
            field_definitions[field_name] = (
                Optional[py_type],
                Field(default=actual_default, description=description),
            )

    """
    第一行：生成类名字符串

    
    class_name = "".join(w.capitalize() for w in tool_name.split("_")) + "Args"
    以 tool_name = "read_file" 为例，逐步拆解：
    
    
    tool_name.split("_")          # ["read", "file"]       按下划线切开
    
    w.capitalize() for w in ...   # ["Read", "File"]        每个单词首字母大写
    
    "".join(...)                  # "ReadFile"              拼在一起，无分隔符
    
    + "Args"                      # "ReadFileArgs"          加后缀，表示这是参数模型
    目的只是给动态生成的类取个有意义的名字，便于调试时看到类名知道是哪个工具的参数。
    
    第二行：动态创建 Pydantic 模型类
    
    
    return create_model(class_name, **field_definitions)
    create_model 是 Pydantic 提供的工厂函数，作用等价于在运行时写了这段代码：
    
    
    # field_definitions 长这样（以 read_file 为例）：
    # {
    #     "path":     (str,         Field(...,          description="文件路径")),
    #     "encoding": (Optional[str], Field("utf-8",   description="文件编码")),
    # }
    
    # create_model 展开后等价于：
    class ReadFileArgs(BaseModel):
        path:     str            = Field(...,    description="文件路径")
        encoding: Optional[str]  = Field("utf-8", description="文件编码")
    **field_definitions 是把字典展开作为关键字参数传入，create_model 规定每个参数的值是 (类型, Field(...)) 的元组，它会自动识别并构建对应字段。
    
    区别在于：普通 class 定义是静态的，写死在源码里；create_model 是运行时根据 MCP Server 返回的 Schema 动态构建，Server 加什么字段就生成什么字段，client 代码不用改。
    """

    # create_model 动态生成 Pydantic 类，类名用工具名驼峰格式
    class_name = "".join(w.capitalize() for w in tool_name.split("_")) + "Args"
    return create_model(class_name, **field_definitions)


# ══════════════════════════════════════════════════════════════════════════════
# 核心二：MCP Tool → StructuredTool（LangChain 工具适配）
# ══════════════════════════════════════════════════════════════════════════════

def mcp_tool_to_structured(mcp_tool: types.Tool, session: ClientSession) -> StructuredTool:
    """
    把 MCP Server 返回的 Tool 对象转换为 LangChain StructuredTool。

    关键设计：
      - args_schema：动态构建的 Pydantic 模型，StructuredTool 用它生成 LLM 可读的 Schema
      - coroutine：异步函数，捕获 session，实际执行时通过 MCP 协议转发到 Server
      - func 也提供（同步包装），让 llm.bind_tools 和 invoke 都能工作

    StructuredTool 内部：
      bind_tools 阶段：读 args_schema → 生成 OpenAI function JSON → 发给 LLM
      执行阶段：     LLM 返回 tool_calls → ainvoke(args) → _arun(**args) → coroutine(**args)
    """
    args_schema = build_args_schema(mcp_tool.name, mcp_tool.inputSchema)  # 动态 Schema
    tool_name = mcp_tool.name  # 闭包捕获工具名

    async def _async_call(**kwargs) -> str:
        """异步执行：通过 MCP session 转发工具调用到 Server"""
        result = await session.call_tool(tool_name, kwargs)  # MCP 协议调用
        # 提取 TextContent 文本内容，多条内容合并
        texts = [c.text for c in result.content if isinstance(c, types.TextContent)]
        return "\n".join(texts) if texts else "（工具返回空结果）"

    def _sync_call(**kwargs) -> str:
        """同步包装（供非异步上下文使用）"""
        return asyncio.get_event_loop().run_until_complete(_async_call(**kwargs))

    return StructuredTool(
        name=mcp_tool.name,          # 工具名（LLM 调用时返回的 name 字段）
        description=mcp_tool.description,  # 工具描述（LLM 靠这个决定是否调用）
        args_schema=args_schema,     # Pydantic Schema（LLM 靠这个知道参数结构）
        func=_sync_call,             # 同步调用入口（_run）
        coroutine=_async_call,       # 异步调用入口（_arun）← ainvoke 走这里
    )


async def load_structured_tools(session: ClientSession) -> list[StructuredTool]:
    """从 MCP Session 加载所有工具，全部转换为 StructuredTool"""
    await session.initialize()  # MCP 握手（必须第一步）
    tools_result = await session.list_tools()  # 拉取工具列表

    structured_tools = []
    for mcp_tool in tools_result.tools:
        st = mcp_tool_to_structured(mcp_tool, session)  # 转换
        structured_tools.append(st)
        # 打印转换结果，方便调试
        props = list(mcp_tool.inputSchema.get("properties", {}).keys())
        print(f"  ✅ MCP Tool → StructuredTool: {mcp_tool.name}（参数：{props}）")

    return structured_tools


# ══════════════════════════════════════════════════════════════════════════════
# 核心三：完整 function_call Agent 循环
# ══════════════════════════════════════════════════════════════════════════════

async def run_agent(
    question: str,
    structured_tools: list[StructuredTool],
    llm: ChatOpenAI,
    max_rounds: int = 5,
) -> None:
    """
    完整的 LLM + StructuredTool + MCP function_call 循环。

    流程（和普通 function_call 完全相同，工具执行底层换成 MCP）：
      1. 用户问题 → LLM（携带工具 Schema）
      2. LLM 返回 tool_calls（包含工具名 + 填好的参数）
      3. 找到对应 StructuredTool，调用 ainvoke(args) → MCP session.call_tool()
      4. 结果包装为 ToolMessage 加入消息历史
      5. 再次调用 LLM，消化工具结果，生成最终回答
      6. 若 LLM 不再调用工具，输出最终回答，结束
    """
    print(f"\n{'─'*60}")
    print(f"用户：{question}")

    # 绑定工具到 LLM（StructuredTool.args_schema → OpenAI function Schema）
    llm_with_tools = llm.bind_tools(structured_tools)
    tool_map = {t.name: t for t in structured_tools}  # 名称 → StructuredTool 路由表

    messages: list[BaseMessage] = [HumanMessage(content=question)]  # 初始消息

    for round_num in range(max_rounds):
        # ── Step 1：LLM 推理 ───────────────────────────────────────
        response = llm_with_tools.invoke(messages)  # 同步调用 LLM
        messages.append(response)

        finish_reason = response.response_metadata.get("finish_reason", "stop")
        print(f"\n[第{round_num+1}轮] finish_reason={finish_reason}, "
              f"tool_calls={[tc['name'] for tc in response.tool_calls]}")

        # ── Step 2：无工具调用 = 最终回答 ─────────────────────────
        if not response.tool_calls:
            print(f"\nLLM：{response.content}")
            return

        # ── Step 3：执行所有工具调用 ──────────────────────────────
        for tc in response.tool_calls:
            tool_name = tc["name"]   # LLM 决定调用的工具名
            tool_args = tc["args"]   # LLM 填好的参数
            tool_id   = tc["id"]     # 调用 ID（ToolMessage 必须对应）

            print(f"\n  [工具] {tool_name}")
            print(f"  [参数] {json.dumps(tool_args, ensure_ascii=False)}")

            tool = tool_map.get(tool_name)
            if not tool:
                result_text = f"错误：工具 '{tool_name}' 不存在"
            else:
                # ── StructuredTool.ainvoke → _arun → MCP session.call_tool ──
                result_text = await tool.ainvoke(tool_args)

            print(f"  [结果] {result_text[:200]}{'...' if len(result_text) > 200 else ''}")

            # ── Step 4：ToolMessage 加入历史 ─────────────────────
            messages.append(ToolMessage(
                content=result_text,  # 工具执行结果
                tool_call_id=tool_id, # 与 LLM 返回的 id 对应（必须！）
            ))

    print("[达到最大轮次，终止]")


# ══════════════════════════════════════════════════════════════════════════════
# 演示一：stdio Transport + Qwen + StructuredTool（主演示）
# ══════════════════════════════════════════════════════════════════════════════

async def demo_stdio_qwen():
    """
    stdio 传输：连接本地 MCP Server（01_stdio_mcp_server.py），
    把工具动态转换为 StructuredTool，用 Qwen 大模型完成完整对话。
    """
    print("\n" + "═"*60)
    print("  演示一：stdio MCP Server × Qwen × StructuredTool")
    print("═"*60)

    server_script = Path(__file__).parent / "01_stdio_mcp_server.py"
    llm = make_llm()

    async with stdio_client(StdioServerParameters(command=sys.executable, args=[str(server_script)])) as (read, write):
        async with ClientSession(read, write) as session:
            print("\n正在从 stdio MCP Server 加载工具（01_stdio_mcp_server.py）...")
            tools = await load_structured_tools(session)
            print(f"\n共加载 {len(tools)} 个 StructuredTool，绑定到 Qwen LLM")

            # 完整 function_call 测试
            questions = [
                "现在是什么时间？",
                "帮我列出当前目录下的文件，并把文件数量写到 /tmp/mcp_result.txt，写完再读出来确认",
                "帮我读取 /tmp/mcp_result.txt 的内容",
            ]
            for q in questions:
                await run_agent(q, tools, llm)


# ══════════════════════════════════════════════════════════════════════════════
# 演示二：SSE Transport + Qwen（需先启动 02_sse_mcp_server.py）
# ══════════════════════════════════════════════════════════════════════════════

async def demo_sse_qwen():
    """SSE 传输：连接 02_sse_mcp_server.py"""
    print("\n" + "═"*60)
    print("  演示二：SSE MCP Server × Qwen（需先启动 02_sse_mcp_server.py）")
    print("═"*60)

    llm = make_llm()
    try:
        async with sse_client("http://localhost:8001/sse") as (read, write):
            async with ClientSession(read, write) as session:
                print("\n正在从 SSE MCP Server 加载工具...")
                tools = await load_structured_tools(session)
                await run_agent("帮我查一下 https://httpbin.org/json 这个接口的返回内容", tools, llm)
    except ConnectionRefusedError:
        print("⚠️  SSE Server 未启动，跳过。请先运行：python 02_sse_mcp_server.py")


# ══════════════════════════════════════════════════════════════════════════════
# 演示三：Streamable HTTP + Qwen（需先启动 03_streamable_http_server.py）
# ══════════════════════════════════════════════════════════════════════════════

async def demo_streamable_http_qwen():
    """Streamable HTTP 传输：连接 03_streamable_http_server.py"""
    print("\n" + "═"*60)
    print("  演示三：Streamable HTTP MCP Server × Qwen（需先启动 03_streamable_http_server.py）")
    print("═"*60)

    llm = make_llm()
    try:
        async with streamablehttp_client("http://localhost:8002/mcp") as (read, write, _):
            async with ClientSession(read, write) as session:
                print("\n正在从 Streamable HTTP MCP Server 加载工具...")
                tools = await load_structured_tools(session)

                questions = [
                    "帮我查一下今年第一季度（2026-01-01 到 2026-03-31）华东地区的营收数据",
                    "华东营收 320 万，目标是 300 万，帮我计算 KPI 完成率，指标名叫「Q1华东营收」",
                    "通过邮件通知 manager@company.com，告诉他 Q1 华东营收超额完成目标",
                ]
                for q in questions:
                    await run_agent(q, tools, llm)
    except ConnectionRefusedError:
        print("⚠️  Streamable HTTP Server 未启动，跳过。请先运行：python 03_streamable_http_server.py")


# ══════════════════════════════════════════════════════════════════════════════
# 演示四：打印 StructuredTool 的完整 Schema（验证 MCP → LangChain 转换是否正确）
# ══════════════════════════════════════════════════════════════════════════════

async def demo_inspect_schemas():
    """打印 StructuredTool 生成的 Schema，验证从 MCP inputSchema 到 LangChain 的转换结果"""
    print("\n" + "═"*60)
    print("  演示四：查看 MCP → StructuredTool Schema 转换结果")
    print("═"*60)

    server_script = Path(__file__).parent / "01_stdio_mcp_server.py"

    async with stdio_client(StdioServerParameters(command=sys.executable, args=[str(server_script)])) as (read, write):
        async with ClientSession(read, write) as session:
            await session.initialize()
            tools_result = await session.list_tools()

            for mcp_tool in tools_result.tools:
                st = mcp_tool_to_structured(mcp_tool, session)

                print(f"\n{'─'*40}")
                print(f"工具名：{st.name}")
                print(f"描述：  {st.description}")

                # 打印 Pydantic args_schema 的 JSON Schema
                print(f"LangChain args_schema（Pydantic 动态模型）：")
                schema = st.args_schema.model_json_schema()
                print(json.dumps(schema, ensure_ascii=False, indent=2))

                # 打印 LLM 实际收到的 OpenAI function 格式（LangChain 0.2+ 标准接口）
                print(f"LLM 收到的 OpenAI function Schema：")
                oai_schema = convert_to_openai_tool(st)  # BaseTool → OpenAI function dict
                print(json.dumps(oai_schema, ensure_ascii=False, indent=2))


# ══════════════════════════════════════════════════════════════════════════════
# 主入口
# ══════════════════════════════════════════════════════════════════════════════

async def main():
    print("=" * 60)
    print("  MCP × LangChain × StructuredTool × Qwen 完整演示")
    print("=" * 60)

    await demo_inspect_schemas()          # 演示四：先看 Schema 转换结果，理解机制
    await demo_stdio_qwen()               # 演示一：主演示（无需额外启动）
    # await demo_sse_qwen()                 # 演示二：SSE（如已启动）
    # await demo_streamable_http_qwen()     # 演示三：Streamable HTTP（如已启动）

    print("\n" + "=" * 60)
    print("  全部演示完成")
    print("=" * 60)


if __name__ == "__main__":
    asyncio.run(main())
