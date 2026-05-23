# load_mcp_tools 原理解析

## 一、整体调用链路

```
load_mcp_tools(session)
  └── _list_all_tools(session)              # 分页拉取全部工具
        └── session.list_tools(cursor=...)  # MCP JSON-RPC: tools/list
  └── convert_mcp_tool_to_langchain_tool()  # 每个 MCP Tool → StructuredTool
        └── StructuredTool(
              args_schema = tool.inputSchema,  # 直接用 JSON Schema dict
              coroutine   = call_tool,          # 闭包：session.call_tool()
            )
```

---

## 二、第一步：_list_all_tools —— 分页拉取

```python
async def _list_all_tools(session: ClientSession) -> list[MCPTool]:
    current_cursor = None
    all_tools = []

    while True:
        result = await session.list_tools(cursor=current_cursor)  # tools/list JSON-RPC
        all_tools.extend(result.tools)

        if not result.nextCursor:   # 没有下一页则结束
            break
        current_cursor = result.nextCursor

    return all_tools
```

**关键点**：支持 MCP 规范中的游标分页（cursor-based pagination），工具数量很多时
不会丢失。最大迭代 1000 次作为保护上限。

---

## 三、第二步：convert_mcp_tool_to_langchain_tool —— 核心转换

**结论：底层确实使用了 StructuredTool**，但 `args_schema` 的传入方式和我们在
`04_mcp_client_demo.py` 中手动实现的不同。

### 3.1 两种方式对比

| | 04_mcp_client_demo.py（手动实现） | langchain_mcp_adapters（官方库） |
|---|---|---|
| `args_schema` | 用 `create_model()` 从 JSON Schema 动态构建 Pydantic 类 | 直接传 `tool.inputSchema`（JSON Schema dict） |
| 原因 | 手动实现，需要 Pydantic 类来做验证 | StructuredTool 支持直接接受 dict |

`StructuredTool.__init__` 的校验逻辑：

```python
# StructuredTool 接受两种 args_schema：
# 1. Pydantic BaseModel 子类
# 2. JSON Schema dict  ← langchain_mcp_adapters 走这条路
if not is_basemodel_subclass(kwargs["args_schema"]) and not isinstance(kwargs["args_schema"], dict):
    raise ValueError("args_schema must be a subclass of BaseModel or a JSON schema dict.")
```

### 3.2 call_tool 闭包

`convert_mcp_tool_to_langchain_tool` 内部定义了一个异步闭包，捕获 `session` 和 `tool.name`：

```python
async def call_tool(**arguments) -> tuple[str | list[str], list[NonTextContent] | None]:
    if session is None:
        # MultiServerMCPClient 场景：每次调用时临时创建 session
        async with create_session(connection) as tool_session:
            await tool_session.initialize()
            call_tool_result = await tool_session.call_tool(tool.name, arguments)
    else:
        # 普通 session 场景：复用已有 session
        call_tool_result = await session.call_tool(tool.name, arguments)

    return _convert_call_tool_result(call_tool_result)  # 格式化结果
```

### 3.3 最终构建的 StructuredTool

```python
return StructuredTool(
    name             = tool.name,
    description      = tool.description or "",
    args_schema      = tool.inputSchema,          # JSON Schema dict，不是 Pydantic 类
    coroutine        = call_tool,                 # 上面定义的异步闭包
    response_format  = "content_and_artifact",    # 支持返回文本 + 附件
    metadata         = metadata,                  # 来自 MCP tool.annotations
)
```

---

## 四、第三步：_convert_call_tool_result —— 结果格式化

MCP 返回的是 `list[TextContent | ImageContent | ...]`，需要转换成 LangChain 期望的格式：

```python
def _convert_call_tool_result(result):
    text_contents     = [c for c in result.content if isinstance(c, TextContent)]
    non_text_contents = [c for c in result.content if not isinstance(c, TextContent)]

    # 文本内容：单条 → str，多条 → list[str]，无内容 → ""
    tool_content = [c.text for c in text_contents]
    if not text_contents:
        tool_content = ""
    elif len(text_contents) == 1:
        tool_content = tool_content[0]

    if result.isError:
        raise ToolException(tool_content)   # 让 AgentExecutor 处理错误重试

    return tool_content, non_text_contents or None  # (文本, 附件)
```

---

## 五、invoke 时的执行路径

```
AgentExecutor 决定调用某个工具
  → tool.ainvoke({"path": "/tmp/a.txt"})
  → StructuredTool._arun(**kwargs)
        if self.coroutine:
            return await self.coroutine(**kwargs)   # 调用闭包
  → call_tool(path="/tmp/a.txt")                   # 闭包执行
  → session.call_tool("read_file", {"path": ...})  # MCP JSON-RPC: tools/call
  → MCP Server 执行 read_file()
  → 返回 CallToolResult
  → _convert_call_tool_result() 格式化
  → StructuredTool 返回结果给 AgentExecutor
```

---

## 六、MultiServerMCPClient 的特殊之处

`MultiServerMCPClient.get_tools()` 传入的是 `session=None`，依赖 `connection` 配置：

```python
# get_tools() 内部
return await load_mcp_tools(None, connection=self.connections[server_name])

# convert_mcp_tool_to_langchain_tool 中的闭包：
if session is None:
    # 每次工具调用时临时建立新 session，用完即关
    async with create_session(connection) as tool_session:
        await tool_session.initialize()
        result = await tool_session.call_tool(tool.name, arguments)
```

**代价**：每次 `tool.invoke()` 都会新建一个 MCP 连接，有握手开销。
**收益**：调用方不需要管理连接生命周期，适合无状态场景。

---

## 七、完整架构图

```
load_mcp_tools(session)
│
├── _list_all_tools(session)
│     └── session.list_tools(cursor)  ──►  MCP Server
│           分页循环直到 nextCursor 为空       tools/list JSON-RPC
│
└── for each MCPTool:
      convert_mcp_tool_to_langchain_tool(session, tool)
      │
      ├── 定义 async def call_tool(**arguments):
      │     session.call_tool(tool.name, arguments)  ──►  MCP Server
      │     _convert_call_tool_result()                    tools/call JSON-RPC
      │
      └── StructuredTool(
            name        = tool.name
            description = tool.description
            args_schema = tool.inputSchema   ← JSON Schema dict（非 Pydantic 类）
            coroutine   = call_tool          ← 闭包，捕获 session + tool.name
          )
```

---

## 八、与手动实现（04_mcp_client_demo.py）的核心差异

| 环节 | 手动实现 | langchain_mcp_adapters |
|---|---|---|
| 工具列表获取 | `session.list_tools()`，不支持分页 | `_list_all_tools()`，游标分页 |
| args_schema | `create_model()` 构建 Pydantic 类 | 直接传 JSON Schema dict |
| anyOf 处理 | 需要 `_extract_py_type()` 手动解析 | 无需处理，StructuredTool 直接接受 dict |
| 结果解析 | 手动拼接 TextContent | `_convert_call_tool_result()` 统一处理 |
| 错误处理 | 返回错误字符串 | `isError=True` 时抛 `ToolException` |
| session 管理 | 调用方持有 session | 支持 `session=None`，按需创建 |
