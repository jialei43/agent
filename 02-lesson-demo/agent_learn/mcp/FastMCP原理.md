# FastMCP 原理解析

## 一、FastMCP 是什么

FastMCP 是对底层 `mcp.server.Server`（MCPServer）的高级封装，目标是让开发者只写业务函数，不手写协议处理代码。

```
你写的代码         FastMCP 层                    底层
@mcp.tool()   →   _tool_manager 注册    →   MCPServer（处理 JSON-RPC）
                  _setup_handlers()           MCP 协议（tools/list, tools/call）
```

---

## 二、@mcp.tool() 注册流程

```python
@mcp.tool()
async def read_file(path: str, encoding: str = "utf-8") -> str:
    ...
```

`@mcp.tool()` 本质是一个装饰器工厂，展开后等价于：

```python
def decorator(fn):
    self.add_tool(fn, name=None, description=None, ...)
    return fn
```

`add_tool()` 将函数注册进 `_tool_manager`：

```
FastMCP
  └── _tool_manager (ToolManager)
        └── {"read_file": <ToolInfo>, "write_file": <ToolInfo>, ...}
```

`ToolInfo` 包含：函数引用、从类型注解自动推导的 JSON Schema、description 等。

---

## 三、_setup_handlers —— 关键衔接点

FastMCP 在 `__init__` 末尾调用 `_setup_handlers()`，把自身方法挂进底层 `MCPServer`：

```python
def _setup_handlers(self):
    self._mcp_server.list_tools()(self.list_tools)   # 等价于原理版 @app.list_tools()
    self._mcp_server.call_tool()(self.call_tool)     # 等价于原理版 @app.call_tool()
    self._mcp_server.list_resources()(self.list_resources)
    self._mcp_server.read_resource()(self.read_resource)
    ...
```

**对比原理版（你手写的部分）：**

| 原理版（低层 Server） | FastMCP（自动完成） |
|---|---|
| `@app.list_tools()` 装饰器 + 手写返回列表 | `FastMCP.list_tools()` 读 `_tool_manager` 自动生成 |
| `@app.call_tool()` 装饰器 + if-elif 派发 | `FastMCP.call_tool()` 通过 `_tool_manager` 自动派发 |
| 手写 `inputSchema` dict | 从函数类型注解自动生成 JSON Schema |

---

## 四、内部 list_tools / call_tool 实现

```python
# FastMCP 内部实现（不需要你写，自动挂载）

async def list_tools(self) -> list[MCPTool]:
    tools = self._tool_manager.list_tools()       # 读注册表
    return [
        MCPTool(
            name=info.name,
            description=info.description,
            inputSchema=info.parameters,          # 类型注解 → JSON Schema
        )
        for info in tools
    ]

async def call_tool(self, name: str, arguments: dict) -> ...:
    context = self.get_context()
    return await self._tool_manager.call_tool(    # 按名字找函数并执行
        name, arguments, context=context
    )
```

---

## 五、Client 侧调用链路

Client 只和 **MCP 协议**交互，完全不感知 Server 是 FastMCP 还是原始 Server。

### load_mcp_tools(session) 的内部流程

```
load_mcp_tools(session)
  → session.list_tools()              # 发送 tools/list JSON-RPC 请求
  → MCPServer 路由 → FastMCP.list_tools()
  → 返回工具列表（name + description + inputSchema）
  → 每个工具包装成 LangChain BaseTool
```

### tool.invoke(args) 的内部流程

```
AgentExecutor 决定调用工具
  → tool.invoke({"path": "/tmp/a.txt"})
  → session.call_tool(name, args)     # 发送 tools/call JSON-RPC 请求
  → MCPServer 路由 → FastMCP.call_tool()
  → _tool_manager 按 name 找到函数
  → 执行 read_file(path="/tmp/a.txt")
  → 返回结果字符串给 Agent
```

---

## 六、完整架构图

```
┌─────────────────────────────────────────────────────┐
│  你写的代码                                           │
│  @mcp.tool()  async def read_file(path: str) -> str │
└────────────────────────┬────────────────────────────┘
                         │ add_tool()
┌────────────────────────▼────────────────────────────┐
│  FastMCP                                             │
│  ┌──────────────────────────────────────────────┐   │
│  │ _tool_manager                                │   │
│  │   {"read_file": fn, "write_file": fn, ...}   │   │
│  └──────────────────────────────────────────────┘   │
│  _setup_handlers() 在 __init__ 时自动执行             │
└────────────────────────┬────────────────────────────┘
                         │ list_tools() / call_tool()
┌────────────────────────▼────────────────────────────┐
│  底层 MCPServer                                      │
│  处理 JSON-RPC：tools/list → tools/call              │
└────────────────────────┬────────────────────────────┘
                         │ MCP 协议（stdio / SSE / HTTP）
┌────────────────────────▼────────────────────────────┐
│  Client                                              │
│  load_mcp_tools(session)  →  LangChain BaseTool      │
│  AgentExecutor.ainvoke()  →  session.call_tool()     │
└─────────────────────────────────────────────────────┘
```

---

## 七、总结

| 问题 | 答案 |
|---|---|
| FastMCP 没有 `list_tools`/`call_tool` 吗？ | 有，是 FastMCP 自身的方法，由 `_setup_handlers()` 自动挂进底层 Server |
| `@mcp.tool()` 做了什么？ | 把函数注册进 `_tool_manager`，JSON Schema 从类型注解自动生成 |
| Client 的 `load_mcp_tools` 怎么工作？ | 发 `tools/list` JSON-RPC → Server 返回工具列表 → 包装成 LangChain Tool |
| `tool.invoke()` 怎么调用到服务端函数？ | 发 `tools/call` JSON-RPC → FastMCP 的 `_tool_manager` 按名字派发到原始函数 |
| FastMCP 和原始 Server 的本质区别？ | FastMCP 是封装壳，原始 Server 是它内部的 `_mcp_server`，协议层完全相同 |
