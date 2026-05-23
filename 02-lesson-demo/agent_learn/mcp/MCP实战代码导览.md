# MCP 实战代码导览

本目录包含 MCP 从原理到实战的完整代码，按顺序阅读/运行即可。

---

## 文件总览

```
mcp/
├── MCP架构与通信方式详解.md       ← 原理文档：架构图 + 三种通信方式 + 企业选型
├── python_a2a使用指南.md          ← A2A 协议文档：Agent-to-Agent 多智能体协作
├── MCP实战代码导览.md             ← 本文件：代码地图
│
├── 01_stdio_mcp_server.py         ← 通信方式一：stdio Server（本地子进程）
├── 02_sse_mcp_server.py           ← 通信方式二：SSE Server（旧版 HTTP）
├── 03_streamable_http_server.py   ← 通信方式三：Streamable HTTP Server（当前主流）
├── 04_mcp_client_demo.py          ← 三种传输方式的客户端调用对比演示
└── 05_langchain_mcp_integration.py ← MCP + LangChain Agent 集成实战
```

---

## 快速启动

### 环境准备

```bash
pip install "mcp[cli]" fastapi uvicorn httpx langchain langchain-openai python-dotenv
```

### 运行顺序

```bash
# ① stdio Server（不需要单独启动，由客户端自动以子进程启动）
#   直接运行客户端即可：
python 04_mcp_client_demo.py

# ② SSE Server（旧版，需要先启动）
python 02_sse_mcp_server.py        # 终端1：启动 Server（端口 8001）
python 04_mcp_client_demo.py       # 终端2：运行客户端演示

# ③ Streamable HTTP Server（当前主流，需要先启动）
python 03_streamable_http_server.py  # 终端1：启动 Server（端口 8002）
python 04_mcp_client_demo.py         # 终端2：运行客户端演示

# ④ LangChain 集成（同时启动 Streamable HTTP Server 效果更完整）
python 03_streamable_http_server.py  # 终端1
python 05_langchain_mcp_integration.py  # 终端2
```

---

## 各文件核心内容

### `01_stdio_mcp_server.py` — stdio 传输

| 要素 | 内容 |
|---|---|
| 传输方式 | `stdio_server()` — stdin/stdout |
| 工具 | `get_current_time` / `read_file` / `write_file` / `list_directory` |
| 资源 | `file:///etc/hostname` / `env://PATH` |
| 启动 | 不直接启动，由 Host 以子进程方式启动 |
| 适用场景 | 本地开发工具、IDE 插件、Claude Desktop |

```python
# 核心代码结构
app = Server("local-tools")

@app.list_tools()
async def list_tools(): ...         # 声明工具

@app.call_tool()
async def call_tool(name, args): ... # 执行工具

async with stdio_server() as (read, write):
    await app.run(read, write, ...)  # 启动事件循环
```

---

### `02_sse_mcp_server.py` — SSE 传输（旧版）

| 要素 | 内容 |
|---|---|
| 传输方式 | `SseServerTransport` + FastAPI |
| 端点 | `GET /sse`（长连接）+ `POST /messages`（请求） |
| 工具 | `http_get` / `http_post` / `query_weather_api` / `get_server_status` |
| 端口 | 8001 |
| 适用场景 | 旧版 MCP 兼容、了解 SSE 原理 |

```python
# 核心代码结构（两个端点是关键）
@web_app.get("/sse")                    # SSE 长连接（Server → Client 推送通道）
async def sse_endpoint(request):
    async with sse_transport.connect_sse(...) as streams:
        await mcp_server.run(...)

@web_app.post("/messages")              # 消息接收（Client → Server 发送通道）
async def messages_endpoint(request):
    await sse_transport.handle_post_message(...)
```

---

### `03_streamable_http_server.py` — Streamable HTTP（当前主流）

| 要素 | 内容 |
|---|---|
| 传输方式 | `FastMCP` + `streamable_http_app()` |
| 端点 | `POST /mcp`（单一端点，统一处理） |
| 工具 | `query_sales_data` / `calculate_kpi` / `send_notification` / `get_server_metrics` |
| 企业特性 | JWT 认证 / 请求日志中间件 / Prometheus 指标 / 健康检查 |
| 端口 | 8002 |
| 适用场景 | 企业生产环境、对外开放 API |

```python
# 核心代码结构（FastMCP 高度简化）
mcp = FastMCP("enterprise-tools")

@mcp.tool()                             # 一个装饰器搞定工具注册
async def query_sales_data(...): ...

web_app.mount("/mcp", mcp.streamable_http_app())  # 挂载到 FastAPI
```

---

### `04_mcp_client_demo.py` — 客户端对比演示

三种传输方式的客户端写法，**建立连接的方式不同，ClientSession API 完全一致**：

```python
# stdio 客户端
async with stdio_client(sys.executable, ["server.py"]) as (read, write):
    async with ClientSession(read, write) as session: ...

# SSE 客户端
async with sse_client(url="http://localhost:8001/sse") as (read, write):
    async with ClientSession(read, write) as session: ...

# Streamable HTTP 客户端
async with streamablehttp_client("http://localhost:8002/mcp") as (read, write, _):
    async with ClientSession(read, write) as session: ...

# ↑ 三种方式，session.initialize() / session.list_tools() / session.call_tool() 完全相同
```

---

### `05_langchain_mcp_integration.py` — LangChain 集成

核心是 `MCPToolWrapper`：把 MCP Tool 包装成 LangChain `BaseTool`，让 LangChain Agent 透明地通过 MCP 协议调用工具。

```
MCP Server（工具服务）
       ↓ MCP 协议
MCPToolWrapper（适配器）
       ↓ LangChain BaseTool 接口
LangChain Agent（llm.bind_tools）
       ↓
LLM（决定调用哪个工具）
```

```python
class MCPToolWrapper(BaseTool):
    async def _arun(self, **kwargs) -> str:
        result = await self.session.call_tool(self.mcp_tool_name, kwargs)
        return result.content[0].text

# 动态加载所有 MCP 工具为 LangChain Tools
lc_tools = await load_mcp_tools_as_langchain(session)
llm_with_tools = llm.bind_tools(lc_tools)  # 正常使用
```

---

## 三种 Transport 一分钟对比

```
             stdio              SSE（旧）          Streamable HTTP（新）
             ─────              ────────           ────────────────────
部署          本地子进程          HTTP 服务            HTTP 服务
端点          stdin/stdout        GET /sse             POST /mcp
                                 POST /messages        （单端点）
Session       无                  有限                 ✅ 内置 Mcp-Session-Id
流式响应       无                  ✅ SSE 推送           ✅ 动态切换
企业认证       无                  HTTP 头认证           HTTP 头认证
官方状态       ✅ 推荐             旧版（仍支持）          ✅ 当前主推
推荐场景       本地工具            旧版兼容              企业生产
```

---

## 依赖版本参考

```
mcp>=1.5.0            # 支持 Streamable HTTP（2025-03-26 规范）
fastapi>=0.110.0
uvicorn>=0.29.0
langchain>=0.2.0
langchain-openai>=0.1.0
httpx>=0.27.0
python-dotenv>=1.0.0
```
