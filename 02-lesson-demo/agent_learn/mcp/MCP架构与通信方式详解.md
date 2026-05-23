# MCP 架构与通信方式详解

> MCP（Model Context Protocol，模型上下文协议）是 Anthropic 于 2024 年 11 月发布的开放标准协议，
> 解决 AI 模型与外部工具/数据源集成混乱的问题，提供统一的"AI 插槽"规范。

---

## 一、为什么需要 MCP

### 没有 MCP 之前的困境

```
Claude ──── 自定义接口 A ────► 数据库工具
Claude ──── 自定义接口 B ────► 文件系统
Claude ──── 自定义接口 C ────► GitHub API
GPT   ──── 自定义接口 D ────► 数据库工具（重复开发）
GPT   ──── 自定义接口 E ────► 文件系统  （重复开发）

问题：N 个模型 × M 个工具 = N×M 套集成代码，维护成本极高
```

### 有了 MCP 之后

```
Claude ─┐
GPT    ─┼──► MCP 标准协议 ──► MCP Server（数据库）
Gemini ─┘                  ► MCP Server（文件系统）
                            ► MCP Server（GitHub API）

优势：N 个模型 + M 个工具 = N+M，每个工具只需实现一次 MCP Server
```

---

## 二、MCP 整体架构

```
╔══════════════════════════════════════════════════════════════╗
║                    MCP Host（宿主应用）                       ║
║                                                              ║
║   如：Claude Desktop、VS Code、Cursor、自研 AI 应用          ║
║                                                              ║
║  ┌──────────────────────────────────────────────────────┐   ║
║  │                  AI 模型（LLM）                       │   ║
║  └────────────────────────┬─────────────────────────────┘   ║
║                           │ 调用工具/读取资源                 ║
║  ┌────────────────────────▼─────────────────────────────┐   ║
║  │              MCP Client（客户端层）                   │   ║
║  │                                                      │   ║
║  │  ┌─────────────┐  ┌─────────────┐  ┌─────────────┐ │   ║
║  │  │  Client 1   │  │  Client 2   │  │  Client 3   │ │   ║
║  │  └──────┬──────┘  └──────┬──────┘  └──────┬──────┘ │   ║
║  └─────────┼────────────────┼─────────────────┼────────┘   ║
╚════════════╪════════════════╪═════════════════╪════════════╝
             │                │                 │
         Transport        Transport         Transport
        (stdio)           (SSE)        (Streamable HTTP)
             │                │                 │
    ╔════════▼═══╗   ╔════════▼═══╗   ╔════════▼═══╗
    ║ MCP Server ║   ║ MCP Server ║   ║ MCP Server ║
    ║  本地工具  ║   ║  远程服务  ║   ║  云端 API  ║
    ║            ║   ║            ║   ║            ║
    ║ • 文件系统 ║   ║ • 数据库   ║   ║ • SaaS 服务║
    ║ • 本地命令 ║   ║ • 内部 API ║   ║ • 第三方平台║
    ╚════════════╝   ╚════════════╝   ╚════════════╝
```

---

## 三、MCP 核心组件

### 3.1 MCP Host（宿主）

运行 AI 模型的应用程序，负责：
- 管理所有 MCP Client 的生命周期
- 在模型和工具之间做调度决策
- 控制权限和安全策略

典型宿主：Claude Desktop、VS Code Copilot、Cursor、自研 Agent 应用

### 3.2 MCP Client（客户端）

内嵌于 Host 中，**每个 Server 对应一个 Client**，负责：
- 维护与 MCP Server 的 1:1 持久连接
- 将 LLM 的工具调用请求转发给 Server
- 将 Server 结果返回给 LLM

### 3.3 MCP Server（服务端）

对外暴露能力的独立进程，提供三类原语：

| 原语 | 说明 | 示例 |
|---|---|---|
| **Tools（工具）** | 模型可执行的函数，有副作用 | 查询数据库、发送邮件、写文件 |
| **Resources（资源）** | 只读的上下文数据，无副作用 | 读取文件内容、获取文档 |
| **Prompts（提示词）** | 预定义的提示词模板 | 代码审查模板、报告生成模板 |

### 3.4 传输层（Transport）

Client 和 Server 之间的通信通道，即下面要详细介绍的三种通信方式。

---

## 四、MCP 消息格式

MCP 底层基于 **JSON-RPC 2.0**，所有消息都是 JSON：

```json
// 客户端发起工具调用请求
{
  "jsonrpc": "2.0",
  "id": "req_001",
  "method": "tools/call",
  "params": {
    "name": "query_database",
    "arguments": {
      "sql": "SELECT * FROM users LIMIT 10"
    }
  }
}

// 服务端返回结果
{
  "jsonrpc": "2.0",
  "id": "req_001",
  "result": {
    "content": [
      {
        "type": "text",
        "text": "[{\"id\": 1, \"name\": \"张三\"}, ...]"
      }
    ]
  }
}
```

---

## 五、三种通信方式详解

---

### 5.1 stdio（标准输入输出）

#### 工作原理

```
MCP Host 进程
    │
    ├── spawn() 启动子进程
    │
    ▼
MCP Server 子进程
    │
    ├── stdin  ◄── Host 写入 JSON-RPC 请求（换行符分隔）
    └── stdout ──► Host 读取 JSON-RPC 响应（换行符分隔）
```

Client 直接把 Server **作为子进程启动**，通过操作系统的标准流通信，**没有网络参与**。

#### 通信流程

```
Host（父进程）                    Server（子进程）
     │                                  │
     │── spawn("python server.py") ────►│ 启动
     │                                  │ 初始化完成
     │◄──── {"result": "ready"} ────────│
     │                                  │
     │──── {"method":"tools/list"} ────►│
     │◄──── {"result": [...tools]} ─────│
     │                                  │
     │──── {"method":"tools/call"} ────►│ 执行工具
     │◄──── {"result": {...}} ──────────│
     │                                  │
     │── 关闭 stdin ───────────────────►│ 进程退出
```

#### 代码示例

```python
# server.py（MCP Server，stdio 模式）
from mcp.server import Server
from mcp.server.stdio import stdio_server
from mcp import types

app = Server("my-local-tool")

@app.list_tools()
async def list_tools():
    return [
        types.Tool(
            name="read_file",
            description="读取本地文件内容",
            inputSchema={
                "type": "object",
                "properties": {"path": {"type": "string"}},
                "required": ["path"]
            }
        )
    ]

@app.call_tool()
async def call_tool(name: str, arguments: dict):
    if name == "read_file":
        with open(arguments["path"]) as f:
            return [types.TextContent(type="text", text=f.read())]

# stdio_server() 自动处理 stdin/stdout 读写
async def main():
    async with stdio_server() as streams:
        await app.run(*streams, app.create_initialization_options())
```

```json
// claude_desktop_config.json（Host 配置）
{
  "mcpServers": {
    "my-local-tool": {
      "command": "python",        // 启动命令
      "args": ["server.py"],      // 参数
      "env": {}                   // 环境变量
    }
  }
}
```

#### 适用场景与限制

| 优点 | 缺点 |
|---|---|
| 零网络配置，部署最简单 | 只能在同一台机器使用 |
| 进程隔离，安全性高 | 不支持多个客户端并发连接 |
| 延迟极低（进程间通信） | Server 崩溃会影响整个 Host |
| 调试方便（直接看进出的 JSON） | 无法横向扩展 |

---

### 5.2 SSE（Server-Sent Events，基于 HTTP）

> 注：这是 MCP 规范 2024-11-05 版本（旧版）的传输方式，部分框架仍在使用。

#### 工作原理

SSE 利用 HTTP 协议的长连接特性，**客户端→服务端用 POST，服务端→客户端用 SSE 推送**：

```
MCP Client                          MCP Server（HTTP 服务）
    │                                       │
    │── GET /sse (长连接) ─────────────────►│
    │◄── text/event-stream ─────────────────│ 服务端保持连接，随时推送
    │                                       │
    │── POST /messages (请求) ─────────────►│
    │                                       │── 处理请求
    │                                       │
    │◄── event: message                     │
    │    data: {"result": {...}} ───────────│ 通过 SSE 通道推回结果
    │                                       │
    │◄── event: message                     │
    │    data: {"method": "notify"} ────────│ Server 主动推送通知
```

**两条通道：**
- `GET /sse`：建立 SSE 长连接，服务端用这条通道**主动推**消息给客户端
- `POST /messages`：客户端通过这条通道**发送**请求给服务端

#### 代码示例

```python
# server.py（SSE 模式，使用 FastAPI）
from mcp.server.sse import SseServerTransport
from mcp.server import Server
from fastapi import FastAPI
import uvicorn

app_web = FastAPI()
mcp_server = Server("remote-tool")

# SSE 传输层（旧版 MCP SDK 写法）
sse = SseServerTransport("/messages")

@app_web.get("/sse")
async def sse_endpoint(request):
    async with sse.connect_sse(request.scope, request.receive, request._send) as streams:
        await mcp_server.run(*streams, mcp_server.create_initialization_options())

@app_web.post("/messages")
async def handle_message(request):
    await sse.handle_post_message(request.scope, request.receive, request._send)

if __name__ == "__main__":
    uvicorn.run(app_web, host="0.0.0.0", port=8000)
```

```json
// Host 配置（SSE 模式）
{
  "mcpServers": {
    "remote-tool": {
      "url": "http://localhost:8000/sse"  // 指向 SSE 端点
    }
  }
}
```

#### 适用场景与限制

| 优点 | 缺点 |
|---|---|
| 支持远程部署，跨机器访问 | 两条 HTTP 通道管理复杂 |
| 支持多客户端同时连接 | 部分代理/防火墙不支持 SSE 长连接 |
| 基于 HTTP，穿透防火墙容易 | 规范已被 Streamable HTTP 取代 |
| 可加 JWT/API Key 认证 | 实现复杂度较高 |

---

### 5.3 Streamable HTTP（流式 HTTP，当前主流标准）

> MCP 规范 2025-03-26 版本引入，取代旧版 SSE 传输，是当前官方推荐的 HTTP 传输方式。

#### 工作原理

**统一用一个 HTTP 端点**，根据请求类型动态决定返回普通 JSON 还是 SSE 流：

```
MCP Client                          MCP Server
    │                                    │
    │── POST /mcp ──────────────────────►│
    │   Accept: application/json         │
    │   或 text/event-stream             │
    │                                    │── 判断响应类型
    │                                    │
    │   情况A（非流式）：                 │
    │◄── 200 OK                          │
    │    Content-Type: application/json  │
    │    {"result": {...}}               │
    │                                    │
    │   情况B（流式）：                   │
    │◄── 200 OK                          │
    │    Content-Type: text/event-stream │
    │    event: message                  │
    │    data: {"partial": "..."}  (第1片│)
    │    data: {"partial": "..."}  (第2片│)
    │    data: [DONE]              (结束 │)
    │                                    │
    │   情况C（Server 主动通知）：        │
    │── GET /mcp (可选 SSE 监听) ───────►│
    │◄── text/event-stream               │
    │    event: notification             │
    │    data: {"method": "..."}         │
```

#### 会话管理

Streamable HTTP 引入了 `Mcp-Session-Id` 头，支持有状态会话：

```
客户端                              服务端
  │── POST /mcp (初始化) ──────────►│
  │   {"method":"initialize"}       │
  │◄── 200 OK ──────────────────────│
  │   Mcp-Session-Id: sess_abc123   │ 服务端颁发 Session ID
  │                                 │
  │── POST /mcp ────────────────────►│
  │   Mcp-Session-Id: sess_abc123   │ 后续请求携带 Session ID
  │   {"method":"tools/call"}        │
  │◄── 200 OK ──────────────────────│
```

#### 代码示例

```python
# server.py（Streamable HTTP，使用最新 MCP SDK）
from mcp.server.fastmcp import FastMCP

# FastMCP 是高级封装，底层自动处理 Streamable HTTP
mcp = FastMCP("enterprise-tool")

@mcp.tool()
async def query_database(sql: str) -> str:
    """执行 SQL 查询，返回结果"""
    # 真实场景：连接数据库执行查询
    return f"查询结果：{sql} 执行成功，返回 100 条记录"

@mcp.resource("config://app")
async def get_config() -> str:
    """获取应用配置"""
    return '{"env": "production", "version": "2.0"}'

if __name__ == "__main__":
    # transport="streamable-http" 启用 Streamable HTTP 模式
    mcp.run(transport="streamable-http", host="0.0.0.0", port=8000)
```

```python
# client.py（连接 Streamable HTTP Server）
from mcp.client.streamable_http import streamablehttp_client
from mcp import ClientSession

async def main():
    async with streamablehttp_client("http://localhost:8000/mcp") as (read, write, _):
        async with ClientSession(read, write) as session:
            await session.initialize()
            tools = await session.list_tools()
            result = await session.call_tool("query_database", {"sql": "SELECT 1"})
            print(result)
```

#### 适用场景与限制

| 优点 | 缺点 |
|---|---|
| 统一单端点，架构最简洁 | 需要客户端支持（旧版 SDK 不支持） |
| 支持流式和非流式，灵活切换 | 2025年3月才加入规范，生态仍在追赶 |
| 会话管理内置，有状态支持好 | — |
| 标准 HTTPS 部署，企业友好 | — |
| 官方长期维护标准 | — |

---

## 六、三种通信方式横向对比

```
特性对比矩阵：

                    stdio      SSE（旧）    Streamable HTTP（新）
─────────────────────────────────────────────────────────────────
部署位置            本地        远程          远程
跨机器访问          ✗          ✓            ✓
多客户端并发         ✗          ✓            ✓
Server 主动推送      ✗          ✓（SSE）      ✓（可选 GET 监听）
流式响应             ✗          ✓            ✓
会话状态管理         ✗          有限          ✓（内置 Session ID）
HTTP 认证（JWT 等）  ✗          ✓            ✓
实现复杂度           低          中            低（SDK 封装好）
官方推荐状态         ✓          旧版          ✓（当前主推）
典型延迟             <1ms       5-20ms        5-20ms
防火墙穿透           不需要      容易          容易
```

---

## 七、企业主流使用方式

### 当前企业生产环境格局（2025年）

```
本地开发工具（IDE 插件、桌面应用）
        │
        └─► stdio（简单，零配置，主流）

内部微服务工具（公司内网部署）
        │
        └─► Streamable HTTP + HTTPS + JWT 认证（企业主流选择）

SaaS 平台对外开放（提供给第三方调用）
        │
        └─► Streamable HTTP + OAuth 2.0（标准企业级方案）
```

### 为什么 Streamable HTTP 成为企业主流

**1. 运维体系兼容**：MCP Server 本质是一个 HTTP 服务，可直接复用 Nginx、K8s Ingress、API Gateway 等现有基础设施。

**2. 安全体系完整**：
```
客户端
  │── POST /mcp ──────────────────────────────────►│
  │   Authorization: Bearer eyJhbGci...            │
  │   X-Tenant-Id: company_abc                     │
  │                                                │
  │            API Gateway / Nginx                  │
  │                    │                           │
  │              JWT 验证 + 限流                    │
  │                    │                           │
  │              MCP Server 集群                    │
```

**3. 可观测性**：标准 HTTP 请求，Prometheus、Jaeger、ELK 等监控链路天然支持。

**4. 横向扩展**：结合 Session ID 做粘性路由，MCP Server 可以水平扩展多个实例。

### 企业级部署参考架构

```
                   ┌──────────────┐
用户/AI 应用        │  API Gateway  │  （认证、限流、路由）
    │              │  Kong/APISIX  │
    └──────────────►              │
                   └──────┬───────┘
                          │
          ┌───────────────┼───────────────┐
          ▼               ▼               ▼
   ┌─────────────┐ ┌─────────────┐ ┌─────────────┐
   │ MCP Server  │ │ MCP Server  │ │ MCP Server  │
   │  数据库工具  │ │  文件服务   │ │  内部 API   │
   │  Pod×3      │ │  Pod×2      │ │  Pod×5      │
   └─────────────┘ └─────────────┘ └─────────────┘
          │               │               │
   ┌──────▼───────────────▼───────────────▼──────┐
   │                  Redis                        │
   │          （Session 状态共享存储）              │
   └──────────────────────────────────────────────┘
```

### 各场景推荐选型

| 场景 | 推荐传输方式 | 理由 |
|---|---|---|
| 本地开发工具、IDE 插件 | **stdio** | 最简单，无需网络，安全 |
| 公司内网 AI 服务 | **Streamable HTTP + HTTPS** | 可管控、可扩展、可监控 |
| 对外开放的 AI 平台 | **Streamable HTTP + OAuth 2.0** | 标准企业鉴权，符合安全合规 |
| 原型验证、快速开发 | **stdio 或 SSE（旧）** | 都有成熟示例，上手快 |
| 需要 Server 主动推送事件 | **Streamable HTTP（GET 监听）** | 支持 Server-Push，功能完整 |

---

## 八、MCP 生命周期

```
1. 初始化（Initialization）
   Client ──► Server: initialize（发送客户端能力）
   Server ──► Client: initialize 响应（发送服务端能力）
   Client ──► Server: initialized 通知（握手完成）

2. 能力发现（Discovery）
   Client ──► Server: tools/list
   Client ──► Server: resources/list
   Client ──► Server: prompts/list

3. 正常运行（Operation）
   Client ──► Server: tools/call（执行工具）
   Client ──► Server: resources/read（读取资源）
   Server ──► Client: notifications/...（主动通知）

4. 关闭（Shutdown）
   Client ──► Server: 关闭连接
   Server: 清理资源，进程退出
```

---

## 九、快速选型决策树

```
我的 MCP Server 需要跨机器访问吗？
    │
    ├── 否 → 使用 stdio（最简单）
    │
    └── 是 → 是否需要流式输出或 Server 主动推送？
                │
                ├── 否 → 普通 HTTP POST 就够了，也可用 Streamable HTTP
                │
                └── 是 → 使用 Streamable HTTP（官方推荐）
                              │
                              └── 需要认证？
                                    ├── 内网 → JWT / API Key
                                    └── 外网 → OAuth 2.0
```
