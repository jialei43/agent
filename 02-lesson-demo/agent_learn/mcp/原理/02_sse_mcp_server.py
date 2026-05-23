"""
MCP 通信方式二：SSE（Server-Sent Events）—— Pydantic 参数约束版
运行方式：python 02_sse_mcp_server.py（启动 HTTP 服务，默认端口 8001）
测试地址：http://localhost:8001/sse（SSE 长连接端点）

Pydantic 改造说明（与 01_stdio_mcp_server.py 改造原则相同）：
  - 每个工具对应一个 Pydantic BaseModel，承担 Schema 生成 + 参数校验双重职责
  - model_json_schema() 替代手写 inputSchema dict
  - field_validator 承载业务规则（URL 合法性、超时范围、城市名非空等）
  - ValidationError 转为 TextContent 错误信息返回给 LLM，让 LLM 自我修正

依赖：pip install mcp fastapi uvicorn httpx pydantic python-dotenv
"""

import asyncio  # 异步
import json  # JSON
import os  # 环境变量
import httpx  # HTTP 客户端
from datetime import datetime  # 时间
from typing import Optional, Dict  # 类型注解
from dotenv import load_dotenv, find_dotenv  # 环境变量

from pydantic import BaseModel, Field, field_validator, ValidationError  # 参数约束
from mcp.server import Server  # MCP Server 核心
from mcp.server.sse import SseServerTransport  # SSE 传输层
from mcp import types  # MCP 类型
from fastapi import FastAPI, Request  # Web 框架
from fastapi.responses import Response  # HTTP 响应
import uvicorn  # ASGI 服务器

load_dotenv(find_dotenv())  # 加载 .env


# ── 创建实例 ─────────────────────────────────────────────────────────────────

mcp_server = Server("remote-api-tools")  # MCP Server 实例
web_app = FastAPI(title="MCP SSE Server", version="1.0.0")  # FastAPI 实例
#  SseServerTransport("/messages") 中的 "/messages" 这个字符串，只做一件事：告诉 SSE 长连接，客户端应该把后续的
#   JSON-RPC 请求 POST 到哪个路径
sse_transport = SseServerTransport("/messages")  # SSE 传输层，指定 POST 端点路径


# ══════════════════════════════════════════════════════════════════════════════
# Pydantic 参数模型 —— Schema 生成 + 业务规则校验
# ══════════════════════════════════════════════════════════════════════════════

class HttpGetInput(BaseModel):
    """http_get 工具参数模型"""
    url: str = Field(description="目标 URL，必须以 http:// 或 https:// 开头")
    headers: Optional[Dict[str, str]] = Field(default=None, description="自定义请求头（键值对），可选")
    timeout: float = Field(default=10.0, ge=1.0, le=60.0, description="超时秒数，范围 1-60，默认 10")

    @field_validator("url")  # URL 合法性校验
    @classmethod
    def validate_url(cls, v: str) -> str:
        v = v.strip()
        if not v.startswith(("http://", "https://")):
            raise ValueError("url 必须以 http:// 或 https:// 开头")
        if len(v) > 2048:  # URL 长度上限
            raise ValueError("url 长度不能超过 2048 字符")
        return v


class HttpPostInput(BaseModel):
    """http_post 工具参数模型"""
    url: str = Field(description="目标 URL，必须以 http:// 或 https:// 开头")
    body: Dict = Field(description="请求体（JSON 格式键值对）")
    headers: Optional[Dict[str, str]] = Field(default=None, description="自定义请求头，可选")

    @field_validator("url")  # URL 合法性校验（与 HttpGetInput 相同规则）
    @classmethod
    def validate_url(cls, v: str) -> str:
        v = v.strip()
        if not v.startswith(("http://", "https://")):
            raise ValueError("url 必须以 http:// 或 https:// 开头")
        return v

    @field_validator("body")  # body 大小限制
    @classmethod
    def validate_body_size(cls, v: Dict) -> Dict:
        if len(json.dumps(v)) > 1024 * 1024:  # 序列化后不超过 1MB
            raise ValueError("body 序列化后不能超过 1MB")
        return v


class WeatherInput(BaseModel):
    """query_weather_api 工具参数模型"""
    city: str = Field(description="城市名称，支持中英文，例如：Beijing、上海")
    format: str = Field(default="json", description="返回格式：json（结构化数据）或 text（简洁文本）")

    @field_validator("city")  # 城市名非空校验
    @classmethod
    def validate_city(cls, v: str) -> str:
        v = v.strip()
        if not v:
            raise ValueError("city 不能为空")
        if len(v) > 100:
            raise ValueError("city 名称不能超过 100 个字符")
        return v

    @field_validator("format")  # 枚举校验
    @classmethod
    def validate_format(cls, v: str) -> str:
        allowed = {"json", "text"}
        if v not in allowed:
            raise ValueError(f"format 必须是 {allowed} 之一，收到: {v}")
        return v


class ServerStatusInput(BaseModel):
    """get_server_status 工具参数模型（无参数，保持一致性）"""
    pass  # 无需参数，Pydantic 模型占位，保持工具定义风格统一


# ── 通用：ValidationError → TextContent ─────────────────────────────────────

def _validation_error_response(e: ValidationError) -> list[types.TextContent]:
    """把 Pydantic 校验失败的详细信息格式化后返回给 LLM，让 LLM 自我修正"""
    errors = [
        f"  参数 '{'.'.join(str(x) for x in err['loc'])}'：{err['msg']}"
        for err in e.errors()
    ]
    return [types.TextContent(type="text", text="参数校验失败（Server 端）：\n" + "\n".join(errors))]


# ══════════════════════════════════════════════════════════════════════════════
# 注册工具：inputSchema 由 Pydantic model_json_schema() 自动生成
# ══════════════════════════════════════════════════════════════════════════════

@mcp_server.list_tools()
async def list_tools() -> list[types.Tool]:
    """Schema 从 Pydantic 模型自动生成，与校验逻辑永远同步"""
    return [
        types.Tool(
            name="http_get",
            description="发起 HTTP GET 请求，获取指定 URL 的响应内容",
            inputSchema=HttpGetInput.model_json_schema(),  # 自动生成，替代手写 dict
        ),
        types.Tool(
            name="http_post",
            description="发起 HTTP POST 请求，向指定 URL 发送 JSON 数据",
            inputSchema=HttpPostInput.model_json_schema(),
        ),
        types.Tool(
            name="query_weather_api",
            description="查询指定城市的实时天气数据（调用 wttr.in 公共 API，无需 Key）",
            inputSchema=WeatherInput.model_json_schema(),
        ),
        types.Tool(
            name="get_server_status",
            description="获取本 MCP Server 的运行状态和工具调用统计信息",
            inputSchema=ServerStatusInput.model_json_schema(),
        ),
    ]


@mcp_server.call_tool()
async def call_tool(name: str, arguments: dict) -> list[types.TextContent]:
    """工具调度：先 Pydantic 校验，再执行业务逻辑；校验失败返回错误信息给 LLM"""
    try:
        if name == "http_get":
            params = HttpGetInput(**arguments)  # Pydantic 校验
            return await _http_get(params)
        elif name == "http_post":
            params = HttpPostInput(**arguments)
            return await _http_post(params)
        elif name == "query_weather_api":
            params = WeatherInput(**arguments)
            return await _query_weather(params)
        elif name == "get_server_status":
            params = ServerStatusInput(**arguments)
            return await _get_server_status()
        else:
            return [types.TextContent(type="text", text=f"错误：未知工具 '{name}'")]
    except ValidationError as e:
        return _validation_error_response(e)  # 校验失败，返回结构化错误


# ── 工具实现（参数类型从 dict 改为 Pydantic 实例）────────────────────────────

_start_time = datetime.now()  # Server 启动时间
_call_count = 0  # 工具调用计数器


async def _http_get(params: HttpGetInput) -> list[types.TextContent]:
    """HTTP GET 请求，参数已由 Pydantic 校验"""
    global _call_count
    _call_count += 1
    try:
        async with httpx.AsyncClient() as client:
            response = await client.get(
                params.url,
                headers=params.headers or {},  # None 转为空 dict
                timeout=params.timeout,
            )
            result = {
                "status_code": response.status_code,
                "url": str(response.url),
                "content_type": response.headers.get("content-type", ""),
                "body": response.text[:2000],
                "truncated": len(response.text) > 2000,
            }
            return [types.TextContent(type="text", text=json.dumps(result, ensure_ascii=False))]
    except httpx.TimeoutException:
        return [types.TextContent(type="text", text=f"错误：请求超时（{params.timeout}秒）")]
    except Exception as e:
        return [types.TextContent(type="text", text=f"错误：{e}")]


async def _http_post(params: HttpPostInput) -> list[types.TextContent]:
    """HTTP POST 请求，参数已由 Pydantic 校验"""
    global _call_count
    _call_count += 1
    merged_headers = {"Content-Type": "application/json", **(params.headers or {})}  # 合并请求头
    try:
        async with httpx.AsyncClient() as client:
            response = await client.post(params.url, json=params.body, headers=merged_headers, timeout=15)
            result = {
                "status_code": response.status_code,
                "body": response.text[:2000],
                "truncated": len(response.text) > 2000,
            }
            return [types.TextContent(type="text", text=json.dumps(result, ensure_ascii=False))]
    except Exception as e:
        return [types.TextContent(type="text", text=f"错误：{e}")]


async def _query_weather(params: WeatherInput) -> list[types.TextContent]:
    """天气查询，调用 wttr.in 公共 API，参数已由 Pydantic 校验"""
    global _call_count
    _call_count += 1
    try:
        url = (f"https://wttr.in/{params.city}?format=j1"
               if params.format == "json"
               else f"https://wttr.in/{params.city}?format=3")
        async with httpx.AsyncClient() as client:
            response = await client.get(url, timeout=10, follow_redirects=True)
            if params.format == "json":
                data = response.json()
                current = data.get("current_condition", [{}])[0]
                weather_info = {
                    "city": params.city,
                    "temp_c": current.get("temp_C"),
                    "feels_like_c": current.get("FeelsLikeC"),
                    "humidity": current.get("humidity"),
                    "description": current.get("weatherDesc", [{}])[0].get("value", ""),
                    "wind_kmph": current.get("windspeedKmph"),
                }
                return [types.TextContent(type="text", text=json.dumps(weather_info, ensure_ascii=False))]
            else:
                return [types.TextContent(type="text", text=response.text)]
    except Exception as e:
        return [types.TextContent(type="text", text=f"天气查询失败：{e}")]


async def _get_server_status() -> list[types.TextContent]:
    """返回 Server 运行状态"""
    uptime = datetime.now() - _start_time
    status = {
        "server_name": "remote-api-tools",
        "transport": "SSE（旧版 HTTP 传输）",
        "started_at": _start_time.strftime("%Y-%m-%d %H:%M:%S"),
        "uptime_seconds": int(uptime.total_seconds()),
        "tool_calls_total": _call_count,
        "endpoints": {"sse": "GET /sse", "messages": "POST /messages", "health": "GET /health"},
    }
    return [types.TextContent(type="text", text=json.dumps(status, ensure_ascii=False, indent=2))]


# ══════════════════════════════════════════════════════════════════════════════
# FastAPI 路由
# ══════════════════════════════════════════════════════════════════════════════

@web_app.get("/sse")
async def sse_endpoint(request: Request):
    """SSE 长连接端点（Server → Client 推送通道）"""
    async with sse_transport.connect_sse(request.scope, request.receive, request._send) as streams:
        await mcp_server.run(streams[0], streams[1], mcp_server.create_initialization_options())


@web_app.post("/messages")
async def messages_endpoint(request: Request):
    """消息接收端点（Client → Server 请求通道）"""
    await sse_transport.handle_post_message(request.scope, request.receive, request._send)
    return Response()


@web_app.get("/health")
async def health_check():
    """健康检查端点"""
    return {"status": "ok", "server": "remote-api-tools", "transport": "sse",
            "uptime_seconds": int((datetime.now() - _start_time).total_seconds())}


if __name__ == "__main__":
    print("启动 MCP SSE Server (Pydantic 约束版)")
    print("SSE 端点：http://localhost:8001/sse")
    uvicorn.run(web_app, host="0.0.0.0", port=8001, log_level="info")
