"""
MCP 通信方式一：stdio（标准输入输出）—— Pydantic 参数约束版
运行方式：不直接运行，由 MCP Host（如 Claude Desktop）以子进程方式启动
本地测试：python 01_stdio_mcp_server.py（会阻塞等待 stdin 输入，Ctrl+C 退出）

改造说明（相对原始版本）：
  原始版本：手写 JSON Schema dict + 裸 dict 访问（args["path"]、args.get(...)）
  改造后：  Pydantic BaseModel 同时承担两个职责：
              1. model_json_schema() → 自动生成 inputSchema（不再手写 dict）
              2. 接收 arguments dict 时做类型校验 + 业务规则校验

  约束放在 Server 的理由：
    MCP Server 会被多个 Client 调用（Claude Desktop、LangChain、curl 等）。
    Server 是唯一能对所有调用方无条件生效的校验层。
    Client 侧的校验是"锦上添花"，不是安全保证。

  StructuredTool 为什么不适合放在 Server：
    StructuredTool 是 LangChain 客户端抽象，职责是把函数+Schema打包供 Agent 调用。
    MCP Server 使用 mcp.types.Tool + @app.call_tool()，是另一套体系。
    两者不冲突，但各司其职：Pydantic（Server 校验）+ StructuredTool（Client 调用）。

配套的 Claude Desktop 配置（~/Library/Application Support/Claude/claude_desktop_config.json）：
{
  "mcpServers": {
    "local-tools": {
      "command": "python",
      "args": ["/绝对路径/01_stdio_mcp_server.py"],
      "env": {"PYTHONPATH": "/绝对路径"}
    }
  }
}
"""

import asyncio  # 异步支持
import json  # JSON 格式化输出
import os  # 环境变量和文件操作
from datetime import datetime  # 时间处理
from pathlib import Path  # 路径操作
from typing import Optional  # 可选类型

from pydantic import BaseModel, Field, field_validator, ValidationError  # 参数约束核心
from mcp.server import Server  # MCP Server 核心类
from mcp.server.stdio import stdio_server  # stdio 传输层
from mcp import types  # MCP 数据类型


# ── 创建 MCP Server 实例 ────────────────────────────────────────────────────

app = Server("local-tools")  # 参数是 Server 名称，会出现在 Agent Card 里


# ══════════════════════════════════════════════════════════════════════════════
# Pydantic 参数模型
# 职责一：model_json_schema() 自动生成 inputSchema，替代手写 dict
# 职责二：接收 arguments 时做类型校验 + 业务规则校验
# ══════════════════════════════════════════════════════════════════════════════

class GetCurrentTimeInput(BaseModel):
    """get_current_time 工具的参数模型"""
    timezone: Optional[str] = Field(
        default=None,
        description="时区名称，例如 Asia/Shanghai、UTC，默认本地时区",
    )


class ReadFileInput(BaseModel):
    """read_file 工具的参数模型"""
    path: str = Field(description="文件的绝对路径或相对路径")
    encoding: str = Field(default="utf-8", description="文件编码，默认 utf-8")

    @field_validator("path")  # 路径安全校验，防止路径穿越攻击
    @classmethod
    def validate_path(cls, v: str) -> str:
        if not v.strip():  # 空路径拒绝
            raise ValueError("path 不能为空")
        if ".." in Path(v).parts:  # 拒绝 ../ 路径穿越
            raise ValueError("path 不允许包含 '..'（路径穿越）")
        return v.strip()  # 去除首尾空白

    @field_validator("encoding")  # 编码白名单，防止传入无效编码导致崩溃
    @classmethod
    def validate_encoding(cls, v: str) -> str:
        allowed = {"utf-8", "utf-8-sig", "gbk", "gb2312", "ascii", "latin-1"}
        if v.lower() not in allowed:
            raise ValueError(f"encoding 必须是 {allowed} 之一，收到: {v}")
        return v.lower()  # 统一小写


class WriteFileInput(BaseModel):
    """write_file 工具的参数模型"""
    path: str = Field(description="目标文件路径")
    content: str = Field(description="要写入的文本内容")
    append: bool = Field(default=False, description="是否追加写入，默认 false（覆盖）")

    @field_validator("path")  # 路径安全校验
    @classmethod
    def validate_path(cls, v: str) -> str:
        if not v.strip():
            raise ValueError("path 不能为空")
        if ".." in Path(v).parts:
            raise ValueError("path 不允许包含 '..'（路径穿越）")
        return v.strip()

    @field_validator("content")  # 内容大小限制，防止写入超大文件
    @classmethod
    def validate_content_size(cls, v: str) -> str:
        max_bytes = 10 * 1024 * 1024  # 10MB 上限
        if len(v.encode("utf-8")) > max_bytes:
            raise ValueError(f"content 超出 10MB 大小限制")
        return v


class ListDirectoryInput(BaseModel):
    """list_directory 工具的参数模型"""
    path: str = Field(default=".", description="目录路径，默认当前目录")
    show_hidden: bool = Field(default=False, description="是否显示隐藏文件（. 开头），默认 false")

    @field_validator("path")  # 路径安全校验
    @classmethod
    def validate_path(cls, v: str) -> str:
        if ".." in Path(v).parts:
            raise ValueError("path 不允许包含 '..'（路径穿越）")
        return v.strip() or "."  # 空字符串归一化为当前目录


# ── 通用：把 Pydantic ValidationError 转成 MCP TextContent 错误响应 ─────────

def _validation_error_response(e: ValidationError) -> list[types.TextContent]:
    """
    Pydantic 校验失败时，把错误信息格式化成 MCP TextContent 返回给 Client。
    不抛异常：LLM 需要读到错误信息才能自我修正参数。
    """
    errors = []
    for err in e.errors():
        field = ".".join(str(x) for x in err["loc"])  # 出错字段路径
        msg = err["msg"]  # 错误描述
        errors.append(f"  参数 '{field}'：{msg}")
    error_text = "参数校验失败（Server 端）：\n" + "\n".join(errors)
    return [types.TextContent(type="text", text=error_text)]


# ══════════════════════════════════════════════════════════════════════════════
# 注册 Tools
# inputSchema 由 Pydantic model_json_schema() 自动生成，不再手写 dict
# ══════════════════════════════════════════════════════════════════════════════

@app.list_tools()  # 响应 tools/list 请求，返回工具清单
async def list_tools() -> list[types.Tool]:
    """
    inputSchema 直接从 Pydantic 模型生成。
    好处：Schema 定义和校验逻辑在同一个 Pydantic 模型里，修改时只改一处，不会不同步。
    """
    return [
        types.Tool(
            name="get_current_time",
            description="获取当前系统时间，可指定时区",
            inputSchema=GetCurrentTimeInput.model_json_schema(),  # 自动生成，替代手写 dict
        ),
        types.Tool(
            name="read_file",
            description="读取指定路径的文件内容，返回文本",
            inputSchema=ReadFileInput.model_json_schema(),  # 自动生成
        ),
        types.Tool(
            name="write_file",
            description="将内容写入文件，文件不存在则创建",
            inputSchema=WriteFileInput.model_json_schema(),  # 自动生成
        ),
        types.Tool(
            name="list_directory",
            description="列出目录下的文件和子目录",
            inputSchema=ListDirectoryInput.model_json_schema(),  # 自动生成
        ),
    ]


@app.call_tool()  # 响应 tools/call 请求，执行工具并返回结果
async def call_tool(name: str, arguments: dict) -> list[types.TextContent]:
    """
    工具调度器：先用 Pydantic 校验参数，再执行业务逻辑。
    校验失败时返回错误文本，不抛异常（让 LLM 读到错误后自我修正）。
    """
    try:
        if name == "get_current_time":
            params = GetCurrentTimeInput(**arguments)   # ① Pydantic 校验
            return await _get_current_time(params)      # ② 校验通过才执行
        elif name == "read_file":
            params = ReadFileInput(**arguments)
            return await _read_file(params)
        elif name == "write_file":
            params = WriteFileInput(**arguments)
            return await _write_file(params)
        elif name == "list_directory":
            params = ListDirectoryInput(**arguments)
            return await _list_directory(params)
        else:
            return [types.TextContent(type="text", text=f"错误：未知工具 '{name}'")]

    except ValidationError as e:
        return _validation_error_response(e)  # Pydantic 校验失败，返回结构化错误信息


# ── 工具实现：参数类型从 dict 改为 Pydantic 模型实例 ─────────────────────────
# 改造前：path = args["path"]（裸 dict 访问，可能 KeyError）
# 改造后：params.path（Pydantic 对象属性，类型已保证，有 IDE 提示）

async def _get_current_time(params: GetCurrentTimeInput) -> list[types.TextContent]:
    """获取当前时间，参数已由 Pydantic 校验"""
    now = datetime.now()  # 获取当前时间
    result = {
        "time": now.strftime("%Y-%m-%d %H:%M:%S"),  # 格式化时间
        "timezone": params.timezone or "本地时区",   # 直接访问属性，无需 .get()
        "timestamp": now.timestamp(),               # Unix 时间戳
        "weekday": ["周一","周二","周三","周四","周五","周六","周日"][now.weekday()],
    }
    return [types.TextContent(type="text", text=json.dumps(result, ensure_ascii=False))]


async def _read_file(params: ReadFileInput) -> list[types.TextContent]:
    """读取文件内容，参数已由 Pydantic 校验（含路径安全检查）"""
    try:
        content = Path(params.path).read_text(encoding=params.encoding)  # 属性访问，类型安全
        return [types.TextContent(type="text", text=content)]
    except FileNotFoundError:
        return [types.TextContent(type="text", text=f"错误：文件不存在 '{params.path}'")]
    except PermissionError:
        return [types.TextContent(type="text", text=f"错误：没有权限读取 '{params.path}'")]
    except Exception as e:
        return [types.TextContent(type="text", text=f"错误：{str(e)}")]


async def _write_file(params: WriteFileInput) -> list[types.TextContent]:
    """写入文件，参数已由 Pydantic 校验（含路径安全 + 内容大小限制）"""
    try:
        Path(params.path).parent.mkdir(parents=True, exist_ok=True)  # 自动创建父目录
        mode = "a" if params.append else "w"  # 追加或覆盖
        Path(params.path).open(mode, encoding="utf-8").write(params.content)
        action = "追加" if params.append else "写入"
        return [types.TextContent(type="text", text=f"成功：已{action} {len(params.content)} 个字符到 '{params.path}'")]
    except Exception as e:
        return [types.TextContent(type="text", text=f"错误：{str(e)}")]


async def _list_directory(params: ListDirectoryInput) -> list[types.TextContent]:
    """列出目录内容，参数已由 Pydantic 校验"""
    try:
        entries = []
        for item in sorted(Path(params.path).iterdir()):
            if not params.show_hidden and item.name.startswith("."):  # 隐藏文件过滤
                continue
            entry_type = "📁" if item.is_dir() else "📄"
            size = f"{item.stat().st_size:,} B" if item.is_file() else ""
            entries.append(f"{entry_type} {item.name}  {size}")
        result = f"目录：{Path(params.path).absolute()}\n共 {len(entries)} 项：\n" + "\n".join(entries)
        return [types.TextContent(type="text", text=result)]
    except Exception as e:
        return [types.TextContent(type="text", text=f"错误：{str(e)}")]


# ══════════════════════════════════════════════════════════════════════════════
# 注册 Resources（只读上下文数据，无副作用）
# ══════════════════════════════════════════════════════════════════════════════

@app.list_resources()  # 响应 resources/list 请求
async def list_resources() -> list[types.Resource]:
    """告诉 Client 本 Server 提供哪些资源"""
    return [
        types.Resource(
            uri="file:///etc/hostname",  # 资源的唯一 URI
            name="主机名",
            description="当前机器的主机名",
            mimeType="text/plain",  # 内容类型
        ),
        types.Resource(
            uri="env://PATH",
            name="系统 PATH",
            description="系统环境变量 PATH 的值",
            mimeType="text/plain",
        ),
    ]


@app.read_resource()  # 响应 resources/read 请求
async def read_resource(uri: str) -> str:
    """读取指定 URI 的资源内容"""
    if uri == "file:///etc/hostname":
        try:
            return Path("/etc/hostname").read_text().strip()  # 读取主机名文件
        except Exception:
            return os.uname().nodename  # fallback：从系统获取
    elif uri == "env://PATH":
        return os.environ.get("PATH", "（PATH 未设置）")  # 读取环境变量
    else:
        raise ValueError(f"未知资源 URI：{uri}")  # 未知 URI 抛出异常


# ══════════════════════════════════════════════════════════════════════════════
# 启动入口：stdio 模式
# ══════════════════════════════════════════════════════════════════════════════

async def main():
    """
    stdio_server() 返回 (read_stream, write_stream)。
    read_stream：从 stdin 读取 Client 发来的 JSON-RPC 消息
    write_stream：向 stdout 写入 Server 的响应
    app.run() 进入事件循环，持续处理消息直到 stdin 关闭
    """
    async with stdio_server() as (read_stream, write_stream):  # 启动 stdio 传输层
        await app.run(
            read_stream,  # 读通道（来自 stdin）
            write_stream,  # 写通道（到 stdout）
            app.create_initialization_options(),  # 初始化选项（能力协商）
        )


if __name__ == "__main__":
    asyncio.run(main())  # 启动异步事件循环
