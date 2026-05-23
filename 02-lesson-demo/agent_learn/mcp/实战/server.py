"""
实战 MCP Server - FastMCP 高级 API
工具：get_current_time / read_file / write_file / list_directory

启动方式：
  python server.py            # stdio（默认，Client 自动启动子进程）
  python server.py sse        # SSE  → http://localhost:8001/sse
  python server.py http       # Streamable HTTP → http://localhost:8002/mcp
"""

import json
import sys
from datetime import datetime
from pathlib import Path

from dotenv import load_dotenv, find_dotenv
from mcp.server.fastmcp import FastMCP
from pydantic import BaseModel, Field, field_validator, ValidationError

load_dotenv(find_dotenv())

_transport = sys.argv[1] if len(sys.argv) > 1 else "stdio"
_port = {"sse": 8001, "http": 8002}.get(_transport, 8000)

mcp = FastMCP("实战工具箱", port=_port)


# ══════════════════════════════════════════════════════════════════════════════
# Pydantic 业务规则模型（FastMCP 负责基础类型，这里负责安全/范围校验）
# ══════════════════════════════════════════════════════════════════════════════

class _ReadInput(BaseModel):
    path: str
    encoding: str = "utf-8"

    @field_validator("path")
    @classmethod
    def no_traversal(cls, v: str) -> str:
        if ".." in Path(v).parts:
            raise ValueError("不允许路径穿越（'..'）")
        return v.strip()

    @field_validator("encoding")
    @classmethod
    def allowed_encoding(cls, v: str) -> str:
        allowed = {"utf-8", "utf-8-sig", "gbk", "gb2312", "ascii"}
        if v.lower() not in allowed:
            raise ValueError(f"不支持的编码 {v}，允许：{allowed}")
        return v.lower()


class _WriteInput(BaseModel):
    path: str
    content: str
    append: bool = False

    @field_validator("path")
    @classmethod
    def no_traversal(cls, v: str) -> str:
        if ".." in Path(v).parts:
            raise ValueError("不允许路径穿越（'..'）")
        return v.strip()

    @field_validator("content")
    @classmethod
    def size_limit(cls, v: str) -> str:
        if len(v.encode()) > 10 * 1024 * 1024:
            raise ValueError("内容超过 10MB 限制")
        return v


# ══════════════════════════════════════════════════════════════════════════════
# 工具定义
# ══════════════════════════════════════════════════════════════════════════════

@mcp.tool()
async def get_current_time(timezone: str | None = None) -> str:
    """获取当前系统时间，可指定时区名称（如 Asia/Shanghai）"""
    now = datetime.now()
    return json.dumps({
        "time": now.strftime("%Y-%m-%d %H:%M:%S"),
        "timezone": timezone or "本地时区",
        "timestamp": now.timestamp(),
        "weekday": ["周一", "周二", "周三", "周四", "周五", "周六", "周日"][now.weekday()],
    }, ensure_ascii=False)


@mcp.tool()
async def read_file(path: str, encoding: str = "utf-8") -> str:
    """读取指定路径的文件内容"""
    try:
        params = _ReadInput(path=path, encoding=encoding)
        return Path(params.path).read_text(encoding=params.encoding)
    except ValidationError as e:
        return f"参数校验失败：{e}"
    except FileNotFoundError:
        return f"错误：文件不存在 '{path}'"
    except PermissionError:
        return f"错误：没有权限读取 '{path}'"
    except Exception as e:
        return f"错误：{e}"


@mcp.tool()
async def write_file(path: str, content: str, append: bool = False) -> str:
    """将内容写入文件，文件不存在则自动创建；append=True 时追加写入"""
    try:
        params = _WriteInput(path=path, content=content, append=append)
        p = Path(params.path)
        p.parent.mkdir(parents=True, exist_ok=True)
        mode = "a" if params.append else "w"
        with p.open(mode, encoding="utf-8") as f:
            f.write(params.content)
        action = "追加" if params.append else "写入"
        return f"成功：已{action} {len(params.content)} 字符到 '{params.path}'"
    except ValidationError as e:
        return f"参数校验失败：{e}"
    except Exception as e:
        return f"错误：{e}"


@mcp.tool()
async def list_directory(path: str = ".", show_hidden: bool = False) -> str:
    """列出目录下的文件和子目录；show_hidden=True 时显示隐藏文件"""
    try:
        if ".." in Path(path).parts:
            return "错误：不允许路径穿越"
        entries = []
        for item in sorted(Path(path).iterdir()):
            if not show_hidden and item.name.startswith("."):
                continue
            tag = "📁" if item.is_dir() else "📄"
            size = f"  {item.stat().st_size:,}B" if item.is_file() else ""
            entries.append(f"{tag} {item.name}{size}")
        header = f"目录：{Path(path).absolute()}  共 {len(entries)} 项"
        return header + "\n" + "\n".join(entries)
    except Exception as e:
        return f"错误：{e}"


# ══════════════════════════════════════════════════════════════════════════════
# 启动
# ══════════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    transport_map = {"sse": "sse", "http": "streamable-http"}
    mcp_transport = transport_map.get(_transport, "stdio")

    if mcp_transport != "stdio":
        print(f"启动 MCP Server [{mcp_transport}]  端口：{_port}")
        if mcp_transport == "sse":
            print(f"SSE 端点：http://localhost:{_port}/sse")
        else:
            print(f"HTTP 端点：http://localhost:{_port}/mcp")

    mcp.run(transport=mcp_transport)
