"""
MCP 通信方式三：Streamable HTTP —— Pydantic 参数约束版
MCP 规范 2025-03-26 引入，官方推荐的 HTTP 传输方式。
运行方式：python 03_streamable_http_server.py（默认端口 8002）
端点：POST http://localhost:8002/mcp

Pydantic 改造说明：
  FastMCP 的 @mcp.tool() 已从函数签名自动生成 JSON Schema，
  但缺乏业务规则校验（日期格式、数值范围、枚举约束等）。
  改造方式：在每个工具函数内部用 Pydantic 模型二次校验，
  field_validator 承载业务规则，ValidationError 转为 dict 错误结果返回。

依赖：pip install "mcp[cli]" fastapi uvicorn pydantic python-dotenv
"""

import json  # JSON
import os  # 环境变量
import time  # 时间戳
from datetime import datetime  # 时间
from typing import Optional  # 类型注解
from contextlib import asynccontextmanager  # 生命周期

from dotenv import load_dotenv, find_dotenv  # 环境变量
from mcp.server.fastmcp import FastMCP  # 高级封装
from fastapi import FastAPI, Request  # Web 框架
from fastapi.middleware.cors import CORSMiddleware  # 跨域
from pydantic import BaseModel, Field, field_validator, ValidationError  # 参数约束

load_dotenv(find_dotenv())


# ══════════════════════════════════════════════════════════════════════════════
# Pydantic 业务规则模型
# FastMCP 负责基础类型校验，这些模型负责业务规则（日期格式、枚举、范围等）
# ══════════════════════════════════════════════════════════════════════════════

class SalesDataInput(BaseModel):
    """query_sales_data 业务规则模型"""
    start_date: str = Field(description="查询开始日期，格式 YYYY-MM-DD")
    end_date: str = Field(description="查询结束日期，格式 YYYY-MM-DD")
    region: Optional[str] = Field(default=None, description="地区：华北/华南/华东，不填返回全国")
    metric: str = Field(default="revenue", description="指标：revenue/orders/users")

    @field_validator("start_date", "end_date")  # 日期格式校验
    @classmethod
    def validate_date_format(cls, v: str) -> str:
        try:
            datetime.strptime(v, "%Y-%m-%d")
        except ValueError:
            raise ValueError(f"日期格式错误，应为 YYYY-MM-DD，收到: {v}")
        return v

    @field_validator("end_date")  # 结束日期不能早于开始日期（跨字段校验在 model_validator 里更合适，这里简化）
    @classmethod
    def validate_end_date(cls, v: str) -> str:
        # 单字段校验器只能拿到当前字段值，跨字段比较需要 model_validator
        # 这里只做格式校验，跨字段比较在业务函数里做
        return v

    @field_validator("region")  # 地区枚举校验
    @classmethod
    def validate_region(cls, v: Optional[str]) -> Optional[str]:
        if v is None:
            return v
        allowed = {"华北", "华南", "华东"}
        if v not in allowed:
            raise ValueError(f"region 必须是 {allowed} 之一，收到: {v}")
        return v

    @field_validator("metric")  # 指标枚举校验
    @classmethod
    def validate_metric(cls, v: str) -> str:
        allowed = {"revenue", "orders", "users"}
        if v not in allowed:
            raise ValueError(f"metric 必须是 {allowed} 之一，收到: {v}")
        return v


class KpiInput(BaseModel):
    """calculate_kpi 业务规则模型"""
    actual: float = Field(description="实际完成值")
    target: float = Field(description="目标值，必须大于 0")
    metric_name: str = Field(default="指标", description="指标名称，用于报告展示")

    @field_validator("target")  # 目标值必须为正数
    @classmethod
    def validate_target(cls, v: float) -> float:
        if v <= 0:
            raise ValueError("target 必须大于 0")
        return v

    @field_validator("metric_name")  # 名称长度限制
    @classmethod
    def validate_name(cls, v: str) -> str:
        if len(v) > 50:
            raise ValueError("metric_name 不能超过 50 个字符")
        return v.strip() or "指标"  # 空字符串归一化为默认值


class NotificationInput(BaseModel):
    """send_notification 业务规则模型"""
    channel: str = Field(description="通知渠道：email/sms/webhook")
    recipient: str = Field(description="接收方：邮箱/手机号/webhook URL")
    message: str = Field(description="通知正文")
    priority: str = Field(default="normal", description="优先级：urgent/normal/low")

    @field_validator("channel")  # 渠道枚举校验
    @classmethod
    def validate_channel(cls, v: str) -> str:
        allowed = {"email", "sms", "webhook"}
        if v not in allowed:
            raise ValueError(f"channel 必须是 {allowed} 之一，收到: {v}")
        return v

    @field_validator("recipient")  # 接收方非空
    @classmethod
    def validate_recipient(cls, v: str) -> str:
        if not v.strip():
            raise ValueError("recipient 不能为空")
        return v.strip()

    @field_validator("message")  # 消息长度限制
    @classmethod
    def validate_message(cls, v: str) -> str:
        if not v.strip():
            raise ValueError("message 不能为空")
        if len(v) > 1000:
            raise ValueError("message 不能超过 1000 个字符")
        return v.strip()

    @field_validator("priority")  # 优先级枚举
    @classmethod
    def validate_priority(cls, v: str) -> str:
        allowed = {"urgent", "normal", "low"}
        if v not in allowed:
            raise ValueError(f"priority 必须是 {allowed} 之一，收到: {v}")
        return v


# ── 通用：ValidationError → 错误 dict（FastMCP 工具返回 dict，不是 TextContent）──

def _pydantic_error_to_dict(e: ValidationError) -> dict:
    """把 ValidationError 格式化为标准错误 dict，FastMCP 工具统一用 dict 返回"""
    errors = [
        {"field": ".".join(str(x) for x in err["loc"]), "message": err["msg"]}
        for err in e.errors()
    ]
    return {"success": False, "error": "参数校验失败（Server 端）", "details": errors}


# ══════════════════════════════════════════════════════════════════════════════
# FastMCP 实例
# ══════════════════════════════════════════════════════════════════════════════

mcp = FastMCP(
    "enterprise-tools",
    instructions="企业级 MCP Server，提供数据查询、KPI 计算和消息通知工具。",
)

_stats = {"total_calls": 0, "calls_by_tool": {}, "errors": 0, "start_time": time.time()}


def _record_call(tool_name: str, success: bool = True):
    """记录工具调用统计"""
    _stats["total_calls"] += 1
    _stats["calls_by_tool"][tool_name] = _stats["calls_by_tool"].get(tool_name, 0) + 1
    if not success:
        _stats["errors"] += 1


# ══════════════════════════════════════════════════════════════════════════════
# FastMCP 工具：函数签名 → FastMCP 基础类型校验 + 内部 Pydantic 业务规则校验
# ══════════════════════════════════════════════════════════════════════════════

@mcp.tool()
async def query_sales_data(
    start_date: str,
    end_date: str,
    region: Optional[str] = None,
    metric: str = "revenue",
) -> dict:
    """
    查询销售数据，支持按日期范围和地区筛选。
    返回指定时间段的销售统计数据。
    """
    # FastMCP 完成基础类型校验后，用 Pydantic 做业务规则校验
    try:
        params = SalesDataInput(
            start_date=start_date, end_date=end_date,
            region=region, metric=metric,
        )
    except ValidationError as e:
        return _pydantic_error_to_dict(e)  # 校验失败，返回结构化错误

    # 跨字段业务规则（start_date <= end_date）
    if params.start_date > params.end_date:
        return {"success": False, "error": "start_date 不能晚于 end_date"}

    _record_call("query_sales_data")
    mock_data = {
        "华北": {"revenue": 1580000, "orders": 3200, "users": 18500},
        "华南": {"revenue": 2100000, "orders": 4800, "users": 25000},
        "华东": {"revenue": 3200000, "orders": 7100, "users": 38000},
    }
    regions = {params.region: mock_data[params.region]} if params.region and params.region in mock_data else mock_data
    return {
        "success": True,
        "query": {"start_date": params.start_date, "end_date": params.end_date,
                  "region": params.region or "全国", "metric": params.metric},
        "data": {r: v[params.metric] for r, v in regions.items()},
        "total": sum(v[params.metric] for v in regions.values()),
        "generated_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
    }


@mcp.tool()
async def calculate_kpi(
    actual: float,
    target: float,
    metric_name: str = "指标",
) -> dict:
    """
    计算 KPI 完成率，并给出达标评级。
    返回完成率、差距和评级（超额完成/达标/待提升/严重不足）。
    """
    try:
        params = KpiInput(actual=actual, target=target, metric_name=metric_name)
    except ValidationError as e:
        return _pydantic_error_to_dict(e)

    _record_call("calculate_kpi")
    rate = params.actual / params.target
    gap = params.actual - params.target

    if rate >= 1.2:
        grade = "超额完成 🏆"
    elif rate >= 1.0:
        grade = "达标 ✅"
    elif rate >= 0.8:
        grade = "待提升 ⚠️"
    else:
        grade = "严重不足 ❌"

    return {
        "success": True,
        "metric_name": params.metric_name,
        "actual": params.actual,
        "target": params.target,
        "completion_rate": f"{rate:.1%}",
        "gap": gap,
        "gap_pct": f"{gap/params.target:+.1%}",
        "grade": grade,
    }


@mcp.tool()
async def send_notification(
    channel: str,
    recipient: str,
    message: str,
    priority: str = "normal",
) -> dict:
    """
    发送通知消息，支持 email（邮件）、sms（短信）、webhook 三种渠道。
    企业场景：告警通知、审批提醒、报告推送等。
    """
    try:
        params = NotificationInput(channel=channel, recipient=recipient,
                                   message=message, priority=priority)
    except ValidationError as e:
        return _pydantic_error_to_dict(e)

    _record_call("send_notification")
    channel_labels = {"email": "📧 邮件", "sms": "📱 短信", "webhook": "🔗 Webhook"}
    return {
        "success": True,
        "message_id": f"msg_{int(time.time())}_{params.channel[:3]}",
        "channel": channel_labels[params.channel],
        "recipient": params.recipient,
        "priority": params.priority,
        "sent_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
    }


@mcp.tool()
async def get_server_metrics() -> dict:
    """获取本 MCP Server 的运行指标和调用统计，用于运维监控。"""
    _record_call("get_server_metrics")
    uptime = time.time() - _stats["start_time"]
    return {
        "server": "enterprise-tools",
        "transport": "Streamable HTTP（MCP 2025-03-26 规范）",
        "uptime_seconds": int(uptime),
        "total_calls": _stats["total_calls"],
        "error_rate": f"{_stats['errors']/max(_stats['total_calls'], 1):.1%}",
        "calls_by_tool": _stats["calls_by_tool"],
    }


@mcp.resource("config://server")
async def get_server_config() -> str:
    """返回 Server 配置信息（只读资源）"""
    config = {
        "version": "1.0.0",
        "transport": "streamable-http",
        "supported_tools": ["query_sales_data", "calculate_kpi", "send_notification", "get_server_metrics"],
        "pydantic_validation": "启用（业务规则校验层）",
    }
    return json.dumps(config, ensure_ascii=False, indent=2)


# ══════════════════════════════════════════════════════════════════════════════
# FastAPI 应用组装
# ══════════════════════════════════════════════════════════════════════════════

@asynccontextmanager
async def lifespan(app: FastAPI):
    print(f"[{datetime.now():%H:%M:%S}] MCP Streamable HTTP Server 启动（Pydantic 约束版）")
    print("端点：POST http://localhost:8002/mcp")
    yield
    print(f"[{datetime.now():%H:%M:%S}] Server 关闭")


web_app = FastAPI(title="Enterprise MCP Server", version="1.0.0", lifespan=lifespan)
web_app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])


@web_app.middleware("http")
async def log_requests(request: Request, call_next):
    """请求日志中间件"""
    import time as t
    start = t.time()
    response = await call_next(request)
    print(f"[{datetime.now():%H:%M:%S}] {request.method} {request.url.path} "
          f"→ {response.status_code} ({(t.time()-start)*1000:.1f}ms)")
    return response


web_app.mount("/mcp", mcp.streamable_http_app())  # 挂载 Streamable HTTP 端点


@web_app.get("/health")
async def health_check():
    return {"status": "healthy", "server": "enterprise-tools",
            "uptime_seconds": int(time.time() - _stats["start_time"])}


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(web_app, host="0.0.0.0", port=8002, log_level="warning")
