import time
from pathlib import Path

import jwt
import requests

# ── 和风天气订阅版配置 ────────────────────────────────────────────────────────
# host 子域名即 Project ID，Credential ID 来自控制台
API_HOST       = "pc7h2tfryx.re.qweatherapi.com"
PROJECT_ID     = "25TQ5R7WEB"   #项目id                       # host 子域名
CREDENTIAL_ID  = "T8GYPRFJQU"   # 项目的jwt生成的凭证id

GEO_PATH       = "/geo/v2/city/lookup"
WEATHER_PATH   = "/v7/weather/30d"

# Ed25519 私钥路径（和风天气控制台下载）
_PRIVATE_KEY_PATH = Path(__file__).parent / "Auth_JWT" / "ed25519-private.pem"
_PRIVATE_KEY = _PRIVATE_KEY_PATH.read_text()


# ── JWT 生成 ──────────────────────────────────────────────────────────────────

def _make_jwt(ttl: int = 900) -> str:
    """
    生成和风天气订阅版所需的 JWT。
    和风要求：
      alg = EdDSA（Ed25519）
      kid = Credential ID
      sub = Project ID
      iat = 签发时间（Unix 秒）
      exp = 过期时间（iat + ttl，最长不超过 900 秒）
    """
    now = int(time.time())
    payload = {"sub": PROJECT_ID, "iat": now, "exp": now + ttl}
    headers = {"alg": "EdDSA","kid": CREDENTIAL_ID}
    return jwt.encode(payload, _PRIVATE_KEY, algorithm="EdDSA", headers=headers)


def _build_headers() -> dict:
    """每次请求前生成新 JWT（避免 token 过期）"""
    return {
        "Authorization": f"Bearer {_make_jwt()}",
        "Accept-Encoding": "gzip, deflate",   # 对应 curl --compressed
    }


# ── API 调用 ──────────────────────────────────────────────────────────────────

def get_location_id(location: str) -> str:
    """通过城市名查询和风天气 LocationID"""
    url = f"https://{API_HOST}{GEO_PATH}"
    resp = requests.get(url, headers=_build_headers(), params={"location": location})
    resp.raise_for_status()
    data = resp.json()
    if data.get("code") != "200":
        raise ValueError(f"GEO API 错误：code={data.get('code')}，location={location}")
    return data["location"][0]["id"]   # location 是列表，取第一个最佳匹配


def get_weather(location: str) -> dict:
    """
    获取指定城市的 30 天天气预报
    :param location: 城市名称（中文或英文）
    :return: 天气信息 dict
    """
    location_id = get_location_id(location)
    url = f"https://{API_HOST}{WEATHER_PATH}"
    resp = requests.get(url, headers=_build_headers(), params={"location": location_id})
    resp.raise_for_status()
    data = resp.json()
    if data.get("code") != "200":
        raise ValueError(f"Weather API 错误：code={data.get('code')}，locationId={location_id}")
    return data


if __name__ == '__main__':
    print(get_weather("乌兰察布"))