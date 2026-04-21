from __future__ import annotations

import json
import os
import socket
from typing import Any, Dict
from urllib.parse import urlparse

import requests
from requests.exceptions import ConnectionError, ConnectTimeout, ProxyError, ReadTimeout, SSLError, Timeout

from services.txt_config_parser import parse_key_value_text


PROXY_ENV_KEYS = ("HTTP_PROXY", "HTTPS_PROXY", "ALL_PROXY", "http_proxy", "https_proxy", "all_proxy")


def _as_bool(value: Any, default: bool = False) -> bool:
    if value is None:
        return default
    if isinstance(value, bool):
        return value
    return str(value).strip().lower() in {"1", "true", "yes", "y", "on"}


def _mask_secret(value: str) -> str:
    if not value:
        return ""
    if len(value) <= 8:
        return "***"
    return f"{value[:4]}...{value[-4:]}"


def _mask_header_value(name: str, value: str) -> str:
    lower_name = name.lower()
    if lower_name in {"authorization", "x-api-key", "api-key"} or "token" in lower_name or "key" in lower_name:
        if value.lower().startswith("bearer "):
            return f"Bearer {_mask_secret(value[7:].strip())}"
        return _mask_secret(value)
    return value


def sanitize_headers(headers: Dict[str, str]) -> Dict[str, str]:
    return {str(key): _mask_header_value(str(key), str(value)) for key, value in headers.items()}


def proxy_env_snapshot() -> Dict[str, str]:
    snapshot = {}
    for key in PROXY_ENV_KEYS:
        value = os.getenv(key)
        if value:
            snapshot[key] = _mask_secret(value)
    return snapshot


def proxies_enabled(disable_proxy: bool) -> bool:
    return (not disable_proxy) and bool(proxy_env_snapshot())


def request_diagnostics(url: str, headers: Dict[str, str], disable_proxy: bool) -> Dict[str, Any]:
    return {
        "final_url": url,
        "disable_proxy": disable_proxy,
        "proxy_enabled": proxies_enabled(disable_proxy),
        "proxy_env": proxy_env_snapshot(),
        "headers": sanitize_headers(headers),
    }


def _format_diagnostics(diagnostics: Dict[str, Any]) -> str:
    return json.dumps(diagnostics, ensure_ascii=False, indent=2)


def _normalize_message_content(content: Any) -> str:
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts = []
        for item in content:
            if isinstance(item, dict) and item.get("type") in {"text", "output_text"} and item.get("text"):
                parts.append(str(item["text"]))
        if parts:
            return "\n".join(parts).strip()
    return ""


def classify_network_error(exc: BaseException, url: str) -> str:
    parsed = urlparse(url)
    host = parsed.hostname or ""
    if isinstance(exc, ProxyError) or "ProxyError" in repr(exc):
        return "代理问题：当前请求尝试通过系统代理转发，但代理连接失败或被远端关闭。可开启“禁用系统代理”。"
    if isinstance(exc, (ConnectTimeout, ReadTimeout, Timeout)):
        return "网络超时：目标服务或中间网络未在 timeout 内响应。"
    if isinstance(exc, SSLError):
        return "TLS/证书问题：HTTPS 握手失败，可能是证书、代理或网络拦截导致。"
    if isinstance(exc, ConnectionError):
        message = str(exc).lower()
        if "name resolution" in message or "getaddrinfo" in message or "temporary failure in name resolution" in message:
            return "DNS 问题：域名解析失败，请检查 endpoint 域名是否正确或本机 DNS。"
        if "connection refused" in message:
            return "网络阻断：目标主机拒绝连接，请检查端口、endpoint 或防火墙。"
        if host:
            try:
                socket.gethostbyname(host)
            except OSError:
                return "DNS 问题：域名无法解析，请检查 endpoint 域名是否正确。"
        return "网络阻断：无法连接目标服务，可能被防火墙、代理或网络策略拦截。"
    if parsed.path.rstrip("/") not in {"", "/v1", "/v1/chat/completions"} and not parsed.path.endswith("/chat/completions"):
        return "endpoint 可能错误：OpenAI-compatible Base URL 通常应是域名根路径或 /v1。"
    return "未知网络错误：请结合最终请求 URL、代理状态和服务端日志继续排查。"


def parse_headers(raw_or_dict) -> Dict[str, str]:
    if isinstance(raw_or_dict, dict):
        return {str(key): str(value) for key, value in raw_or_dict.items() if value not in (None, "")}
    if not raw_or_dict or not str(raw_or_dict).strip():
        return {}

    raw = str(raw_or_dict)
    try:
        parsed = json.loads(raw)
        if not isinstance(parsed, dict):
            raise ValueError("Headers 必须是 JSON 对象或 key=value 文本。")
        return {str(key): str(value) for key, value in parsed.items() if value not in (None, "")}
    except json.JSONDecodeError:
        parsed, warnings = parse_key_value_text(raw, "Headers")
        if warnings and not parsed:
            raise ValueError("Headers 格式无法解析，请使用每行 key=value。")
        return parsed


def parse_extra_params(raw_or_dict) -> Dict[str, Any]:
    if isinstance(raw_or_dict, dict):
        return raw_or_dict
    if not raw_or_dict or not str(raw_or_dict).strip():
        return {}
    parsed, warnings = parse_key_value_text(str(raw_or_dict), "Extra Params")
    if warnings and not parsed:
        raise ValueError("Extra Params 格式无法解析，请使用每行 key=value。")
    return parsed


def normalize_base_url(base_url: str) -> str:
    cleaned = base_url.strip().rstrip("/")
    if not cleaned:
        raise ValueError("Base URL 不能为空。")
    return cleaned


def chat_completions_url(base_url: str) -> str:
    cleaned = normalize_base_url(base_url)
    if cleaned.endswith("/chat/completions"):
        return cleaned
    if cleaned.endswith("/v1"):
        return f"{cleaned}/chat/completions"
    return f"{cleaned}/v1/chat/completions"


def validate_custom_llm_config(config: Dict[str, Any]) -> None:
    if not config:
        raise ValueError("请先保存 Custom LLM API 配置。")
    if not config.get("base_url", "").strip():
        raise ValueError("Base URL 不能为空。")
    if not config.get("api_key", ""):
        raise ValueError("API Key 不能为空。")
    model_name = config.get("model_name") or config.get("model") or ""
    if not model_name.strip():
        raise ValueError("Model Name 不能为空。")
    parse_headers(config.get("headers", config.get("additional_headers", "")))
    parse_extra_params(config.get("extra_params", ""))


def _post_json(url: str, headers: Dict[str, str], payload: Dict[str, Any], disable_proxy: bool, timeout: int = 180):
    if disable_proxy:
        session = requests.Session()
        session.trust_env = False
        return session.post(url, headers=headers, json=payload, timeout=timeout)
    return requests.post(url, headers=headers, json=payload, timeout=timeout)


def run_custom_llm_chat_completion(
    config: Dict[str, Any],
    messages: list,
    temperature: float,
) -> str:
    validate_custom_llm_config(config)

    headers = {
        "Authorization": f"Bearer {config['api_key']}",
        "Content-Type": "application/json",
    }
    headers.update(parse_headers(config.get("headers", config.get("additional_headers", ""))))
    disable_proxy = _as_bool(config.get("disable_proxy"), default=True)

    payload = {
        "model": (config.get("model_name") or config.get("model", "")).strip(),
        "temperature": temperature,
        "messages": messages,
    }
    payload.update(parse_extra_params(config.get("extra_params", "")))

    url = chat_completions_url(config["base_url"])
    diagnostics = request_diagnostics(url, headers, disable_proxy)
    print(f"OpenAI-compatible request diagnostics:\n{_format_diagnostics(diagnostics)}")

    try:
        response = _post_json(url, headers, payload, disable_proxy, timeout=180)
    except requests.RequestException as exc:
        category = classify_network_error(exc, url)
        raise RuntimeError(
            "OpenAI-compatible 请求失败："
            f"{category}\n"
            f"原始错误：{exc}\n"
            f"请求诊断：{_format_diagnostics(diagnostics)}"
        ) from exc

    if response.status_code != 200:
        endpoint_hint = ""
        if response.status_code == 404:
            endpoint_hint = " 可能是 endpoint 路径错误，请确认 Base URL 是否应填写到 /v1。"
        raise RuntimeError(
            "OpenAI-compatible 请求失败："
            f"status_code={response.status_code}，body={response.text[:500]}{endpoint_hint}\n"
            f"请求诊断：{_format_diagnostics(diagnostics)}"
        )

    try:
        data = response.json()
    except ValueError as exc:
        raise RuntimeError("OpenAI-compatible 请求失败：响应不是合法 JSON。") from exc

    print(
        f"[OPENAI_COMPAT_RESPONSE] status={response.status_code} "
        f"body_preview={response.text[:300]!r}"
    )

    choices = data.get("choices")
    if not choices:
        raise RuntimeError("LLM 返回为空。")

    message = choices[0].get("message") if isinstance(choices[0], dict) else None
    content = message.get("content") if isinstance(message, dict) else None
    text = _normalize_message_content(content)
    if not (text or content):
        raise RuntimeError("未提取到 message.content。")
    return text or content


def test_openai_compatible_endpoint(config: Dict[str, Any], timeout: int = 20) -> Dict[str, Any]:
    """Minimal standalone connectivity test for an OpenAI-compatible endpoint."""
    validate_custom_llm_config(config)
    headers = {
        "Authorization": f"Bearer {config['api_key']}",
        "Content-Type": "application/json",
    }
    headers.update(parse_headers(config.get("headers", config.get("additional_headers", ""))))
    disable_proxy = _as_bool(config.get("disable_proxy"), default=True)
    url = chat_completions_url(config["base_url"])
    payload = {
        "model": (config.get("model_name") or config.get("model", "")).strip(),
        "temperature": 0,
        "max_tokens": 8,
        "messages": [{"role": "user", "content": "ping"}],
    }
    diagnostics = request_diagnostics(url, headers, disable_proxy)
    try:
        response = _post_json(url, headers, payload, disable_proxy, timeout=timeout)
    except requests.RequestException as exc:
        return {
            "ok": False,
            "category": classify_network_error(exc, url),
            "error": str(exc),
            "diagnostics": diagnostics,
        }
    return {
        "ok": response.status_code == 200,
        "status_code": response.status_code,
        "body_preview": response.text[:500],
        "diagnostics": diagnostics,
        "category": "连接成功" if response.status_code == 200 else "endpoint错误或鉴权/模型错误",
    }
