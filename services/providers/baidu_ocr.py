from __future__ import annotations

import base64
import time
from typing import Any, Dict

import requests


SUPPORTED_EXTENSIONS = {"png", "jpg", "jpeg", "webp"}
TOKEN_URL = "https://aip.baidubce.com/oauth/2.0/token"
OCR_ENDPOINTS = {
    "general": "https://aip.baidubce.com/rest/2.0/ocr/v1/general",
    "accurate": "https://aip.baidubce.com/rest/2.0/ocr/v1/accurate_basic",
}


def _file_extension(file_name: str) -> str:
    return file_name.lower().rsplit(".", 1)[-1] if "." in file_name else ""


def _validate_config(config: Dict[str, Any]) -> Dict[str, str]:
    if not config:
        raise ValueError("请先保存 Baidu OCR 配置。")

    api_key = config.get("api_key", "").strip()
    secret_key = config.get("secret_key", "")
    mode = config.get("mode", "accurate")

    if not api_key:
        raise ValueError("Baidu OCR API Key 不能为空。")
    if not secret_key:
        raise ValueError("Baidu OCR Secret Key 不能为空。")
    if mode not in {"general", "accurate"}:
        raise ValueError("Baidu OCR 模式必须是 general 或 accurate。")

    return {
        "api_key": api_key,
        "secret_key": secret_key,
        "mode": mode,
    }


def get_access_token(config: Dict[str, Any]) -> str:
    settings = _validate_config(config)
    cached_token = config.get("access_token", "")
    expires_at = float(config.get("access_token_expires_at") or 0)
    if cached_token and expires_at > time.time() + 60:
        return cached_token

    params = {
        "grant_type": "client_credentials",
        "client_id": settings["api_key"],
        "client_secret": settings["secret_key"],
    }

    try:
        response = requests.post(TOKEN_URL, params=params, timeout=30)
    except requests.RequestException as exc:
        raise RuntimeError(f"Baidu OCR access_token 获取失败：{exc}") from exc

    if response.status_code != 200:
        raise RuntimeError(f"Baidu OCR access_token 获取失败：status_code={response.status_code}，body={response.text[:300]}")

    try:
        data = response.json()
    except ValueError as exc:
        raise RuntimeError("Baidu OCR access_token 获取失败：响应不是合法 JSON。") from exc

    access_token = data.get("access_token")
    if not access_token:
        error = data.get("error_description") or data.get("error") or "响应中没有 access_token"
        raise RuntimeError(f"Baidu OCR access_token 获取失败：{error}")

    expires_in = int(data.get("expires_in") or 2592000)
    config["access_token"] = access_token
    config["access_token_expires_at"] = time.time() + expires_in
    return access_token


def recognize(file_name: str, content: bytes, config: Dict[str, Any], language_hint: str = "") -> Dict[str, Any]:
    ext = _file_extension(file_name)
    if ext not in SUPPORTED_EXTENSIONS:
        raise ValueError("Baidu OCR 第一版仅支持 png、jpg、jpeg、webp 图片文件。")

    settings = _validate_config(config)
    access_token = get_access_token(config)
    endpoint = OCR_ENDPOINTS[settings["mode"]]
    image_base64 = base64.b64encode(content).decode("ascii")

    headers = {"Content-Type": "application/x-www-form-urlencoded"}
    data = {
        "image": image_base64,
        "detect_direction": "false",
        "paragraph": "false",
    }

    try:
        response = requests.post(endpoint, params={"access_token": access_token}, data=data, headers=headers, timeout=180)
    except requests.RequestException as exc:
        raise RuntimeError(f"Baidu OCR 请求失败：{exc}") from exc

    if response.status_code != 200:
        raise RuntimeError(f"Baidu OCR 请求失败：status_code={response.status_code}，body={response.text[:500]}")

    try:
        full_response = response.json()
    except ValueError as exc:
        raise RuntimeError("Baidu OCR 请求失败：响应不是合法 JSON。") from exc

    if "error_code" in full_response:
        raise RuntimeError(
            f"Baidu OCR 请求失败：error_code={full_response.get('error_code')}，"
            f"error_msg={full_response.get('error_msg')}"
        )

    words_result = full_response.get("words_result", []) or []
    blocks = []
    lines = []
    for item in words_result:
        text = item.get("words", "")
        if text:
            lines.append(text)
        blocks.append(item)

    raw_text = "\n".join(lines).strip()
    if not raw_text:
        raise RuntimeError("Baidu OCR 未返回可用文本。")

    return {
        "provider": "Baidu OCR",
        "raw_text": raw_text,
        "blocks": blocks,
        "full_response": full_response,
        "meta": {
            "engine": "Baidu OCR",
            "mode": settings["mode"],
            "api": "general" if settings["mode"] == "general" else "accurate_basic",
            "language": language_hint,
            "source_type": "image",
        },
    }
