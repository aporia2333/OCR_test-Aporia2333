from __future__ import annotations

import base64
import os
from typing import Tuple

import requests
from requests.exceptions import ConnectionError, ConnectTimeout, ProxyError, ReadTimeout, SSLError, Timeout

from services.providers import baidu_ocr, tencent_ocr


def extract_text_mock(file_name: str, content: bytes) -> Tuple[str, dict]:
    try:
        decoded = content.decode("utf-8")
        text = decoded[:12000]
        source = "decoded_text"
    except UnicodeDecodeError:
        text = (
            f"[Mock OCR output] 已接收文件 {file_name}。\n"
            "这是一个占位 OCR 结果。你可以在 services/ocr.py 中接入真实 OCR API。\n"
            f"文件大小：{len(content)} bytes。"
        )
        source = "mock_binary"
    meta = {"source": source, "bytes": len(content)}
    return text, meta


def _guess_file_type(file_name: str) -> int:
    ext = file_name.lower().rsplit(".", 1)[-1] if "." in file_name else ""
    if ext == "pdf":
        return 0
    return 1


def _as_bool(value, default: bool = False) -> bool:
    if value is None:
        return default
    if isinstance(value, bool):
        return value
    return str(value).strip().lower() in {"1", "true", "yes", "y", "on"}


def _post_json(url: str, payload: dict, headers: dict, disable_proxy: bool, timeout: int = 180):
    if disable_proxy:
        session = requests.Session()
        session.trust_env = False
        return session.post(url, json=payload, headers=headers, timeout=timeout)
    return requests.post(url, json=payload, headers=headers, timeout=timeout)


def _classify_request_error(exc: BaseException) -> str:
    if isinstance(exc, ProxyError) or "ProxyError" in repr(exc):
        return "代理错误：当前 OCR 请求尝试经过系统代理，但代理连接失败或被远端关闭。"
    if isinstance(exc, (ConnectTimeout, ReadTimeout, Timeout)):
        return "网络超时：OCR 服务在规定时间内没有响应。"
    if isinstance(exc, SSLError):
        return "TLS/证书错误：HTTPS 握手失败，可能与代理或网络拦截有关。"
    if isinstance(exc, ConnectionError):
        return "网络连接错误：无法连接到 OCR 服务，请检查服务地址、网络或防火墙。"
    return "未知网络错误。"


def extract_text_paddle_layout_parsing(file_name: str, content: bytes, config: dict | None = None) -> Tuple[str, dict]:
    config = config or {}
    api_url = config.get("api_url") or os.getenv("PADDLEOCR_API_URL")
    token = config.get("token") or os.getenv("PADDLEOCR_TOKEN")
    disable_proxy = _as_bool(
        config.get("disable_proxy", os.getenv("PADDLEOCR_DISABLE_PROXY")),
        default=True,
    )
    if not api_url or not token:
        raise RuntimeError("未检测到 PADDLEOCR_API_URL 或 PADDLEOCR_TOKEN。")

    file_data = base64.b64encode(content).decode("ascii")
    payload = {
        "file": file_data,
        "fileType": _guess_file_type(file_name),
        "useDocOrientationClassify": False,
        "useDocUnwarping": False,
        "useChartRecognition": False,
    }
    headers = {
        "Authorization": f"token {token}",
        "Content-Type": "application/json",
    }

    print(
        f"[PADDLE_OCR_REQUEST] file={file_name} url={api_url} "
        f"disable_proxy={disable_proxy} bytes={len(content)} file_type={payload['fileType']}"
    )

    try:
        response = _post_json(api_url, payload, headers, disable_proxy, timeout=180)
    except requests.RequestException as exc:
        category = _classify_request_error(exc)
        raise RuntimeError(
            f"PaddleOCR 请求失败：{category} "
            f"disable_proxy={disable_proxy} url={api_url} original_error={exc}"
        ) from exc
    if response.status_code != 200:
        raise RuntimeError(f"PaddleOCR 请求失败：status_code={response.status_code}，body={response.text[:500]}")

    data = response.json()
    result = data.get("result", {})
    layout_results = result.get("layoutParsingResults", [])

    markdown_parts = []
    image_count = 0
    output_image_count = 0
    for idx, item in enumerate(layout_results):
        markdown = item.get("markdown", {})
        md_text = markdown.get("text", "")
        if md_text:
            markdown_parts.append(f"\n\n# Document Part {idx + 1}\n\n{md_text}")
        image_count += len(markdown.get("images", {}) or {})
        output_image_count += len(item.get("outputImages", {}) or {})

    final_text = "\n".join(markdown_parts).strip()
    if not final_text:
        final_text = "[PaddleOCR 已返回结果，但未解析出 markdown.text 内容。]"

    meta = {
        "source": "paddleocr_layout_parsing",
        "provider": "PaddleOCR Layout Parsing",
        "bytes": len(content),
        "file_type": _guess_file_type(file_name),
        "parts": len(layout_results),
        "markdown_image_count": image_count,
        "output_image_count": output_image_count,
        "response_keys": list(data.keys()),
    }
    return final_text, meta


def extract_text(
    file_name: str,
    content: bytes,
    engine: str,
    provider_config: dict | None = None,
    language_hint: str = "",
) -> Tuple[str, dict]:
    if engine == "Mock OCR":
        return extract_text_mock(file_name, content)
    if engine == "PaddleOCR Layout Parsing":
        return extract_text_paddle_layout_parsing(file_name, content, provider_config)
    if engine == "Tencent OCR":
        result = tencent_ocr.recognize(file_name, content, provider_config or {}, language_hint)
        return result["raw_text"], result
    if engine == "Baidu OCR":
        result = baidu_ocr.recognize(file_name, content, provider_config or {}, language_hint)
        return result["raw_text"], result

    raise NotImplementedError(f"当前示例尚未接入 {engine}，请在 services/ocr.py 中补充对应 SDK 调用。")
