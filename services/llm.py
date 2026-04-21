from __future__ import annotations

import json
import os
from typing import Dict, Tuple

from services.custom_llm_api import run_custom_llm_chat_completion
from services.registry import MODEL_REGISTRY


GENERIC_SYSTEM_PROMPT = "你是一个文档识别与结构化助手。严格基于用户提供的文本内容完成任务，不要编造。"


CONSERVATIVE_OCR_FORMAT_SYSTEM_PROMPT = """你现在的任务不是重新识别内容，也不是总结内容，而是仅对 OCR 输出文本做保守整理。

必须严格遵守以下规则：

只允许做断句、分段、换行、基础格式整理。
不允许补充任何原文没有的信息。
不允许删除任何有效信息。
不允许修改数字、日期、金额、编号、人名、地名、专有名词。
不允许纠正 OCR 识别错误。
不允许改写原句含义，不允许润色，不允许总结。
如果原文格式混乱，只把它整理得更易读。
输出内容必须尽可能忠实于原始 OCR 文本。

请直接输出整理后的文本，不要添加解释。"""


def run_mock_llm(text: str, prompt: str, output_format: str) -> Tuple[str, Dict]:
    preview = text[:2000]
    final_output = (
        "[Mock LLM output]\n\n"
        f"Prompt:\n{prompt}\n\n"
        f"Content Preview:\n{preview}"
    )
    structured = None
    if output_format == "json":
        structured = {
            "task": "mock_llm",
            "prompt": prompt,
            "content_preview": preview,
        }
        final_output = json.dumps(structured, ensure_ascii=False, indent=2)
    elif output_format == "markdown":
        final_output = f"# Mock LLM Output\n\n## Prompt\n{prompt}\n\n## Content Preview\n{preview}"
    return final_output, structured or {}


def _openai_compatible_client(provider: str, provider_config: dict | None = None):
    try:
        from openai import OpenAI
        import httpx
    except Exception as exc:
        raise RuntimeError("未安装 openai 包，请先执行 pip install -r requirements.txt。") from exc

    provider_info = MODEL_REGISTRY.get(provider, {})
    api_key_env = provider_info.get("env_key")
    base_url_env = provider_info.get("base_url_env")
    provider_config = provider_config or {}

    api_key = provider_config.get("api_key") or (os.getenv(api_key_env) if api_key_env else None)
    if not api_key:
        raise RuntimeError(f"未检测到 {api_key_env} 环境变量。")

    disable_proxy_value = provider_config.get("disable_proxy", True)
    disable_proxy = (
        disable_proxy_value
        if isinstance(disable_proxy_value, bool)
        else str(disable_proxy_value).strip().lower() not in {"false", "0", "no", "off"}
    )

    kwargs = {"api_key": api_key}
    base_url = (
        provider_config.get("base_url")
        or (os.getenv(base_url_env) if base_url_env else None)
        or provider_info.get("default_base_url")
    )
    if base_url:
        kwargs["base_url"] = base_url
    if disable_proxy:
        kwargs["http_client"] = httpx.Client(trust_env=False, timeout=180)

    return OpenAI(**kwargs)


def _extract_chat_text(response) -> str:
    if getattr(response, "choices", None):
        message = response.choices[0].message
        if getattr(message, "content", None):
            return message.content
    raise RuntimeError("LLM 返回为空，未提取到 message.content。")


def run_openai_compatible_llm(
    provider: str,
    text: str,
    prompt: str,
    model: str,
    temperature: float,
    output_format: str,
    conservative_formatting: bool = False,
    provider_config: dict | None = None,
) -> Tuple[str, Dict]:
    client = _openai_compatible_client(provider, provider_config)

    system_instruction = CONSERVATIVE_OCR_FORMAT_SYSTEM_PROMPT if conservative_formatting else GENERIC_SYSTEM_PROMPT

    if output_format == "json":
        if conservative_formatting:
            user_text = (
                f"整理要求：{prompt}\n\n"
                f"OCR 输出文本如下：\n{text}\n\n"
                "请仅基于 OCR 输出文本整理为合法 JSON，不要新增、删除、改写任何有效信息。"
            )
        else:
            user_text = (
                f"用户要求：{prompt}\n\n"
                f"文档文本如下：\n{text}\n\n"
                '请仅输出合法 JSON，对象结构建议为 {"result": "...", "fields": {...}}。'
            )
    elif output_format == "markdown":
        if conservative_formatting:
            user_text = f"整理要求：{prompt}\n\nOCR 输出文本如下：\n{text}\n\n请整理为 Markdown，保持原文信息不变。"
        else:
            user_text = f"用户要求：{prompt}\n\n文档文本如下：\n{text}\n\n请输出 Markdown。"
    elif output_format == "word":
        if conservative_formatting:
            user_text = f"整理要求：{prompt}\n\nOCR 输出文本如下：\n{text}\n\n请整理为适合 Word 文档粘贴的纯文本，保持原文信息不变。"
        else:
            user_text = f"用户要求：{prompt}\n\n文档文本如下：\n{text}\n\n请输出适合 Word 文档粘贴的纯文本。"
    else:
        if conservative_formatting:
            user_text = f"整理要求：{prompt}\n\nOCR 输出文本如下：\n{text}"
        else:
            user_text = f"用户要求：{prompt}\n\n文档文本如下：\n{text}"

    response = client.chat.completions.create(
        model=model,
        temperature=temperature,
        messages=[
            {"role": "system", "content": system_instruction},
            {"role": "user", "content": user_text},
        ],
    )
    output_text = _extract_chat_text(response)

    if output_format == "json":
        try:
            parsed = json.loads(output_text)
            return json.dumps(parsed, ensure_ascii=False, indent=2), parsed
        except Exception:
            fallback = {"result": output_text, "fields": {}}
            return json.dumps(fallback, ensure_ascii=False, indent=2), fallback

    return output_text, {}


def _build_messages(text: str, prompt: str, output_format: str, conservative_formatting: bool = False) -> list:
    system_instruction = CONSERVATIVE_OCR_FORMAT_SYSTEM_PROMPT if conservative_formatting else GENERIC_SYSTEM_PROMPT

    if output_format == "json":
        if conservative_formatting:
            user_text = (
                f"整理要求：{prompt}\n\n"
                f"OCR 输出文本如下：\n{text}\n\n"
                "请仅基于 OCR 输出文本整理为合法 JSON，不要新增、删除、改写任何有效信息。"
            )
        else:
            user_text = (
                f"用户要求：{prompt}\n\n"
                f"文档文本如下：\n{text}\n\n"
                '请仅输出合法 JSON，对象结构建议为 {"result": "...", "fields": {...}}。'
            )
    elif output_format == "markdown":
        if conservative_formatting:
            user_text = f"整理要求：{prompt}\n\nOCR 输出文本如下：\n{text}\n\n请整理为 Markdown，保持原文信息不变。"
        else:
            user_text = f"用户要求：{prompt}\n\n文档文本如下：\n{text}\n\n请输出 Markdown。"
    elif output_format == "word":
        if conservative_formatting:
            user_text = f"整理要求：{prompt}\n\nOCR 输出文本如下：\n{text}\n\n请整理为适合 Word 文档粘贴的纯文本，保持原文信息不变。"
        else:
            user_text = f"用户要求：{prompt}\n\n文档文本如下：\n{text}\n\n请输出适合 Word 文档粘贴的纯文本。"
    else:
        if conservative_formatting:
            user_text = f"整理要求：{prompt}\n\nOCR 输出文本如下：\n{text}"
        else:
            user_text = f"用户要求：{prompt}\n\n文档文本如下：\n{text}"

    return [
        {"role": "system", "content": system_instruction},
        {"role": "user", "content": user_text},
    ]


def _normalize_json_output(output_text: str) -> Tuple[str, Dict]:
    try:
        parsed = json.loads(output_text)
        return json.dumps(parsed, ensure_ascii=False, indent=2), parsed
    except Exception:
        fallback = {"result": output_text, "fields": {}}
        return json.dumps(fallback, ensure_ascii=False, indent=2), fallback


def run_custom_llm(
    text: str,
    prompt: str,
    temperature: float,
    output_format: str,
    custom_llm_api_config: dict | None,
    conservative_formatting: bool = False,
) -> Tuple[str, Dict]:
    output_text = run_custom_llm_chat_completion(
        custom_llm_api_config or {},
        _build_messages(text, prompt, output_format, conservative_formatting),
        temperature,
    )
    if output_format == "json":
        return _normalize_json_output(output_text)
    return output_text, {}


def run_llm(
    text: str,
    prompt: str,
    provider: str,
    model: str,
    temperature: float,
    output_format: str,
    custom_llm_api_config: dict | None = None,
    conservative_formatting: bool = False,
) -> Tuple[str, Dict]:
    if provider == "Mock LLM":
        return run_mock_llm(text, prompt, output_format)
    if provider == "Custom LLM API":
        return run_custom_llm(text, prompt, temperature, output_format, custom_llm_api_config, conservative_formatting)
    if provider in ["DeepSeek", "OpenAI", "GLM"]:
        return run_openai_compatible_llm(
            provider,
            text,
            prompt,
            model,
            temperature,
            output_format,
            conservative_formatting,
            custom_llm_api_config,
        )
    raise NotImplementedError(f"暂不支持 provider={provider}")
