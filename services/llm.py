from __future__ import annotations

import json
import os
from typing import Dict, Tuple

from services.registry import MODEL_REGISTRY



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



def _openai_compatible_client(provider: str):
    try:
        from openai import OpenAI
    except Exception as e:
        raise RuntimeError("未安装 openai 包，请先执行 pip install -r requirements.txt") from e

    provider_info = MODEL_REGISTRY.get(provider, {})
    api_key_env = provider_info.get("env_key")
    base_url_env = provider_info.get("base_url_env")

    api_key = os.getenv(api_key_env) if api_key_env else None
    if not api_key:
        raise RuntimeError(f"未检测到 {api_key_env} 环境变量。")

    kwargs = {"api_key": api_key}
    base_url = (os.getenv(base_url_env) if base_url_env else None) or provider_info.get("default_base_url")
    if base_url:
        kwargs["base_url"] = base_url

    return OpenAI(**kwargs)



def _extract_chat_text(response) -> str:
    if getattr(response, "choices", None):
        message = response.choices[0].message
        if getattr(message, "content", None):
            return message.content
    raise RuntimeError("LLM 返回为空，未提取到 message.content")



def run_openai_compatible_llm(
    provider: str,
    text: str,
    prompt: str,
    model: str,
    temperature: float,
    output_format: str,
) -> Tuple[str, Dict]:
    client = _openai_compatible_client(provider)

    system_instruction = "你是一个文档识别与结构化助手。严格基于用户提供的文本内容完成任务，不要编造。"

    if output_format == "json":
        user_text = (
            f"用户要求：{prompt}\n\n"
            f"文档文本如下：\n{text}\n\n"
            "请仅输出合法 JSON，对象结构必须为："
            '{"result": "...", "fields": {...}}'
        )
    elif output_format == "markdown":
        user_text = f"用户要求：{prompt}\n\n文档文本如下：\n{text}\n\n请输出 markdown。"
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



def run_llm(text: str, prompt: str, provider: str, model: str, temperature: float, output_format: str) -> Tuple[str, Dict]:
    if provider == "Mock LLM":
        return run_mock_llm(text, prompt, output_format)
    if provider == "DeepSeek":
        return run_openai_compatible_llm(provider, text, prompt, model, temperature, output_format)
    raise NotImplementedError(f"暂不支持 provider={provider}")
