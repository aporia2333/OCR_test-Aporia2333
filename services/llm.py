from __future__ import annotations

import json
import os
from typing import Dict, Tuple


def run_mock_llm(text: str, prompt: str, output_format: str) -> Tuple[str, Dict]:
    preview = text[:2000]
    final_output = (
        "[Mock LLM output]\n\n"
        f"Prompt:\n{prompt}\n\n"
        f"Content Preview:\n{preview}"
    )
    structured = None
    if output_format == 'json':
        structured = {
            'task': 'mock_llm',
            'prompt': prompt,
            'content_preview': preview,
        }
        final_output = json.dumps(structured, ensure_ascii=False, indent=2)
    elif output_format == 'markdown':
        final_output = f"# Mock LLM Output\n\n## Prompt\n{prompt}\n\n## Content Preview\n{preview}"
    return final_output, structured or {}


def run_openai_llm(text: str, prompt: str, model: str, temperature: float, output_format: str) -> Tuple[str, Dict]:
    try:
        from openai import OpenAI
    except Exception as e:
        raise RuntimeError('未安装 openai 包，请先执行 pip install -r requirements.txt') from e

    api_key = os.getenv('OPENAI_API_KEY')
    if not api_key:
        raise RuntimeError('未检测到 OPENAI_API_KEY 环境变量。')

    client = OpenAI(api_key=api_key)

    system_instruction = (
        "你是一个文档识别与结构化助手。严格基于用户提供的文本内容完成任务，不要编造。"
    )

    if output_format == 'json':
        schema = {
            'type': 'object',
            'properties': {
                'result': {'type': 'string'},
                'fields': {'type': 'object', 'additionalProperties': True},
            },
            'required': ['result', 'fields'],
            'additionalProperties': False,
        }
        response = client.responses.create(
            model=model,
            temperature=temperature,
            input=[
                {'role': 'system', 'content': [{'type': 'input_text', 'text': system_instruction}]},
                {
                    'role': 'user',
                    'content': [
                        {'type': 'input_text', 'text': f'用户要求：{prompt}\n\n文档文本如下：\n{text}'}
                    ],
                },
            ],
            text={
                'format': {
                    'type': 'json_schema',
                    'name': 'document_extraction',
                    'schema': schema,
                    'strict': True,
                }
            },
        )
        parsed = json.loads(response.output_text)
        return json.dumps(parsed, ensure_ascii=False, indent=2), parsed

    response = client.responses.create(
        model=model,
        temperature=temperature,
        input=[
            {'role': 'system', 'content': [{'type': 'input_text', 'text': system_instruction}]},
            {
                'role': 'user',
                'content': [
                    {'type': 'input_text', 'text': f'用户要求：{prompt}\n\n文档文本如下：\n{text}'}
                ],
            },
        ],
    )
    return response.output_text, {}


def run_llm(text: str, prompt: str, provider: str, model: str, temperature: float, output_format: str) -> Tuple[str, Dict]:
    if provider == 'Mock LLM':
        return run_mock_llm(text, prompt, output_format)
    if provider == 'OpenAI':
        return run_openai_llm(text, prompt, model, temperature, output_format)
    raise NotImplementedError(f'暂不支持 provider={provider}')
