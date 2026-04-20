from __future__ import annotations

import base64
import os
from typing import Tuple

import requests


def extract_text_mock(file_name: str, content: bytes) -> Tuple[str, dict]:
    try:
        decoded = content.decode('utf-8')
        text = decoded[:12000]
        source = 'decoded_text'
    except UnicodeDecodeError:
        text = (
            f"[Mock OCR output] 已接收文件 {file_name}。\n"
            f"这是一个占位 OCR 结果。你可以在 services/ocr.py 中接入真实 OCR API。\n"
            f"文件大小：{len(content)} bytes。"
        )
        source = 'mock_binary'
    meta = {'source': source, 'bytes': len(content)}
    return text, meta



def _guess_file_type(file_name: str) -> int:
    ext = file_name.lower().rsplit('.', 1)[-1] if '.' in file_name else ''
    if ext == 'pdf':
        return 0
    return 1



def extract_text_paddle_layout_parsing(file_name: str, content: bytes) -> Tuple[str, dict]:
    api_url = os.getenv('PADDLEOCR_API_URL')
    token = os.getenv('PADDLEOCR_TOKEN')
    if not api_url or not token:
        raise RuntimeError('未检测到 PADDLEOCR_API_URL 或 PADDLEOCR_TOKEN 环境变量。')

    file_data = base64.b64encode(content).decode('ascii')
    payload = {
        'file': file_data,
        'fileType': _guess_file_type(file_name),
        'useDocOrientationClassify': False,
        'useDocUnwarping': False,
        'useChartRecognition': False,
    }
    headers = {
        'Authorization': f'token {token}',
        'Content-Type': 'application/json',
    }

    response = requests.post(api_url, json=payload, headers=headers, timeout=180)
    if response.status_code != 200:
        raise RuntimeError(f'PaddleOCR 请求失败，status_code={response.status_code}，body={response.text[:500]}')

    data = response.json()
    result = data.get('result', {})
    layout_results = result.get('layoutParsingResults', [])

    markdown_parts = []
    image_count = 0
    output_image_count = 0
    for idx, item in enumerate(layout_results):
        markdown = item.get('markdown', {})
        md_text = markdown.get('text', '')
        if md_text:
            markdown_parts.append(f'\n\n# Document Part {idx + 1}\n\n{md_text}')
        image_count += len(markdown.get('images', {}) or {})
        output_image_count += len(item.get('outputImages', {}) or {})

    final_text = '\n'.join(markdown_parts).strip()
    if not final_text:
        final_text = '[PaddleOCR 已返回结果，但未解析出 markdown.text 内容。]'

    meta = {
        'source': 'paddleocr_layout_parsing',
        'bytes': len(content),
        'file_type': _guess_file_type(file_name),
        'parts': len(layout_results),
        'markdown_image_count': image_count,
        'output_image_count': output_image_count,
        'response_keys': list(data.keys()),
    }
    return final_text, meta



def extract_text(file_name: str, content: bytes, engine: str) -> Tuple[str, dict]:
    if engine == 'Mock OCR':
        return extract_text_mock(file_name, content)
    if engine == 'PaddleOCR Layout Parsing':
        return extract_text_paddle_layout_parsing(file_name, content)

    raise NotImplementedError(
        f"当前示例未接入 {engine}，但接口已预留。可在此添加 Azure / Google / AWS 的 SDK 调用。"
    )
