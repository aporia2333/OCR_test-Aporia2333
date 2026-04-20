from __future__ import annotations

from typing import Tuple


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


def extract_text(file_name: str, content: bytes, engine: str) -> Tuple[str, dict]:
    if engine == 'Mock OCR':
        return extract_text_mock(file_name, content)

    raise NotImplementedError(
        f"当前示例未接入 {engine}，但接口已预留。可在此添加 Azure / Google / AWS 的 SDK 调用。"
    )
