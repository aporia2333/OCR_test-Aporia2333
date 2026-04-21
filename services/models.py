from __future__ import annotations

import json
from dataclasses import dataclass, field
from typing import Any, Dict, Optional


@dataclass
class UserConfig:
    mode: str
    ocr_engine: str
    llm_provider: str
    llm_model: str
    output_format: str
    language_hint: str
    prompt_template_name: str
    custom_prompt: str
    temperature: float = 0.1
    custom_llm_api_config: Optional[Dict[str, Any]] = None
    ocr_provider_config: Optional[Dict[str, Any]] = None


@dataclass
class ProcessResult:
    file_name: str
    status: str
    mode: str
    raw_text: str
    final_output: str
    formatted_output: str = ""
    structured_data: Optional[Dict[str, Any]] = None
    ocr_engine_used: Optional[str] = None
    llm_provider_used: Optional[str] = None
    llm_model_used: Optional[str] = None
    error: Optional[str] = None
    meta: Dict[str, Any] = field(default_factory=dict)

    def export_name(self) -> str:
        base = self.file_name.rsplit('.', 1)[0]
        if self.structured_data and self.meta.get('output_format') == 'json':
            return f"{base}.json"
        if self.meta.get('output_format') == 'word':
            return f"{base}.rtf"
        if self.meta.get('output_format') == 'markdown':
            return f"{base}.md"
        return f"{base}.txt"

    def raw_export_name(self) -> str:
        base = self.file_name.rsplit('.', 1)[0]
        return f"{base}_raw_ocr.txt"

    def formatted_export_name(self) -> str:
        base = self.file_name.rsplit('.', 1)[0]
        if self.structured_data and self.meta.get('output_format') == 'json':
            return f"{base}_formatted_output.json"
        if self.meta.get('output_format') == 'word':
            return f"{base}_formatted_output.rtf"
        if self.meta.get('output_format') == 'markdown':
            return f"{base}_formatted_output.md"
        return f"{base}_formatted_output.txt"

    def export_mime(self) -> str:
        if self.structured_data and self.meta.get('output_format') == 'json':
            return 'application/json'
        if self.meta.get('output_format') == 'word':
            return 'application/rtf'
        if self.meta.get('output_format') == 'markdown':
            return 'text/markdown'
        return 'text/plain'

    def export_bytes(self) -> bytes:
        if self.structured_data and self.meta.get('output_format') == 'json':
            return json.dumps(self.structured_data, ensure_ascii=False, indent=2).encode('utf-8')
        if self.meta.get('output_format') == 'word':
            return self._rtf_bytes(self.final_output)
        return self.final_output.encode('utf-8')

    def raw_export_bytes(self) -> bytes:
        return self.raw_text.encode('utf-8')

    def formatted_export_bytes(self) -> bytes:
        if self.structured_data and self.meta.get('output_format') == 'json':
            return json.dumps(self.structured_data, ensure_ascii=False, indent=2).encode('utf-8')
        if self.meta.get('output_format') == 'word':
            return self._rtf_bytes(self.formatted_output or self.final_output)
        return (self.formatted_output or self.final_output).encode('utf-8')

    @staticmethod
    def _rtf_bytes(text: str) -> bytes:
        escaped = text.replace('\\', '\\\\').replace('{', '\\{').replace('}', '\\}')
        escaped = escaped.replace('\n', '\\par\n')
        return ("{\\rtf1\\ansi\\deff0\n" + escaped + "\n}").encode('utf-8')
