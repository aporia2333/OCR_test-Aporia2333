from __future__ import annotations

import json
from dataclasses import dataclass, field
from typing import Any, Dict, Optional


@dataclass
class UserConfig:
    ocr_engine: str
    use_llm_postprocess: bool
    llm_provider: str = ""
    llm_model: str = ""
    output_format: str = "text"
    language_hint: str = "自动"
    prompt_template_name: str = ""
    custom_prompt: str = ""
    temperature: float = 0.1
    llm_provider_config: Optional[Dict[str, Any]] = None
    ocr_provider_config: Optional[Dict[str, Any]] = None


@dataclass
class ProcessResult:
    file_name: str
    status: str
    raw_text: str
    final_output: str
    use_llm_postprocess: bool
    ocr_output_text: str = ""
    llm_output_text: str = ""
    structured_data: Optional[Dict[str, Any]] = None
    ocr_engine_used: Optional[str] = None
    llm_provider_used: Optional[str] = None
    llm_model_used: Optional[str] = None
    llm_attempted: bool = False
    llm_succeeded: bool = False
    error: Optional[str] = None
    meta: Dict[str, Any] = field(default_factory=dict)

    def ocr_export_name(self) -> str:
        base = self.file_name.rsplit(".", 1)[0]
        return f"{base}_ocr.txt"

    def ocr_export_bytes(self) -> bytes:
        return self.ocr_output_text.encode("utf-8")

    def ocr_export_mime(self) -> str:
        return "text/plain"

    def llm_export_name(self) -> str:
        base = self.file_name.rsplit(".", 1)[0]
        output_format = self.meta.get("output_format")
        if self.structured_data and output_format == "json":
            return f"{base}_llm.json"
        if output_format == "word":
            return f"{base}_llm.rtf"
        if output_format == "markdown":
            return f"{base}_llm.md"
        return f"{base}_llm.txt"

    def llm_export_mime(self) -> str:
        output_format = self.meta.get("output_format")
        if self.structured_data and output_format == "json":
            return "application/json"
        if output_format == "word":
            return "application/rtf"
        if output_format == "markdown":
            return "text/markdown"
        return "text/plain"

    def llm_export_bytes(self) -> bytes:
        output_format = self.meta.get("output_format")
        if self.structured_data and output_format == "json":
            return json.dumps(self.structured_data, ensure_ascii=False, indent=2).encode("utf-8")
        if output_format == "word":
            return self._rtf_bytes(self.llm_output_text)
        return self.llm_output_text.encode("utf-8")

    def export_name(self) -> str:
        return self.llm_export_name() if self.llm_succeeded else self.ocr_export_name()

    def export_mime(self) -> str:
        return self.llm_export_mime() if self.llm_succeeded else self.ocr_export_mime()

    def export_bytes(self) -> bytes:
        return self.llm_export_bytes() if self.llm_succeeded else self.ocr_export_bytes()

    def raw_export_name(self) -> str:
        return self.ocr_export_name()

    def raw_export_bytes(self) -> bytes:
        return self.ocr_export_bytes()

    def formatted_export_name(self) -> str:
        return self.llm_export_name()

    def formatted_export_bytes(self) -> bytes:
        return self.llm_export_bytes()

    @staticmethod
    def _rtf_bytes(text: str) -> bytes:
        escaped = text.replace("\\", "\\\\").replace("{", "\\{").replace("}", "\\}")
        escaped = escaped.replace("\n", "\\par\n")
        return ("{\\rtf1\\ansi\\deff0\n" + escaped + "\n}").encode("utf-8")
