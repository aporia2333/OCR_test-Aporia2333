from __future__ import annotations

from services.llm import run_llm
from services.models import ProcessResult, UserConfig
from services.ocr import extract_text


MODE_OCR = "视觉识别 API"
MODE_LLM = "大模型"
MODE_OCR_FORMAT = "视觉识别结果整理"


def process_files(uploaded_file, config: UserConfig) -> ProcessResult:
    content = uploaded_file.getvalue()
    file_name = uploaded_file.name

    try:
        raw_text = ""
        ocr_meta = {}
        structured_data = {}
        final_output = ""
        ocr_engine_used = None
        llm_provider_used = None
        llm_model_used = None

        if config.mode in [MODE_OCR, MODE_OCR_FORMAT]:
            raw_text, ocr_meta = extract_text(
                file_name,
                content,
                config.ocr_engine,
                config.ocr_provider_config,
                config.language_hint,
            )
            ocr_engine_used = config.ocr_engine

        if config.mode == MODE_LLM:
            try:
                raw_text = content.decode("utf-8")[:12000]
            except UnicodeDecodeError:
                raw_text = (
                    f"文件 {file_name} 已读取为二进制。当前原型的纯大模型模式只支持 txt/md 文本输入；"
                    "图片或 PDF 请先使用视觉识别 API 模式提取文本。"
                )

        if config.mode in [MODE_LLM, MODE_OCR_FORMAT]:
            final_output, structured_data = run_llm(
                raw_text,
                config.custom_prompt,
                config.llm_provider,
                config.llm_model,
                config.temperature,
                config.output_format,
                config.custom_llm_api_config,
                config.mode == MODE_OCR_FORMAT,
            )
            llm_provider_used = config.llm_provider
            if config.llm_provider == "Custom LLM API" and config.custom_llm_api_config:
                llm_model_used = config.custom_llm_api_config.get(
                    "model",
                    config.custom_llm_api_config.get("model_name", config.llm_model),
                )
            else:
                llm_model_used = config.llm_model

            formatted_output = final_output if config.mode == MODE_OCR_FORMAT else ""
        else:
            formatted_output = ""
            if config.output_format == "json":
                structured_data = {"text": raw_text}
                final_output = raw_text
            elif config.output_format == "markdown":
                final_output = f"# OCR Result\n\n{raw_text}"
            else:
                final_output = raw_text

        return ProcessResult(
            file_name=file_name,
            status="success",
            mode=config.mode,
            raw_text=raw_text,
            final_output=final_output,
            formatted_output=formatted_output,
            structured_data=structured_data,
            ocr_engine_used=ocr_engine_used,
            llm_provider_used=llm_provider_used,
            llm_model_used=llm_model_used,
            meta={
                "output_format": config.output_format,
                "language_hint": config.language_hint,
                "prompt_template": config.prompt_template_name,
                "ocr_meta": ocr_meta,
            },
        )
    except Exception as exc:
        return ProcessResult(
            file_name=file_name,
            status="error",
            mode=config.mode,
            raw_text="",
            final_output=f"处理失败：{exc}",
            error=str(exc),
            meta={"output_format": config.output_format},
        )
