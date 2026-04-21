from __future__ import annotations

from services.llm import run_llm
from services.models import ProcessResult, UserConfig
from services.ocr import extract_text


def process_files(uploaded_file, config: UserConfig) -> ProcessResult:
    content = uploaded_file.getvalue()
    file_name = uploaded_file.name

    try:
        raw_text, ocr_meta = extract_text(
            file_name,
            content,
            config.ocr_engine,
            config.ocr_provider_config,
            config.language_hint,
        )
    except Exception as exc:
        return ProcessResult(
            file_name=file_name,
            status="error",
            raw_text="",
            final_output="",
            use_llm_postprocess=config.use_llm_postprocess,
            ocr_output_text="",
            ocr_engine_used=config.ocr_engine,
            error=str(exc),
            meta={
                "output_format": config.output_format,
                "language_hint": config.language_hint,
                "prompt_template": config.prompt_template_name,
                "ocr_meta": {},
                "stage": "ocr",
                "llm_skipped": True,
            },
        )

    line_count = raw_text.count("\n") + 1 if raw_text else 0
    print(
        f"[OCR] file={file_name} engine={config.ocr_engine} "
        f"raw_text_chars={len(raw_text)} lines={line_count} meta={ocr_meta}"
    )

    base_meta = {
        "output_format": config.output_format,
        "language_hint": config.language_hint,
        "prompt_template": config.prompt_template_name,
        "ocr_meta": ocr_meta,
        "raw_text_chars": len(raw_text),
    }

    if not config.use_llm_postprocess:
        return ProcessResult(
            file_name=file_name,
            status="success",
            raw_text=raw_text,
            final_output=raw_text,
            use_llm_postprocess=False,
            ocr_output_text=raw_text,
            ocr_engine_used=config.ocr_engine,
            meta={**base_meta, "stage": "ocr_only", "llm_skipped": True},
        )

    if not config.llm_provider:
        return ProcessResult(
            file_name=file_name,
            status="partial_success",
            raw_text=raw_text,
            final_output=raw_text,
            use_llm_postprocess=True,
            ocr_output_text=raw_text,
            ocr_engine_used=config.ocr_engine,
            error="已完成 OCR，但未选择 LLM provider，未执行 LLM 整理。",
            meta={**base_meta, "stage": "llm_config", "llm_skipped": True},
        )

    if not raw_text.strip():
        return ProcessResult(
            file_name=file_name,
            status="partial_success",
            raw_text=raw_text,
            final_output=raw_text,
            use_llm_postprocess=True,
            ocr_output_text=raw_text,
            ocr_engine_used=config.ocr_engine,
            llm_provider_used=config.llm_provider,
            llm_model_used=config.llm_model,
            error="OCR 已完成，但没有提取到可供 LLM 整理的文本。",
            meta={**base_meta, "stage": "llm_input", "llm_skipped": True},
        )

    if len(raw_text) > 120000:
        return ProcessResult(
            file_name=file_name,
            status="partial_success",
            raw_text=raw_text,
            final_output=raw_text,
            use_llm_postprocess=True,
            ocr_output_text=raw_text,
            ocr_engine_used=config.ocr_engine,
            llm_provider_used=config.llm_provider,
            llm_model_used=config.llm_model,
            error=f"OCR 文本过长（{len(raw_text)} chars），当前版本未做分块整理，未执行 LLM。",
            meta={**base_meta, "stage": "llm_input", "llm_skipped": True},
        )

    print(
        f"[LLM_INPUT] file={file_name} provider={config.llm_provider} model={config.llm_model} "
        f"output_format={config.output_format} conservative=True "
        f"prompt_chars={len(config.custom_prompt or '')} raw_text_chars={len(raw_text)} "
        f"raw_preview={raw_text[:300]!r}"
    )

    try:
        llm_output, structured_data = run_llm(
            raw_text,
            config.custom_prompt,
            config.llm_provider,
            config.llm_model,
            config.temperature,
            config.output_format,
            config.llm_provider_config,
            True,
        )
    except Exception as exc:
        return ProcessResult(
            file_name=file_name,
            status="partial_success",
            raw_text=raw_text,
            final_output=raw_text,
            use_llm_postprocess=True,
            ocr_output_text=raw_text,
            ocr_engine_used=config.ocr_engine,
            llm_provider_used=config.llm_provider,
            llm_model_used=(
                config.llm_provider_config.get("model", config.llm_model)
                if config.llm_provider == "Custom LLM API" and config.llm_provider_config
                else config.llm_model
            ),
            llm_attempted=True,
            llm_succeeded=False,
            error=str(exc),
            meta={**base_meta, "stage": "llm"},
        )

    llm_model_used = (
        config.llm_provider_config.get("model", config.llm_model)
        if config.llm_provider == "Custom LLM API" and config.llm_provider_config
        else config.llm_model
    )
    return ProcessResult(
        file_name=file_name,
        status="success",
        raw_text=raw_text,
        final_output=llm_output,
        use_llm_postprocess=True,
        ocr_output_text=raw_text,
        llm_output_text=llm_output,
        structured_data=structured_data,
        ocr_engine_used=config.ocr_engine,
        llm_provider_used=config.llm_provider,
        llm_model_used=llm_model_used,
        llm_attempted=True,
        llm_succeeded=True,
        meta={**base_meta, "stage": "llm"},
    )
