from __future__ import annotations

from services.llm import run_llm
from services.models import ProcessResult, UserConfig
from services.ocr import extract_text


def process_files(uploaded_file, config: UserConfig) -> ProcessResult:
    content = uploaded_file.getvalue()
    file_name = uploaded_file.name

    try:
        raw_text = ''
        ocr_meta = {}
        structured_data = {}
        final_output = ''
        ocr_engine_used = None
        llm_provider_used = None
        llm_model_used = None

        if config.mode in ['视觉识别 API', '视觉识别 API + 大模型']:
            raw_text, ocr_meta = extract_text(file_name, content, config.ocr_engine)
            ocr_engine_used = config.ocr_engine

        if config.mode == '大模型':
            try:
                raw_text = content.decode('utf-8')[:12000]
            except UnicodeDecodeError:
                raw_text = (
                    f'文件 {file_name} 已读取为二进制。当前示例中，纯大模型模式对图片/PDF 的直接视觉输入未在本地原型中启用。'
                    '如需启用，可在 services/llm.py 中改为直接上传图像/PDF 给模型。'
                )

        if config.mode in ['大模型', '视觉识别 API + 大模型']:
            final_output, structured_data = run_llm(
                raw_text,
                config.custom_prompt,
                config.llm_provider,
                config.llm_model,
                config.temperature,
                config.output_format,
            )
            llm_provider_used = config.llm_provider
            llm_model_used = config.llm_model
        else:
            if config.output_format == 'json':
                structured_data = {'text': raw_text}
                final_output = raw_text
            elif config.output_format == 'markdown':
                final_output = f'# OCR Result\n\n{raw_text}'
            else:
                final_output = raw_text

        return ProcessResult(
            file_name=file_name,
            status='success',
            mode=config.mode,
            raw_text=raw_text,
            final_output=final_output,
            structured_data=structured_data,
            ocr_engine_used=ocr_engine_used,
            llm_provider_used=llm_provider_used,
            llm_model_used=llm_model_used,
            meta={
                'output_format': config.output_format,
                'language_hint': config.language_hint,
                'prompt_template': config.prompt_template_name,
                'ocr_meta': ocr_meta,
            },
        )
    except Exception as e:
        return ProcessResult(
            file_name=file_name,
            status='error',
            mode=config.mode,
            raw_text='',
            final_output=f'处理失败：{e}',
            error=str(e),
            meta={'output_format': config.output_format},
        )
