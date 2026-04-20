import io
import json
from dataclasses import asdict
from typing import List

import streamlit as st

from services.models import ProcessResult, UserConfig
from services.ocr_registry import get_available_ocr_engines, get_ocr_engine_status
from services.pipeline import process_files
from services.registry import (
    MODEL_REGISTRY,
    get_capabilities,
    get_llm_provider_status,
    get_models_for_provider,
)
from prompts.templates import DEFAULT_PROMPTS, TEMPLATE_DESCRIPTIONS

st.set_page_config(page_title="OCR / LLM Workbench", page_icon="📄", layout="wide")


def get_provider_status() -> dict:
    return {
        "ocr": get_ocr_engine_status(),
        "llm": get_llm_provider_status(),
    }


status = get_provider_status()

st.title("OCR / LLM Workbench")
st.caption("批量上传文件，选择 OCR / 大模型 / 结合模式，并支持默认或自定义提示词。")

with st.sidebar:
    st.header("服务连接状态")
    for engine, ready in status["ocr"].items():
        if engine == "Mock OCR":
            continue
        st.markdown(f"**{engine}**：{'已连接' if ready else '未连接'}")
    for provider, ready in status["llm"].items():
        if provider == "Mock LLM":
            continue
        st.markdown(f"**{provider}**：{'已连接' if ready else '未连接'}")

    st.caption("正式方案：API Key 仅由服务器环境变量读取，不在网页中直接填写。")

    st.divider()
    st.header("处理参数")
    mode = st.radio(
        "处理模式",
        options=["视觉识别 API", "大模型", "视觉识别 API + 大模型"],
        index=2,
    )

    if mode in ["视觉识别 API", "视觉识别 API + 大模型"]:
        ocr_engine = st.selectbox(
            "OCR 引擎",
            options=get_available_ocr_engines(),
            index=0,
        )
    else:
        ocr_engine = ""

    if mode in ["大模型", "视觉识别 API + 大模型"]:
        provider_options = [p for p in MODEL_REGISTRY.keys() if p == "Mock LLM" or status["llm"].get(p)]
        if not provider_options:
            provider_options = ["Mock LLM"]

        llm_provider = st.selectbox(
            "LLM 提供方",
            options=provider_options,
            index=0,
        )

        model_options = get_models_for_provider(llm_provider)
        llm_model = st.selectbox(
            "LLM 模型",
            options=model_options,
            index=0,
        )

        capabilities = get_capabilities(llm_provider, llm_model)
        st.caption(
            f"能力：视觉={'是' if capabilities.get('supports_vision') else '否'}，"
            f"JSON={'是' if capabilities.get('supports_json') else '否'}，"
            f"长上下文={'是' if capabilities.get('supports_long_context') else '否'}"
        )
    else:
        llm_provider = ""
        llm_model = ""

    output_format = st.selectbox(
        "输出格式",
        options=["text", "json", "markdown"],
        index=0,
    )

    language_hint = st.selectbox(
        "语言提示",
        options=["自动", "中文", "英文", "中英混合"],
        index=0,
    )

    st.divider()
    st.subheader("提示词")
    prompt_template_name = st.selectbox(
        "默认模板",
        options=list(DEFAULT_PROMPTS.keys()),
        index=0,
        help="可先选模板，再按需修改下面的自定义提示词。",
    )
    st.caption(TEMPLATE_DESCRIPTIONS[prompt_template_name])

    show_prompt_editor = mode in ["大模型", "视觉识别 API + 大模型"]
    custom_prompt = st.text_area(
        "自定义提示词",
        value=DEFAULT_PROMPTS[prompt_template_name],
        height=220,
        disabled=not show_prompt_editor,
        help="仅大模型相关模式启用。纯 OCR 模式下不使用提示词。",
    )

    temperature = st.slider(
        "Temperature",
        min_value=0.0,
        max_value=1.0,
        value=0.1,
        step=0.1,
        disabled=not show_prompt_editor or llm_provider == "Mock LLM",
    )

left, right = st.columns([1.1, 1.2])

with left:
    st.subheader("上传文件")
    uploaded_files = st.file_uploader(
        "支持多文件上传",
        type=["png", "jpg", "jpeg", "pdf", "webp", "txt", "md"],
        accept_multiple_files=True,
    )

    st.info(
        "建议先用少量文件验证流程。纯 OCR 适合稳定提取；纯大模型适合理解和定制输出；结合模式通常更稳。"
    )

    run_clicked = st.button("开始处理", type="primary", use_container_width=True)

with right:
    st.subheader("当前配置说明")
    st.markdown(
        f"""
- **处理模式**：{mode}
- **OCR 引擎**：{ocr_engine or '-'}
- **LLM 提供方**：{llm_provider or '-'}
- **LLM 模型**：{llm_model or '-'}
        """
    )
    st.markdown(
        """
- **视觉识别 API**：只做 OCR / 文档识别。
- **大模型**：直接让模型识别并按提示词输出。
- **视觉识别 API + 大模型**：先 OCR，再交给模型整理、抽取、重写。
        """
    )

if run_clicked:
    if not uploaded_files:
        st.warning("请先上传至少一个文件。")
    elif mode in ["视觉识别 API", "视觉识别 API + 大模型"] and not ocr_engine:
        st.warning("当前模式需要选择 OCR 引擎。")
    elif mode in ["大模型", "视觉识别 API + 大模型"] and not llm_provider:
        st.warning("当前模式需要选择 LLM 提供方。")
    else:
        config = UserConfig(
            mode=mode,
            ocr_engine=ocr_engine,
            llm_provider=llm_provider,
            llm_model=llm_model,
            output_format=output_format,
            language_hint=language_hint,
            prompt_template_name=prompt_template_name,
            custom_prompt=custom_prompt if show_prompt_editor else "",
            temperature=temperature,
        )

        progress = st.progress(0, text="准备处理中...")
        results: List[ProcessResult] = []

        for idx, file in enumerate(uploaded_files, start=1):
            progress.progress((idx - 1) / len(uploaded_files), text=f"处理中：{file.name}")
            result = process_files(file, config)
            results.append(result)
            progress.progress(idx / len(uploaded_files), text=f"已完成：{file.name}")

        st.session_state["results"] = results
        st.session_state["config"] = config
        progress.empty()
        st.success(f"处理完成，共 {len(results)} 个文件。")

if "results" in st.session_state:
    results: List[ProcessResult] = st.session_state["results"]
    st.divider()
    st.subheader("处理结果")

    tabs = st.tabs([r.file_name for r in results])
    for tab, result in zip(tabs, results):
        with tab:
            col1, col2 = st.columns(2)
            with col1:
                st.markdown(f"**文件名**：{result.file_name}")
                st.markdown(f"**状态**：{result.status}")
                st.markdown(f"**模式**：{result.mode}")
                st.markdown(f"**OCR 引擎**：{result.ocr_engine_used or '-'}")
                st.markdown(f"**LLM**：{result.llm_provider_used or '-'} / {result.llm_model_used or '-'}")
            with col2:
                st.json(result.meta)

            st.markdown("**原始 OCR 文本 / 输入文本**")
            st.text_area(
                f"raw_{result.file_name}",
                value=result.raw_text,
                height=180,
                label_visibility="collapsed",
            )

            st.markdown("**最终输出**")
            st.text_area(
                f"final_{result.file_name}",
                value=result.final_output,
                height=240,
                label_visibility="collapsed",
            )

            if result.structured_data:
                st.markdown("**结构化输出**")
                st.json(result.structured_data)

            st.download_button(
                label=f"下载 {result.file_name} 结果",
                data=result.export_bytes(),
                file_name=result.export_name(),
                mime=result.export_mime(),
            )

    zip_buffer = io.BytesIO()
    with zipfile.ZipFile(zip_buffer, mode="w", compression=zipfile.ZIP_DEFLATED) as zf:
        manifest = []
        for result in results:
            zf.writestr(result.export_name(), result.export_bytes())
            manifest.append(asdict(result))
        zf.writestr("manifest.json", json.dumps(manifest, ensure_ascii=False, indent=2))
    zip_buffer.seek(0)

    st.download_button(
        label="下载全部结果（ZIP）",
        data=zip_buffer.getvalue(),
        file_name="ocr_llm_results.zip",
        mime="application/zip",
        use_container_width=True,
    )


status = get_provider_status()

st.title("OCR / LLM Workbench")
st.caption("当前版本只保留你真正需要的两条链路：PaddleOCR + DeepSeek。")

with st.sidebar:
    st.header("服务连接状态")
    for engine, ready in status["ocr"].items():
        if engine == "Mock OCR":
            continue
        st.markdown(f"**{engine}**：{'已连接' if ready else '未连接'}")
    for provider, ready in status["llm"].items():
        if provider == "Mock LLM":
            continue
        st.markdown(f"**{provider}**：{'已连接' if ready else '未连接'}")

    st.caption("API Key 仅从服务器环境变量 / Streamlit secrets 读取。")

    st.divider()
    st.header("处理参数")
    mode = st.radio(
        "处理模式",
        options=["视觉识别 API", "大模型", "视觉识别 API + 大模型"],
        index=2,
    )

    if mode in ["视觉识别 API", "视觉识别 API + 大模型"]:
        ocr_options = get_available_ocr_engines()
        ocr_engine = st.selectbox("OCR 引擎", options=ocr_options, index=0)
    else:
        ocr_engine = ""

    if mode in ["大模型", "视觉识别 API + 大模型"]:
        provider_options = [
            provider for provider in MODEL_REGISTRY.keys() if provider == "Mock LLM" or status["llm"].get(provider)
        ]
        if not provider_options:
            provider_options = ["Mock LLM"]

        default_provider_index = 0
        if "DeepSeek" in provider_options:
            default_provider_index = provider_options.index("DeepSeek")

        llm_provider = st.selectbox("LLM 提供方", options=provider_options, index=default_provider_index)
        model_options = get_models_for_provider(llm_provider)
        llm_model = st.selectbox("LLM 模型", options=model_options, index=0)

        capabilities = get_capabilities(llm_provider, llm_model)
        st.caption(
            f"能力：视觉={'是' if capabilities.get('supports_vision') else '否'}，"
            f"JSON={'是' if capabilities.get('supports_json') else '否'}，"
            f"长上下文={'是' if capabilities.get('supports_long_context') else '否'}"
        )
    else:
        llm_provider = ""
        llm_model = ""

    output_format = st.selectbox("输出格式", options=["text", "json", "markdown"], index=0)
    language_hint = st.selectbox("语言提示", options=["自动", "中文", "英文", "中英混合"], index=0)

    st.divider()
    st.subheader("提示词")
    prompt_template_name = st.selectbox(
        "默认模板",
        options=list(DEFAULT_PROMPTS.keys()),
        index=0,
        help="可先选模板，再按需修改下面的自定义提示词。",
    )
    st.caption(TEMPLATE_DESCRIPTIONS[prompt_template_name])

    show_prompt_editor = mode in ["大模型", "视觉识别 API + 大模型"]
    custom_prompt = st.text_area(
        "自定义提示词",
        value=DEFAULT_PROMPTS[prompt_template_name],
        height=220,
        disabled=not show_prompt_editor,
        help="仅大模型相关模式启用。纯 OCR 模式下不使用提示词。",
    )

    temperature = st.slider(
        "Temperature",
        min_value=0.0,
        max_value=1.0,
        value=0.1,
        step=0.1,
        disabled=not show_prompt_editor or llm_provider == "Mock LLM",
    )

left, right = st.columns([1.1, 1.2])

with left:
    st.subheader("上传文件")
    uploaded_files = st.file_uploader(
        "支持多文件上传",
        type=["png", "jpg", "jpeg", "pdf", "webp", "txt", "md"],
        accept_multiple_files=True,
    )

    st.info("建议先用少量文件验证流程。你当前这版主打：PaddleOCR 抽文本，DeepSeek 做整理。")
    run_clicked = st.button("开始处理", type="primary", use_container_width=True)

with right:
    st.subheader("当前配置说明")
    st.markdown(
        f"""
- **处理模式**：{mode}
- **OCR 引擎**：{ocr_engine or '-'}
- **LLM 提供方**：{llm_provider or '-'}
- **LLM 模型**：{llm_model or '-'}
        """
    )
    st.markdown(
        """
- **视觉识别 API**：只做 OCR / 文档识别。
- **大模型**：直接对文本做结构化、清洗、重写。
- **视觉识别 API + 大模型**：先用 PaddleOCR 抽文本，再交给 DeepSeek 整理。
        """
    )

if run_clicked:
    if not uploaded_files:
        st.warning("请先上传至少一个文件。")
    elif mode in ["视觉识别 API", "视觉识别 API + 大模型"] and not ocr_engine:
        st.warning("当前模式需要选择 OCR 引擎。")
    elif mode in ["大模型", "视觉识别 API + 大模型"] and not llm_provider:
        st.warning("当前模式需要选择 LLM 提供方。")
    else:
        config = UserConfig(
            mode=mode,
            ocr_engine=ocr_engine,
            llm_provider=llm_provider,
            llm_model=llm_model,
            output_format=output_format,
            language_hint=language_hint,
            prompt_template_name=prompt_template_name,
            custom_prompt=custom_prompt if show_prompt_editor else "",
            temperature=temperature,
        )

        progress = st.progress(0, text="准备处理中...")
        results: List[ProcessResult] = []

        for idx, file in enumerate(uploaded_files, start=1):
            progress.progress((idx - 1) / len(uploaded_files), text=f"处理中：{file.name}")
            result = process_files(file, config)
            results.append(result)
            progress.progress(idx / len(uploaded_files), text=f"已完成：{file.name}")

        st.session_state["results"] = results
        st.session_state["config"] = config
        progress.empty()
        st.success(f"处理完成，共 {len(results)} 个文件。")

if "results" in st.session_state:
    results: List[ProcessResult] = st.session_state["results"]
    st.divider()
    st.subheader("处理结果")

    tabs = st.tabs([r.file_name for r in results])
    for tab, result in zip(tabs, results):
        with tab:
            col1, col2 = st.columns(2)
            with col1:
                st.markdown(f"**文件名**：{result.file_name}")
                st.markdown(f"**状态**：{result.status}")
                st.markdown(f"**模式**：{result.mode}")
                st.markdown(f"**OCR 引擎**：{result.ocr_engine_used or '-'}")
                st.markdown(f"**LLM**：{result.llm_provider_used or '-'} / {result.llm_model_used or '-'}")
            with col2:
                st.json(result.meta)

            st.markdown("**原始 OCR 文本 / 输入文本**")
            st.text_area(
                f"raw_{result.file_name}",
                value=result.raw_text,
                height=180,
                label_visibility="collapsed",
            )

            st.markdown("**最终输出**")
            st.text_area(
                f"final_{result.file_name}",
                value=result.final_output,
                height=240,
                label_visibility="collapsed",
            )

            if result.structured_data:
                st.markdown("**结构化输出**")
                st.json(result.structured_data)

            st.download_button(
                label=f"下载 {result.file_name} 结果",
                data=result.export_bytes(),
                file_name=result.export_name(),
                mime=result.export_mime(),
            )

    zip_buffer = io.BytesIO()
    with zipfile.ZipFile(zip_buffer, mode="w", compression=zipfile.ZIP_DEFLATED) as zf:
        manifest = []
        for result in results:
            zf.writestr(result.export_name(), result.export_bytes())
            manifest.append(asdict(result))
        zf.writestr("manifest.json", json.dumps(manifest, ensure_ascii=False, indent=2))
    zip_buffer.seek(0)

    st.download_button(
        label="下载全部结果（ZIP）",
        data=zip_buffer.getvalue(),
        file_name="ocr_llm_results.zip",
        mime="application/zip",
        use_container_width=True,
    )
