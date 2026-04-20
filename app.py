import io
import json
import os
import zipfile
from dataclasses import asdict
from typing import List

import streamlit as st

from services.models import ProcessResult, UserConfig
from services.pipeline import process_files
from prompts.templates import DEFAULT_PROMPTS, TEMPLATE_DESCRIPTIONS

st.set_page_config(page_title="OCR / LLM Workbench", page_icon="📄", layout="wide")


def get_provider_status() -> dict:
    google_ready = bool(os.getenv("GOOGLE_API_KEY") or os.getenv("GOOGLE_APPLICATION_CREDENTIALS"))
    openai_ready = bool(os.getenv("OPENAI_API_KEY"))
    return {
        "google": google_ready,
        "openai": openai_ready,
    }


status = get_provider_status()

st.title("OCR / LLM Workbench")
st.caption("批量上传文件，选择 OCR / 大模型 / 结合模式，并支持默认或自定义提示词。")

with st.sidebar:
    st.header("服务连接状态")
    st.markdown(
        f"**Google OCR**：{'已连接' if status['google'] else '未连接'}  \n"
        f"**OpenAI**：{'已连接' if status['openai'] else '未连接'}"
    )
    st.caption("正式方案：API Key 仅由服务器环境变量读取，不在网页中直接填写。")

    enable_google = st.toggle(
        "启用 Google OCR",
        value=status["google"],
        disabled=not status["google"],
        help="需要服务端已配置 GOOGLE_API_KEY 或 GOOGLE_APPLICATION_CREDENTIALS。",
    )
    enable_openai = st.toggle(
        "启用 OpenAI",
        value=status["openai"],
        disabled=not status["openai"],
        help="需要服务端已配置 OPENAI_API_KEY。",
    )

    st.divider()
    st.header("处理参数")
    mode = st.radio(
        "处理模式",
        options=["视觉识别 API", "大模型", "视觉识别 API + 大模型"],
        index=2,
    )

    available_ocr_engines = ["Mock OCR"]
    if enable_google:
        available_ocr_engines.append("Google Document AI")

    if mode in ["视觉识别 API", "视觉识别 API + 大模型"]:
        ocr_engine = st.selectbox(
            "OCR 引擎",
            options=available_ocr_engines,
            index=0 if "Mock OCR" in available_ocr_engines else 0,
        )
    else:
        ocr_engine = ""

    available_llm_providers = ["Mock LLM"]
    if enable_openai:
        available_llm_providers.append("OpenAI")

    if mode in ["大模型", "视觉识别 API + 大模型"]:
        llm_provider = st.selectbox(
            "LLM 提供方",
            options=available_llm_providers,
            index=0 if "Mock LLM" in available_llm_providers else 0,
        )
        llm_model = st.text_input("LLM 模型", value="gpt-4.1-mini")
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
        disabled=not show_prompt_editor,
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
- **Google OCR 状态**：{'已启用' if enable_google else '未启用'}
- **OpenAI 状态**：{'已启用' if enable_openai else '未启用'}
- **处理模式**：{mode}
- **OCR 引擎**：{ocr_engine or '-'}
- **LLM 提供方**：{llm_provider or '-'}
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
