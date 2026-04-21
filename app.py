import io
import json
import os
import hashlib
import zipfile
from dataclasses import asdict
from pathlib import Path
from typing import List

import streamlit as st

from prompts.templates import DEFAULT_PROMPTS, TEMPLATE_DESCRIPTIONS
from services.custom_llm_api import test_openai_compatible_endpoint, validate_custom_llm_config
from services.models import ProcessResult, UserConfig
from services.ocr_registry import get_available_ocr_engines, get_ocr_engine_status
from services.pipeline import process_files
from services.registry import (
    MODEL_REGISTRY,
    get_capabilities,
    get_llm_provider_status,
    get_models_for_provider,
)
from services.txt_config_parser import (
    key_values_to_text,
    parse_key_value_text,
    parse_multi_api_txt_config,
)


st.set_page_config(page_title="OCR / LLM Workbench", page_icon=":page_facing_up:", layout="wide")


CUSTOM_LLM_PROVIDER = "Custom LLM API"
TENCENT_OCR_ENGINE = "Tencent OCR"
BAIDU_OCR_ENGINE = "Baidu OCR"

API_CONFIG_TEMPLATE = """# API 配置模板：可一次配置多个 API

# 自定义 LLM（OpenAI-compatible）
[llm]
provider_type: openai_compatible
base_url: https://example.com/v1
api_key: your_llm_api_key_here
model: your_model_name_here
disable_proxy: true

# 可选 headers:
# Authorization=Bearer your_llm_api_key_here

extra_params:
# temperature=0

# 腾讯 OCR
[tencent_ocr]
secret_id: your_tencent_secret_id
secret_key: your_tencent_secret_key
region: ap-guangzhou
mode: accurate

# 百度 OCR
[baidu_ocr]
api_key: your_baidu_api_key
secret_key: your_baidu_secret_key
mode: accurate

# PaddleOCR Layout Parsing
[paddleocr]
api_url: https://your-paddleocr-endpoint
token: your_paddleocr_token
disable_proxy: true

# DeepSeek
[deepseek]
api_key: your_deepseek_api_key
base_url: https://api.deepseek.com
model: deepseek-chat
"""


def apply_uploaded_api_configs(parsed: dict) -> None:
    configs = parsed.get("configs", {})

    if "llm" in configs:
        llm = configs["llm"]
        draft = {
            "source": "uploaded_txt",
            "provider_type": llm.get("provider_type", "openai_compatible"),
            "base_url": llm.get("base_url", ""),
            "api_key": llm.get("api_key", ""),
            "model": llm.get("model", ""),
            "model_name": llm.get("model", ""),
            "disable_proxy": llm.get("disable_proxy", "true"),
            "headers": llm.get("headers", {}),
            "headers_text": key_values_to_text(llm.get("headers", {})),
            "extra_params": llm.get("extra_params", {}),
            "extra_params_text": key_values_to_text(llm.get("extra_params", {})),
        }
        st.session_state["custom_llm_api_config_draft"] = draft
        st.session_state["custom_llm_api_config"] = draft

    if "tencent_ocr" in configs:
        tencent = configs["tencent_ocr"]
        st.session_state["tencent_ocr_config"] = {
            "source": "uploaded_txt",
            "secret_id": tencent.get("secret_id", ""),
            "secret_key": tencent.get("secret_key", ""),
            "region": tencent.get("region", "ap-guangzhou") or "ap-guangzhou",
            "mode": tencent.get("mode", "accurate") or "accurate",
        }

    if "baidu_ocr" in configs:
        previous = st.session_state.get("baidu_ocr_config") or {}
        baidu = configs["baidu_ocr"]
        st.session_state["baidu_ocr_config"] = {
            "source": "uploaded_txt",
            "api_key": baidu.get("api_key", ""),
            "secret_key": baidu.get("secret_key", ""),
            "mode": baidu.get("mode", "accurate") or "accurate",
            "access_token": previous.get("access_token", ""),
            "access_token_expires_at": previous.get("access_token_expires_at", 0),
        }

    if "paddleocr" in configs:
        paddle = configs["paddleocr"]
        st.session_state["paddleocr_config"] = {
            "source": "uploaded_txt",
            "api_url": paddle.get("api_url", ""),
            "token": paddle.get("token", ""),
            "disable_proxy": paddle.get("disable_proxy", "true"),
        }

    if "deepseek" in configs:
        deepseek = configs["deepseek"]
        st.session_state["deepseek_config"] = {
            "source": "uploaded_txt",
            "api_key": deepseek.get("api_key", ""),
            "base_url": deepseek.get("base_url", "https://api.deepseek.com") or "https://api.deepseek.com",
            "model": deepseek.get("model", "deepseek-chat") or "deepseek-chat",
        }

    if "paddleocr" in configs:
        st.session_state["ocr_engine_select"] = "PaddleOCR Layout Parsing"
    elif "tencent_ocr" in configs:
        st.session_state["ocr_engine_select"] = TENCENT_OCR_ENGINE
    elif "baidu_ocr" in configs:
        st.session_state["ocr_engine_select"] = BAIDU_OCR_ENGINE

    if "llm" in configs:
        st.session_state["llm_provider_select"] = CUSTOM_LLM_PROVIDER
    elif "deepseek" in configs:
        st.session_state["llm_provider_select"] = "DeepSeek"


def render_api_config_importer() -> None:
    with st.sidebar.expander("批量导入 API TXT 配置", expanded=False):
        st.caption("上传 TXT 只用于填充当前会话配置，不会长期保存。")
        st.download_button(
            "下载多 API TXT 模板",
            data=API_CONFIG_TEMPLATE.encode("utf-8"),
            file_name="api_config_template.txt",
            mime="text/plain",
            key="download_multi_api_txt_template",
        )
        uploaded_config = st.file_uploader(
            "上传多 API TXT 配置文件",
            type=["txt"],
            accept_multiple_files=False,
            key="multi_api_txt_uploader",
        )
        if uploaded_config is None:
            return

        try:
            content = uploaded_config.getvalue()
            if not content:
                raise ValueError("上传文件为空。")
            fingerprint = hashlib.sha256(content).hexdigest()
            upload_key = f"{uploaded_config.name}:{fingerprint}"
            already_applied = st.session_state.get("last_api_txt_upload_key") == upload_key
            parsed = parse_multi_api_txt_config(content.decode("utf-8-sig"))
        except UnicodeDecodeError:
            st.error("TXT 文件编码无法识别，请使用 UTF-8 保存后重试。")
        except Exception as exc:
            st.error(str(exc))
        else:
            if not already_applied:
                apply_uploaded_api_configs(parsed)
                st.session_state["last_api_txt_upload_key"] = upload_key
            loaded = ", ".join(parsed.get("configs", {}).keys()) or "-"
            if already_applied:
                st.info(f"当前 TXT 配置已载入：{loaded}")
            else:
                st.success(f"已成功解析并填充：{loaded}")
            for warning in parsed.get("warnings", []):
                st.warning(warning)


def load_secrets_into_env() -> None:
    loaded = {}

    try:
        for key, value in st.secrets.items():
            if isinstance(value, (str, int, float, bool)):
                loaded[str(key)] = str(value)
    except Exception:
        pass

    local_secrets = Path("secrets.toml")
    if local_secrets.exists():
        try:
            import tomllib

            data = tomllib.loads(local_secrets.read_text(encoding="utf-8"))
            for key, value in data.items():
                if isinstance(value, (str, int, float, bool)):
                    loaded[str(key)] = str(value)
        except Exception as exc:
            st.sidebar.warning(f"读取本地 secrets.toml 失败：{exc}")

    for key, value in loaded.items():
        os.environ.setdefault(key, value)


def get_provider_status() -> dict:
    return {
        "ocr": get_ocr_engine_status(
            {
                "PaddleOCR Layout Parsing": st.session_state.get("paddleocr_config"),
                TENCENT_OCR_ENGINE: st.session_state.get("tencent_ocr_config"),
                BAIDU_OCR_ENGINE: st.session_state.get("baidu_ocr_config"),
            }
        ),
        "llm": get_llm_provider_status(
            {
                "DeepSeek": st.session_state.get("deepseek_config"),
            }
        ),
    }


def render_custom_llm_form() -> None:
    saved = st.session_state.get("custom_llm_api_config_draft") or st.session_state.get("custom_llm_api_config") or {}
    st.info("Custom LLM API 配置只保存在当前会话中。")

    with st.form("custom_llm_api_form"):
        base_url = st.text_input("Base URL（必填）", value=saved.get("base_url", ""))
        api_key = st.text_input("API Key（必填）", value=saved.get("api_key", ""), type="password")
        model_name = st.text_input("Model（必填）", value=saved.get("model", saved.get("model_name", "")))
        disable_proxy = st.checkbox(
            "禁用系统代理（HTTP_PROXY / HTTPS_PROXY / ALL_PROXY）",
            value=str(saved.get("disable_proxy", "true")).lower() not in {"false", "0", "no", "off"},
            help="遇到 ProxyError 或代理断连时建议开启。",
        )
        headers_text = st.text_area(
            "Headers（可选，每行 key=value）",
            value=saved.get("headers_text", key_values_to_text(saved.get("headers", {}))),
            height=90,
        )
        extra_params_text = st.text_area(
            "Extra Params（可选，每行 key=value）",
            value=saved.get("extra_params_text", key_values_to_text(saved.get("extra_params", {}))),
            height=80,
        )

        save_clicked = st.form_submit_button("保存到当前会话", type="primary")
        test_clicked = st.form_submit_button("测试连接")
        if save_clicked or test_clicked:
            headers, header_warnings = parse_key_value_text(headers_text, "Headers")
            extra_params, extra_warnings = parse_key_value_text(extra_params_text, "Extra Params")
            config = {
                "source": "uploaded_txt" if saved.get("source") == "uploaded_txt" else "manual",
                "provider_type": "openai_compatible",
                "base_url": base_url.strip(),
                "api_key": api_key,
                "model": model_name.strip(),
                "model_name": model_name.strip(),
                "disable_proxy": disable_proxy,
                "headers": headers,
                "headers_text": headers_text,
                "extra_params": extra_params,
                "extra_params_text": extra_params_text,
            }
            try:
                validate_custom_llm_config(config)
            except Exception as exc:
                st.error(str(exc))
            else:
                if test_clicked:
                    result = test_openai_compatible_endpoint(config, timeout=20)
                    if result.get("ok"):
                        st.success("连接测试成功。")
                    else:
                        st.error(f"连接测试失败：{result.get('category', '未知错误')}")
                    st.json(result)
                if save_clicked:
                    st.session_state["custom_llm_api_config"] = config
                    st.session_state["custom_llm_api_config_draft"] = config
                    st.success("Custom LLM API 已保存到当前会话。")
                for warning in header_warnings + extra_warnings:
                    st.warning(warning)

    if st.button("清除当前会话的 Custom LLM 配置", key="clear_custom_llm_api_config"):
        st.session_state.pop("custom_llm_api_config", None)
        st.session_state.pop("custom_llm_api_config_draft", None)
        st.success("已清除当前会话的 Custom LLM API 配置。")


def render_tencent_ocr_form() -> None:
    saved = st.session_state.get("tencent_ocr_config") or {}
    st.info("Tencent OCR 使用 SecretId / SecretKey，配置只保存在当前会话。")

    with st.form("tencent_ocr_form"):
        secret_id = st.text_input("SecretId", value=saved.get("secret_id", ""))
        secret_key = st.text_input("SecretKey", value=saved.get("secret_key", ""), type="password")
        region = st.text_input("Region", value=saved.get("region", "ap-guangzhou"))
        mode = st.selectbox(
            "OCR 模式",
            options=["accurate", "basic"],
            format_func=lambda item: "Accurate（高精度）" if item == "accurate" else "Basic（标准）",
            index=0 if saved.get("mode", "accurate") == "accurate" else 1,
        )

        save_clicked = st.form_submit_button("保存到当前会话", type="primary")
        if save_clicked:
            if not secret_id.strip():
                st.error("Tencent OCR SecretId 不能为空。")
            elif not secret_key:
                st.error("Tencent OCR SecretKey 不能为空。")
            else:
                st.session_state["tencent_ocr_config"] = {
                    "secret_id": secret_id.strip(),
                    "secret_key": secret_key,
                    "region": region.strip() or "ap-guangzhou",
                    "mode": mode,
                }
                st.success("Tencent OCR 配置已保存到当前会话。")


def render_baidu_ocr_form() -> None:
    saved = st.session_state.get("baidu_ocr_config") or {}
    st.info("Baidu OCR 使用 API Key / Secret Key，配置只保存在当前会话。")

    with st.form("baidu_ocr_form"):
        api_key = st.text_input("API Key", value=saved.get("api_key", ""))
        secret_key = st.text_input("Secret Key", value=saved.get("secret_key", ""), type="password")
        mode = st.selectbox(
            "OCR 模式",
            options=["accurate", "general"],
            format_func=lambda item: "Accurate（高精度）" if item == "accurate" else "General（标准）",
            index=0 if saved.get("mode", "accurate") == "accurate" else 1,
        )

        save_clicked = st.form_submit_button("保存到当前会话", type="primary")
        if save_clicked:
            if not api_key.strip():
                st.error("Baidu OCR API Key 不能为空。")
            elif not secret_key:
                st.error("Baidu OCR Secret Key 不能为空。")
            else:
                previous = st.session_state.get("baidu_ocr_config") or {}
                st.session_state["baidu_ocr_config"] = {
                    "api_key": api_key.strip(),
                    "secret_key": secret_key,
                    "mode": mode,
                    "access_token": previous.get("access_token", ""),
                    "access_token_expires_at": previous.get("access_token_expires_at", 0),
                }
                st.success("Baidu OCR 配置已保存到当前会话。")


def get_ocr_provider_config(engine: str) -> dict | None:
    if engine == "PaddleOCR Layout Parsing":
        return st.session_state.get("paddleocr_config")
    if engine == TENCENT_OCR_ENGINE:
        return st.session_state.get("tencent_ocr_config")
    if engine == BAIDU_OCR_ENGINE:
        return st.session_state.get("baidu_ocr_config")
    return None


def get_llm_provider_config(provider: str) -> dict | None:
    if provider == "DeepSeek":
        return st.session_state.get("deepseek_config")
    if provider == CUSTOM_LLM_PROVIDER:
        return st.session_state.get("custom_llm_api_config")
    return None


load_secrets_into_env()
status = get_provider_status()

st.title("OCR / LLM Workbench")
st.caption("固定主流程：先 OCR 提取文本，再按需使用 LLM 对 OCR 文本做后处理。LLM 不直接识图。")

with st.sidebar:
    st.header("服务连接状态")
    for engine, ready in status["ocr"].items():
        if engine == "Mock OCR":
            continue
        if engine == "PaddleOCR Layout Parsing" and st.session_state.get("paddleocr_config"):
            st.markdown(f"**{engine}**：已保存当前会话配置")
        elif engine == TENCENT_OCR_ENGINE and st.session_state.get("tencent_ocr_config"):
            st.markdown(f"**{engine}**：已保存当前会话配置")
        elif engine == BAIDU_OCR_ENGINE and st.session_state.get("baidu_ocr_config"):
            st.markdown(f"**{engine}**：已保存当前会话配置")
        else:
            st.markdown(f"**{engine}**：{'已连接' if ready else '未连接'}")

    for provider, ready in status["llm"].items():
        if provider == "Mock LLM":
            continue
        if provider == "DeepSeek" and st.session_state.get("deepseek_config"):
            st.markdown(f"**{provider}**：已保存当前会话配置")
        else:
            st.markdown(f"**{provider}**：{'已连接' if ready else '未连接'}")

    st.caption("PaddleOCR / DeepSeek 可从环境变量、Streamlit secrets 或项目根目录 secrets.toml 读取。")
    render_api_config_importer()

    st.divider()
    st.header("处理参数")

    ocr_options = get_available_ocr_engines(
        {
            "PaddleOCR Layout Parsing": st.session_state.get("paddleocr_config"),
            TENCENT_OCR_ENGINE: st.session_state.get("tencent_ocr_config"),
            BAIDU_OCR_ENGINE: st.session_state.get("baidu_ocr_config"),
        }
    )
    ocr_engine = st.selectbox("OCR 引擎", options=ocr_options, index=0, key="ocr_engine_select")
    if ocr_engine == TENCENT_OCR_ENGINE:
        render_tencent_ocr_form()
    elif ocr_engine == BAIDU_OCR_ENGINE:
        render_baidu_ocr_form()

    language_hint = st.selectbox(
        "语言提示",
        options=["自动", "中文", "英文", "中英混合"],
        index=0,
        key="language_hint_select",
    )

    st.divider()
    use_llm_postprocess = st.checkbox("使用 LLM 整理 OCR 结果", value=False, key="use_llm_postprocess")
    llm_provider = ""
    llm_model = ""
    output_format = "text"
    prompt_template_name = list(DEFAULT_PROMPTS.keys())[0]
    custom_prompt = DEFAULT_PROMPTS[prompt_template_name]
    temperature = 0.1

    if use_llm_postprocess:
        provider_options = [
            provider
            for provider in MODEL_REGISTRY.keys()
            if (
                provider == "Mock LLM"
                or status["llm"].get(provider)
                or provider == CUSTOM_LLM_PROVIDER
                or (provider == "DeepSeek" and st.session_state.get("deepseek_config"))
            )
        ]
        if not provider_options:
            provider_options = [CUSTOM_LLM_PROVIDER]

        selected_provider = st.session_state.get("llm_provider_select")
        if selected_provider not in provider_options:
            if st.session_state.get("deepseek_config") and "DeepSeek" in provider_options:
                selected_provider = "DeepSeek"
            elif st.session_state.get("custom_llm_api_config") and CUSTOM_LLM_PROVIDER in provider_options:
                selected_provider = CUSTOM_LLM_PROVIDER
            else:
                selected_provider = provider_options[0]

        llm_provider = st.selectbox(
            "LLM Provider",
            options=provider_options,
            index=provider_options.index(selected_provider),
            key="llm_provider_select",
        )

        if llm_provider == CUSTOM_LLM_PROVIDER:
            render_custom_llm_form()
            saved_custom_llm = st.session_state.get("custom_llm_api_config") or {}
            llm_model = saved_custom_llm.get("model", saved_custom_llm.get("model_name", "custom-session-model"))
            st.caption(f"当前模型：{llm_model}")
        elif llm_provider == "DeepSeek" and st.session_state.get("deepseek_config"):
            deepseek_config = st.session_state["deepseek_config"]
            model_options = get_models_for_provider(llm_provider)
            configured_model = deepseek_config.get("model", "deepseek-chat")
            if configured_model not in model_options:
                model_options = [configured_model] + model_options
            llm_model = st.selectbox(
                "LLM 模型",
                options=model_options,
                index=model_options.index(configured_model),
                key="llm_model_select",
            )
        else:
            model_options = get_models_for_provider(llm_provider)
            llm_model = st.selectbox("LLM 模型", options=model_options, index=0, key="llm_model_select")

        capabilities = get_capabilities(llm_provider, llm_model)
        st.caption(
            f"能力：JSON={'是' if capabilities.get('supports_json') else '否'}，"
            f"长上下文={'是' if capabilities.get('supports_long_context') else '否'}"
        )

        output_format = st.selectbox(
            "输出格式",
            options=["text", "json", "markdown", "word"],
            index=0,
            key="output_format_select",
        )
        prompt_template_name = st.selectbox(
            "提示词模板",
            options=list(DEFAULT_PROMPTS.keys()),
            index=0,
            key="prompt_template_select",
        )
        st.caption(TEMPLATE_DESCRIPTIONS[prompt_template_name])
        custom_prompt = st.text_area(
            "自定义提示词",
            value=DEFAULT_PROMPTS[prompt_template_name],
            height=220,
            help="LLM 只基于 OCR 输出文本做整理，不会重新识别图片。",
            key="custom_prompt_text",
        )
        temperature = st.slider(
            "Temperature",
            min_value=0.0,
            max_value=1.0,
            value=0.1,
            step=0.1,
            disabled=llm_provider == "Mock LLM",
            key="temperature_slider",
        )

left, right = st.columns([1.1, 1.2])

with left:
    st.subheader("上传文件")
    uploaded_files = st.file_uploader(
        "支持多文件上传",
        type=["png", "jpg", "jpeg", "pdf", "webp"],
        accept_multiple_files=True,
        key="file_uploader",
    )
    st.info("系统会始终执行 OCR，并保留原始 OCR 文本。")
    run_clicked = st.button("开始处理", type="primary", use_container_width=True, key="run_button")

with right:
    st.subheader("当前配置说明")
    st.markdown(
        f"""
- **OCR 引擎**：{ocr_engine or "-"}
- **语言提示**：{language_hint}
- **启用 LLM 整理**：{"是" if use_llm_postprocess else "否"}
- **LLM Provider**：{llm_provider or "-"}
- **LLM 模型**：{llm_model or "-"}
- **输出格式**：{output_format if use_llm_postprocess else "-"}
        """
    )
    st.markdown(
        """
- **PaddleOCR Layout Parsing**：使用 `PADDLEOCR_API_URL` 和 `PADDLEOCR_TOKEN`
- **Tencent OCR**：使用 SecretId / SecretKey
- **Baidu OCR**：使用 API Key / Secret Key
- **DeepSeek**：使用 `DEEPSEEK_API_KEY`，可选 `DEEPSEEK_BASE_URL`
- **LLM 后处理**：始终基于 OCR 文本做保守整理，不直接识图
        """
    )

if run_clicked:
    if not uploaded_files:
        st.warning("请先上传至少一个文件。")
    elif not ocr_engine:
        st.warning("请先选择 OCR 引擎。")
    elif (
        ocr_engine == "PaddleOCR Layout Parsing"
        and not st.session_state.get("paddleocr_config")
        and not status["ocr"].get("PaddleOCR Layout Parsing")
    ):
        st.warning("请先通过 secrets/env 或 TXT 上传配置 PaddleOCR。")
    elif ocr_engine == TENCENT_OCR_ENGINE and not st.session_state.get("tencent_ocr_config"):
        st.warning("请先保存 Tencent OCR 配置。")
    elif ocr_engine == BAIDU_OCR_ENGINE and not st.session_state.get("baidu_ocr_config"):
        st.warning("请先保存 Baidu OCR 配置。")
    elif use_llm_postprocess and not llm_provider:
        st.warning("已启用 LLM 整理，请先选择 LLM provider。")
    elif use_llm_postprocess and llm_provider == "DeepSeek" and not st.session_state.get("deepseek_config") and not status["llm"].get("DeepSeek"):
        st.warning("请先通过 secrets/env 或 TXT 上传配置 DeepSeek。")
    elif use_llm_postprocess and llm_provider == CUSTOM_LLM_PROVIDER and not st.session_state.get("custom_llm_api_config"):
        st.warning("请先保存 Custom LLM API 配置。")
    else:
        config = UserConfig(
            ocr_engine=ocr_engine,
            use_llm_postprocess=use_llm_postprocess,
            llm_provider=llm_provider if use_llm_postprocess else "",
            llm_model=llm_model if use_llm_postprocess else "",
            output_format=output_format if use_llm_postprocess else "text",
            language_hint=language_hint,
            prompt_template_name=prompt_template_name if use_llm_postprocess else "",
            custom_prompt=custom_prompt if use_llm_postprocess else "",
            temperature=temperature if use_llm_postprocess else 0.1,
            llm_provider_config=get_llm_provider_config(llm_provider) if use_llm_postprocess else None,
            ocr_provider_config=get_ocr_provider_config(ocr_engine),
        )

        progress = st.progress(0, text="准备处理...")
        results: List[ProcessResult] = []

        with st.spinner("正在处理文件，请稍候..."):
            for idx, file in enumerate(uploaded_files, start=1):
                progress.progress((idx - 1) / len(uploaded_files), text=f"处理中：{file.name}")
                results.append(process_files(file, config))
                progress.progress(idx / len(uploaded_files), text=f"已完成：{file.name}")

        st.session_state["results"] = results
        st.session_state["config"] = config
        progress.empty()

        ocr_failures = sum(1 for result in results if result.meta.get("stage") == "ocr")
        llm_failures = sum(1 for result in results if result.meta.get("stage") == "llm" and result.status == "partial_success")
        if ocr_failures:
            st.error(f"处理完成，但有 {ocr_failures} 个文件在 OCR 阶段失败。")
        elif llm_failures:
            st.warning(f"OCR 已完成，但有 {llm_failures} 个文件的 LLM 后处理失败。")
        else:
            st.success(f"处理完成，共 {len(results)} 个文件。")

if "results" in st.session_state:
    results = st.session_state["results"]
    st.divider()
    st.subheader("处理结果")

    tabs = st.tabs([r.file_name for r in results])
    for tab, result in zip(tabs, results):
        with tab:
            col1, col2 = st.columns(2)
            with col1:
                st.markdown(f"**文件名**：{result.file_name}")
                st.markdown(f"**状态**：{result.status}")
                st.markdown(f"**OCR 引擎**：{result.ocr_engine_used or '-'}")
                if result.use_llm_postprocess:
                    st.markdown(f"**LLM**：{result.llm_provider_used or '-'} / {result.llm_model_used or '-'}")
            with col2:
                st.json(result.meta)

            if result.error:
                stage = result.meta.get("stage")
                if stage == "ocr":
                    st.error(f"OCR 失败：{result.error}")
                elif stage == "llm":
                    st.error(f"OCR 成功，但 LLM 整理失败：{result.error}")
                elif stage == "llm_config":
                    st.warning(result.error)
                else:
                    st.warning(result.error)

            st.markdown("**原始 OCR 文本**")
            st.text_area(
                f"raw_{result.file_name}",
                value=result.raw_text,
                height=220,
                label_visibility="collapsed",
                key=f"raw_{result.file_name}",
            )
            if result.ocr_output_text:
                st.download_button(
                    label=f"下载 {result.file_name} OCR 结果",
                    data=result.ocr_export_bytes(),
                    file_name=result.ocr_export_name(),
                    mime=result.ocr_export_mime(),
                    key=f"download_ocr_{result.file_name}",
                )

            if result.use_llm_postprocess:
                st.markdown("**LLM 整理结果**")
                st.text_area(
                    f"llm_{result.file_name}",
                    value=result.llm_output_text,
                    height=240,
                    label_visibility="collapsed",
                    key=f"llm_{result.file_name}",
                )
                if result.structured_data:
                    st.markdown("**结构化输出**")
                    st.json(result.structured_data)
                if result.llm_succeeded:
                    st.download_button(
                        label=f"下载 {result.file_name} LLM 结果",
                        data=result.llm_export_bytes(),
                        file_name=result.llm_export_name(),
                        mime=result.llm_export_mime(),
                        key=f"download_llm_{result.file_name}",
                    )

    zip_buffer = io.BytesIO()
    with zipfile.ZipFile(zip_buffer, mode="w", compression=zipfile.ZIP_DEFLATED) as zf:
        manifest = []
        for result in results:
            if result.ocr_output_text:
                zf.writestr(result.ocr_export_name(), result.ocr_export_bytes())
            if result.llm_succeeded:
                zf.writestr(result.llm_export_name(), result.llm_export_bytes())
            manifest.append(asdict(result))
        zf.writestr("manifest.json", json.dumps(manifest, ensure_ascii=False, indent=2))
    zip_buffer.seek(0)

    st.download_button(
        label="下载全部结果（ZIP）",
        data=zip_buffer.getvalue(),
        file_name="ocr_llm_results.zip",
        mime="application/zip",
        use_container_width=True,
        key="download_zip_button",
    )
