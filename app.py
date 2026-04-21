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


MODE_OCR = "视觉识别 API"
MODE_LLM = "大模型"
MODE_OCR_FORMAT = "视觉识别结果整理"
MODES = [MODE_OCR, MODE_LLM, MODE_OCR_FORMAT]
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

# 可选
headers:
# Authorization=Bearer your_llm_api_key_here

extra_params:
# temperature=0

# 腾讯云 OCR
[tencent_ocr]
secret_id: your_tencent_secret_id
secret_key: your_tencent_secret_key
region: ap-guangzhou
mode: accurate

# 百度智能云 OCR
[baidu_ocr]
api_key: your_baidu_api_key
secret_key: your_baidu_secret_key
mode: accurate

# PaddleOCR Layout Parsing
[paddleocr]
api_url: https://your-paddleocr-endpoint
token: your_paddleocr_token

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
        st.caption("上传 TXT 仅用于自动填充本次会话配置，不会长期保存。可同时配置 LLM、DeepSeek、PaddleOCR、Tencent OCR、Baidu OCR。")
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
        if uploaded_config is not None:
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
                loaded = ", ".join(parsed.get("configs", {}).keys()) or "无"
                if already_applied:
                    st.info(f"当前 TXT 配置已载入：{loaded}")
                else:
                    st.success(f"已成功解析并填充：{loaded}")
                if "llm" in parsed.get("configs", {}):
                    st.caption(f"LLM base_url：{parsed['configs']['llm'].get('base_url', '-')}")
                    st.caption(f"LLM model：{parsed['configs']['llm'].get('model', '-')}")
                if "tencent_ocr" in parsed.get("configs", {}):
                    st.caption(f"Tencent OCR mode：{parsed['configs']['tencent_ocr'].get('mode', '-')}")
                if "baidu_ocr" in parsed.get("configs", {}):
                    st.caption(f"Baidu OCR mode：{parsed['configs']['baidu_ocr'].get('mode', '-')}")
                if "paddleocr" in parsed.get("configs", {}):
                    st.caption("PaddleOCR：已填充")
                if "deepseek" in parsed.get("configs", {}):
                    st.caption(f"DeepSeek model：{parsed['configs']['deepseek'].get('model', '-')}")
                for warning in parsed.get("warnings", []):
                    st.warning(warning)


def load_secrets_into_env() -> None:
    """Expose Streamlit secrets and local secrets.toml to the existing services."""
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
    st.info("上传 TXT 仅用于自动填充本次会话的配置，不会长期保存。最终运行以当前页面表单值为准。")

    with st.form("custom_llm_api_form"):
        base_url = st.text_input("Base URL（必填）", value=saved.get("base_url", ""))
        api_key = st.text_input("API Key（必填）", value=saved.get("api_key", ""), type="password")
        model_name = st.text_input("Model（必填）", value=saved.get("model", saved.get("model_name", "")))
        disable_proxy = st.checkbox(
            "禁用系统代理（HTTP_PROXY / HTTPS_PROXY / ALL_PROXY）",
            value=str(saved.get("disable_proxy", "true")).lower() not in {"false", "0", "no", "off"},
            help="遇到 ProxyError 或代理断连时建议开启。关闭后会继承系统代理环境变量。",
        )
        headers_text = st.text_area(
            "Headers（可选，每行 key=value）",
            value=saved.get("headers_text", key_values_to_text(saved.get("headers", {}))),
            height=90,
            help="示例：Authorization=Bearer your_api_key_here",
        )
        extra_params_text = st.text_area(
            "Extra Params（可选，每行 key=value）",
            value=saved.get("extra_params_text", key_values_to_text(saved.get("extra_params", {}))),
            height=80,
            help="示例：temperature=0",
        )

        save_clicked = st.form_submit_button("Save for This Session", type="primary")
        test_clicked = st.form_submit_button("Test Connection")
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
                    st.success("Custom LLM API 已保存到本次会话。")
                for warning in header_warnings + extra_warnings:
                    st.warning(warning)

    if st.button("Clear Session Config", key="clear_custom_llm_api_config"):
        st.session_state.pop("custom_llm_api_config", None)
        st.session_state.pop("custom_llm_api_config_draft", None)
        st.success("已清除本次会话的 Custom LLM API 配置。")

    if st.session_state.get("custom_llm_api_config"):
        current = st.session_state["custom_llm_api_config"]
        masked_key = "已填写" if current.get("api_key") else "未填写"
        st.caption(f"状态：已保存，模型：{current.get('model', current.get('model_name', '-'))}，API Key：{masked_key}")
        st.caption(f"配置来源：{current.get('source', 'manual')}")
    else:
        st.caption("状态：尚未保存。")


def render_tencent_ocr_form() -> None:
    saved = st.session_state.get("tencent_ocr_config") or {}
    st.info("Tencent OCR 使用 SecretId / SecretKey。配置只保存在本次会话中。")

    with st.form("tencent_ocr_form"):
        secret_id = st.text_input("SecretId", value=saved.get("secret_id", ""))
        secret_key = st.text_input("SecretKey", value=saved.get("secret_key", ""), type="password")
        region = st.text_input("Region", value=saved.get("region", "ap-guangzhou"))
        mode = st.selectbox(
            "OCR 模式",
            options=["accurate", "basic"],
            format_func=lambda item: "Accurate（高精度）" if item == "accurate" else "Basic（标准/普通）",
            index=0 if saved.get("mode", "accurate") == "accurate" else 1,
        )

        save_clicked = st.form_submit_button("Save for This Session", type="primary")
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
                st.success("Tencent OCR 配置已保存到本次会话。")

    if st.button("Clear Tencent OCR Config", key="clear_tencent_ocr_config"):
        st.session_state.pop("tencent_ocr_config", None)
        st.success("已清除本次会话的 Tencent OCR 配置。")

    current = st.session_state.get("tencent_ocr_config")
    if current:
        st.caption(f"状态：已保存，Region：{current.get('region', '-')}，模式：{current.get('mode', '-')}")
    else:
        st.caption("状态：尚未保存。")


def render_baidu_ocr_form() -> None:
    saved = st.session_state.get("baidu_ocr_config") or {}
    st.info("Baidu OCR 使用 API Key / Secret Key，并自动换取和缓存 access_token。配置只保存在本次会话中。")

    with st.form("baidu_ocr_form"):
        api_key = st.text_input("API Key", value=saved.get("api_key", ""))
        secret_key = st.text_input("Secret Key", value=saved.get("secret_key", ""), type="password")
        mode = st.selectbox(
            "OCR 模式",
            options=["accurate", "general"],
            format_func=lambda item: "Accurate（高精度）" if item == "accurate" else "General（标准）",
            index=0 if saved.get("mode", "accurate") == "accurate" else 1,
        )

        save_clicked = st.form_submit_button("Save for This Session", type="primary")
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
                st.success("Baidu OCR 配置已保存到本次会话。")

    if st.button("Clear Baidu OCR Config", key="clear_baidu_ocr_config"):
        st.session_state.pop("baidu_ocr_config", None)
        st.success("已清除本次会话的 Baidu OCR 配置。")

    current = st.session_state.get("baidu_ocr_config")
    if current:
        token_status = "已缓存 access_token" if current.get("access_token") else "尚未获取 access_token"
        st.caption(f"状态：已保存，模式：{current.get('mode', '-')}，{token_status}")
    else:
        st.caption("状态：尚未保存。")


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
st.caption("当前主链路：PaddleOCR 提取文本；如需整理，LLM 只处理 OCR 输出文本，不参与图片识别。")

with st.sidebar:
    st.header("服务连接状态")
    for engine, ready in status["ocr"].items():
        if engine == "Mock OCR":
            continue
        if engine == TENCENT_OCR_ENGINE:
            st.markdown(f"**{engine}**：{'已保存本次会话配置' if st.session_state.get('tencent_ocr_config') else '可配置'}")
            continue
        if engine == BAIDU_OCR_ENGINE:
            st.markdown(f"**{engine}**：{'已保存本次会话配置' if st.session_state.get('baidu_ocr_config') else '可配置'}")
            continue
        if engine == "PaddleOCR Layout Parsing" and st.session_state.get("paddleocr_config"):
            st.markdown(f"**{engine}**：已保存本次会话配置")
            continue
        st.markdown(f"**{engine}**：{'已连接' if ready else '未连接'}")

    for provider, ready in status["llm"].items():
        if provider == "Mock LLM":
            continue
        if provider == "DeepSeek" and st.session_state.get("deepseek_config"):
            st.markdown(f"**{provider}**：已保存本次会话配置")
            continue
        st.markdown(f"**{provider}**：{'已连接' if ready else '未连接'}")

    st.caption("PaddleOCR / DeepSeek 从环境变量、Streamlit secrets 或项目根目录 secrets.toml 读取。Custom LLM 只保存在本次会话。")

    render_api_config_importer()

    st.divider()
    st.header("处理参数")
    mode = st.radio(
        "处理模式",
        options=MODES,
        index=2,
        key="mode_radio",
    )

    if mode in [MODE_OCR, MODE_OCR_FORMAT]:
        ocr_options = get_available_ocr_engines({"PaddleOCR Layout Parsing": st.session_state.get("paddleocr_config")})
        ocr_engine = st.selectbox("OCR 引擎", options=ocr_options, index=0, key="ocr_engine_select")
        if ocr_engine == TENCENT_OCR_ENGINE:
            render_tencent_ocr_form()
        elif ocr_engine == BAIDU_OCR_ENGINE:
            render_baidu_ocr_form()
    else:
        ocr_engine = ""

    if mode in [MODE_LLM, MODE_OCR_FORMAT]:
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
            elif "DeepSeek" in provider_options:
                selected_provider = "DeepSeek"
            else:
                selected_provider = provider_options[0]

        llm_provider = st.selectbox(
            "LLM 提供方",
            options=provider_options,
            index=provider_options.index(selected_provider),
            key="llm_provider_select",
        )

        if llm_provider == CUSTOM_LLM_PROVIDER:
            render_custom_llm_form()
            saved_custom_llm = st.session_state.get("custom_llm_api_config") or {}
            llm_model = saved_custom_llm.get("model", saved_custom_llm.get("model_name", "custom-session-model"))
            st.caption(f"LLM 模型：{llm_model}")
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
            f"能力：视觉={'是' if capabilities.get('supports_vision') else '否'}；"
            f"JSON={'是' if capabilities.get('supports_json') else '否'}；"
            f"长上下文={'是' if capabilities.get('supports_long_context') else '否'}"
        )
    else:
        llm_provider = ""
        llm_model = ""

    output_format = st.selectbox("输出格式", options=["text", "json", "markdown", "word"], index=0, key="output_format_select")
    language_hint = st.selectbox(
        "语言提示",
        options=["自动", "中文", "英文", "中英混合"],
        index=0,
        key="language_hint_select",
    )

    st.divider()
    st.subheader("提示词")
    prompt_template_name = st.selectbox(
        "默认模板",
        options=list(DEFAULT_PROMPTS.keys()),
        index=0,
        help="可以先选择模板，再按需修改下面的自定义提示词。",
        key="prompt_template_select",
    )
    st.caption(TEMPLATE_DESCRIPTIONS[prompt_template_name])

    show_prompt_editor = mode in [MODE_LLM, MODE_OCR_FORMAT]
    custom_prompt = st.text_area(
        "整理提示词" if mode == MODE_OCR_FORMAT else "自定义提示词",
        value=DEFAULT_PROMPTS[prompt_template_name],
        height=220,
        disabled=not show_prompt_editor,
        help="视觉识别结果整理模式下，LLM 仅用于整理 OCR 输出，不会修改原始识别结果。",
        key="custom_prompt_text",
    )

    if mode == MODE_OCR_FORMAT:
        st.warning("LLM 仅用于整理 OCR 输出，不会修改原始识别结果，也不会重新识别图片。")

    temperature = st.slider(
        "Temperature",
        min_value=0.0,
        max_value=1.0,
        value=0.1,
        step=0.1,
        disabled=not show_prompt_editor or llm_provider == "Mock LLM",
        key="temperature_slider",
    )

left, right = st.columns([1.1, 1.2])

with left:
    st.subheader("上传文件")
    uploaded_files = st.file_uploader(
        "支持多文件上传",
        type=["png", "jpg", "jpeg", "pdf", "webp", "txt", "md"],
        accept_multiple_files=True,
        key="file_uploader",
    )

    st.info("建议先用少量文件验证流程。视觉识别结果整理会同时保留 raw OCR 和 LLM 整理输出。")
    run_clicked = st.button("开始处理", type="primary", use_container_width=True, key="run_button")

with right:
    st.subheader("当前配置说明")
    st.markdown(
        f"""
- **处理模式**：{mode}
- **OCR 引擎**：{ocr_engine or "-"}
- **LLM 提供方**：{llm_provider or "-"}
- **LLM 模型**：{llm_model or "-"}
        """
    )
    st.markdown(
        """
- **PaddleOCR Layout Parsing**：使用 `PADDLEOCR_API_URL` 和 `PADDLEOCR_TOKEN`。
- **Tencent OCR**：使用 SecretId / SecretKey，默认高精度识别。
- **Baidu OCR**：使用 API Key / Secret Key，并自动换取 access_token。
- **DeepSeek**：使用 `DEEPSEEK_API_KEY`，可选 `DEEPSEEK_BASE_URL`。
- **视觉识别结果整理**：先 OCR，后 LLM 保守整理；原始 OCR 与整理结果分开保存和下载。
- **Custom LLM API**：在左侧选择该 LLM 提供方后显示临时配置表单。
        """
    )

if run_clicked:
    if not uploaded_files:
        st.warning("请先上传至少一个文件。")
    elif mode in [MODE_OCR, MODE_OCR_FORMAT] and not ocr_engine:
        st.warning("当前模式需要选择 OCR 引擎。")
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
    elif mode in [MODE_LLM, MODE_OCR_FORMAT] and not llm_provider:
        st.warning("当前模式需要选择 LLM 提供方。")
    elif llm_provider == "DeepSeek" and not st.session_state.get("deepseek_config") and not status["llm"].get("DeepSeek"):
        st.warning("请先通过 secrets/env 或 TXT 上传配置 DeepSeek。")
    elif llm_provider == CUSTOM_LLM_PROVIDER and not st.session_state.get("custom_llm_api_config"):
        st.warning("请先保存 Custom LLM API 配置。")
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
            custom_llm_api_config=get_llm_provider_config(llm_provider),
            ocr_provider_config=get_ocr_provider_config(ocr_engine),
        )

        progress = st.progress(0, text="准备处理...")
        results: List[ProcessResult] = []

        with st.spinner("正在处理文件，请稍候..."):
            for idx, file in enumerate(uploaded_files, start=1):
                progress.progress((idx - 1) / len(uploaded_files), text=f"处理中：{file.name}")
                result = process_files(file, config)
                results.append(result)
                progress.progress(idx / len(uploaded_files), text=f"已完成：{file.name}")

        st.session_state["results"] = results
        st.session_state["config"] = config
        progress.empty()

        error_count = sum(1 for result in results if result.status == "error")
        if error_count:
            st.error(f"处理完成，但有 {error_count} 个文件失败。请在下方结果页查看错误详情。")
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
                st.markdown(f"**模式**：{result.mode}")
                st.markdown(f"**OCR 引擎**：{result.ocr_engine_used or '-'}")
                st.markdown(f"**LLM**：{result.llm_provider_used or '-'} / {result.llm_model_used or '-'}")
            with col2:
                st.json(result.meta)

            if result.error:
                st.error(result.error)

            raw_label = "**原始 OCR 文本**" if result.ocr_engine_used else "**输入文本**"
            st.markdown(raw_label)
            st.text_area(
                f"raw_{result.file_name}",
                value=result.raw_text,
                height=180,
                label_visibility="collapsed",
                key=f"raw_{result.file_name}",
            )

            if result.mode == MODE_OCR_FORMAT:
                st.download_button(
                    label=f"下载 {result.file_name} 原始 OCR",
                    data=result.raw_export_bytes(),
                    file_name=result.raw_export_name(),
                    mime="text/plain",
                    key=f"download_raw_{result.file_name}",
                )

            final_label = "**LLM 整理后的文本**" if result.mode == MODE_OCR_FORMAT else "**最终输出**"
            st.markdown(final_label)
            st.text_area(
                f"final_{result.file_name}",
                value=result.final_output,
                height=240,
                label_visibility="collapsed",
                key=f"final_{result.file_name}",
            )

            if result.structured_data:
                st.markdown("**结构化输出**")
                st.json(result.structured_data)

            if result.mode == MODE_OCR_FORMAT:
                st.download_button(
                    label=f"下载 {result.file_name} 整理结果",
                    data=result.formatted_export_bytes(),
                    file_name=result.formatted_export_name(),
                    mime=result.export_mime(),
                    key=f"download_formatted_{result.file_name}",
                )
            else:
                st.download_button(
                    label=f"下载 {result.file_name} 结果",
                    data=result.export_bytes(),
                    file_name=result.export_name(),
                    mime=result.export_mime(),
                    key=f"download_{result.file_name}",
                )

    zip_buffer = io.BytesIO()
    with zipfile.ZipFile(zip_buffer, mode="w", compression=zipfile.ZIP_DEFLATED) as zf:
        manifest = []
        for result in results:
            if result.mode == MODE_OCR_FORMAT:
                zf.writestr(result.raw_export_name(), result.raw_export_bytes())
                zf.writestr(result.formatted_export_name(), result.formatted_export_bytes())
            else:
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
        key="download_zip_button",
    )
