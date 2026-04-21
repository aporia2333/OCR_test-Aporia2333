from __future__ import annotations

from typing import Any, Dict, List, Tuple


SECTION_KEYS = {"headers", "extra_params"}
ROOT_KEYS = {
    "provider_type",
    "base_url",
    "api_url",
    "api_key",
    "model",
    "disable_proxy",
    "token",
    "secret_id",
    "secret_key",
    "region",
    "mode",
}
PROVIDER_SECTIONS = {
    "llm",
    "custom_llm",
    "openai_compatible",
    "deepseek",
    "paddleocr",
    "paddle_ocr",
    "tencent_ocr",
    "baidu_ocr",
}

KEY_ALIASES = {
    "model_name": "model",
    "paddle_api_url": "api_url",
    "paddleocr_api_url": "api_url",
    "paddle_token": "token",
    "paddleocr_token": "token",
    "secretid": "secret_id",
    "secretkey": "secret_key",
    "api key": "api_key",
        "secret key": "secret_key",
        "no_proxy": "disable_proxy",
    }


def _normalize_key(key: str) -> str:
    normalized = key.lower().strip()
    return KEY_ALIASES.get(normalized, normalized)


def _normalize_provider(value: str) -> str:
    normalized = value.lower().strip()
    if normalized in {"llm", "custom_llm", "openai_compatible"}:
        return "llm"
    if normalized in {"paddle", "paddleocr", "paddle_ocr"}:
        return "paddleocr"
    if normalized == "deepseek":
        return "deepseek"
    if normalized in {"tencent", "tencent_ocr"}:
        return "tencent_ocr"
    if normalized in {"baidu", "baidu_ocr"}:
        return "baidu_ocr"
    return ""


def _split_key_value(line: str) -> Tuple[str, str] | None:
    positions = [
        (line.find(separator), separator)
        for separator in (":", "\uff1a", "=")
        if line.find(separator) >= 0
    ]
    if not positions:
        return None

    _, separator = min(positions, key=lambda item: item[0])
    key, value = line.split(separator, 1)
    key = key.strip()
    value = value.strip()
    if not key:
        return None
    return key, value


def parse_multi_api_txt_config(text: str) -> Dict[str, Any]:
    if not text or not text.strip():
        raise ValueError("Uploaded TXT file is empty.")

    sections = {
        "llm": {
            "source": "uploaded_txt",
            "provider_type": "openai_compatible",
            "base_url": "",
            "api_key": "",
            "model": "",
            "disable_proxy": "true",
            "headers": {},
            "extra_params": {},
        },
        "tencent_ocr": {
            "source": "uploaded_txt",
            "secret_id": "",
            "secret_key": "",
            "region": "ap-guangzhou",
            "mode": "accurate",
        },
        "baidu_ocr": {
            "source": "uploaded_txt",
            "api_key": "",
            "secret_key": "",
            "mode": "accurate",
        },
        "paddleocr": {
            "source": "uploaded_txt",
            "api_url": "",
            "token": "",
        },
        "deepseek": {
            "source": "uploaded_txt",
            "api_key": "",
            "base_url": "https://api.deepseek.com",
            "model": "deepseek-chat",
        },
    }
    touched = set()
    warnings: List[str] = []
    current_provider = "llm"
    current_nested = None
    valid_pairs = 0

    for raw_line in text.splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#"):
            continue

        if line.startswith("[") and line.endswith("]"):
            provider = _normalize_provider(line[1:-1])
            if provider:
                current_provider = provider
            else:
                warnings.append(f"Ignored unknown config section: {line}")
                current_provider = ""
            current_nested = None
            continue

        parsed = _split_key_value(line)
        if not parsed:
            warnings.append(f"Ignored line that is not key-value: {line}")
            continue

        key, value = parsed
        normalized_key = _normalize_key(key)

        if normalized_key in {"provider", "provider_type"}:
            provider = _normalize_provider(value)
            if provider:
                current_provider = provider
                current_nested = None
                touched.add(current_provider)
                if "provider_type" in sections[current_provider]:
                    sections[current_provider]["provider_type"] = value.strip()
                valid_pairs += 1
                continue

        if not current_provider:
            warnings.append(f"Ignored line without a known provider section: {line}")
            continue

        if current_provider == "llm" and normalized_key in SECTION_KEYS and not value:
            current_nested = normalized_key
            touched.add(current_provider)
            valid_pairs += 1
            continue

        if current_provider == "llm" and current_nested in SECTION_KEYS and normalized_key not in ROOT_KEYS:
            if value:
                sections[current_provider][current_nested][key.strip()] = value
                touched.add(current_provider)
                valid_pairs += 1
            else:
                warnings.append(f"Ignored empty {current_nested} value: {line}")
            continue

        current_nested = None
        target = sections[current_provider]
        if normalized_key in target:
            target[normalized_key] = value
            touched.add(current_provider)
            valid_pairs += 1
        elif current_provider == "llm" and normalized_key in SECTION_KEYS:
            current_nested = normalized_key
            touched.add(current_provider)
            valid_pairs += 1
            if value:
                warnings.append(f"Ignored value after {key}; use key=value on following lines.")
        else:
            warnings.append(f"Ignored unknown field for {current_provider}: {key}")

    if valid_pairs == 0:
        raise ValueError("TXT content does not contain valid key-value configuration.")

    configs = {}
    errors = []

    if "llm" in touched:
        missing = [key for key in ("base_url", "api_key", "model") if not sections["llm"].get(key)]
        if missing:
            errors.append("LLM config is incomplete. Please provide base_url, api_key, and model.")
        else:
            configs["llm"] = sections["llm"]

    if "tencent_ocr" in touched:
        missing = [key for key in ("secret_id", "secret_key") if not sections["tencent_ocr"].get(key)]
        if missing:
            errors.append("Tencent OCR config is incomplete. Please provide secret_id and secret_key.")
        else:
            configs["tencent_ocr"] = sections["tencent_ocr"]

    if "baidu_ocr" in touched:
        missing = [key for key in ("api_key", "secret_key") if not sections["baidu_ocr"].get(key)]
        if missing:
            errors.append("Baidu OCR config is incomplete. Please provide api_key and secret_key.")
        else:
            configs["baidu_ocr"] = sections["baidu_ocr"]

    if "paddleocr" in touched:
        missing = [key for key in ("api_url", "token") if not sections["paddleocr"].get(key)]
        if missing:
            errors.append("PaddleOCR config is incomplete. Please provide api_url and token.")
        else:
            configs["paddleocr"] = sections["paddleocr"]

    if "deepseek" in touched:
        missing = [key for key in ("api_key",) if not sections["deepseek"].get(key)]
        if missing:
            errors.append("DeepSeek config is incomplete. Please provide api_key.")
        else:
            configs["deepseek"] = sections["deepseek"]

    if errors and not configs:
        raise ValueError(" ".join(errors))

    return {
        "source": "uploaded_txt",
        "configs": configs,
        "warnings": warnings + errors,
    }


def key_values_to_text(values: Dict[str, Any]) -> str:
    if not values:
        return ""
    return "\n".join(f"{key}={value}" for key, value in values.items())


def parse_key_value_text(text: str, label: str) -> Tuple[Dict[str, str], List[str]]:
    parsed: Dict[str, str] = {}
    warnings: List[str] = []
    if not text or not text.strip():
        return parsed, warnings

    for raw_line in text.splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#"):
            continue

        item = _split_key_value(line)
        if not item:
            warnings.append(f"{label}: ignored line that is not key-value: {line}")
            continue

        key, value = item
        if not value:
            warnings.append(f"{label}: ignored empty value for {key}")
            continue
        parsed[key] = value

    return parsed, warnings
