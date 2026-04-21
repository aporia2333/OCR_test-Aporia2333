from __future__ import annotations

import os
from typing import Dict, List


MODEL_REGISTRY: Dict[str, Dict] = {
    "Mock LLM": {
        "env_key": None,
        "base_url_env": None,
        "models": {
            "mock-default": {
                "supports_vision": False,
                "supports_json": True,
                "supports_long_context": True,
            }
        },
    },
    "OpenAI": {
        "env_key": "OPENAI_API_KEY",
        "base_url_env": "OPENAI_BASE_URL",
        "models": {
            "gpt-4.1-mini": {
                "supports_vision": True,
                "supports_json": True,
                "supports_long_context": True,
            },
            "gpt-4.1": {
                "supports_vision": True,
                "supports_json": True,
                "supports_long_context": True,
            },
        },
    },
    "DeepSeek": {
        "env_key": "DEEPSEEK_API_KEY",
        "base_url_env": "DEEPSEEK_BASE_URL",
        "default_base_url": "https://api.deepseek.com",
        "models": {
            "deepseek-chat": {
                "supports_vision": False,
                "supports_json": True,
                "supports_long_context": True,
            },
            "deepseek-reasoner": {
                "supports_vision": False,
                "supports_json": True,
                "supports_long_context": True,
            },
        },
    },
    "Custom LLM API": {
        "env_key": None,
        "base_url_env": None,
        "session_config": "custom_llm_api_config",
        "models": {
            "custom-session-model": {
                "supports_vision": False,
                "supports_json": True,
                "supports_long_context": True,
            }
        },
    },
    "GLM": {
        "env_key": "GLM_API_KEY",
        "base_url_env": "GLM_BASE_URL",
        "default_base_url": "https://open.bigmodel.cn/api/paas/v4",
        "models": {
            "glm-4-flash": {
                "supports_vision": False,
                "supports_json": True,
                "supports_long_context": True,
            },
            "glm-4-plus": {
                "supports_vision": False,
                "supports_json": True,
                "supports_long_context": True,
            },
        },
    },
}


def provider_connected(provider: str, session_config: dict | None = None) -> bool:
    if provider == "Custom LLM API":
        return True
    if provider == "DeepSeek" and session_config:
        return bool(session_config.get("api_key"))
    env_key = MODEL_REGISTRY.get(provider, {}).get("env_key")
    if not env_key:
        return provider == "Mock LLM"
    return bool(os.getenv(env_key))



def get_llm_provider_status(session_configs: dict | None = None) -> Dict[str, bool]:
    session_configs = session_configs or {}
    return {
        provider: provider_connected(provider, session_configs.get(provider))
        for provider in MODEL_REGISTRY.keys()
    }



def get_models_for_provider(provider: str) -> List[str]:
    return list(MODEL_REGISTRY.get(provider, {}).get("models", {}).keys())



def get_capabilities(provider: str, model: str) -> Dict:
    return MODEL_REGISTRY.get(provider, {}).get("models", {}).get(model, {})
