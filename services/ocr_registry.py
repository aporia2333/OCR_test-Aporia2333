from __future__ import annotations

import os
from typing import Dict, List


OCR_REGISTRY: Dict[str, Dict] = {
    "Mock OCR": {
        "required_envs": [],
    },
    "PaddleOCR Layout Parsing": {
        "required_envs": ["PADDLEOCR_API_URL", "PADDLEOCR_TOKEN"],
    },
    "Tencent OCR": {
        "required_envs": [],
        "session_config": "tencent_ocr_config",
    },
    "Baidu OCR": {
        "required_envs": [],
        "session_config": "baidu_ocr_config",
    },
    "Google Document AI": {
        "required_envs": ["GOOGLE_API_KEY"],
    },
}



def ocr_engine_connected(engine: str, session_config: dict | None = None) -> bool:
    if engine in {"Tencent OCR", "Baidu OCR"}:
        return True
    if engine == "PaddleOCR Layout Parsing" and session_config:
        return bool(session_config.get("api_url") and session_config.get("token"))
    envs = OCR_REGISTRY.get(engine, {}).get("required_envs", [])
    if not envs:
        return engine == "Mock OCR"
    return all(bool(os.getenv(k)) for k in envs)



def get_ocr_engine_status(session_configs: dict | None = None) -> Dict[str, bool]:
    session_configs = session_configs or {}
    return {
        engine: ocr_engine_connected(engine, session_configs.get(engine))
        for engine in OCR_REGISTRY.keys()
    }



def get_available_ocr_engines(session_configs: dict | None = None) -> List[str]:
    status = get_ocr_engine_status(session_configs)
    return [engine for engine in OCR_REGISTRY.keys() if engine == "Mock OCR" or status.get(engine)]
