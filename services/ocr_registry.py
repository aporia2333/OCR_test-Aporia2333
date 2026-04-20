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
    "Google Document AI": {
        "required_envs": ["GOOGLE_API_KEY"],
    },
}



def ocr_engine_connected(engine: str) -> bool:
    envs = OCR_REGISTRY.get(engine, {}).get("required_envs", [])
    if not envs:
        return engine == "Mock OCR"
    return all(bool(os.getenv(k)) for k in envs)



def get_ocr_engine_status() -> Dict[str, bool]:
    return {engine: ocr_engine_connected(engine) for engine in OCR_REGISTRY.keys()}



def get_available_ocr_engines() -> List[str]:
    status = get_ocr_engine_status()
    return [engine for engine in OCR_REGISTRY.keys() if engine == "Mock OCR" or status.get(engine)]
