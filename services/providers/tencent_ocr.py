from __future__ import annotations

import base64
import json
from typing import Any, Dict


SUPPORTED_EXTENSIONS = {"png", "jpg", "jpeg", "webp"}


def _file_extension(file_name: str) -> str:
    return file_name.lower().rsplit(".", 1)[-1] if "." in file_name else ""


def _validate_config(config: Dict[str, Any]) -> Dict[str, str]:
    if not config:
        raise ValueError("请先保存 Tencent OCR 配置。")

    secret_id = config.get("secret_id", "").strip()
    secret_key = config.get("secret_key", "")
    region = config.get("region", "ap-guangzhou").strip() or "ap-guangzhou"
    mode = config.get("mode", "accurate")

    if not secret_id:
        raise ValueError("Tencent OCR SecretId 不能为空。")
    if not secret_key:
        raise ValueError("Tencent OCR SecretKey 不能为空。")
    if mode not in {"basic", "accurate"}:
        raise ValueError("Tencent OCR 模式必须是 basic 或 accurate。")

    return {
        "secret_id": secret_id,
        "secret_key": secret_key,
        "region": region,
        "mode": mode,
    }


def recognize(file_name: str, content: bytes, config: Dict[str, Any], language_hint: str = "") -> Dict[str, Any]:
    ext = _file_extension(file_name)
    if ext not in SUPPORTED_EXTENSIONS:
        raise ValueError("Tencent OCR 第一版仅支持 png、jpg、jpeg、webp 图片文件。")

    settings = _validate_config(config)

    try:
        from tencentcloud.common import credential
        from tencentcloud.common.exception.tencent_cloud_sdk_exception import TencentCloudSDKException
        from tencentcloud.ocr.v20181119 import models, ocr_client
    except Exception as exc:
        raise RuntimeError("未安装腾讯云 OCR SDK，请先安装 tencentcloud-sdk-python。") from exc

    try:
        from tencentcloud.common.profile.client_profile import ClientProfile
        from tencentcloud.common.profile.http_profile import HttpProfile

        http_profile = HttpProfile()
        http_profile.endpoint = "ocr.tencentcloudapi.com"
        http_profile.reqTimeout = 180
        client_profile = ClientProfile()
        client_profile.httpProfile = http_profile

        cred = credential.Credential(settings["secret_id"], settings["secret_key"])
        client = ocr_client.OcrClient(cred, settings["region"], client_profile)

        image_base64 = base64.b64encode(content).decode("ascii")
        if settings["mode"] == "basic":
            request = models.GeneralBasicOCRRequest()
            api_name = "GeneralBasicOCR"
            request.from_json_string(json.dumps({"ImageBase64": image_base64}))
            response = client.GeneralBasicOCR(request)
        else:
            request = models.GeneralAccurateOCRRequest()
            api_name = "GeneralAccurateOCR"
            request.from_json_string(json.dumps({"ImageBase64": image_base64}))
            response = client.GeneralAccurateOCR(request)
    except TencentCloudSDKException as exc:
        raise RuntimeError(f"Tencent OCR SDK 请求失败：{exc.get_code()} - {exc.get_message()}") from exc
    except Exception as exc:
        raise RuntimeError(f"Tencent OCR 请求失败：{exc}") from exc

    full_response = json.loads(response.to_json_string())
    detections = full_response.get("TextDetections", []) or []
    blocks = []
    lines = []
    for item in detections:
        text = item.get("DetectedText", "")
        if text:
            lines.append(text)
        blocks.append(item)

    raw_text = "\n".join(lines).strip()
    if not raw_text:
        raise RuntimeError("Tencent OCR 未返回可用文本。")

    return {
        "provider": "Tencent OCR",
        "raw_text": raw_text,
        "blocks": blocks,
        "full_response": full_response,
        "meta": {
            "engine": "Tencent OCR",
            "mode": settings["mode"],
            "api": api_name,
            "language": language_hint,
            "source_type": "image",
            "region": settings["region"],
        },
    }
