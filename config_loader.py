import json
import sys
from pathlib import Path
from typing import Any, Dict, Optional

CONFIG_PATH = Path(__file__).resolve().parent / "bitget_credentials.json"


def _load_config(path: Path = CONFIG_PATH) -> Dict[str, Any]:
    try:
        with path.open("r", encoding="utf-8") as fp:
            return json.load(fp)
    except FileNotFoundError as exc:
        raise RuntimeError(f"Config file not found: {path}") from exc
    except json.JSONDecodeError as exc:
        raise RuntimeError(f"Invalid JSON in config file: {path}") from exc


def _platform_value(section: str, platform: Optional[str] = None, config: Optional[Dict[str, Any]] = None) -> Any:
    platform_key = platform or sys.platform
    cfg = config or _load_config()
    section_data = cfg.get(section, {})
    if isinstance(section_data, dict):
        platform_names = {"win32", "linux", "darwin", "default"}
        if any(k in platform_names for k in section_data):
            return section_data.get(platform_key) or section_data.get("default")
    return section_data


def load_api_keys(platform: Optional[str] = None, config: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    apis = _platform_value("apis", platform, config)
    return apis if isinstance(apis, dict) else {}


def get_webhook_url(platform: Optional[str] = None, config: Optional[Dict[str, Any]] = None) -> str:
    webhook = _platform_value("webhooks", platform, config)
    if isinstance(webhook, dict):
        for value in webhook.values():
            if isinstance(value, str):
                return value
        return ""
    return str(webhook) if webhook else ""
