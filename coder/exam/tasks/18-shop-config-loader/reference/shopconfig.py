"""Load the shop server's settings: defaults < JSON file < SHOP_* environment variables."""

from __future__ import annotations

import json
import os
from collections.abc import Mapping
from dataclasses import dataclass, field
from typing import Any

ENV_PREFIX = "SHOP_"
_TRUE = {"1", "true", "yes", "on"}
_FALSE = {"0", "false", "no", "off"}


class ConfigError(Exception):
    """The configuration is missing something or has an invalid value."""


@dataclass(frozen=True)
class Config:
    name: str
    port: int = 8080
    debug: bool = False
    low_stock_threshold: int = 5
    sms_api_key: str | None = field(default=None, repr=False)

    def __repr__(self) -> str:
        key = "None" if self.sms_api_key is None else "'***'"
        return (
            f"Config(name={self.name!r}, port={self.port!r}, debug={self.debug!r}, "
            f"low_stock_threshold={self.low_stock_threshold!r}, sms_api_key={key})"
        )

    __str__ = __repr__


_DEFAULTS: dict[str, Any] = {
    "port": 8080,
    "debug": False,
    "low_stock_threshold": 5,
    "sms_api_key": None,
}
_FIELDS = ("name", "port", "debug", "low_stock_threshold", "sms_api_key")


def _read_file(path: str | os.PathLike) -> dict[str, Any]:
    try:
        with open(path, encoding="utf-8") as fh:
            data = json.load(fh)
    except FileNotFoundError:
        raise ConfigError(f"config file not found: {os.fspath(path)}") from None
    except (OSError, UnicodeDecodeError) as exc:
        raise ConfigError(f"cannot read config file {os.fspath(path)}: {exc}") from None
    except json.JSONDecodeError as exc:
        raise ConfigError(f"config file {os.fspath(path)} is not valid JSON: {exc}") from None
    if not isinstance(data, dict):
        raise ConfigError(f"config file {os.fspath(path)} must contain a JSON object")
    unknown = sorted(set(data) - set(_FIELDS))
    if unknown:
        raise ConfigError(f"unknown setting(s) in {os.fspath(path)}: {', '.join(unknown)}")
    return data


def _env_name(key: str) -> str:
    return ENV_PREFIX + key.upper()


def _as_int(key: str, value: Any, source: str) -> int:
    if isinstance(value, bool):
        raise ConfigError(f"{key} from {source} must be a whole number, got {value!r}")
    if isinstance(value, int):
        return value
    if isinstance(value, str):
        try:
            return int(value.strip())
        except ValueError:
            pass
    raise ConfigError(f"{key} from {source} must be a whole number, got {value!r}")


def _as_bool(key: str, value: Any, source: str) -> bool:
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        text = value.strip().lower()
        if text in _TRUE:
            return True
        if text in _FALSE:
            return False
    raise ConfigError(f"{key} from {source} must be true or false, got {value!r}")


def load_config(path: str | os.PathLike | None = None, env: Mapping[str, str] | None = None) -> Config:
    if env is None:
        env = os.environ
    raw: dict[str, tuple[Any, str]] = {k: (v, "defaults") for k, v in _DEFAULTS.items()}
    if path is not None:
        source = f"file {os.fspath(path)}"
        for key, value in _read_file(path).items():
            raw[key] = (value, source)
    for key in _FIELDS:
        name = _env_name(key)
        if name in env:
            raw[key] = (env[name], f"environment variable {name}")

    if "name" not in raw:
        raise ConfigError(f"name is required: set it in the config file or {_env_name('name')}")
    name, source = raw["name"]
    if not isinstance(name, str) or not name.strip():
        raise ConfigError(f"name from {source} must be a non-empty string")

    port = _as_int("port", *raw["port"])
    if not 1 <= port <= 65535:
        raise ConfigError(f"port from {raw['port'][1]} must be between 1 and 65535, got {port}")

    threshold = _as_int("low_stock_threshold", *raw["low_stock_threshold"])
    if threshold < 0:
        raise ConfigError(f"low_stock_threshold from {raw['low_stock_threshold'][1]} must not be negative")

    debug = _as_bool("debug", *raw["debug"])

    key, source = raw["sms_api_key"]
    if key is not None and not isinstance(key, str):
        raise ConfigError(f"sms_api_key from {source} must be a string")
    if isinstance(key, str) and not key.strip():
        key = None

    return Config(name=name.strip(), port=port, debug=debug, low_stock_threshold=threshold, sms_api_key=key)
