import json

import pytest

from shopconfig import ConfigError, load_config

SECRET = "sk_live_9f8e7d6c5b4a"


def write_json(tmp_path, data, name="shop.json"):
    p = tmp_path / name
    p.write_text(json.dumps(data, ensure_ascii=False), encoding="utf-8")
    return p


def test_defaults_with_only_the_name():
    cfg = load_config(env={"SHOP_NAME": "Sharma Kirana"})
    assert cfg.name == "Sharma Kirana"
    assert cfg.port == 8080
    assert cfg.debug is False
    assert cfg.low_stock_threshold == 5
    assert cfg.sms_api_key is None


def test_file_overrides_defaults(tmp_path):
    p = write_json(tmp_path, {"name": "शर्मा किराना", "port": 9000, "debug": True, "low_stock_threshold": 2})
    cfg = load_config(p, env={})
    assert cfg.name == "शर्मा किराना"
    assert (cfg.port, cfg.debug, cfg.low_stock_threshold) == (9000, True, 2)


def test_env_overrides_file(tmp_path):
    p = write_json(tmp_path, {"name": "File Name", "port": 9000, "debug": True})
    cfg = load_config(str(p), env={"SHOP_PORT": "9100", "SHOP_DEBUG": "false", "SHOP_SMS_API_KEY": SECRET})
    assert cfg.name == "File Name"
    assert cfg.port == 9100
    assert cfg.debug is False
    assert cfg.sms_api_key == SECRET


@pytest.mark.parametrize("text,expected", [("true", True), ("TRUE", True), ("1", True), ("false", False),
                                           ("False", False), ("0", False)])
def test_env_booleans(text, expected):
    cfg = load_config(env={"SHOP_NAME": "x", "SHOP_DEBUG": text})
    assert cfg.debug is expected


def test_uses_os_environ_by_default(monkeypatch):
    monkeypatch.setenv("SHOP_NAME", "Env Shop")
    monkeypatch.setenv("SHOP_LOW_STOCK_THRESHOLD", "12")
    cfg = load_config()
    assert cfg.name == "Env Shop"
    assert cfg.low_stock_threshold == 12


def test_does_not_modify_given_env():
    env = {"SHOP_NAME": "x", "SHOP_PORT": "9001"}
    before = dict(env)
    load_config(env=env)
    assert env == before


@pytest.mark.parametrize(
    "env",
    [
        {},
        {"SHOP_NAME": ""},
        {"SHOP_NAME": "   "},
        {"SHOP_NAME": "x", "SHOP_PORT": "abc"},
        {"SHOP_NAME": "x", "SHOP_PORT": "0"},
        {"SHOP_NAME": "x", "SHOP_PORT": "70000"},
        {"SHOP_NAME": "x", "SHOP_PORT": "-5"},
        {"SHOP_NAME": "x", "SHOP_PORT": "80.5"},
        {"SHOP_NAME": "x", "SHOP_DEBUG": "maybe"},
        {"SHOP_NAME": "x", "SHOP_LOW_STOCK_THRESHOLD": "-1"},
        {"SHOP_NAME": "x", "SHOP_LOW_STOCK_THRESHOLD": "few"},
    ],
)
def test_bad_env_values_raise_config_error(env):
    with pytest.raises(ConfigError):
        load_config(env=env)


@pytest.mark.parametrize(
    "data",
    [
        {"port": 9000},
        {"name": "x", "port": True},
        {"name": "x", "port": 0},
        {"name": "x", "port": "lots"},
        {"name": "x", "debug": "sometimes"},
        {"name": "x", "low_stock_threshold": 2.5},
        {"name": 123},
    ],
)
def test_bad_file_values_raise_config_error(tmp_path, data):
    with pytest.raises(ConfigError):
        load_config(write_json(tmp_path, data), env={})


def test_missing_file_raises_config_error(tmp_path):
    with pytest.raises(ConfigError):
        load_config(tmp_path / "nope.json", env={"SHOP_NAME": "x"})


@pytest.mark.parametrize("content", ["{not json", "", "[1, 2]", '"shop"'])
def test_broken_file_raises_config_error(tmp_path, content):
    p = tmp_path / "shop.json"
    p.write_text(content, encoding="utf-8")
    with pytest.raises(ConfigError):
        load_config(p, env={"SHOP_NAME": "x"})


def test_config_error_is_an_exception():
    assert issubclass(ConfigError, Exception)


def test_config_is_read_only():
    cfg = load_config(env={"SHOP_NAME": "x"})
    with pytest.raises((AttributeError, TypeError)):
        cfg.port = 1
    assert cfg.port == 8080


def test_secret_is_not_shown_when_printed(tmp_path):
    p = write_json(tmp_path, {"name": "Sharma Kirana", "sms_api_key": SECRET})
    cfg = load_config(p, env={})
    assert cfg.sms_api_key == SECRET
    assert SECRET not in repr(cfg)
    assert SECRET not in str(cfg)
    assert SECRET not in f"{cfg}"
    assert "Sharma Kirana" in repr(cfg)  # the rest is still useful in the log

