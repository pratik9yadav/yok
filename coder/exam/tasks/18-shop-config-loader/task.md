Our shop server reads its settings from a JSON file and from environment variables, and I want one place that does it.
Please write `shopconfig.py` with `load_config(path=None, env=None) -> Config`: `path` is the optional JSON file and `env` is a mapping of environment variables that defaults to `os.environ`.
Settings are `name` (str, required), `port` (int, default 8080), `debug` (bool, default False), `low_stock_threshold` (int, default 5) and `sms_api_key` (str or None, default None).
The environment variables are `SHOP_` plus the upper-case setting name (`SHOP_PORT=9000`, `SHOP_DEBUG=true`, ...) and they win over the file, which wins over the defaults.
If the configuration is wrong in any way, raise `ConfigError` (define it in the module) with a message that tells whoever is deploying what to fix; last month a server started on port 0 because of a typo and nobody noticed.
The returned `Config` should be read-only, and keep in mind `sms_api_key` is a secret while our startup log prints the config.
