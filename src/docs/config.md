# 配置

TokenCode 的配置文件位于 `~/.tokencode/config.toml`，支持项目级 `.tokencode/config.toml` 覆盖。常用项包括 `model`、`model_provider`、`model_providers`、审批策略、沙箱与日志等。

完整字段以 `config.schema.json` 为准；一份最小示例见 [配置示例](./example-config.md)。

## 生命周期钩子（Lifecycle hooks）

管理员可以在 `requirements.toml` 顶层设置 `allow_managed_hooks_only = true`，以忽略用户、项目与会话级的 hook 配置，同时仍允许来自 requirements 与托管配置层（managed config layers）的托管钩子。

该设置仅在 `requirements.toml` 中生效；写在 `config.toml` 中不会启用「仅托管钩子」模式。
