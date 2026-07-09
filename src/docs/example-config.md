# 配置示例

一份最小的 `~/.tokencode/config.toml` 示例：

```toml
model_provider = "my-provider"
model = "gpt-4o"

[model_providers.my-provider]
name = "My Provider"
base_url = "https://api.example.com/v1"
wire_api = "chat"
env_key = "MY_API_KEY"
```

然后在环境中提供 Key：

```shell
export MY_API_KEY=sk-...
```

完整字段以 `config.schema.json` 为准，概述见 [配置](./config.md)。
