# 鉴权

TokenCode 通过模型提供商的鉴权方式访问大模型，凭据配置在 `~/.tokencode/config.toml` 与环境变量中。

- **API Key**：在 `model_providers` 中为每个提供商设置 `env_key`，并在对应环境变量中提供 API Key。
- **多协议**：不同 `wire_api`（OpenAI Responses / Chat Completions、Anthropic、Gemini、Ollama 等）按各自规范鉴权。

TokenCode 不绑定任何 SaaS 账号；只需配置所选模型提供商的凭据即可。

配置示例见 [配置示例](./example-config.md)。
