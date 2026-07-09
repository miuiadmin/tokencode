# TokenCode

<p align="center"><strong>TokenCode</strong> 是一个开源的超级代码智能体（AI Coding Agent），在终端本地运行。</p>

---

## 快速开始

### 安装

通过 GitHub Releases 安装：前往 [最新发布](https://github.com/miuiadmin/tokencode/releases/latest)，按你的平台下载对应归档，解压后将 `tokencode` 二进制放入 `PATH`，然后运行：

```shell
tokencode
```

从源码构建见 [安装与构建](./docs/install.md)。

### 配置模型

TokenCode 原生支持多协议模型接入（OpenAI Responses / Chat Completions、Anthropic、Gemini、Ollama 及兼容网关）。在 `~/.tokencode/config.toml` 中配置 `model_providers` 与所选 `model_provider`，并为其设置 `env_key`（API Key 环境变量）即可开始使用。

详见 [配置](./docs/config.md) 与 [鉴权](./docs/authentication.md)。

## 文档

- [配置](./docs/config.md)
- [鉴权](./docs/authentication.md)
- [沙箱与审批](./docs/sandbox.md)
- [安装与构建](./docs/install.md)
- [贡献指南](./docs/contributing.md)
- [安全策略](./SECURITY.md)

更多文档见仓库根目录 [docs/](../docs/)。

## 协议

TokenCode 以 [GPL-3.0](../LICENSE) 发布。
