# tools/ — 开发脚本

TokenCode 的开发脚本目录：构建、测试、启动等流程入口。

> 📌 这是**对外开发目录**（会随仓库上传），与项目根的「常用工具/」（🔒 维护者隐私自用脚本）严格区分：
> - `tools/` —— 项目开发流程脚本，对外公开。
> - `常用工具/` —— 个人临时脚本、密钥处理、本地实验，永不上传。

## 当前状态（2026-07-06 基线切换后）

TokenCode 已切换到 **Codex（Rust）基线**。源码位于 `../src/`（Rust workspace `codex-rs/` + Node CLI `codex-cli/`）。**构建 / 测试 / 运行工具链由上游提供**，直接在 `src/` 下使用：

| 操作 | 命令（在 `src/` 下执行） |
|---|---|
| 构建（Rust） | `cargo build`（或 `just build` / `bazel build //...`） |
| 运行 CLI | `cargo run` / 上游 `just` 目标 |
| 测试 | `cargo test` |
| 任务编排 | `just <task>`（见 `src/justfile`） |
| Nix 开发环境 | `nix develop`（见 `src/flake.nix`） |

> 上游同时提供 `cargo`、`just`、`bazel` 三套构建入口，详见 `src/justfile` / `src/MODULE.bazel` / `src/codex-rs/Cargo.toml`。

## 本目录规划

旧的 OpenCode（bun + turbo）专用脚本（`install.sh` / `dev.sh` / `build.sh` 等）已随基线切换移除。后续会在此目录补充 **TokenCode 专属**的便捷脚本（如本地启动、冒烟测试、打包），封装 `src/` 下的上游命令。补充时同步更新本表。
