use schemars::JsonSchema;
use serde::{Deserialize, Serialize};

/// 模型后端「说话方式」的分类键，驱动 LanguageModel 实现的选择。
///
/// ⛔ 死规则（协议范围）：本枚举只保留 OpenAI / Anthropic / Gemini 三大协议对应的变体。
/// Ollama 原生协议、独立「OpenAI 兼容网关」adapter 一律不设变体——所有 OpenAI 兼容端点
/// （vLLM / new-api / OpenRouter / DeepSeek / Kimi / 通义等）统一走 `OpenaiChat` + 自定义
/// `base_url` 接入，不计为独立协议。新增第 4 个协议 adapter 必须先推翻
/// `设计文档/协议适配范围决策.md` 的死规则（默认答案：不加）。
///
/// P0 仅 `OpenaiResponses` 一个变体生效；其余变体待 P1/P2 adapter 落地后启用。
/// 未显式配置时，由 ModelProviderInfo 依据 wire_api 推导（Responses → OpenaiResponses），
/// 保证老配置向后兼容。
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq, Serialize, Deserialize, JsonSchema)]
#[serde(rename_all = "snake_case")]
pub enum AdapterType {
    /// OpenAI Responses API（POST /responses）。默认值，与既有 Responses 客户端行为一致。
    #[default]
    OpenaiResponses,
    // ---- 以下为 P1+ 预留，P0 不实例化 ----
    /// OpenAI Chat Completions API（POST /v1/chat/completions）。
    /// 同时覆盖所有 OpenAI 兼容端点（vLLM / new-api / OpenRouter / DeepSeek / Kimi / 通义等，配 base_url）。
    OpenaiChat,
    /// Anthropic Messages API（POST /v1/messages）。
    Anthropic,
    /// Google Gemini API（POST generateContent）。
    Gemini,
}
