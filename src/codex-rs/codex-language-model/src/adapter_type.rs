use schemars::JsonSchema;
use serde::{Deserialize, Serialize};

/// 模型后端「说话方式」的分类键，驱动 LanguageModel 实现的选择。
///
/// P0 仅 `OpenaiResponses` 一个变体生效；其余变体为预留位，待 P1/P2 各协议
/// adapter 落地后启用。未显式配置时，由 ModelProviderInfo 依据 wire_api 推导
/// （Responses → OpenaiResponses），保证老配置向后兼容。
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq, Serialize, Deserialize, JsonSchema)]
#[serde(rename_all = "snake_case")]
pub enum AdapterType {
    /// OpenAI Responses API（POST /responses）。默认值，与既有 Responses 客户端行为一致。
    #[default]
    OpenaiResponses,
    // ---- 以下为 P1+ 预留，P0 不实例化 ----
    /// OpenAI Chat Completions API（POST /v1/chat/completions）。
    OpenaiChat,
    /// OpenAI 兼容端点（vLLM / new-api / OpenRouter 等网关）。
    OpenaiCompatible,
    /// Anthropic Messages API（POST /v1/messages）。
    Anthropic,
    /// Google Gemini API（POST generateContent）。
    Gemini,
    /// Ollama 原生 API（POST /api/chat）。
    Ollama,
}
