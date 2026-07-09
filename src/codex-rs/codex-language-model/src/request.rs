use codex_protocol::config_types::ReasoningSummary as ReasoningSummaryConfig;
use codex_protocol::models::ResponseItem;
use codex_protocol::openai_models::ReasoningEffort as ReasoningEffortConfig;
use codex_protocol::protocol::SessionSource;
use http::HeaderMap;
use serde::Serialize;
use std::collections::HashMap;
use std::sync::Arc;
use std::sync::OnceLock;

/// 中立请求 schema。
///
/// P0 字段与 codex_api::ResponsesApiRequest 1:1 对齐（语义等价、类型独立），由各
/// adapter 负责向自家 wire body 转换。本结构不参与 HTTP 序列化——序列化由具体协议
/// 的请求类型（如 ResponsesApiRequest）承担，以保证 wire 字节一致。
#[derive(Debug, Clone)]
pub struct UnifiedRequest {
    pub model: String,
    pub instructions: String,
    pub input: Vec<ResponseItem>,
    pub tools: Option<Vec<serde_json::Value>>,
    pub tool_choice: String,
    pub parallel_tool_calls: bool,
    pub reasoning: Option<UnifiedReasoning>,
    pub store: bool,
    pub stream: bool,
    pub include: Vec<String>,
    pub service_tier: Option<String>,
    pub prompt_cache_key: Option<String>,
    pub text: Option<UnifiedTextControls>,
    pub client_metadata: Option<HashMap<String, String>>,
}

/// 中立 reasoning 配置。叶子类型复用 protocol（ReasoningEffort / ReasoningSummary）。
#[derive(Debug, Clone)]
pub struct UnifiedReasoning {
    pub effort: Option<ReasoningEffortConfig>,
    pub summary: Option<ReasoningSummaryConfig>,
    pub context: Option<UnifiedReasoningContext>,
}

/// reasoning 上下文范围（语义对齐 ReasoningContext）。
#[derive(Debug, Clone, Serialize, PartialEq, Eq)]
#[serde(rename_all = "snake_case")]
pub enum UnifiedReasoningContext {
    Auto,
    CurrentTurn,
    AllTurns,
}

/// 文本输出控制（verbosity + 可选 JSON schema 约束）。
#[derive(Debug, Clone, Default)]
pub struct UnifiedTextControls {
    pub verbosity: Option<UnifiedOpenAiVerbosity>,
    pub format: Option<UnifiedTextFormat>,
}

/// 输出详细度档位。
#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub enum UnifiedOpenAiVerbosity {
    Low,
    #[default]
    Medium,
    High,
}

/// 文本输出格式约束（JSON schema）。
#[derive(Debug, Clone, Default)]
pub struct UnifiedTextFormat {
    pub r#type: UnifiedTextFormatType,
    pub strict: bool,
    pub schema: serde_json::Value,
    pub name: String,
}

/// 文本格式类型。P0 仅 JSON schema。
#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub enum UnifiedTextFormatType {
    #[default]
    JsonSchema,
}

/// 请求体压缩方式。
#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub enum UnifiedCompression {
    #[default]
    None,
    Zstd,
}

/// 流式请求的传输层选项（会话标识、自定义头、压缩、turn 状态）。
///
/// 对齐 codex_api::ResponsesOptions 的语义子集。session_source 直接复用 protocol 的
/// SessionSource 类型（协议无关），避免 String 往返解析带来的行为偏差。
#[derive(Debug, Clone, Default)]
pub struct UnifiedRequestOptions {
    pub session_id: Option<String>,
    pub thread_id: Option<String>,
    pub session_source: Option<SessionSource>,
    pub extra_headers: HeaderMap,
    pub compression: UnifiedCompression,
    pub turn_state: Option<Arc<OnceLock<String>>>,
}
