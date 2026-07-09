use crate::error::ApiError;
use codex_protocol::config_types::ReasoningSummary as ReasoningSummaryConfig;
use codex_protocol::config_types::Verbosity as VerbosityConfig;
use codex_protocol::models::ResponseItem;
use codex_protocol::openai_models::ReasoningEffort as ReasoningEffortConfig;
use codex_protocol::protocol::ModelVerification;
use codex_protocol::protocol::RateLimitSnapshot;
use codex_protocol::protocol::TokenUsage;
use codex_protocol::protocol::TurnModerationMetadataEvent;
use codex_protocol::protocol::W3cTraceContext;
use futures::Stream;
use serde::Deserialize;
use serde::Serialize;
use serde_json::Value;
use std::collections::HashMap;
use std::pin::Pin;
use std::task::Context;
use std::task::Poll;
use tokio::sync::mpsc;

pub const WS_REQUEST_HEADER_TRACEPARENT_CLIENT_METADATA_KEY: &str = "ws_request_header_traceparent";
pub const WS_REQUEST_HEADER_TRACESTATE_CLIENT_METADATA_KEY: &str = "ws_request_header_tracestate";

/// Canonical input payload for the compaction endpoint.
#[derive(Debug, Clone, Serialize)]
pub struct CompactionInput<'a> {
    pub model: &'a str,
    pub input: &'a [ResponseItem],
    #[serde(skip_serializing_if = "str::is_empty")]
    pub instructions: &'a str,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub tools: Option<Vec<Value>>,
    pub parallel_tool_calls: bool,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub reasoning: Option<Reasoning>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub service_tier: Option<&'a str>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub prompt_cache_key: Option<&'a str>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub text: Option<TextControls>,
}

/// Canonical input payload for the memory summarize endpoint.
#[derive(Debug, Clone, Serialize)]
pub struct MemorySummarizeInput {
    pub model: String,
    #[serde(rename = "traces")]
    pub raw_memories: Vec<RawMemory>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub reasoning: Option<Reasoning>,
}

#[derive(Debug, Clone, Serialize)]
pub struct RawMemory {
    pub id: String,
    pub metadata: RawMemoryMetadata,
    pub items: Vec<Value>,
}

#[derive(Debug, Clone, Serialize)]
pub struct RawMemoryMetadata {
    pub source_path: String,
}

#[derive(Debug, Clone, Deserialize, PartialEq, Eq)]
pub struct MemorySummarizeOutput {
    #[serde(rename = "trace_summary", alias = "raw_memory")]
    pub raw_memory: String,
    pub memory_summary: String,
}

#[derive(Debug)]
pub enum ResponseEvent {
    Created,
    SafetyBuffering(SafetyBuffering),
    OutputItemDone(ResponseItem),
    OutputItemAdded(ResponseItem),
    /// Emitted when the server includes `OpenAI-Model` on the stream response.
    /// This can differ from the requested model when backend safety routing applies.
    ServerModel(String),
    /// Emitted when the server recommends additional account verification.
    ModelVerifications(Vec<ModelVerification>),
    /// Emitted when the server includes moderation metadata for first-party turn presentation.
    TurnModerationMetadata(TurnModerationMetadataEvent),
    /// Emitted when `X-Reasoning-Included: true` is present on the response,
    /// meaning the server already accounted for past reasoning tokens and the
    /// client should not re-estimate them.
    ServerReasoningIncluded(bool),
    Completed {
        response_id: String,
        token_usage: Option<TokenUsage>,
        /// Did the model affirmatively end its turn? Some providers do not set this,
        /// so we rely on fallback logic when this is `None`.
        end_turn: Option<bool>,
    },
    OutputTextDelta(String),
    ToolCallInputDelta {
        item_id: String,
        call_id: Option<String>,
        delta: String,
    },
    ReasoningSummaryDelta {
        delta: String,
        summary_index: i64,
    },
    ReasoningContentDelta {
        delta: String,
        content_index: i64,
    },
    ReasoningSummaryPartAdded {
        summary_index: i64,
    },
    RateLimits(RateLimitSnapshot),
    ModelsEtag(String),
}

#[derive(Debug, Clone, Deserialize, PartialEq, Eq)]
pub struct SafetyBuffering {
    pub use_cases: Vec<String>,
    pub reasons: Vec<String>,
    #[serde(skip)]
    pub show_buffering_ui: bool,
    pub faster_model: Option<String>,
}

#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub(crate) struct SafetyBufferingTreatment {
    pub faster_model: Option<String>,
}

#[derive(Debug, Serialize, Clone, PartialEq)]
#[serde(rename_all = "snake_case")]
pub enum ReasoningContext {
    Auto,
    CurrentTurn,
    AllTurns,
}

#[derive(Debug, Serialize, Clone, PartialEq)]
pub struct Reasoning {
    #[serde(skip_serializing_if = "Option::is_none")]
    pub effort: Option<ReasoningEffortConfig>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub summary: Option<ReasoningSummaryConfig>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub context: Option<ReasoningContext>,
}

#[derive(Debug, Serialize, Default, Clone, PartialEq)]
#[serde(rename_all = "snake_case")]
pub enum TextFormatType {
    #[default]
    JsonSchema,
}

#[derive(Debug, Serialize, Default, Clone, PartialEq)]
pub struct TextFormat {
    /// Format type used by the OpenAI text controls.
    pub r#type: TextFormatType,
    /// When true, the server is expected to strictly validate responses.
    pub strict: bool,
    /// JSON schema for the desired output.
    pub schema: Value,
    /// Friendly name for the format, used in telemetry/debugging.
    pub name: String,
}

/// Controls the `text` field for the Responses API, combining verbosity and
/// optional JSON schema output formatting.
#[derive(Debug, Serialize, Default, Clone, PartialEq)]
pub struct TextControls {
    #[serde(skip_serializing_if = "Option::is_none")]
    pub verbosity: Option<OpenAiVerbosity>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub format: Option<TextFormat>,
}

#[derive(Debug, Serialize, Default, Clone, PartialEq)]
#[serde(rename_all = "lowercase")]
pub enum OpenAiVerbosity {
    Low,
    #[default]
    Medium,
    High,
}

impl From<VerbosityConfig> for OpenAiVerbosity {
    fn from(v: VerbosityConfig) -> Self {
        match v {
            VerbosityConfig::Low => OpenAiVerbosity::Low,
            VerbosityConfig::Medium => OpenAiVerbosity::Medium,
            VerbosityConfig::High => OpenAiVerbosity::High,
        }
    }
}

#[derive(Debug, Serialize, Clone, PartialEq)]
pub struct ResponsesApiRequest {
    pub model: String,
    #[serde(skip_serializing_if = "String::is_empty")]
    pub instructions: String,
    pub input: Vec<ResponseItem>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub tools: Option<Vec<serde_json::Value>>,
    pub tool_choice: String,
    pub parallel_tool_calls: bool,
    pub reasoning: Option<Reasoning>,
    pub store: bool,
    pub stream: bool,
    pub include: Vec<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub service_tier: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub prompt_cache_key: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub text: Option<TextControls>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub client_metadata: Option<HashMap<String, String>>,
}

impl From<&ResponsesApiRequest> for ResponseCreateWsRequest {
    fn from(request: &ResponsesApiRequest) -> Self {
        Self {
            model: request.model.clone(),
            instructions: request.instructions.clone(),
            previous_response_id: None,
            input: request.input.clone(),
            tools: request.tools.clone(),
            tool_choice: request.tool_choice.clone(),
            parallel_tool_calls: request.parallel_tool_calls,
            reasoning: request.reasoning.clone(),
            store: request.store,
            stream: request.stream,
            include: request.include.clone(),
            service_tier: request.service_tier.clone(),
            prompt_cache_key: request.prompt_cache_key.clone(),
            text: request.text.clone(),
            generate: None,
            client_metadata: request.client_metadata.clone(),
        }
    }
}

#[derive(Debug, Serialize)]
pub struct ResponseCreateWsRequest {
    pub model: String,
    #[serde(skip_serializing_if = "String::is_empty")]
    pub instructions: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub previous_response_id: Option<String>,
    pub input: Vec<ResponseItem>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub tools: Option<Vec<Value>>,
    pub tool_choice: String,
    pub parallel_tool_calls: bool,
    pub reasoning: Option<Reasoning>,
    pub store: bool,
    pub stream: bool,
    pub include: Vec<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub service_tier: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub prompt_cache_key: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub text: Option<TextControls>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub generate: Option<bool>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub client_metadata: Option<HashMap<String, String>>,
}

pub fn response_create_client_metadata(
    client_metadata: Option<HashMap<String, String>>,
    trace: Option<&W3cTraceContext>,
) -> Option<HashMap<String, String>> {
    let mut client_metadata = client_metadata.unwrap_or_default();

    if let Some(traceparent) = trace.and_then(|trace| trace.traceparent.as_deref()) {
        client_metadata.insert(
            WS_REQUEST_HEADER_TRACEPARENT_CLIENT_METADATA_KEY.to_string(),
            traceparent.to_string(),
        );
    }
    if let Some(tracestate) = trace.and_then(|trace| trace.tracestate.as_deref()) {
        client_metadata.insert(
            WS_REQUEST_HEADER_TRACESTATE_CLIENT_METADATA_KEY.to_string(),
            tracestate.to_string(),
        );
    }

    (!client_metadata.is_empty()).then_some(client_metadata)
}

#[derive(Debug, Serialize)]
#[serde(tag = "type")]
#[allow(clippy::large_enum_variant)]
pub enum ResponsesWsRequest {
    #[serde(rename = "response.create")]
    ResponseCreate(ResponseCreateWsRequest),
}

pub fn create_text_param_for_request(
    verbosity: Option<VerbosityConfig>,
    output_schema: &Option<Value>,
    output_schema_strict: bool,
) -> Option<TextControls> {
    if verbosity.is_none() && output_schema.is_none() {
        return None;
    }

    Some(TextControls {
        verbosity: verbosity.map(std::convert::Into::into),
        format: output_schema.as_ref().map(|schema| TextFormat {
            r#type: TextFormatType::JsonSchema,
            strict: output_schema_strict,
            schema: schema.clone(),
            name: "codex_output_schema".to_string(),
        }),
    })
}

pub struct ResponseStream {
    pub rx_event: mpsc::Receiver<Result<ResponseEvent, ApiError>>,
    /// Server-assigned `x-request-id` response header, when present.
    pub upstream_request_id: Option<String>,
}

impl Stream for ResponseStream {
    type Item = Result<ResponseEvent, ApiError>;

    fn poll_next(mut self: Pin<&mut Self>, cx: &mut Context<'_>) -> Poll<Option<Self::Item>> {
        self.rx_event.poll_recv(cx)
    }
}

// ======================================================================
// OpenAI Chat Completions 请求结构（原生多协议支持）。
// ----------------------------------------------------------------------
// 与 `ResponsesApiRequest` 并列，仅用于 OpenAI Chat Completions 协议
// （`POST /v1/chat/completions`）的 wire 序列化。中立 `UnifiedRequest`
// → `ChatApiRequest` 的字段翻译在 `endpoint/chat_adapter.rs`。
// ======================================================================

fn is_false(value: &bool) -> bool {
    !value
}

/// OpenAI Chat Completions 单条消息（`messages[]` 元素）。
///
/// `content` 用 `serde_json::Value` 而非 `String`：纯文本走 `Value::String`，
/// 多模态走 `Value::Array([{type:"text"|"image_url",...}])`。`tool_calls` 与
/// `tool_call_id` 互斥——前者出现在 assistant 发起调用的消息上，后者出现在
/// 回喂工具结果的 `role:"tool"` 消息上。
#[derive(Debug, Clone, Serialize, PartialEq)]
pub struct ChatMessage {
    pub role: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub content: Option<Value>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub name: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub tool_call_id: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub tool_calls: Option<Vec<ChatToolCall>>,
}

/// Chat Completions 的函数调用项（assistant 消息 `tool_calls[]` 元素）。
#[derive(Debug, Clone, Serialize, PartialEq)]
pub struct ChatToolCall {
    /// 工具调用 id，对齐 Responses 的 `call_id`，用于回喂 `tool_call_id`。
    pub id: String,
    /// 固定为 `"function"`。
    pub r#type: String,
    pub function: ChatToolCallFunction,
}

/// `tool_calls[].function` 子对象。
#[derive(Debug, Clone, Serialize, PartialEq)]
pub struct ChatToolCallFunction {
    pub name: String,
    /// 参数 JSON 串（对齐 Responses `arguments`，约定为含 JSON 的字符串）。
    pub arguments: String,
}

/// Chat Completions 流式选项。`include_usage: true` 让服务端在流末尾追加一帧 `usage`。
#[derive(Debug, Clone, Serialize, PartialEq)]
pub struct ChatStreamOptions {
    pub include_usage: bool,
}

/// OpenAI Chat Completions 顶层请求体（wire）。
///
/// 与 `ResponsesApiRequest` 字段语义对齐但形态不同：Responses 用 `input: Vec<ResponseItem>`
/// + 顶层 `reasoning`/`text`/`store`/`include`/`prompt_cache_key`/`client_metadata`；Chat
/// 用扁平的 `messages[]` + `reasoning_effort` + `response_format`，无 store/include 语义
/// （多轮靠 `messages` 累积，由上层保证）。
#[derive(Debug, Clone, Serialize, PartialEq)]
pub struct ChatApiRequest {
    pub model: String,
    pub messages: Vec<ChatMessage>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub tools: Option<Vec<Value>>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub tool_choice: Option<Value>,
    #[serde(skip_serializing_if = "is_false")]
    pub parallel_tool_calls: bool,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub reasoning_effort: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub service_tier: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub response_format: Option<Value>,
    /// 单次回复的最大输出 token 上限（可选）。None 时不发送，由服务端决定；
    /// 来自 `Provider.max_output_tokens`，由 `OpenaiChatAdapter::stream` 注入。
    #[serde(skip_serializing_if = "Option::is_none")]
    pub max_tokens: Option<u32>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub stream_options: Option<ChatStreamOptions>,
    pub stream: bool,
}

// ======================================================================
// Anthropic Messages 请求结构（原生多协议支持）。
// ----------------------------------------------------------------------
// 与 `ResponsesApiRequest` / `ChatApiRequest` 并列，仅用于 Anthropic Messages
// 协议（`POST /v1/messages`）的 wire 序列化。中立 `UnifiedRequest`
// → `AnthropicApiRequest` 的字段翻译在 `endpoint/anthropic_adapter.rs`。
//
// 与 Chat 的核心形态差异：① system 提示是顶层 `system` 字段（非 messages 里的
// 角色）；② 消息内容是 `content[]` 内容块数组（非扁平 String）；③ 工具调用是
// assistant 消息里的 `tool_use` 块、结果回喂是 user 消息里的 `tool_result` 块。
// ======================================================================

/// Anthropic 顶层 system 文本块（`{type:"text", text}`）。
///
/// Anthropic 把系统提示放在顶层 `system` 字段（字符串或内容块数组），而非
/// `messages[]` 里。统一用数组形态（便于后续追加 `cache_control`）。
#[derive(Debug, Clone, Serialize, PartialEq)]
pub struct AnthropicSystemTextBlock {
    #[serde(rename = "type")]
    type_: String,
    text: String,
}

impl AnthropicSystemTextBlock {
    pub fn new(text: String) -> Self {
        Self {
            type_: "text".to_string(),
            text,
        }
    }
}

/// Anthropic 消息内容块（`messages[].content[]` 元素）。
///
/// `#[serde(tag = "type", rename_all = "snake_case")]` 把变体名映射为 `type`
/// 字段：`Text`→`text`、`Image`→`image`、`ToolUse`→`tool_use`、
/// `ToolResult`→`tool_result`，各变体字段扁平其旁。
#[derive(Debug, Clone, Serialize, PartialEq)]
#[serde(tag = "type", rename_all = "snake_case")]
pub enum AnthropicContentBlock {
    /// 纯文本块。
    Text { text: String },
    /// 图片块（base64 或 url 两种来源）。
    Image { source: AnthropicImageSource },
    /// assistant 发起的工具调用（`id` 对齐 Responses `call_id`，`input` 为已解析 JSON 对象）。
    ToolUse {
        id: String,
        name: String,
        input: Value,
    },
    /// user 回喂的工具结果（`tool_use_id` 对齐被回应的 `ToolUse.id`）。
    ToolResult {
        tool_use_id: String,
        content: String,
        /// 失败标志（对齐 Anthropic `is_error`）：`Some(true)` 表示工具执行失败，
        /// 让服务端据此让模型区分成败（避免按成功输出继续错误推理）；`None` 不序列化（默认成功）。
        #[serde(skip_serializing_if = "Option::is_none")]
        is_error: Option<bool>,
    },
}

/// Anthropic 图片来源（`image.source`）。
#[derive(Debug, Clone, Serialize, PartialEq)]
#[serde(tag = "type", rename_all = "snake_case")]
pub enum AnthropicImageSource {
    /// base64 内嵌图片：`{type:"base64", media_type, data}`。
    Base64 { media_type: String, data: String },
    /// 外链图片：`{type:"url", url}`。
    Url { url: String },
}

/// Anthropic 单条消息（`messages[]` 元素）。`role` 限 `user` / `assistant`。
#[derive(Debug, Clone, Serialize, PartialEq)]
pub struct AnthropicMessage {
    pub role: String,
    pub content: Vec<AnthropicContentBlock>,
}

/// Anthropic 工具描述（`tools[]` 元素）。
///
/// 形态 `{name, description?, input_schema}`，与 OpenAI 的
/// `{type:"function", function:{name, parameters}}` 不同：无外层 function 包装，
/// schema 字段名为 `input_schema`。
#[derive(Debug, Clone, Serialize, PartialEq)]
pub struct AnthropicTool {
    pub name: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub description: Option<String>,
    pub input_schema: Value,
}

/// Anthropic tool_choice（`{type, name?, disable_parallel_tool_use?}`）。
///
/// `auto` 由模型决定是否调用；`any` 强制调用任一工具（对齐 OpenAI `required`）；
/// `tool` 指定调用某工具（带 `name`）；`none` 显式禁用工具调用。`disable_parallel_tool_use`
/// 仅在 tool_choice 存在时有效（Anthropic 约束），`Some(true)` 强制串行工具调用
/// （对齐 OpenAI `parallel_tool_calls=false`），`None` 不序列化。
#[derive(Debug, Clone, Serialize, PartialEq)]
pub struct AnthropicToolChoice {
    #[serde(rename = "type")]
    pub type_: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub name: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub disable_parallel_tool_use: Option<bool>,
}

/// Anthropic Messages 顶层请求体（wire）。
///
/// 与 `ResponsesApiRequest` / `ChatApiRequest` 语义对齐但形态不同：Anthropic 用
/// 顶层 `system` + `messages[].content[]` 内容块，工具调用/结果以 `tool_use` /
/// `tool_result` 块承载。`max_tokens` 为 Anthropic 必填项（Responses/Chat 不传），
/// 由 adapter 以常量默认填充（见 `anthropic_adapter::ANTHROPIC_DEFAULT_MAX_TOKENS`）。
#[derive(Debug, Clone, Serialize, PartialEq)]
pub struct AnthropicApiRequest {
    pub model: String,
    pub max_tokens: u32,
    pub messages: Vec<AnthropicMessage>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub system: Option<Vec<AnthropicSystemTextBlock>>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub tools: Option<Vec<AnthropicTool>>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub tool_choice: Option<AnthropicToolChoice>,
    pub stream: bool,
}
