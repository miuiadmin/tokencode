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

/// Anthropic 结构化输出的虚拟工具名（adapter 内部 tool-mode 注入，非 registry 工具）。
///
/// Anthropic 无原生 JSON-schema 输出约束；adapter 在请求带 `text.format` 时注入此工具
/// 并强制 `tool_choice` 指向它，迫使模型把结构化结果以 `tool_use.input` 回吐（JSON 对象）。
/// parser 侧识别此名，把 `tool_use` 还原成 `Message(OutputText)`（input_json 当输出文本），
/// 使 harness 零感知——不派发该工具、不产出 FunctionCall。
///
/// **不含 `__`**：`ToolName::from_flat_wire_name` 按首个 `__` 拆 namespace，若虚拟名含 `__`
/// 会被误判为 namespaced，导致派发表错位。adapter 与 parser 共用此常量，避免两侧名漂移。
pub const ANTHROPIC_STRUCTURED_OUTPUT_TOOL: &str = "respond_structured";

/// Anthropic prompt 缓存断点（`cache_control`：`{type:"ephemeral"}`）。
///
/// Anthropic 按请求内容前缀缓存（无 key 概念）；在稳定前缀的末元素上打 `ephemeral` 断点，
/// 使其前的内容进入缓存。token 成本：首次写入 1.25x、后续命中 0.1x，session ≥2 turn 即净赚。
/// Anthropic 约束单请求 ≤4 个断点——本项目固定打 2 个（system 末块 + tools 末工具），天然满足。
/// 缓存命中要求前缀逐字节相同，故只标稳定前缀（system + tools），**不**标每 turn 变动的对话消息
/// （标了反而会 bust 缓存 + 白付写惩罚）。adapter 始终开启，非按 `prompt_cache_key` 门控。
#[derive(Debug, Clone, Serialize, PartialEq)]
pub struct AnthropicCacheControl {
    #[serde(rename = "type")]
    type_: String,
}

impl AnthropicCacheControl {
    /// 短命缓存断点（默认 5 分钟 TTL，命中后续请求）。本项目唯一使用的类型。
    pub fn ephemeral() -> Self {
        Self {
            type_: "ephemeral".to_string(),
        }
    }
}

/// Anthropic 顶层 system 文本块（`{type:"text", text, cache_control?}`）。
///
/// Anthropic 把系统提示放在顶层 `system` 字段（字符串或内容块数组），而非
/// `messages[]` 里。统一用数组形态（便于在末块打 `cache_control` 缓存断点）。
#[derive(Debug, Clone, Serialize, PartialEq)]
pub struct AnthropicSystemTextBlock {
    #[serde(rename = "type")]
    type_: String,
    text: String,
    /// prompt 缓存断点：system 末块标 `ephemeral`（稳定前缀，跨 turn 命中）。None 不序列化。
    #[serde(skip_serializing_if = "Option::is_none")]
    pub cache_control: Option<AnthropicCacheControl>,
}

impl AnthropicSystemTextBlock {
    pub fn new(text: String) -> Self {
        Self {
            type_: "text".to_string(),
            text,
            cache_control: None,
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
    /// 扩展思考块（assistant 的 thinking 内容）。
    ///
    /// 仅在跨 turn 回喂历史思考时由 adapter 产出：从 `ResponseItem::Reasoning`
    /// （`continuity_token` 承载 Anthropic signature）还原。`signature` 是思考连续
    /// 凭证，无则不序列化（首 turn 无签名的情况罕见）。wire 形如
    /// `{"type":"thinking","thinking":"...","signature":"..."}`，对齐 Anthropic 协议。
    Thinking {
        thinking: String,
        #[serde(skip_serializing_if = "Option::is_none")]
        signature: Option<String>,
    },
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
/// 形态 `{name, description?, input_schema, cache_control?}`，与 OpenAI 的
/// `{type:"function", function:{name, parameters}}` 不同：无外层 function 包装，
/// schema 字段名为 `input_schema`。`cache_control` 标在 tools 末工具上以缓存稳定工具定义前缀。
#[derive(Debug, Clone, Serialize, PartialEq)]
pub struct AnthropicTool {
    pub name: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub description: Option<String>,
    pub input_schema: Value,
    /// prompt 缓存断点：tools 末工具标 `ephemeral`（稳定前缀，跨 turn 命中）。None 不序列化。
    #[serde(skip_serializing_if = "Option::is_none")]
    pub cache_control: Option<AnthropicCacheControl>,
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

/// Anthropic extended thinking 配置（顶层 `thinking` 字段）。
///
/// 形态 `{type:"enabled", budget_tokens}`：启用模型扩展思考，`budget_tokens` 为
/// 思考 token 预算（Anthropic 约束：≥ 1024 且严格小于 `max_tokens`）。由 adapter 从
/// 中立 `UnifiedReasoning.effort` 档位推导（None/Minimal 不启用），让 Anthropic 协议
/// 获得与其他协议对齐的「思考强度」能力——请求侧发 `thinking`，响应侧把 `thinking_delta`
/// 流式归一为 `ResponseEvent::ReasoningContentDelta`。
#[derive(Debug, Clone, Serialize, PartialEq)]
pub struct AnthropicThinking {
    #[serde(rename = "type")]
    pub type_: String,
    pub budget_tokens: u32,
}

/// Anthropic Messages 顶层请求体（wire）。
///
/// 与 `ResponsesApiRequest` / `ChatApiRequest` 语义对齐但形态不同：Anthropic 用
/// 顶层 `system` + `messages[].content[]` 内容块，工具调用/结果以 `tool_use` /
/// `tool_result` 块承载。`max_tokens` 为 Anthropic 必填项（Responses/Chat 不传），
/// 由 adapter 以常量默认填充（见 `anthropic_adapter::ANTHROPIC_DEFAULT_MAX_TOKENS`）。
/// `thinking` 由 adapter 从 `UnifiedReasoning.effort` 推导（见
/// `anthropic_adapter::thinking_from_effort`），未启用时为 None 不序列化。
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
    /// 扩展思考配置；None 时不序列化（不发 `thinking` 字段）。
    #[serde(skip_serializing_if = "Option::is_none")]
    pub thinking: Option<AnthropicThinking>,
    pub stream: bool,
}

// ======================================================================
// Gemini generateContent 协议 wire 结构（与 Responses / Chat / Anthropic 并列）
//
// 与前三者的核心形态差异：
// ① contents[] 只认 user/model 角色（assistant→model），system 必须走顶层
//   `systemInstruction`（进 contents 会被服务端 400）；
// ② 消息内容是 parts[]，靠字段名区分类型（`text`/`inlineData`/`functionCall`/
//   `functionResponse`），而非 OpenAI 的 `{type:"..."}` tag；
// ③ 工具调用是 model 消息里的 `functionCall` part，结果回喂是 **user** 消息里的
//   `functionResponse` part（注意：是 user 而非专用 role，与 OpenAI tool role 不同）；
// ④ model 不在 body，而在 URL path（`v1beta/models/{model}:streamGenerateContent`）；
//   流式由端点名决定（streamGenerateContent vs generateContent），故 body 无 model/stream 字段。
// ======================================================================

/// Gemini 内嵌数据（图片/文件 base64）——`parts[].inlineData`。
#[derive(Debug, Clone, Serialize, PartialEq)]
pub struct GeminiInlineData {
    #[serde(rename = "mimeType")]
    pub mime_type: String,
    /// base64 编码的字节流（与 Anthropic `image.source.data` 同源）。
    pub data: String,
}

/// Gemini 函数调用载荷（`parts[].functionCall`）：`{name, args}`。
///
/// `args` 为已解析 JSON 对象（对齐 Responses `FunctionCall.arguments` 解析后形态）。
#[derive(Debug, Clone, Serialize, PartialEq)]
pub struct GeminiFunctionCall {
    pub name: String,
    pub args: Value,
}

/// Gemini 函数结果回喂载荷（`parts[].functionResponse`）：`{name, response}`。
///
/// `response` 为结果对象（对齐 Responses `FunctionCallOutput.output` 解析后形态）。
/// 失败时 `response` 内塞 `{error: "..."}`（Gemini 无独立 is_error 字段，靠内容表达）。
#[derive(Debug, Clone, Serialize, PartialEq)]
pub struct GeminiFunctionResponse {
    pub name: String,
    pub response: Value,
}

/// Gemini 内容块（`contents[].parts[]` 元素）。
///
/// Gemini 的 part 靠字段名区分类型（非 `type` tag），故用 `#[serde(untagged)]`：
/// 序列化时直接输出变体字段（无 tag 前缀），与 Gemini wire 完全一致。各变体字段必然
/// 存在（非 Option），保证一个 part 恒为单态。
///
/// `Thought` 须在 `Text` 之前声明——虽然 Serialize 方向 untagged 不依赖顺序，但保持
/// 「带标记者优先」的阅读顺序，便于理解思考流 part 的 `thought:true` 标记语义。
#[derive(Debug, Clone, Serialize, PartialEq)]
#[serde(untagged)]
pub enum GeminiPart {
    /// 思考片段（`includeThoughts:true` 时模型返回；`thought:true` 标记 + 文本）。
    ///
    /// `thought_signature` 承载 Gemini 的 `thoughtSignature`（跨 turn 思考连续凭证）。仅在 adapter
    /// 回放历史思考时由 `ResponseItem::Reasoning`（`continuity_token` 还原）填入；首 turn 无签名时
    /// 缺席（`skip_serializing_if`）。回喂历史思考须带 `thought:true` + `thoughtSignature`，缺一会被
    /// 服务端拒。
    Thought {
        thought: bool,
        text: String,
        #[serde(rename = "thoughtSignature", skip_serializing_if = "Option::is_none")]
        thought_signature: Option<String>,
    },
    /// 纯文本块。
    Text {
        text: String,
    },
    /// 内嵌图片/文件（base64）。
    InlineData {
        #[serde(rename = "inlineData")]
        inline_data: GeminiInlineData,
    },
    /// 模型发起的工具调用（出现在 model 消息 parts）。
    FunctionCall {
        #[serde(rename = "functionCall")]
        function_call: GeminiFunctionCall,
    },
    /// 工具结果回喂（出现在 user 消息 parts）。
    FunctionResponse {
        #[serde(rename = "functionResponse")]
        function_response: GeminiFunctionResponse,
    },
}

/// Gemini 单条内容（`contents[]` 元素）。`role` 限 `user` / `model`（assistant→model）。
#[derive(Debug, Clone, Serialize, PartialEq)]
pub struct GeminiContent {
    pub role: String,
    pub parts: Vec<GeminiPart>,
}

/// Gemini 系统指令（顶层 `systemInstruction`）。
///
/// Gemini 不允许 contents[] 出现 system 角色：系统提示必须独立放在顶层
/// `systemInstruction`。沿用 `{role, parts}` 形态（role 省略，parts 为文本块）。
#[derive(Debug, Clone, Serialize, PartialEq)]
pub struct GeminiSystemInstruction {
    #[serde(skip_serializing_if = "Option::is_none")]
    pub role: Option<String>,
    pub parts: Vec<GeminiPart>,
}

/// Gemini 函数声明（`tools[].functionDeclarations[]` 元素）。
///
/// 形态 `{name, description?, parameters?}`：`parameters` 为 JSON Schema（须经
/// `sanitize_gemini_schema` 清洗，否则 Gemini 严格校验会 400）。
#[derive(Debug, Clone, Serialize, PartialEq)]
pub struct GeminiToolDeclaration {
    pub name: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub description: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub parameters: Option<Value>,
}

/// Gemini 工具集（`tools[]` 元素）。
///
/// 形态 `{functionDeclarations: [...]}`：函数声明包在 `functionDeclarations` 数组里
/// （与 OpenAI `{type:"function", function:{...}}` 扁平、Anthropic `{name, input_schema}` 不同）。
#[derive(Debug, Clone, Serialize, PartialEq)]
pub struct GeminiTool {
    #[serde(rename = "functionDeclarations")]
    pub function_declarations: Vec<GeminiToolDeclaration>,
}

/// Gemini 函数调用配置（`toolConfig.functionCallingConfig`）。
///
/// `mode`：`AUTO`（模型决定）/ `ANY`（强制调用，可配 `allowed_function_names` 限定）/
/// `NONE`（禁用）。`allowed_function_names` 仅 ANY 模式有意义，其余模式不发。
#[derive(Debug, Clone, Serialize, PartialEq)]
pub struct GeminiFunctionCallingConfig {
    pub mode: String,
    #[serde(skip_serializing_if = "Option::is_none", rename = "allowedFunctionNames")]
    pub allowed_function_names: Option<Vec<String>>,
}

/// Gemini 工具选择配置（顶层 `toolConfig`）。
#[derive(Debug, Clone, Serialize, PartialEq)]
pub struct GeminiToolConfig {
    #[serde(rename = "functionCallingConfig")]
    pub function_calling_config: GeminiFunctionCallingConfig,
}

/// Gemini thinking 配置（`generationConfig.thinkingConfig`）。
///
/// 双模态共存（按 model 名分叉，由 adapter 推导发其中之一）：
/// - gemini-2.5：`thinking_budget`（int，0=禁用，>0 为思考 token 预算）；
/// - gemini-3.x：`thinking_level`（enum 字面量：minimal/low/medium/high）。
/// `include_thoughts`：true 时返回思考流（parts 带 `thought:true`），adapter 据此
/// 归一为 `ResponseEvent::ReasoningContentDelta`。三者皆 Option，按需发。
#[derive(Debug, Clone, Serialize, PartialEq)]
pub struct GeminiThinkingConfig {
    #[serde(skip_serializing_if = "Option::is_none", rename = "thinkingBudget")]
    pub thinking_budget: Option<i32>,
    #[serde(skip_serializing_if = "Option::is_none", rename = "thinkingLevel")]
    pub thinking_level: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none", rename = "includeThoughts")]
    pub include_thoughts: Option<bool>,
}

/// Gemini 生成配置（`generationConfig`）。
///
/// `max_output_tokens` 由 adapter 从 `Provider.max_output_tokens` 注入（None 时不发，
/// 走服务端默认）。`thinking_config` 由 adapter 从 `UnifiedReasoning.effort` + model
/// 名推导（双模态）。`response_mime_type` / `response_schema` 由 adapter 从
/// `UnifiedTextFormat`（`text.format`）注入，承载结构化输出（JSON schema）；二者皆
/// `Option`，仅在请求带 `format` 时填充。`Default` 便于 adapter 用
/// `get_or_insert_default()` 先占位后填字段。
#[derive(Debug, Clone, Serialize, PartialEq, Default)]
pub struct GeminiGenerationConfig {
    #[serde(skip_serializing_if = "Option::is_none")]
    pub temperature: Option<f32>,
    #[serde(skip_serializing_if = "Option::is_none", rename = "topP")]
    pub top_p: Option<f32>,
    #[serde(skip_serializing_if = "Option::is_none", rename = "topK")]
    pub top_k: Option<f32>,
    #[serde(skip_serializing_if = "Option::is_none", rename = "maxOutputTokens")]
    pub max_output_tokens: Option<u32>,
    #[serde(skip_serializing_if = "Option::is_none", rename = "thinkingConfig")]
    pub thinking_config: Option<GeminiThinkingConfig>,
    /// 结构化输出 MIME：带 `text.format` 时固定 `application/json`，指示 Gemini 输出 JSON。
    #[serde(skip_serializing_if = "Option::is_none", rename = "responseMimeType")]
    pub response_mime_type: Option<String>,
    /// 结构化输出 schema：`text.format.schema` 经 strip 禁键 + sanitize 后填入。
    /// Gemini responseSchema 仅认 OpenAPI 3.0 子集，须先剥 `$schema`/`title`/`$defs`/
    /// `$ref`/`default`/`examples` 再过工具参数同款 `sanitize_gemini_schema`。
    #[serde(skip_serializing_if = "Option::is_none", rename = "responseSchema")]
    pub response_schema: Option<Value>,
}

/// Gemini generateContent 顶层请求体（wire）。
///
/// 与 `ResponsesApiRequest` / `ChatApiRequest` / `AnthropicApiRequest` 语义对齐，但为
/// 第四种 wire 风格：`contents[]` + 顶层 `systemInstruction` + `tools` + `toolConfig` +
/// `generationConfig`。**无 model 字段**（model 在 URL path）；**无 stream 字段**
/// （流式由端点 `streamGenerateContent` 决定）。`contents` 必须非空（Gemini 要求至少
/// 一条 user 内容），由 `GeminiClient::stream_request` 守卫。
#[derive(Debug, Clone, Serialize, PartialEq)]
pub struct GeminiApiRequest {
    pub contents: Vec<GeminiContent>,
    #[serde(skip_serializing_if = "Option::is_none", rename = "systemInstruction")]
    pub system_instruction: Option<GeminiSystemInstruction>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub tools: Option<Vec<GeminiTool>>,
    #[serde(skip_serializing_if = "Option::is_none", rename = "toolConfig")]
    pub tool_config: Option<GeminiToolConfig>,
    #[serde(skip_serializing_if = "Option::is_none", rename = "generationConfig")]
    pub generation_config: Option<GeminiGenerationConfig>,
}
