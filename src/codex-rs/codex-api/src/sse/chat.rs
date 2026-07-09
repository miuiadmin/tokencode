//! OpenAI Chat Completions 流式解析器（原生多协议支持）。
//!
//! 与 `sse::responses` 同构：消费一条 `StreamResponse`（HTTP 已是 2xx；非 2xx 由传输层
//! 归一为 `TransportError::Http`，401/重试由上层 core 处理），把 Chat Completions 的
//! SSE chunk 流翻译为 core 已消费的 `ResponseEvent` 词表。
//!
//! Chat 与 Responses 的协议差异（本文件要弥合的）：
//! - Chat 以 `data: [DONE]` 收尾；Responses 靠 `response.completed`。
//! - Chat 不发装配好的 item：assistant 文本靠 `choices[].delta.content` 分片、工具调用靠
//!   `choices[].delta.tool_calls` 分片——本解析器累积后，在 `[DONE]`/EOF 一次性装配出
//!   `OutputItemDone(Message)` / `OutputItemDone(FunctionCall)`。
//! - Chat 的 usage 在末尾单独一帧（`choices:[]` + `usage:{...}`，需 `stream_options.
//!   include_usage`），格式为 `prompt_tokens`/`completion_tokens`/`total_tokens`。

use crate::common::ResponseEvent;
use crate::common::ResponseStream;
use crate::error::ApiError;
use crate::rate_limits::parse_all_rate_limits;
use crate::telemetry::SseTelemetry;
use codex_client::ByteStream;
use codex_client::StreamResponse;
use codex_protocol::models::ContentItem;
use codex_protocol::models::ResponseItem;
use codex_protocol::protocol::TokenUsage;
use eventsource_stream::Eventsource;
use futures::StreamExt;
use serde::Deserialize;
use std::collections::BTreeMap;
use std::sync::Arc;
use std::sync::OnceLock;
use std::time::Duration;
use tokio::sync::mpsc;
use tokio::time::Instant;
use tokio::time::timeout;
use tracing::debug;
use tracing::trace;

const OPENAI_MODEL_HEADER: &str = "openai-model";
const REQUEST_ID_HEADER: &str = "x-request-id";
const MODELS_ETAG_HEADER: &str = "X-Models-Etag";
const X_CODEX_TURN_STATE_HEADER: &str = "x-codex-turn-state";

/// `[DONE]` 终止标记（OpenAI Chat 流式约定）。
const DONE_MARKER: &str = "[DONE]";

/// 把一条 Chat Completions 的 `StreamResponse` 翻译为 `ResponseStream`。
///
/// 与 `spawn_response_stream` 同构：先从响应头解析出协议无关的派生事件
/// （ServerModel / RateLimits / ModelsEtag），再 spawn 一个任务跑 SSE 字节循环。
pub fn spawn_chat_stream(
    stream_response: StreamResponse,
    idle_timeout: Duration,
    telemetry: Option<Arc<dyn SseTelemetry>>,
    turn_state: Option<Arc<OnceLock<String>>>,
) -> ResponseStream {
    let rate_limit_snapshots = parse_all_rate_limits(&stream_response.headers);
    let models_etag = stream_response
        .headers
        .get(MODELS_ETAG_HEADER)
        .and_then(|v| v.to_str().ok())
        .map(ToString::to_string);
    let server_model = stream_response
        .headers
        .get(OPENAI_MODEL_HEADER)
        .and_then(|v| v.to_str().ok())
        .map(ToString::to_string);
    let upstream_request_id = stream_response
        .headers
        .get(REQUEST_ID_HEADER)
        .and_then(|value| value.to_str().ok())
        .map(str::to_string);
    if let Some(turn_state) = turn_state.as_ref()
        && let Some(header_value) = stream_response
            .headers
            .get(X_CODEX_TURN_STATE_HEADER)
            .and_then(|value| value.to_str().ok())
    {
        let _ = turn_state.set(header_value.to_string());
    }

    let (tx_event, rx_event) = mpsc::channel::<Result<ResponseEvent, ApiError>>(1600);
    tokio::spawn(async move {
        if let Some(model) = server_model {
            let _ = tx_event.send(Ok(ResponseEvent::ServerModel(model))).await;
        }
        for snapshot in rate_limit_snapshots {
            let _ = tx_event.send(Ok(ResponseEvent::RateLimits(snapshot))).await;
        }
        if let Some(etag) = models_etag {
            let _ = tx_event.send(Ok(ResponseEvent::ModelsEtag(etag))).await;
        }
        process_chat_sse(stream_response.bytes, tx_event, idle_timeout, telemetry).await;
    });

    ResponseStream {
        rx_event,
        upstream_request_id,
    }
}

// ===== Chat SSE chunk 反序列化结构 ============================================

/// 单个 Chat Completions 流式 chunk（`data: {...}`）。
#[derive(Debug, Deserialize)]
struct ChatStreamChunk {
    #[serde(default)]
    id: Option<String>,
    #[serde(default)]
    model: Option<String>,
    #[serde(default)]
    choices: Vec<ChatChoice>,
    #[serde(default)]
    usage: Option<ChatUsage>,
    /// 部分网关在流中以 `{ "error": {...} }` 帧报错（HTTP 仍是 2xx）。
    #[serde(default)]
    error: Option<ChatStreamError>,
}

#[derive(Debug, Deserialize)]
struct ChatChoice {
    #[serde(default)]
    delta: ChatDelta,
    #[serde(default)]
    finish_reason: Option<String>,
}

#[derive(Debug, Default, Deserialize)]
#[allow(dead_code)] // role 仅作 wire 兼容（首个 chunk 的 delta.role="assistant"），翻译侧不消费。
struct ChatDelta {
    #[serde(default)]
    role: Option<String>,
    #[serde(default)]
    content: Option<String>,
    /// o-series / 部分网关的推理内容字段（glm-5.2 不发；best-effort 转发）。
    #[serde(default)]
    reasoning_content: Option<String>,
    #[serde(default)]
    tool_calls: Option<Vec<ChatDeltaToolCall>>,
}

#[derive(Debug, Deserialize)]
#[allow(dead_code)] // type 恒为 "function"，翻译侧按 function 语义处理，不读该字段。
struct ChatDeltaToolCall {
    /// 索引：决定参数分片累积到哪一条 tool_call。
    index: i64,
    #[serde(default)]
    id: Option<String>,
    #[serde(default)]
    r#type: Option<String>,
    #[serde(default)]
    function: Option<ChatDeltaToolCallFunction>,
}

#[derive(Debug, Default, Deserialize)]
struct ChatDeltaToolCallFunction {
    #[serde(default)]
    name: Option<String>,
    #[serde(default)]
    arguments: Option<String>,
}

#[derive(Debug, Clone, Copy, Deserialize)]
struct ChatUsage {
    prompt_tokens: i64,
    completion_tokens: i64,
    total_tokens: i64,
}

impl From<ChatUsage> for TokenUsage {
    fn from(val: ChatUsage) -> Self {
        // Chat usage 无 cached / reasoning 拆分，对应字段置 0。
        TokenUsage {
            input_tokens: val.prompt_tokens,
            cached_input_tokens: 0,
            output_tokens: val.completion_tokens,
            reasoning_output_tokens: 0,
            total_tokens: val.total_tokens,
        }
    }
}

#[derive(Debug, Deserialize)]
struct ChatStreamError {
    #[serde(default)]
    message: Option<String>,
}

// ===== 累积装配状态 ==========================================================

/// 单条工具调用的累积器（按 `index` 归集参数分片）。
#[derive(Debug, Default)]
struct ToolCallAccumulator {
    /// 工具调用 id（对齐 Responses `call_id`）。
    id: String,
    name: String,
    arguments: String,
}

/// 一个 assistant turn 的累积状态。
#[derive(Debug, Default)]
struct ChatStreamState {
    /// 是否已发过 `Created`。
    created_emitted: bool,
    /// 是否已发过 `OutputItemAdded(Message)`（文本/reasoning 增量前置需要 active item）。
    message_item_added: bool,
    assistant_text: String,
    reasoning_content: String,
    /// 按 index 排序的工具调用累积（输出顺序确定）。
    tool_calls: BTreeMap<i64, ToolCallAccumulator>,
    response_id: Option<String>,
    server_model: Option<String>,
    finish_reason: Option<String>,
    usage: Option<TokenUsage>,
}

// ===== SSE 字节循环 ==========================================================

async fn process_chat_sse(
    stream: ByteStream,
    tx_event: mpsc::Sender<Result<ResponseEvent, ApiError>>,
    idle_timeout: Duration,
    telemetry: Option<Arc<dyn SseTelemetry>>,
) {
    let mut stream = stream.eventsource();
    let mut state = ChatStreamState::default();

    loop {
        let start = Instant::now();
        let response = timeout(idle_timeout, stream.next()).await;
        if let Some(t) = telemetry.as_ref() {
            t.on_sse_poll(&response, start.elapsed());
        }
        let sse = match response {
            Ok(Some(Ok(sse))) => sse,
            Ok(Some(Err(e))) => {
                debug!("Chat SSE 错误: {e:#}");
                let _ = tx_event.send(Err(ApiError::Stream(e.to_string()))).await;
                return;
            }
            Ok(None) => {
                // 流自然结束（未见 `[DONE]`）：尽力冲刷已累积内容后收尾。
                flush(&mut state, &tx_event).await;
                return;
            }
            Err(_) => {
                let _ = tx_event
                    .send(Err(ApiError::Stream(
                        "idle timeout waiting for Chat SSE".into(),
                    )))
                    .await;
                return;
            }
        };

        trace!("Chat SSE 事件: {}", &sse.data);

        if sse.data.trim() == DONE_MARKER {
            flush(&mut state, &tx_event).await;
            return;
        }

        let chunk: ChatStreamChunk = match serde_json::from_str(&sse.data) {
            Ok(chunk) => chunk,
            Err(e) => {
                debug!("Chat SSE 解析失败: {e}, data: {}", &sse.data);
                continue;
            }
        };

        // 流内错误帧（HTTP 已 2xx，但 body 报错）。
        if let Some(err) = chunk.error {
            let message = err
                .message
                .filter(|m| !m.trim().is_empty())
                .unwrap_or_else(|| "chat stream error".to_string());
            let _ = tx_event
                .send(Err(ApiError::Stream(message)))
                .await;
            return;
        }

        if !process_chunk(&mut state, &chunk, &tx_event).await {
            return;
        }
    }
}

/// 处理单个 chunk：更新累积状态 + 发送增量事件。
///
/// 返回 `false` 表示消费端已关闭，调用方应结束任务。
async fn process_chunk(
    state: &mut ChatStreamState,
    chunk: &ChatStreamChunk,
    tx_event: &mpsc::Sender<Result<ResponseEvent, ApiError>>,
) -> bool {
    // Created：仅发一次（首个有效 chunk）。
    if !state.created_emitted {
        if send(tx_event, Ok(ResponseEvent::Created)).await {
            return false;
        }
        state.created_emitted = true;
    }

    // ServerModel 变更（chunk 内的 model 字段；与响应头互补）。
    if let Some(model) = &chunk.model
        && state.server_model.as_deref() != Some(model.as_str())
    {
        if send(tx_event, Ok(ResponseEvent::ServerModel(model.clone()))).await {
            return false;
        }
        state.server_model = Some(model.clone());
    }

    if state.response_id.is_none() {
        state.response_id = chunk.id.clone();
    }
    if let Some(usage) = &chunk.usage {
        state.usage = Some(usage.clone().into());
    }

    for choice in &chunk.choices {
        let delta = &choice.delta;
        if let Some(finish) = &choice.finish_reason {
            state.finish_reason = Some(finish.clone());
        }

        // 文本增量：先确保 active message item，再发 delta（core 要求 active item）。
        if let Some(content) = delta.content.as_deref()
            && !content.is_empty()
        {
            if !ensure_message_item(state, tx_event).await {
                return false;
            }
            state.assistant_text.push_str(content);
            if send(
                tx_event,
                Ok(ResponseEvent::OutputTextDelta(content.to_string())),
            )
            .await
            {
                return false;
            }
        }

        // reasoning 增量（best-effort）：仅在已有 active message item 时转发，
        // 避免为纯推理 turn 凭空创建空 assistant 消息。
        if let Some(reasoning) = delta.reasoning_content.as_deref()
            && !reasoning.is_empty()
            && state.message_item_added
        {
            state.reasoning_content.push_str(reasoning);
            if send(
                tx_event,
                Ok(ResponseEvent::ReasoningContentDelta {
                    delta: reasoning.to_string(),
                    content_index: 0,
                }),
            )
            .await
            {
                return false;
            }
        }

        // 工具调用增量：按 index 累积；新 index 发 OutputItemAdded 占位。
        if let Some(tool_calls) = &delta.tool_calls {
            for tc in tool_calls {
                let id_opt = tc.id.clone();
                let name_opt = tc.function.as_ref().and_then(|f| f.name.clone());
                let args_opt = tc.function.as_ref().and_then(|f| f.arguments.clone());

                let is_new = !state.tool_calls.contains_key(&tc.index);
                let entry = state.tool_calls.entry(tc.index).or_default();
                if is_new {
                    let id = id_opt
                        .clone()
                        .unwrap_or_else(|| format!("call_{}", tc.index));
                    entry.id = id.clone();
                    if let Some(name) = &name_opt {
                        entry.name = name.clone();
                    }
                    let placeholder = ResponseItem::FunctionCall {
                        id: None,
                        name: entry.name.clone(),
                        namespace: None,
                        arguments: String::new(),
                        call_id: id,
                        internal_chat_message_metadata_passthrough: None,
                    };
                    if send(tx_event, Ok(ResponseEvent::OutputItemAdded(placeholder))).await {
                        return false;
                    }
                } else {
                    if entry.name.is_empty() {
                        if let Some(name) = &name_opt {
                            entry.name = name.clone();
                        }
                    }
                    if entry.id.is_empty() {
                        if let Some(id) = &id_opt {
                            entry.id = id.clone();
                        }
                    }
                }
                if let Some(args) = args_opt
                    && !args.is_empty()
                {
                    entry.arguments.push_str(&args);
                }
            }
        }
    }

    true
}

/// 首次需要流式文本/reasoning 增量时，发一次 `OutputItemAdded(Message)` 建立 active item。
///
/// core 的 `OutputTextDelta` / `ReasoningContentDelta` 处理要求 `active_item` 已存在，
/// 否则会 panic；active item 由 `OutputItemAdded` 设置。
async fn ensure_message_item(
    state: &mut ChatStreamState,
    tx_event: &mpsc::Sender<Result<ResponseEvent, ApiError>>,
) -> bool {
    if state.message_item_added {
        return true;
    }
    let item = ResponseItem::Message {
        id: None,
        role: "assistant".to_string(),
        content: vec![],
        phase: None,
        internal_chat_message_metadata_passthrough: None,
    };
    if send(tx_event, Ok(ResponseEvent::OutputItemAdded(item))).await {
        return false;
    }
    state.message_item_added = true;
    true
}

/// 在 `[DONE]`/EOF 一次性装配并发出收尾事件：`OutputItemDone(Message)` →
/// `OutputItemDone(FunctionCall)`× → `Completed`。
async fn flush(
    state: &mut ChatStreamState,
    tx_event: &mpsc::Sender<Result<ResponseEvent, ApiError>>,
) {
    // 1. assistant 文本消息收尾（仅当确实累积到文本）。
    if state.message_item_added && !state.assistant_text.is_empty() {
        let message = ResponseItem::Message {
            id: None,
            role: "assistant".to_string(),
            content: vec![ContentItem::OutputText {
                text: state.assistant_text.clone(),
            }],
            phase: None,
            internal_chat_message_metadata_passthrough: None,
        };
        if send(tx_event, Ok(ResponseEvent::OutputItemDone(message))).await {
            return;
        }
    }

    // 2. 工具调用收尾（BTreeMap 按 index 升序）。
    for (_, tc) in std::mem::take(&mut state.tool_calls) {
        if tc.name.is_empty() && tc.arguments.is_empty() {
            // 跳过既无 name 又无参数的空累积（防御）。
            continue;
        }
        let call = ResponseItem::FunctionCall {
            id: None,
            name: tc.name,
            namespace: None,
            arguments: tc.arguments,
            call_id: tc.id,
            internal_chat_message_metadata_passthrough: None,
        };
        if send(tx_event, Ok(ResponseEvent::OutputItemDone(call))).await {
            return;
        }
    }

    // 3. Completed：response_id（缺省空串）、usage（缺省 None）、end_turn（stop → true）。
    let end_turn = Some(state.finish_reason.as_deref() == Some("stop"));
    let completed = ResponseEvent::Completed {
        response_id: state.response_id.clone().unwrap_or_default(),
        token_usage: state.usage.take(),
        end_turn,
    };
    let _ = tx_event.send(Ok(completed)).await;
}

/// 发送一条事件；返回 `true` 表示通道已关闭（调用方应结束）。
async fn send(
    tx_event: &mpsc::Sender<Result<ResponseEvent, ApiError>>,
    event: Result<ResponseEvent, ApiError>,
) -> bool {
    tx_event.send(event).await.is_err()
}
