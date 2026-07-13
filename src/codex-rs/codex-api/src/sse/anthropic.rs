//! Anthropic Messages 流式解析器（原生多协议支持）。
//!
//! 与 `sse::chat` 同构：消费一条 `StreamResponse`（HTTP 已是 2xx；非 2xx 由传输层归一为
//! `TransportError::Http`，401/重试由上层 core 处理），把 Anthropic Messages 的 SSE 事件流
//! 翻译为 core 已消费的 `ResponseEvent` 词表，再经 `response_stream_to_unified` 归一为
//! `UnifiedEvent`。
//!
//! Anthropic 与 Chat 的协议差异（本文件要弥合的）：
//! - 收尾靠 `message_stop` 事件（非 `[DONE]` 标记）；仍兜底在 EOF 冲刷，防止半截输出。
//! - 内容以 `content_block_*` 系列事件分片：`content_block_start` 给出块的 `type`（text/tool_use）
//!   与工具的 id/name；`content_block_delta` 给文本增量（`text_delta`）或工具参数增量
//!   （`input_json_delta`）；均带 `index`，须按 index 累积。
//! - usage 分两处：`message_start` 给 input/cache tokens，`message_delta` 给 output tokens。
//! - 收尾时（`message_stop`）统一装配 `OutputItemDone(Message)` / `OutputItemDone(FunctionCall)`，
//!   与 Chat 在 `[DONE]` 冲刷同构（文本走 live delta，工具参数静默累积 + 末尾装配）。

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
use codex_protocol::ToolName;
use eventsource_stream::Eventsource;
use futures::StreamExt;
use serde::Deserialize;
use serde_json::Value;
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

/// 把一条 Anthropic Messages 的 `StreamResponse` 翻译为 `ResponseStream`。
///
/// 与 `spawn_chat_stream` 同构：先从响应头解析出协议无关的派生事件（ServerModel / RateLimits /
/// ModelsEtag，原生 Anthropic 通常不带这些头，best-effort 兼容网关），再 spawn 任务跑 SSE 循环。
/// Anthropic 的权威 `model` 来自 `message_start` 事件（非响应头），在循环内补发 `ServerModel`。
pub fn spawn_anthropic_stream(
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
        process_anthropic_sse(stream_response.bytes, tx_event, idle_timeout, telemetry).await;
    });

    ResponseStream {
        rx_event,
        upstream_request_id,
    }
}

// ===== Anthropic SSE 事件反序列化结构 =========================================
//
// Anthropic SSE 每帧 `data` 是一个 JSON 对象，顶层带 `type` 字段区分事件类型；不同事件的
// 字段差异较大（`message` / `content_block` / `delta` / `usage` 形状各不相同）。为兼顾
// 类型安全与对未来事件/块类型（如 thinking、redacted_thinking）的容错，统一反序列化为一个
// 带 `type` + 一组 `Option<Value>` 字段的结构体，再在匹配侧按 `type` 提取所需字段。

/// 单个 Anthropic SSE 事件帧（`data: {...}`）。
#[derive(Debug, Deserialize)]
struct AnthropicEvent {
    #[serde(rename = "type")]
    type_: String,
    #[serde(default)]
    index: Option<i64>,
    /// `message_start` 的载荷（含 id / model / usage）。
    #[serde(default)]
    message: Option<Value>,
    /// `content_block_start` 的内容块描述（含 type / id / name / input）。
    #[serde(default)]
    content_block: Option<Value>,
    /// `content_block_delta` 的增量（text_delta / input_json_delta）；
    /// `message_delta` 的收尾载荷（stop_reason / stop_sequence）。
    #[serde(default)]
    delta: Option<Value>,
    /// `message_start` / `message_delta` 末尾的 usage 快照。
    #[serde(default)]
    usage: Option<Value>,
    /// 流内错误帧（部分网关在 2xx body 内以 `{type:"error", error:{...}}` 报错）。
    #[serde(default)]
    error: Option<Value>,
}

// ===== 累积装配状态 ==========================================================

/// 单条工具调用的累积器（按内容块 `index` 归集 `input_json_delta` 分片）。
#[derive(Debug, Default)]
struct ToolBlockAccumulator {
    /// 工具调用 id（对齐 Responses `call_id`，来自 `tool_use.id`）。
    id: String,
    name: String,
    /// 累积的参数 JSON 串（若干 `input_json_delta.partial_json` 拼接）。
    input_json: String,
}

/// usage 累积器：`message_start` 给 input/cache，`message_delta` 给 output。
///
/// Anthropic 的 `input_tokens` 是「非缓存」计数（缓存读写另报
/// `cache_creation_input_tokens` / `cache_read_input_tokens`），与本仓库
/// `TokenUsage.input_tokens`「含缓存超集」的语义不同（见 protocol.rs `non_cached_input`
/// 不变量：`non_cached = input - cached`）。故此处保留三个原始字段，超集在
/// `build_token_usage` 统一计算，避免不变量被破坏。
#[derive(Debug, Default)]
struct UsageAccumulator {
    /// Anthropic 原始 `input_tokens`（非缓存部分）。
    input_tokens: Option<i64>,
    /// `cache_creation_input_tokens`（本次写入缓存的输入）。
    cache_creation_input_tokens: Option<i64>,
    /// `cache_read_input_tokens`（命中缓存的输入）。
    cache_read_input_tokens: Option<i64>,
    output_tokens: Option<i64>,
}

/// 一个 assistant turn 的累积状态。
#[derive(Debug, Default)]
struct AnthropicStreamState {
    /// 是否已发过 `Created`。
    created_emitted: bool,
    /// 是否已发过 `OutputItemAdded(Message)`（文本增量前置需要 active item）。
    message_item_added: bool,
    assistant_text: String,
    /// 按 index 排序的工具块累积（输出顺序确定）。
    tool_blocks: BTreeMap<i64, ToolBlockAccumulator>,
    response_id: Option<String>,
    server_model: Option<String>,
    finish_reason: Option<String>,
    usage: UsageAccumulator,
}

// ===== SSE 字节循环 ==========================================================

async fn process_anthropic_sse(
    stream: ByteStream,
    tx_event: mpsc::Sender<Result<ResponseEvent, ApiError>>,
    idle_timeout: Duration,
    telemetry: Option<Arc<dyn SseTelemetry>>,
) {
    let mut stream = stream.eventsource();
    let mut state = AnthropicStreamState::default();

    loop {
        let start = Instant::now();
        let response = timeout(idle_timeout, stream.next()).await;
        if let Some(t) = telemetry.as_ref() {
            t.on_sse_poll(&response, start.elapsed());
        }
        let sse = match response {
            Ok(Some(Ok(sse))) => sse,
            Ok(Some(Err(e))) => {
                debug!("Anthropic SSE 错误: {e:#}");
                let _ = tx_event.send(Err(ApiError::Stream(e.to_string()))).await;
                return;
            }
            Ok(None) => {
                // 流自然结束（未见 `message_stop`）：尽力冲刷已累积内容后收尾。
                flush(&mut state, &tx_event).await;
                return;
            }
            Err(_) => {
                let _ = tx_event
                    .send(Err(ApiError::Stream(
                        "idle timeout waiting for Anthropic SSE".into(),
                    )))
                    .await;
                return;
            }
        };

        trace!("Anthropic SSE 事件: {}", &sse.data);

        let event: AnthropicEvent = match serde_json::from_str(&sse.data) {
            Ok(event) => event,
            Err(e) => {
                debug!("Anthropic SSE 解析失败: {e}, data: {}", &sse.data);
                continue;
            }
        };

        // 流内错误帧（HTTP 已 2xx，但 body 报错）。
        if event.type_ == "error" {
            let message = event
                .error
                .as_ref()
                .and_then(|e| e.get("error").or(Some(e)))
                .and_then(|e| e.get("message"))
                .and_then(|m| m.as_str())
                .map(str::to_string)
                .filter(|m| !m.trim().is_empty())
                .unwrap_or_else(|| "anthropic stream error".to_string());
            let _ = tx_event.send(Err(ApiError::Stream(message))).await;
            return;
        }

        if !process_event(&mut state, &event, &tx_event).await {
            return;
        }

        // `message_stop` 是 Anthropic 流的收尾信号：冲刷后结束任务。
        if event.type_ == "message_stop" {
            flush(&mut state, &tx_event).await;
            return;
        }
    }
}

/// 处理单个事件：更新累积状态 + 发送增量事件。
///
/// 返回 `false` 表示消费端已关闭，调用方应结束任务。
async fn process_event(
    state: &mut AnthropicStreamState,
    event: &AnthropicEvent,
    tx_event: &mpsc::Sender<Result<ResponseEvent, ApiError>>,
) -> bool {
    match event.type_.as_str() {
        "message_start" => {
            // Created 仅发一次。
            if !state.created_emitted {
                if send(tx_event, Ok(ResponseEvent::Created)).await {
                    return false;
                }
                state.created_emitted = true;
            }

            if let Some(message) = &event.message {
                if state.response_id.is_none() {
                    state.response_id = message
                        .get("id")
                        .and_then(|v| v.as_str())
                        .map(str::to_string);
                }
                if let Some(model) = message.get("model").and_then(|v| v.as_str()) {
                    if state.server_model.as_deref() != Some(model) {
                        if send(tx_event, Ok(ResponseEvent::ServerModel(model.to_string()))).await {
                            return false;
                        }
                        state.server_model = Some(model.to_string());
                    }
                }
                if let Some(usage) = message.get("usage") {
                    apply_usage(&mut state.usage, usage);
                }
            }
        }
        "content_block_start" => {
            let Some(block) = &event.content_block else {
                return true;
            };
            let kind = block.get("type").and_then(|v| v.as_str()).unwrap_or("");
            let index = event.index.unwrap_or(0);
            match kind {
                "tool_use" => {
                    let id = block
                        .get("id")
                        .and_then(|v| v.as_str())
                        .unwrap_or("")
                        .to_string();
                    let name = block
                        .get("name")
                        .and_then(|v| v.as_str())
                        .unwrap_or("")
                        .to_string();
                    // 首帧 `content_block_start` 的 tool_use 通常带空 input，由后续
                    // input_json_delta 填充；此处先建累积器并占位（对齐 Chat 的占位语义）。
                    let entry = state
                        .tool_blocks
                        .entry(index)
                        .or_insert_with(ToolBlockAccumulator::default);
                    let id = if id.is_empty() {
                        format!("call_{}", index)
                    } else {
                        id
                    };
                    entry.id = id.clone();
                    entry.name = name.clone();
                    // Anthropic wire 工具名为单字符串（协议无 namespace 字段）；非 Responses
                    // 协议下 harness 已把 namespace 工具展平成 `{ns}__{name}` 下发，这里还原。
                    let parsed = ToolName::from_flat_wire_name(&name);
                    let placeholder = ResponseItem::FunctionCall {
                        id: None,
                        name: parsed.name,
                        namespace: parsed.namespace,
                        arguments: String::new(),
                        call_id: id,
                        internal_chat_message_metadata_passthrough: None,
                    };
                    if send(tx_event, Ok(ResponseEvent::OutputItemAdded(placeholder))).await {
                        return false;
                    }
                }
                "text" => {
                    // 文本块：首个 delta 到来时再建 active message item，此处无需操作。
                }
                "thinking" | "redacted_thinking" => {
                    // 扩展思考块：thinking_delta 由后续 content_block_delta 流式归一为
                    // ReasoningContentDelta；redacted_thinking 是服务端加密块（无可读文本），
                    // v1 仅记录、不消费。块本身无需预建状态（delta 按 index 直发）。
                    debug!(block_type = kind, "Anthropic 扩展思考内容块");
                }
                _ => {
                    // 未知块类型：静默忽略（容错未来新块类型）。
                    debug!(block_type = kind, "跳过未知 Anthropic 内容块类型");
                }
            }
        }
        "content_block_delta" => {
            let Some(delta) = &event.delta else {
                return true;
            };
            let kind = delta.get("type").and_then(|v| v.as_str()).unwrap_or("");
            match kind {
                "text_delta" => {
                    if let Some(text) = delta.get("text").and_then(|v| v.as_str()) {
                        if !text.is_empty() {
                            if !ensure_message_item(state, tx_event).await {
                                return false;
                            }
                            state.assistant_text.push_str(text);
                            if send(
                                tx_event,
                                Ok(ResponseEvent::OutputTextDelta(text.to_string())),
                            )
                            .await
                            {
                                return false;
                            }
                        }
                    }
                }
                "input_json_delta" => {
                    if let Some(index) = event.index {
                        if let Some(partial) = delta.get("partial_json").and_then(|v| v.as_str()) {
                            if let Some(entry) = state.tool_blocks.get_mut(&index) {
                                entry.input_json.push_str(partial);
                            }
                        }
                    }
                }
                "thinking_delta" => {
                    // 扩展思考文本增量：归一为 ReasoningContentDelta（与其他协议推理增量对齐），
                    // 让上层能看到模型思考过程。core 消费要求 active item（与 text_delta 同因），
                    // 故先 ensure。content_index 用内容块 index。
                    if let Some(text) = delta.get("thinking").and_then(|v| v.as_str())
                        && !text.is_empty()
                    {
                        if !ensure_message_item(state, tx_event).await {
                            return false;
                        }
                        let index = event.index.unwrap_or(0);
                        if send(
                            tx_event,
                            Ok(ResponseEvent::ReasoningContentDelta {
                                delta: text.to_string(),
                                content_index: index,
                            }),
                        )
                        .await
                        {
                            return false;
                        }
                    }
                }
                "signature_delta" => {
                    // 思考签名增量（redacted_thinking 回喂凭证）：v1 不回喂历史思考
                    // （adapter 不发历史 thinking 块），签名仅记录、不累积、不外发。
                    debug!("Anthropic signature_delta（v1 不回喂思考，忽略）");
                }
                _ => {
                    debug!(delta_type = kind, "跳过未知 Anthropic 增量类型");
                }
            }
        }
        "content_block_stop" => {
            // 累积器在此 index 收口；装配统一在 message_stop 冲刷，此处无需操作。
        }
        "message_delta" => {
            if let Some(delta) = &event.delta {
                if let Some(stop_reason) = delta.get("stop_reason").and_then(|v| v.as_str()) {
                    state.finish_reason = Some(stop_reason.to_string());
                }
            }
            if let Some(usage) = &event.usage {
                apply_usage(&mut state.usage, usage);
            }
        }
        "ping" => {
            // 心跳，忽略。
        }
        _ => {
            // 未知事件类型：best-effort 忽略（Anthropic 新增事件类型时不致崩）。
            trace!(event_type = %event.type_, "跳过未知 Anthropic 事件类型");
        }
    }
    true
}

/// 把一帧 usage（`message_start` 或 `message_delta`）合并进累积器（后者覆盖 output 计数）。
fn apply_usage(acc: &mut UsageAccumulator, usage: &Value) {
    if let Some(v) = usage.get("input_tokens").and_then(|v| v.as_i64()) {
        acc.input_tokens = Some(v);
    }
    if let Some(v) = usage
        .get("cache_creation_input_tokens")
        .and_then(|v| v.as_i64())
    {
        acc.cache_creation_input_tokens = Some(v);
    }
    if let Some(v) = usage
        .get("cache_read_input_tokens")
        .and_then(|v| v.as_i64())
    {
        acc.cache_read_input_tokens = Some(v);
    }
    if let Some(v) = usage.get("output_tokens").and_then(|v| v.as_i64()) {
        acc.output_tokens = Some(v);
    }
}

/// 首次需要流式文本增量时，发一次 `OutputItemAdded(Message)` 建立 active item。
///
/// core 的 `OutputTextDelta` 处理要求 `active_item` 已存在，否则会 panic；active item 由
/// `OutputItemAdded` 设置。
async fn ensure_message_item(
    state: &mut AnthropicStreamState,
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

/// 在 `message_stop`/EOF 一次性装配并发出收尾事件：`OutputItemDone(Message)` →
/// `OutputItemDone(FunctionCall)`× → `Completed`。
async fn flush(
    state: &mut AnthropicStreamState,
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
    for (_, tb) in std::mem::take(&mut state.tool_blocks) {
        if tb.name.is_empty() && tb.input_json.is_empty() {
            // 跳过既无 name 又无参数的空累积（防御）。
            continue;
        }
        let parsed = ToolName::from_flat_wire_name(&tb.name);
        let call = ResponseItem::FunctionCall {
            id: None,
            name: parsed.name,
            namespace: parsed.namespace,
            arguments: tb.input_json,
            call_id: tb.id,
            internal_chat_message_metadata_passthrough: None,
        };
        if send(tx_event, Ok(ResponseEvent::OutputItemDone(call))).await {
            return;
        }
    }

    // 3. Completed：response_id（缺省空串）、usage（缺省 None）、end_turn。
    let end_turn = match state.finish_reason.as_deref() {
        // 自然结束（模型主动收尾 / 命中 stop_sequence）：回合结束。
        Some("end_turn") | Some("stop_sequence") => Some(true),
        // 已知「未结束 / 非正常截断」：工具待续、达上限、被内容过滤、思考暂停（pause_turn）。
        Some("tool_use") | Some("max_tokens") | Some("content_filter") | Some("pause_turn") => {
            Some(false)
        }
        // 未知 stop_reason（如网关返回非标准值）：不臆断回合结束，交上层裁决。
        Some(_) | None => None,
    };
    let token_usage = build_token_usage(&mut state.usage);
    let completed = ResponseEvent::Completed {
        response_id: state.response_id.clone().unwrap_or_default(),
        token_usage,
        end_turn,
    };
    let _ = tx_event.send(Ok(completed)).await;
}

/// 把 usage 累积器装配为 `TokenUsage`（全缺省时返回 None）。
///
/// `input_tokens` 存「含缓存的超集」= 非缓存输入 + cache_creation + cache_read，
/// 以满足 `TokenUsage` 不变量（`non_cached_input = input - cached`；protocol.rs）；
/// `cached_input_tokens` 存 `cache_read`（不变量中的缓存子集）。各项用 `saturating_add`
/// 防止恶意 / 异常计数的算术溢出。
fn build_token_usage(acc: &mut UsageAccumulator) -> Option<TokenUsage> {
    if acc.input_tokens.is_none()
        && acc.cache_creation_input_tokens.is_none()
        && acc.cache_read_input_tokens.is_none()
        && acc.output_tokens.is_none()
    {
        return None;
    }
    let raw_input = acc.input_tokens.unwrap_or(0);
    let cache_creation = acc.cache_creation_input_tokens.unwrap_or(0);
    let cache_read = acc.cache_read_input_tokens.unwrap_or(0);
    let output_tokens = acc.output_tokens.unwrap_or(0);
    let input_tokens = raw_input
        .saturating_add(cache_creation)
        .saturating_add(cache_read);
    Some(TokenUsage {
        input_tokens,
        cached_input_tokens: cache_read,
        output_tokens,
        reasoning_output_tokens: 0,
        total_tokens: input_tokens.saturating_add(output_tokens),
    })
}

/// 发送一条事件；返回 `true` 表示通道已关闭（调用方应结束）。
async fn send(
    tx_event: &mpsc::Sender<Result<ResponseEvent, ApiError>>,
    event: Result<ResponseEvent, ApiError>,
) -> bool {
    tx_event.send(event).await.is_err()
}
