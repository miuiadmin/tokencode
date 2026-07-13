//! Gemini generateContent 流式解析器（原生多协议支持）。
//!
//! 与 `sse::anthropic` 同构：消费一条 `StreamResponse`（HTTP 已是 2xx；非 2xx 由传输层归一为
//! `TransportError::Http`，401/重试由上层 core 处理），把 Gemini `streamGenerateContent?alt=sse`
//! 的 SSE 帧流翻译为 core 已消费的 `ResponseEvent` 词表，再经 `response_stream_to_unified` 归一为
//! `UnifiedEvent`。
//!
//! Gemini 与 Anthropic/Chat 的协议差异（本文件要弥合的）：
//! - **无 type 字段事件分发**：每帧是 `{candidates:[{content:{parts:[...],role}, finishReason?, index}],
//!   usageMetadata?}`——candidates 增量直出（text 片段累积 / functionCall 通常一帧完整），不像
//!   Anthropic 的 `content_block_start`/`content_block_delta` 两段式。
//! - **收尾靠 finishReason**（在 candidate 层，非 `[DONE]` / `message_stop`）；EOF 仍兜底冲刷。
//! - **usage 单帧**（usageMetadata，常在最后一帧）：`promptTokenCount` 是**含 cached 的超集**
//!   （与 Anthropic 的 `input_tokens` 非缓存子集不同——故本文件 input 直接取 prompt，无需再加 cache）。
//! - **思考流**（`includeThoughts:true`）：parts 里 `{thought:true, text}` → `ReasoningContentDelta`。

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
use serde_json::Value;
use serde_json::json;
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

/// 把一条 Gemini `streamGenerateContent` 的 `StreamResponse` 翻译为 `ResponseStream`。
///
/// 与 `spawn_anthropic_stream` 同构：先从响应头解析出协议无关的派生事件（ServerModel / RateLimits /
/// ModelsEtag，原生 Gemini 通常不带这些头，best-effort 兼容网关），再 spawn 任务跑 SSE 循环。
pub fn spawn_gemini_stream(
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
        process_gemini_sse(stream_response.bytes, tx_event, idle_timeout, telemetry).await;
    });

    ResponseStream {
        rx_event,
        upstream_request_id,
    }
}

// ===== 累积装配状态 ==========================================================

/// 单条工具调用的累积器（Gemini functionCall 通常一帧完整，但仍保留累积位以容错分片）。
#[derive(Debug, Default)]
struct ToolCallAccumulator {
    /// 工具调用 id（Gemini functionCall 无 id，按序生成 `call_{index}`，对齐内部追踪语义）。
    call_id: String,
    name: String,
    /// 已序列化的 args JSON 串（对齐 Responses `FunctionCall.arguments` 的 String 形态）。
    args: String,
}

/// usage 累积器：保留 Gemini usageMetadata 的各原始字段，超集语义在 `build_token_usage` 统一表达。
///
/// Gemini 的 `promptTokenCount` **已含 cached**（是输入超集，与 Anthropic 的 `input_tokens`
/// 非缓存子集不同），故 `build_token_usage` 直接取 prompt 为 `input_tokens`，无需再加 cache——
/// 这样仍满足 `TokenUsage` 不变量（`non_cached = input - cached`；protocol.rs）。
#[derive(Debug, Default)]
struct GeminiUsageAccumulator {
    /// 输入 token（含 cached 超集）。
    prompt_token_count: Option<i64>,
    /// 输出 token（不含思考）。
    candidates_token_count: Option<i64>,
    /// 思考 token（2.5 thinking 模型单列）。
    thoughts_token_count: Option<i64>,
    /// 命中缓存的输入子集。
    cached_content_token_count: Option<i64>,
    /// 总 token（通常 = prompt + candidates + thoughts）。
    total_token_count: Option<i64>,
}

/// 一个 assistant turn 的累积状态。
#[derive(Debug, Default)]
struct GeminiStreamState {
    /// 是否已发过 `Created`。
    created_emitted: bool,
    /// 是否已发过 `OutputItemAdded(Message)`（文本/思考增量前置需要 active item）。
    message_item_added: bool,
    assistant_text: String,
    /// functionCall part 按出现顺序累积（输出顺序确定）。
    tool_calls: Vec<ToolCallAccumulator>,
    /// Gemini generateContent 无显式 response id（常 None → Completed.response_id 为空串）。
    response_id: Option<String>,
    finish_reason: Option<String>,
    usage: GeminiUsageAccumulator,
}

// ===== SSE 字节循环 ==========================================================

async fn process_gemini_sse(
    stream: ByteStream,
    tx_event: mpsc::Sender<Result<ResponseEvent, ApiError>>,
    idle_timeout: Duration,
    telemetry: Option<Arc<dyn SseTelemetry>>,
) {
    let mut stream = stream.eventsource();
    let mut state = GeminiStreamState::default();

    loop {
        let start = Instant::now();
        let response = timeout(idle_timeout, stream.next()).await;
        if let Some(t) = telemetry.as_ref() {
            t.on_sse_poll(&response, start.elapsed());
        }
        let sse = match response {
            Ok(Some(Ok(sse))) => sse,
            Ok(Some(Err(e))) => {
                debug!("Gemini SSE 错误: {e:#}");
                let _ = tx_event.send(Err(ApiError::Stream(e.to_string()))).await;
                return;
            }
            Ok(None) => {
                // 流自然结束：尽力冲刷已累积内容后收尾（finishReason 已记录到 state）。
                flush(&mut state, &tx_event).await;
                return;
            }
            Err(_) => {
                let _ = tx_event
                    .send(Err(ApiError::Stream(
                        "idle timeout waiting for Gemini SSE".into(),
                    )))
                    .await;
                return;
            }
        };

        trace!("Gemini SSE 事件: {}", &sse.data);

        let frame: Value = match serde_json::from_str(&sse.data) {
            Ok(frame) => frame,
            Err(e) => {
                debug!("Gemini SSE 解析失败: {e}, data: {}", &sse.data);
                continue;
            }
        };

        // 流内错误帧（HTTP 已 2xx，但 body 报错——部分网关以 `{error:{message}}` 形态）。
        if let Some(message) = extract_stream_error(&frame) {
            let _ = tx_event.send(Err(ApiError::Stream(message))).await;
            return;
        }

        if !process_frame(&mut state, &frame, &tx_event).await {
            return;
        }
    }
}

/// 从一帧中提取流内错误消息（`{error:{message}}` 或 `{error:"..."}`）；无 error 字段返回 None。
fn extract_stream_error(frame: &Value) -> Option<String> {
    let err = frame.get("error")?;
    if let Some(msg) = err.get("message").and_then(|v| v.as_str()) {
        return Some(msg.to_string());
    }
    err.as_str().map(str::to_string)
}

/// 处理单帧：更新累积状态 + 发送增量事件。返回 `false` 表示消费端已关闭，调用方应结束任务。
async fn process_frame(
    state: &mut GeminiStreamState,
    frame: &Value,
    tx_event: &mpsc::Sender<Result<ResponseEvent, ApiError>>,
) -> bool {
    // Created 仅发一次（首帧即发——Gemini 无 message_start 等价事件，首帧到来即代表流已建立）。
    if !state.created_emitted {
        if send(tx_event, Ok(ResponseEvent::Created)).await {
            return false;
        }
        state.created_emitted = true;
    }

    // Gemini 无显式 response id；best-effort 从帧顶层 `id` 提取（兼容注入 id 的网关）。
    if state.response_id.is_none() {
        if let Some(id) = frame.get("id").and_then(|v| v.as_str()) {
            state.response_id = Some(id.to_string());
        }
    }

    // candidates[]：默认 candidateCount=1，但仍遍历全部（parts 累积到同一 state，假设单候选）。
    if let Some(candidates) = frame.get("candidates").and_then(|v| v.as_array()) {
        for cand in candidates {
            if let Some(fr) = cand.get("finishReason").and_then(|v| v.as_str()) {
                state.finish_reason = Some(fr.to_string());
            }
            if let Some(parts) = cand
                .get("content")
                .and_then(|c| c.get("parts"))
                .and_then(|p| p.as_array())
            {
                for part in parts {
                    if !process_part(state, part, tx_event).await {
                        return false;
                    }
                }
            }
        }
    }

    // usageMetadata（常在最后一帧，也可能独立成帧）。
    if let Some(usage) = frame.get("usageMetadata") {
        apply_usage(&mut state.usage, usage);
    }

    true
}

/// 处理单个 part：text（含 thought 标记）/ functionCall，发送对应增量事件或累积。
async fn process_part(
    state: &mut GeminiStreamState,
    part: &Value,
    tx_event: &mpsc::Sender<Result<ResponseEvent, ApiError>>,
) -> bool {
    // functionCall part：工具调用（通常一帧完整）。发 placeholder（OutputItemAdded）+ 累积完整 args。
    if let Some(function_call) = part.get("functionCall") {
        let name = function_call
            .get("name")
            .and_then(|v| v.as_str())
            .unwrap_or("")
            .to_string();
        let args_value = function_call.get("args").cloned().unwrap_or_else(|| json!({}));
        let args_str = serde_json::to_string(&args_value).unwrap_or_else(|_| "{}".to_string());
        let index = state.tool_calls.len() as i64;
        let call_id = format!("call_{index}");
        // Gemini wire 工具名为单字符串（functionCall.name，协议无 namespace 字段）；非 Responses
        // 协议下 harness 已把 namespace 工具展平成 `{ns}__{name}` 下发，这里还原。注意 `name`
        // 仍以 flat 形式存入累积器， flush 时再次还原，保证占位与收尾一致。
        let parsed = ToolName::from_flat_wire_name(&name);
        let placeholder = ResponseItem::FunctionCall {
            id: None,
            name: parsed.name,
            namespace: parsed.namespace,
            arguments: String::new(),
            call_id: call_id.clone(),
            internal_chat_message_metadata_passthrough: None,
        };
        if send(tx_event, Ok(ResponseEvent::OutputItemAdded(placeholder))).await {
            return false;
        }
        state.tool_calls.push(ToolCallAccumulator {
            call_id,
            name,
            args: args_str,
        });
        return true;
    }

    // text part：thought=true → ReasoningContentDelta；否则 → OutputTextDelta（累积）。
    let is_thought = part
        .get("thought")
        .and_then(|v| v.as_bool())
        .unwrap_or(false);
    if let Some(text) = part.get("text").and_then(|v| v.as_str())
        && !text.is_empty()
    {
        if !ensure_message_item(state, tx_event).await {
            return false;
        }
        if is_thought {
            // 思考增量：归一为 ReasoningContentDelta（与其他协议推理增量对齐）。
            if send(
                tx_event,
                Ok(ResponseEvent::ReasoningContentDelta {
                    delta: text.to_string(),
                    content_index: 0,
                }),
            )
            .await
            {
                return false;
            }
        } else {
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

    // inlineData / functionResponse 在响应 parts 中一般不出现（模型不输出图片；functionResponse
    // 是请求侧回喂），忽略。未知字段静默跳过（容错未来新 part 类型）。
    true
}

/// 把一帧 usageMetadata 合并进累积器（后者覆盖——Gemini usage 是快照式，最终值在末帧）。
fn apply_usage(acc: &mut GeminiUsageAccumulator, usage: &Value) {
    if let Some(v) = usage.get("promptTokenCount").and_then(|v| v.as_i64()) {
        acc.prompt_token_count = Some(v);
    }
    if let Some(v) = usage.get("candidatesTokenCount").and_then(|v| v.as_i64()) {
        acc.candidates_token_count = Some(v);
    }
    if let Some(v) = usage.get("thoughtsTokenCount").and_then(|v| v.as_i64()) {
        acc.thoughts_token_count = Some(v);
    }
    if let Some(v) = usage.get("cachedContentTokenCount").and_then(|v| v.as_i64()) {
        acc.cached_content_token_count = Some(v);
    }
    if let Some(v) = usage.get("totalTokenCount").and_then(|v| v.as_i64()) {
        acc.total_token_count = Some(v);
    }
}

/// 首次需要流式文本/思考增量时，发一次 `OutputItemAdded(Message)` 建立 active item。
///
/// core 的 `OutputTextDelta` / `ReasoningContentDelta` 处理要求 `active_item` 已存在，否则会 panic。
async fn ensure_message_item(
    state: &mut GeminiStreamState,
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

/// 在 EOF 一次性装配并发出收尾事件：`OutputItemDone(Message)` → `OutputItemDone(FunctionCall)`× → `Completed`。
async fn flush(
    state: &mut GeminiStreamState,
    tx_event: &mpsc::Sender<Result<ResponseEvent, ApiError>>,
) {
    // 1. assistant 文本消息收尾（仅当确实累积到文本；纯思考无文本时不发 Message）。
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

    // 2. 工具调用收尾（按出现顺序）。
    for tc in std::mem::take(&mut state.tool_calls) {
        if tc.name.is_empty() && tc.args.is_empty() {
            continue;
        }
        let parsed = ToolName::from_flat_wire_name(&tc.name);
        let call = ResponseItem::FunctionCall {
            id: None,
            name: parsed.name,
            namespace: parsed.namespace,
            arguments: tc.args,
            call_id: tc.call_id,
            internal_chat_message_metadata_passthrough: None,
        };
        if send(tx_event, Ok(ResponseEvent::OutputItemDone(call))).await {
            return;
        }
    }

    // 3. Completed：response_id（Gemini 无显式 id，缺省空串）、usage、end_turn。
    let end_turn = match state.finish_reason.as_deref() {
        // 自然结束（模型主动收尾）。
        Some("STOP") => Some(true),
        // 已知「未结束 / 非正常截断」：达上限、安全拦截、抄袭、其它、各类内容拦截。
        Some("MAX_TOKENS")
        | Some("SAFETY")
        | Some("RECITATION")
        | Some("OTHER")
        | Some("BLOCKLIST")
        | Some("PROHIBITED_CONTENT")
        | Some("SPII") => Some(false),
        // 未知 finishReason（如网关返回非标准值）或缺省：不臆断回合结束，交上层裁决。
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
/// `input_tokens` 直接取 `promptTokenCount`（Gemini 该字段已含 cached，是超集——满足
/// `non_cached_input = input - cached` 不变量）；`cached_input_tokens` 取 `cachedContentTokenCount`
/// （子集）；`reasoning_output_tokens` 取 `thoughtsTokenCount`（单列）；`total_tokens` 取
/// `totalTokenCount`（缺省则 input+output+reasoning 自算）。各项 `unwrap_or(0)` + 无溢出风险
/// （Gemini 计数已是绝对值，非累加）。
fn build_token_usage(acc: &mut GeminiUsageAccumulator) -> Option<TokenUsage> {
    if acc.prompt_token_count.is_none()
        && acc.candidates_token_count.is_none()
        && acc.thoughts_token_count.is_none()
        && acc.cached_content_token_count.is_none()
        && acc.total_token_count.is_none()
    {
        return None;
    }
    let input_tokens = acc.prompt_token_count.unwrap_or(0);
    let output_tokens = acc.candidates_token_count.unwrap_or(0);
    let reasoning_output_tokens = acc.thoughts_token_count.unwrap_or(0);
    let cached_input_tokens = acc.cached_content_token_count.unwrap_or(0);
    let total_tokens = acc.total_token_count.unwrap_or_else(|| {
        input_tokens
            .saturating_add(output_tokens)
            .saturating_add(reasoning_output_tokens)
    });
    Some(TokenUsage {
        input_tokens,
        cached_input_tokens,
        output_tokens,
        reasoning_output_tokens,
        total_tokens,
    })
}

/// 发送一条事件；返回 `true` 表示通道已关闭（调用方应结束）。
async fn send(
    tx_event: &mpsc::Sender<Result<ResponseEvent, ApiError>>,
    event: Result<ResponseEvent, ApiError>,
) -> bool {
    tx_event.send(event).await.is_err()
}
