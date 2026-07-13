//! Fixture 回放测试：验证 `AnthropicAdapter` 把 Anthropic Messages 的 SSE 事件流
//! 正确翻译为 core 已消费的 `UnifiedEvent` 序列。
//!
//! 三份捕获形态的 Anthropic SSE：① 纯文本完成（含 ping 心跳）；② 工具调用累积装配；
//! ③ 流内错误终止。再附一个请求翻译测试，验证 instructions→system、role 归一、
//! tool_use/tool_result 装配、max_tokens 默认等 wire 形态。脚手架（fixture 传输 /
//! 鉴权 / provider / body 构造）与 `chat_adapter_fixtures.rs` 同构。

#![allow(clippy::expect_used)]

use std::sync::Arc;
use std::sync::Mutex;
use std::time::Duration;

use bytes::Bytes;
use codex_api::AnthropicAdapter;
use codex_api::AnthropicApiRequest;
use codex_api::AuthProvider;
use codex_api::Provider;
use codex_api::RetryConfig;
use codex_client::HttpTransport;
use codex_client::Request;
use codex_client::RequestBody;
use codex_client::Response;
use codex_client::StreamResponse;
use codex_client::TransportError;
use codex_language_model::LanguageModel;
use codex_language_model::UnifiedError;
use codex_language_model::UnifiedEvent;
use codex_language_model::UnifiedEventStream;
use codex_language_model::UnifiedReasoning;
use codex_language_model::UnifiedRequest;
use codex_language_model::UnifiedRequestOptions;
use codex_language_model::UnifiedTextControls;
use codex_language_model::UnifiedTextFormat;
use codex_language_model::UnifiedTextFormatType;
use codex_protocol::models::ContentItem;
use codex_protocol::models::ReasoningItemContent;
use codex_protocol::models::ResponseItem;
use codex_protocol::openai_models::ReasoningEffort;
use futures::StreamExt;
use http::HeaderMap;
use http::StatusCode;
use serde_json::Value;
use serde_json::json;

// --- 与 chat_adapter_fixtures.rs 同构的 fixture 传输 / 鉴权 / provider --------

#[derive(Clone)]
struct FixtureSseTransport {
    body: String,
}

impl FixtureSseTransport {
    fn new(body: String) -> Self {
        Self { body }
    }
}

impl HttpTransport for FixtureSseTransport {
    async fn execute(&self, _req: Request) -> Result<Response, TransportError> {
        Err(TransportError::Build("execute should not run".to_string()))
    }

    async fn stream(&self, _req: Request) -> Result<StreamResponse, TransportError> {
        let stream = futures::stream::iter(vec![Ok::<Bytes, TransportError>(Bytes::from(
            self.body.clone(),
        ))]);
        Ok(StreamResponse {
            status: StatusCode::OK,
            headers: HeaderMap::new(),
            bytes: Box::pin(stream),
        })
    }
}

/// 捕获发往传输的请求体（wire JSON），用于断言 adapter 注入的字段（如 max_tokens）。
/// 同时回放一段固定 SSE body，保证 stream() 不报错。
#[derive(Clone)]
struct CapturingTransport {
    body: String,
    captured: Arc<Mutex<Option<Value>>>,
}

impl CapturingTransport {
    fn new(body: String, captured: Arc<Mutex<Option<Value>>>) -> Self {
        Self { body, captured }
    }
}

impl HttpTransport for CapturingTransport {
    async fn execute(&self, _req: Request) -> Result<Response, TransportError> {
        Err(TransportError::Build("execute should not run".to_string()))
    }

    async fn stream(&self, req: Request) -> Result<StreamResponse, TransportError> {
        // 反序列化请求体（兼容 Json / EncodedJson / Raw 三种形态）。
        let json = match req.body.as_ref() {
            Some(RequestBody::Json(v)) => Some(v.clone()),
            Some(RequestBody::EncodedJson(e)) => serde_json::from_slice(e.as_bytes()).ok(),
            Some(RequestBody::Raw(b)) => serde_json::from_slice(b).ok(),
            None => None,
        };
        if let Some(v) = json {
            *self.captured.lock().expect("capture mutex poisoned") = Some(v);
        }
        let stream = futures::stream::iter(vec![Ok::<Bytes, TransportError>(Bytes::from(
            self.body.clone(),
        ))]);
        Ok(StreamResponse {
            status: StatusCode::OK,
            headers: HeaderMap::new(),
            bytes: Box::pin(stream),
        })
    }
}

#[derive(Clone, Default)]
struct NoAuth;

impl AuthProvider for NoAuth {
    fn add_auth_headers(&self, _headers: &mut HeaderMap) {}
}

fn provider() -> Provider {
    Provider {
        name: "anthropic".to_string(),
        base_url: "https://example.com/v1".to_string(),
        query_params: None,
        headers: HeaderMap::new(),
        retry: RetryConfig {
            max_attempts: 1,
            base_delay: Duration::from_millis(1),
            retry_429: false,
            retry_5xx: false,
            retry_transport: true,
        },
        stream_idle_timeout: Duration::from_millis(50),
        max_output_tokens: None,
    }
}

/// 把一组 Anthropic 事件（JSON）拼成 SSE body（每帧仅 `data:` 行，解析侧按 JSON `type` 分派）。
fn build_anthropic_body(events: &[Value]) -> String {
    let mut body = String::new();
    for event in events {
        body.push_str(&format!("data: {event}\n\n"));
    }
    body
}

// --- 中立请求样本 ----------------------------------------------------------

fn sample_unified_request() -> UnifiedRequest {
    UnifiedRequest {
        model: "claude-3-5-sonnet".to_string(),
        instructions: "你是一个助手。".to_string(),
        input: vec![ResponseItem::Message {
            id: None,
            role: "user".to_string(),
            content: vec![ContentItem::InputText {
                text: "你好".to_string(),
            }],
            phase: None,
            internal_chat_message_metadata_passthrough: None,
        }],
        tools: None,
        tool_choice: "auto".to_string(),
        parallel_tool_calls: false,
        reasoning: None,
        store: false,
        stream: true,
        include: vec![],
        service_tier: None,
        prompt_cache_key: None,
        text: None,
        client_metadata: None,
    }
}

/// 在 Anthropic wire 请求体里找首个 `tool_result` content 块（跨 messages[].content[] 扁平）。
/// 两个 is_error 映射测试共用此提取，避免逐字重复 wire 形态遍历逻辑（wire 形态若变更只改一处）。
fn find_tool_result_block(json: &serde_json::Value) -> serde_json::Value {
    json.get("messages")
        .and_then(|m| m.as_array())
        .expect("应有 messages")
        .iter()
        .flat_map(|m| {
            m.get("content")
                .and_then(|c| c.as_array())
                .cloned()
                .unwrap_or_default()
        })
        .find(|b| b.get("type").and_then(|v| v.as_str()) == Some("tool_result"))
        .expect("应有 tool_result 块")
}

fn sample_unified_options() -> UnifiedRequestOptions {
    UnifiedRequestOptions {
        session_id: None,
        thread_id: None,
        session_source: None,
        extra_headers: HeaderMap::new(),
        compression: Default::default(),
        turn_state: None,
    }
}

// --- 流收集：返回 (事件序列, 终止错误串) ------------------------------------

async fn drain(s: UnifiedEventStream) -> (Vec<UnifiedEvent>, Option<String>) {
    let mut s = s;
    let mut evs = Vec::new();
    let mut err = None;
    while let Some(item) = s.next().await {
        match item {
            Ok(ev) => {
                // RateLimits 源自 HTTP 头（fixture 无相关头），过滤掉保证可比。
                if !matches!(ev, UnifiedEvent::RateLimits(_)) {
                    evs.push(ev);
                }
            }
            Err(e) => {
                err = Some(match e {
                    UnifiedError::Passthrough(boxed) => boxed.to_string(),
                    UnifiedError::Mapping(m) => format!("Mapping({m})"),
                });
                break;
            }
        }
    }
    (evs, err)
}

/// 断言事件序列中存在一条 assistant 文本 `OutputItemDone(Message)`，且文本 == 期望。
fn expect_assistant_text(evs: &[UnifiedEvent], expected: &str) {
    let found = evs.iter().any(|ev| match ev {
        UnifiedEvent::OutputItemDone(ResponseItem::Message { content, .. }) => {
            content.iter().any(|c| match c {
                ContentItem::OutputText { text } => text == expected,
                _ => false,
            })
        }
        _ => false,
    });
    assert!(found, "未找到文本为 {expected:?} 的 assistant OutputItemDone；事件序列: {evs:?}");
}

/// 断言事件序列中存在一条 `OutputItemDone(FunctionCall)`，name/call_id/arguments 符合期望。
fn expect_function_call(evs: &[UnifiedEvent], name: &str, call_id: &str, arguments: &str) {
    let found = evs.iter().any(|ev| match ev {
        UnifiedEvent::OutputItemDone(ResponseItem::FunctionCall {
            name: n,
            call_id: c,
            arguments: a,
            ..
        }) => n == name && c == call_id && a == arguments,
        _ => false,
    });
    assert!(
        found,
        "未找到 FunctionCall(name={name:?}, call_id={call_id:?}, arguments={arguments:?})；事件序列: {evs:?}"
    );
}

// --- 驱动 ------------------------------------------------------------------

/// 驱动 adapter 跑给定 Anthropic SSE body，返回事件序列 + 错误。
async fn run(body: String) -> (Vec<UnifiedEvent>, Option<String>) {
    let adapter = AnthropicAdapter::new(
        FixtureSseTransport::new(body),
        provider(),
        Arc::new(NoAuth),
    );
    let stream = LanguageModel::stream(
        &adapter,
        sample_unified_request(),
        sample_unified_options(),
    )
    .await
    .expect("adapter stream 不应失败");
    drain(stream).await
}

// --- 三份 fixture -----------------------------------------------------------

#[tokio::test]
async fn fixture_text_completion() {
    // 标准纯文本完成：message_start（usage 起算）→ text 块 → 文本增量 → ping 心跳
    // → content_block_stop → message_delta（stop_reason:end_turn + output tokens）→ message_stop。
    let events = vec![
        json!({
            "type": "message_start",
            "message": {
                "id": "msg_1",
                "model": "claude-3-5-sonnet",
                "role": "assistant",
                "content": [],
                "usage": {"input_tokens": 10, "output_tokens": 1}
            }
        }),
        json!({"type": "content_block_start", "index": 0, "content_block": {"type": "text", "text": ""}}),
        json!({"type": "content_block_delta", "index": 0, "delta": {"type": "text_delta", "text": "Hello"}}),
        json!({"type": "ping"}),
        json!({"type": "content_block_delta", "index": 0, "delta": {"type": "text_delta", "text": ", world"}}),
        json!({"type": "content_block_stop", "index": 0}),
        json!({
            "type": "message_delta",
            "delta": {"stop_reason": "end_turn", "stop_sequence": null},
            "usage": {"output_tokens": 3}
        }),
        json!({"type": "message_stop"}),
    ];
    let (evs, err) = run(build_anthropic_body(&events)).await;

    assert!(err.is_none(), "纯文本完成不应有错误：{err:?}");
    // 首事件 Created；末事件 Completed(end_turn=Some(true), token_usage 非空)。
    assert!(matches!(evs.first(), Some(UnifiedEvent::Created)), "首事件应为 Created: {evs:?}");
    match evs.iter().find(|ev| matches!(ev, UnifiedEvent::Completed { .. })) {
        Some(UnifiedEvent::Completed {
            end_turn, token_usage, ..
        }) => {
            assert_eq!(*end_turn, Some(true), "stop_reason=end_turn → end_turn=Some(true)");
            let usage = token_usage
                .as_ref()
                .expect("Completed 应携带 token_usage");
            // input 来自 message_start；output 由 message_delta 覆盖为 3；total = input + output。
            assert_eq!(usage.input_tokens, 10);
            assert_eq!(usage.output_tokens, 3);
            assert_eq!(usage.total_tokens, 13);
        }
        other => panic!("应存在 Completed 事件，实际: {other:?}"),
    }
    // 文本增量逐条到达（ping 不产出任何事件）。
    let deltas: Vec<&str> = evs
        .iter()
        .filter_map(|ev| match ev {
            UnifiedEvent::OutputTextDelta(s) => Some(s.as_str()),
            _ => None,
        })
        .collect();
    assert_eq!(deltas, vec!["Hello", ", world"], "文本增量序列: {evs:?}");
    // 装配出完整 assistant 文本。
    expect_assistant_text(&evs, "Hello, world");
    // OutputItemAdded(Message) 必须先于 OutputItemDone（core 要求 active item）。
    let added = evs
        .iter()
        .position(|ev| matches!(ev, UnifiedEvent::OutputItemAdded(ResponseItem::Message { .. })));
    let done = evs
        .iter()
        .position(|ev| matches!(ev, UnifiedEvent::OutputItemDone(ResponseItem::Message { .. })));
    assert!(added.is_some() && done.is_some(), "应同时有 Message 的 Added/Done: {evs:?}");
    assert!(added.unwrap() < done.unwrap(), "Message Added 必须先于 Done");
}

#[tokio::test]
async fn fixture_tool_call_accumulation() {
    // 工具调用：message_start → tool_use 块（id+name+空 input）→ input_json_delta 分片
    // → content_block_stop → message_delta（stop_reason:tool_use）→ message_stop。
    let events = vec![
        json!({
            "type": "message_start",
            "message": {
                "id": "msg_2",
                "model": "claude-3-5-sonnet",
                "usage": {"input_tokens": 20, "output_tokens": 1}
            }
        }),
        json!({
            "type": "content_block_start",
            "index": 0,
            "content_block": {"type": "tool_use", "id": "toolu_1", "name": "get_weather", "input": {}}
        }),
        json!({"type": "content_block_delta", "index": 0, "delta": {"type": "input_json_delta", "partial_json": "{\"city\":"}}),
        json!({"type": "content_block_delta", "index": 0, "delta": {"type": "input_json_delta", "partial_json": "\"SF\"}"}}),
        json!({"type": "content_block_stop", "index": 0}),
        json!({
            "type": "message_delta",
            "delta": {"stop_reason": "tool_use", "stop_sequence": null},
            "usage": {"output_tokens": 5}
        }),
        json!({"type": "message_stop"}),
    ];
    let (evs, err) = run(build_anthropic_body(&events)).await;

    assert!(err.is_none(), "工具调用完成不应有错误：{err:?}");
    // tool_use 块触发 OutputItemAdded(FunctionCall) 占位（call_id = tool_use.id）。
    assert!(
        evs.iter().any(|ev| matches!(
            ev,
            UnifiedEvent::OutputItemAdded(ResponseItem::FunctionCall { name, call_id, .. })
                if name == "get_weather" && call_id == "toolu_1"
        )),
        "应有 FunctionCall 的 OutputItemAdded 占位: {evs:?}"
    );
    // 装配出完整 FunctionCall（input_json_delta 分片已累积为 arguments）。
    expect_function_call(&evs, "get_weather", "toolu_1", r#"{"city":"SF"}"#);
    // stop_reason=tool_use → end_turn=Some(false)（turn 待工具结果继续）。
    match evs.iter().find(|ev| matches!(ev, UnifiedEvent::Completed { .. })) {
        Some(UnifiedEvent::Completed { end_turn, .. }) => {
            assert_eq!(*end_turn, Some(false), "stop_reason=tool_use → end_turn=Some(false)");
        }
        other => panic!("应存在 Completed 事件，实际: {other:?}"),
    }
}

#[tokio::test]
async fn fixture_in_stream_error_terminates() {
    // 流内错误帧（HTTP 2xx，body 为 {type:error, error:{message}}）：应终止并产出错误。
    let events = vec![json!({
        "type": "error",
        "error": {"type": "overloaded_error", "message": "rate limited"}
    })];
    let (evs, err) = run(build_anthropic_body(&events)).await;

    assert!(evs.is_empty(), "错误帧不应产出任何 UnifiedEvent: {evs:?}");
    let err = err.expect("应终止于错误");
    assert!(
        err.contains("rate limited"),
        "错误信息应透传 'rate limited'，实际: {err}"
    );
}

// --- 请求翻译（wire 形态）--------------------------------------------------

#[test]
fn request_translation_shapes_anthropic_wire() {
    // instructions → 顶层 system；user 文本 → messages[].content[].text；
    // assistant FunctionCall → tool_use 块；user FunctionCallOutput → tool_result 块。
    let request = UnifiedRequest {
        model: "claude-3-5-sonnet".to_string(),
        instructions: "系统级指令".to_string(),
        input: vec![
            ResponseItem::Message {
                id: None,
                role: "user".to_string(),
                content: vec![ContentItem::InputText {
                    text: "查天气".to_string(),
                }],
                phase: None,
                internal_chat_message_metadata_passthrough: None,
            },
            ResponseItem::FunctionCall {
                id: None,
                name: "get_weather".to_string(),
                namespace: None,
                arguments: r#"{"city":"SF"}"#.to_string(),
                call_id: "toolu_1".to_string(),
                internal_chat_message_metadata_passthrough: None,
            },
            ResponseItem::FunctionCallOutput {
                id: None,
                call_id: "toolu_1".to_string(),
                output: codex_protocol::models::FunctionCallOutputPayload {
                    body: codex_protocol::models::FunctionCallOutputBody::Text("晴天".to_string()),
                    success: None,
                },
                internal_chat_message_metadata_passthrough: None,
            },
        ],
        tools: None,
        tool_choice: "auto".to_string(),
        parallel_tool_calls: false,
        reasoning: None,
        store: false,
        stream: true,
        include: vec![],
        service_tier: None,
        prompt_cache_key: None,
        text: None,
        client_metadata: None,
    };

    let api_request: AnthropicApiRequest = request.into();
    let json = serde_json::to_value(&api_request).expect("AnthropicApiRequest 应可序列化");

    // 必填 max_tokens 走常量默认（16384：内置 Claude 模型均支持）。
    assert_eq!(
        json.get("max_tokens").and_then(|v| v.as_u64()),
        Some(16384),
        "max_tokens 应为常量默认 16384"
    );
    // stream 透传。
    assert_eq!(json.get("stream").and_then(|v| v.as_bool()), Some(true));

    // instructions → 顶层 system（数组形态 [{type:text, text}]），不出现在 messages 里。
    let system = json
        .get("system")
        .and_then(|v| v.as_array())
        .expect("应有顶层 system 数组");
    assert_eq!(system.len(), 1);
    assert_eq!(
        system[0].get("type").and_then(|v| v.as_str()),
        Some("text")
    );
    assert_eq!(
        system[0].get("text").and_then(|v| v.as_str()),
        Some("系统级指令")
    );

    // messages：user(文本) → assistant(tool_use) → user(tool_result)，三条。
    let messages = json
        .get("messages")
        .and_then(|m| m.as_array())
        .expect("应有 messages 数组");
    assert_eq!(messages.len(), 3, "消息序列: {messages:?}");

    // ① user 文本块。
    let (role, block_type, text) = block_at(messages, 0, 0);
    assert_eq!(role, "user");
    assert_eq!(block_type, "text");
    assert_eq!(text.as_deref(), Some("查天气"));

    // ② assistant tool_use 块（id=call_id, name, input 为解析后对象）。
    let msg1 = &messages[1];
    assert_eq!(msg1.get("role").and_then(|v| v.as_str()), Some("assistant"));
    let tool_use = &msg1.get("content").and_then(|c| c.as_array()).unwrap()[0];
    assert_eq!(
        tool_use.get("type").and_then(|v| v.as_str()),
        Some("tool_use")
    );
    assert_eq!(
        tool_use.get("id").and_then(|v| v.as_str()),
        Some("toolu_1")
    );
    assert_eq!(
        tool_use.get("name").and_then(|v| v.as_str()),
        Some("get_weather")
    );
    // arguments JSON 串解析为对象 {"city":"SF"}。
    assert_eq!(
        tool_use.get("input").and_then(|v| v.get("city")).and_then(|v| v.as_str()),
        Some("SF")
    );

    // ③ user tool_result 块（tool_use_id 对齐被回应的 tool_use.id，content 为文本）。
    let (role, block_type, content) = block_at(messages, 2, 0);
    assert_eq!(role, "user");
    assert_eq!(block_type, "tool_result");
    assert_eq!(content.as_deref(), Some("晴天"));
    // tool_use_id 字段对齐。
    let tool_result = &messages[2].get("content").and_then(|c| c.as_array()).unwrap()[0];
    assert_eq!(
        tool_result.get("tool_use_id").and_then(|v| v.as_str()),
        Some("toolu_1")
    );
}

#[test]
fn request_translation_marks_prompt_cache_breakpoints() {
    // prompt 缓存断点：非空 system + 非空 tools 时，system 末块与 tools 末工具各带
    // cache_control:{type:"ephemeral"}；对话消息（tool_use / tool_result 块）不带。
    let request = UnifiedRequest {
        model: "claude-sonnet-5".to_string(),
        instructions: "系统级指令".to_string(),
        input: vec![
            ResponseItem::Message {
                id: None,
                role: "user".to_string(),
                content: vec![ContentItem::InputText {
                    text: "调用工具".to_string(),
                }],
                phase: None,
                internal_chat_message_metadata_passthrough: None,
            },
            ResponseItem::FunctionCall {
                id: None,
                name: "get_weather".to_string(),
                namespace: None,
                arguments: r#"{"city":"SF"}"#.to_string(),
                call_id: "toolu_1".to_string(),
                internal_chat_message_metadata_passthrough: None,
            },
            ResponseItem::FunctionCallOutput {
                id: None,
                call_id: "toolu_1".to_string(),
                output: codex_protocol::models::FunctionCallOutputPayload {
                    body: codex_protocol::models::FunctionCallOutputBody::Text("晴天".to_string()),
                    success: None,
                },
                internal_chat_message_metadata_passthrough: None,
            },
        ],
        tools: Some(vec![
            json!({"type":"function","name":"get_weather","parameters":{"type":"object","properties":{"city":{"type":"string"}}}}),
            json!({"type":"function","name":"get_time","parameters":{"type":"object"}}),
        ]),
        tool_choice: "auto".to_string(),
        parallel_tool_calls: false,
        reasoning: None,
        store: false,
        stream: true,
        include: vec![],
        service_tier: None,
        prompt_cache_key: None,
        text: None,
        client_metadata: None,
    };

    let api_request: AnthropicApiRequest = request.into();
    let json = serde_json::to_value(&api_request).expect("AnthropicApiRequest 应可序列化");

    // system 末块（此处仅 1 块）带 cache_control:{type:"ephemeral"}。
    let system = json
        .get("system")
        .and_then(|v| v.as_array())
        .expect("应有 system 数组");
    assert_eq!(system.len(), 1);
    assert_eq!(
        system[0]
            .get("cache_control")
            .and_then(|c| c.get("type"))
            .and_then(|v| v.as_str()),
        Some("ephemeral"),
        "system 末块应带 cache_control ephemeral"
    );

    // tools：仅末工具带 cache_control，其余工具不带（中间断点无意义且消耗 ≤4 断点额度）。
    let tools = json
        .get("tools")
        .and_then(|v| v.as_array())
        .expect("应有 tools 数组");
    assert_eq!(tools.len(), 2);
    assert!(
        tools[0].get("cache_control").is_none(),
        "非末工具不应带 cache_control"
    );
    assert_eq!(
        tools[1]
            .get("cache_control")
            .and_then(|c| c.get("type"))
            .and_then(|v| v.as_str()),
        Some("ephemeral"),
        "tools 末工具应带 cache_control ephemeral"
    );

    // 对话消息（tool_use / tool_result 块）一律不带 cache_control：每 turn 变动，
    // 标了反而 bust 缓存 + 白付 1.25x 写惩罚。
    let messages = json
        .get("messages")
        .and_then(|m| m.as_array())
        .expect("应有 messages 数组");
    for (mi, msg) in messages.iter().enumerate() {
        let blocks = msg
            .get("content")
            .and_then(|c| c.as_array())
            .expect("content 应为数组");
        for (bi, block) in blocks.iter().enumerate() {
            assert!(
                block.get("cache_control").is_none(),
                "messages[{mi}].content[{bi}] 不应带 cache_control"
            );
        }
    }
}

/// 取 messages[msg_idx].content[block_idx] 的 (role, block.type, 文本类字段的字符串值)。
fn block_at(messages: &[Value], msg_idx: usize, block_idx: usize) -> (&str, &str, Option<String>) {
    let msg = &messages[msg_idx];
    let role = msg.get("role").and_then(|v| v.as_str()).unwrap_or("");
    let block = msg
        .get("content")
        .and_then(|c| c.as_array())
        .expect("content 应为数组")
        .get(block_idx)
        .expect("应有对应内容块");
    let block_type = block.get("type").and_then(|v| v.as_str()).unwrap_or("");
    let text = block
        .get("text")
        .and_then(|v| v.as_str())
        .map(String::from)
        .or_else(|| block.get("content").and_then(|v| v.as_str()).map(String::from));
    (role, block_type, text)
}

/// Anthropic 把非 user/assistant 角色（如 developer）从 messages 中丢弃，避免非法角色；
/// system 提示一律经顶层 instructions → system 承载。
#[test]
fn request_translation_drops_non_user_assistant_roles() {
    let request = UnifiedRequest {
        model: "claude-3-5-sonnet".to_string(),
        instructions: String::new(),
        input: vec![ResponseItem::Message {
            id: None,
            role: "developer".to_string(),
            content: vec![ContentItem::InputText {
                text: "开发者指令".to_string(),
            }],
            phase: None,
            internal_chat_message_metadata_passthrough: None,
        }],
        tools: None,
        tool_choice: String::new(),
        parallel_tool_calls: false,
        reasoning: None,
        store: false,
        stream: true,
        include: vec![],
        service_tier: None,
        prompt_cache_key: None,
        text: None,
        client_metadata: None,
    };

    let api_request: AnthropicApiRequest = request.into();
    let json = serde_json::to_value(&api_request).expect("应可序列化");

    // developer 消息被丢弃 → messages 为空；system 因 instructions 为空也不出现。
    let messages = json
        .get("messages")
        .and_then(|m| m.as_array())
        .expect("应有 messages 数组");
    assert!(messages.is_empty(), "developer 角色应被丢弃，messages 应为空: {messages:?}");
    assert!(
        json.get("system").is_none(),
        "instructions 为空时不应出现 system 字段"
    );
}

// --- 回归（code-review 修复）------------------------------------------------

/// 回归 A1：开启 prompt caching 时，`message_start.usage` 同时报 `input_tokens`（非缓存）、
/// `cache_creation_input_tokens`、`cache_read_input_tokens`。`TokenUsage.input_tokens` 须为
/// 「含缓存超集」（非缓存 + cache_creation + cache_read），`cached_input_tokens` = cache_read，
/// 否则破坏 `non_cached_input = input - cached` 不变量（protocol.rs）——之前直接透传原始
/// `input_tokens`（非缓存计数）会令 total 偏小、`non_cached_input` 反算溢出。
#[tokio::test]
async fn fixture_usage_counts_prompt_cache_as_superset() {
    let events = vec![
        json!({
            "type": "message_start",
            "message": {
                "id": "msg_c",
                "model": "claude-3-5-sonnet",
                "usage": {
                    "input_tokens": 10,
                    "cache_creation_input_tokens": 200,
                    "cache_read_input_tokens": 5000,
                    "output_tokens": 1
                }
            }
        }),
        json!({"type": "content_block_start", "index": 0, "content_block": {"type": "text", "text": ""}}),
        json!({"type": "content_block_delta", "index": 0, "delta": {"type": "text_delta", "text": "ok"}}),
        json!({"type": "content_block_stop", "index": 0}),
        json!({
            "type": "message_delta",
            "delta": {"stop_reason": "end_turn", "stop_sequence": null},
            "usage": {"output_tokens": 3}
        }),
        json!({"type": "message_stop"}),
    ];
    let (evs, err) = run(build_anthropic_body(&events)).await;
    assert!(err.is_none(), "缓存 usage 不应有错误：{err:?}");
    match evs.iter().find(|ev| matches!(ev, UnifiedEvent::Completed { .. })) {
        Some(UnifiedEvent::Completed { token_usage, .. }) => {
            let usage = token_usage.as_ref().expect("应有 token_usage");
            // 超集 = 10（非缓存）+ 200（cache_creation）+ 5000（cache_read）= 5210。
            assert_eq!(usage.input_tokens, 5210, "input_tokens 应为含缓存超集");
            // cached_input_tokens = cache_read（不变量中的缓存子集）。
            assert_eq!(usage.cached_input_tokens, 5000);
            assert_eq!(usage.output_tokens, 3);
            // total = 超集 input + output。
            assert_eq!(usage.total_tokens, 5213, "total = 超集 input + output");
        }
        other => panic!("应存在 Completed 事件，实际: {other:?}"),
    }
}

/// 回归 wire 形态（code-review 多条）：
/// ① 无 tools 时即便 tool_choice="auto" 也不发 tool_choice（Anthropic 无 tools 时拒绝 tool_choice）；
/// ② 有 tools + auto + parallel=false → {type:auto, disable_parallel_tool_use:true}；
/// ③ 有 tools + auto + parallel=true → 不带 disable_parallel_tool_use（保持默认并行）；
/// ④ 有 tools + none → {type:none}（不带 disable_parallel）；
/// ⑤ 工具失败 success=Some(false) → tool_result 带 is_error:true（之前丢失该标志）。
#[test]
fn request_translation_tool_choice_parallel_and_is_error_shapes() {
    let tool = json!({"type":"function","name":"get_weather","parameters":{"type":"object","properties":{}}});

    // ① 无 tools → 不发 tool_choice（即便 tool_choice=auto）。
    let req = sample_unified_request();
    let api: AnthropicApiRequest = req.into();
    let json = serde_json::to_value(&api).expect("可序列化");
    assert!(
        json.get("tool_choice").is_none(),
        "无 tools 时不应发 tool_choice（即便 tool_choice=auto）"
    );

    // ② 有 tools + auto + parallel=false → disable_parallel_tool_use:true。
    let mut req = sample_unified_request();
    req.tools = Some(vec![tool.clone()]);
    req.tool_choice = "auto".to_string();
    req.parallel_tool_calls = false;
    let api: AnthropicApiRequest = req.into();
    let json = serde_json::to_value(&api).expect("可序列化");
    let tc = json.get("tool_choice").expect("有 tools + auto 应发 tool_choice");
    assert_eq!(tc.get("type").and_then(|v| v.as_str()), Some("auto"));
    assert_eq!(
        tc.get("disable_parallel_tool_use").and_then(|v| v.as_bool()),
        Some(true),
        "parallel=false → disable_parallel_tool_use=true"
    );

    // ③ 有 tools + auto + parallel=true → 不带 disable_parallel_tool_use。
    let mut req = sample_unified_request();
    req.tools = Some(vec![tool.clone()]);
    req.tool_choice = "auto".to_string();
    req.parallel_tool_calls = true;
    let api: AnthropicApiRequest = req.into();
    let json = serde_json::to_value(&api).expect("可序列化");
    let tc = json.get("tool_choice").expect("有 tools + auto 应发 tool_choice");
    assert_eq!(tc.get("type").and_then(|v| v.as_str()), Some("auto"));
    assert!(
        tc.get("disable_parallel_tool_use").is_none(),
        "parallel=true 时不应带 disable_parallel_tool_use"
    );

    // ④ 有 tools + none → {type:none}，无 disable_parallel。
    let mut req = sample_unified_request();
    req.tools = Some(vec![tool]);
    req.tool_choice = "none".to_string();
    req.parallel_tool_calls = false;
    let api: AnthropicApiRequest = req.into();
    let json = serde_json::to_value(&api).expect("可序列化");
    let tc = json.get("tool_choice").expect("none 应显式发出 {type:none}");
    assert_eq!(tc.get("type").and_then(|v| v.as_str()), Some("none"));
    assert!(
        tc.get("disable_parallel_tool_use").is_none(),
        "none 时不应带 disable_parallel_tool_use"
    );

    // ⑤ 工具失败 success=Some(false) → tool_result.is_error=true。
    let mut req = sample_unified_request();
    req.input.push(ResponseItem::FunctionCallOutput {
        id: None,
        call_id: "toolu_1".to_string(),
        output: codex_protocol::models::FunctionCallOutputPayload {
            body: codex_protocol::models::FunctionCallOutputBody::Text("出错了".to_string()),
            success: Some(false),
        },
        internal_chat_message_metadata_passthrough: None,
    });
    let api: AnthropicApiRequest = req.into();
    let json = serde_json::to_value(&api).expect("可序列化");
    let tool_result = find_tool_result_block(&json);
    assert_eq!(
        tool_result.get("is_error").and_then(|v| v.as_bool()),
        Some(true),
        "success=Some(false) → is_error=true"
    );
}

/// 回归 is_error 映射的三个状态（扫尾补缺：原测试只覆盖 success=Some(false) 一个方向，缺成功与
/// 未表态两个分支）：
/// - `success=Some(false)`（失败）→ `is_error=Some(true)`；
/// - `success=Some(true)`（成功）→ `is_error=Some(false)`（取反，非同义透传）；
/// - `success=None`（未表态）→ `is_error` 字段省略（`skip_serializing_if`，默认成功）。
#[test]
fn request_translation_tool_result_is_error_covers_all_success_states() {
    fn translate(success: Option<bool>) -> serde_json::Value {
        let mut req = sample_unified_request();
        req.input.push(ResponseItem::FunctionCallOutput {
            id: None,
            call_id: "toolu_1".to_string(),
            output: codex_protocol::models::FunctionCallOutputPayload {
                body: codex_protocol::models::FunctionCallOutputBody::Text("x".to_string()),
                success,
            },
            internal_chat_message_metadata_passthrough: None,
        });
        let api: AnthropicApiRequest = req.into();
        serde_json::to_value(&api).expect("可序列化")
    }

    // 失败 → is_error=true。
    assert_eq!(
        find_tool_result_block(&translate(Some(false)))
            .get("is_error")
            .and_then(|v| v.as_bool()),
        Some(true),
        "success=Some(false) → is_error=true"
    );
    // 成功 → is_error=false（取反，验证不是同义透传）。
    assert_eq!(
        find_tool_result_block(&translate(Some(true)))
            .get("is_error")
            .and_then(|v| v.as_bool()),
        Some(false),
        "success=Some(true) → is_error=false"
    );
    // 未表态 → is_error 字段省略。
    assert!(
        find_tool_result_block(&translate(None))
            .get("is_error")
            .is_none(),
        "success=None 时 is_error 不应序列化"
    );
}

/// 回归空 messages 守卫：input 仅含被丢弃的 developer 角色 → 翻译后 messages 为空；adapter 应
/// 在发请求前明确失败（而非把空 messages 送到 Anthropic 拿 400）。
#[tokio::test]
async fn stream_rejects_empty_messages() {
    let mut request = sample_unified_request();
    request.instructions = String::new();
    request.input = vec![ResponseItem::Message {
        id: None,
        role: "developer".to_string(),
        content: vec![ContentItem::InputText {
            text: "开发者指令".to_string(),
        }],
        phase: None,
        internal_chat_message_metadata_passthrough: None,
    }];

    let adapter = AnthropicAdapter::new(
        FixtureSseTransport::new(String::new()),
        provider(),
        Arc::new(NoAuth),
    );
    let result = LanguageModel::stream(&adapter, request, sample_unified_options()).await;
    let err = match result {
        Ok(_) => panic!("空 messages 应在客户端被拒绝，但 stream 成功了"),
        Err(e) => e,
    };
    let msg = match err {
        UnifiedError::Passthrough(boxed) => boxed.to_string(),
        UnifiedError::Mapping(m) => m,
    };
    assert!(
        msg.contains("无任何 message"),
        "错误应说明无 messages：{msg}"
    );
}

#[tokio::test]
async fn provider_max_output_tokens_drives_wire_max_tokens() {
    // 经 CapturingTransport 捕获 adapter 发出的 wire 请求体，断言 max_tokens：
    //   provider.max_output_tokens = None  → 沿用 From 写入的 16384 默认；
    //   provider.max_output_tokens = Some  → 覆盖为配置值。
    for (configured, expected) in [(None, 16_384u64), (Some(8_192u32), 8_192)] {
        let captured = Arc::new(Mutex::new(None::<Value>));
        // 最小可完成的 SSE，仅供 stream() 跑通；本用例只断言请求体。
        let body = build_anthropic_body(&[
            json!({
                "type": "message_start",
                "message": {
                    "id": "msg_cap",
                    "model": "claude-3-5-sonnet",
                    "usage": { "input_tokens": 1, "output_tokens": 0 }
                }
            }),
            json!({ "type": "message_stop" }),
        ]);
        let mut provider = provider();
        provider.max_output_tokens = configured;
        let adapter = AnthropicAdapter::new(
            CapturingTransport::new(body, captured.clone()),
            provider,
            Arc::new(NoAuth),
        );
        let _ = LanguageModel::stream(
            &adapter,
            sample_unified_request(),
            sample_unified_options(),
        )
        .await
        .expect("adapter stream 不应失败");

        let wire = captured
            .lock()
            .expect("capture mutex poisoned")
            .clone()
            .expect("应已捕获 Anthropic 请求体");
        assert_eq!(
            wire.get("max_tokens").and_then(|v| v.as_u64()),
            Some(expected),
            "provider.max_output_tokens={configured:?} 时 wire max_tokens 应为 {expected}"
        );
    }
}

/// effort 档位 → wire `thinking` 字段：发 `enabled` + budget_tokens，None/Minimal/Custom 不发。
/// 经 CapturingTransport 捕获 adapter stream 发出的 wire 请求体，覆盖 `thinking_from_effort`
/// 的档位→budget 映射与「不发」档位（clamp 由 max_tokens 测试的覆盖路径间接保障）。
#[tokio::test]
async fn reasoning_effort_drives_wire_thinking() {
    let body = build_anthropic_body(&[
        json!({"type": "message_start", "message": {"id": "msg_t", "model": "claude", "usage": {"input_tokens": 1, "output_tokens": 0}}}),
        json!({"type": "message_stop"}),
    ]);
    // (effort, 期望 budget_tokens；None 表示不应发 thinking 字段)
    let cases: Vec<(Option<ReasoningEffort>, Option<u64>)> = vec![
        (Some(ReasoningEffort::Low), Some(1_024)),
        (Some(ReasoningEffort::Medium), Some(4_096)),
        (Some(ReasoningEffort::High), Some(8_192)),
        (Some(ReasoningEffort::XHigh), Some(12_288)),
        (Some(ReasoningEffort::Max), Some(15_360)),
        (Some(ReasoningEffort::Ultra), Some(15_360)),
        (Some(ReasoningEffort::None), None),
        (Some(ReasoningEffort::Minimal), None),
        (Some(ReasoningEffort::Custom("future".to_string())), None),
        (None, None),
    ];
    for (effort, expect_budget) in cases {
        let captured = Arc::new(Mutex::new(None::<Value>));
        let mut req = sample_unified_request();
        req.reasoning = effort.clone().map(|e| UnifiedReasoning {
            effort: Some(e),
            summary: None,
            context: None,
        });
        let adapter = AnthropicAdapter::new(
            CapturingTransport::new(body.clone(), captured.clone()),
            provider(),
            Arc::new(NoAuth),
        );
        let _ = LanguageModel::stream(&adapter, req, sample_unified_options())
            .await
            .expect("adapter stream 不应失败");
        let wire = captured
            .lock()
            .expect("capture mutex poisoned")
            .clone()
            .expect("应已捕获请求体");
        let thinking = wire.get("thinking");
        match expect_budget {
            Some(budget) => {
                let t = thinking
                    .unwrap_or_else(|| panic!("effort={effort:?} 应产出 thinking 字段"));
                assert_eq!(
                    t.get("type").and_then(|v| v.as_str()),
                    Some("enabled"),
                    "effort={effort:?} thinking.type 应为 enabled"
                );
                assert_eq!(
                    t.get("budget_tokens").and_then(|v| v.as_u64()),
                    Some(budget),
                    "effort={effort:?} budget_tokens 应为 {budget}"
                );
            }
            None => assert!(
                thinking.is_none(),
                "effort={effort:?} 不应发 thinking，实际: {thinking:?}"
            ),
        }
    }
}

/// 扩展思考 SSE 回放：`thinking_delta` 归一为 `UnifiedEvent::ReasoningContentDelta`（live 显示），
/// `signature_delta` 捕获为思考签名；text 块开始时收口出 `Reasoning` item（content=思考文本，
/// continuity_token=signature），且其 `OutputItemDone` 早于 assistant 文本（对齐 wire 序）。
#[tokio::test]
async fn fixture_thinking_delta_emits_reasoning_event() {
    let events = vec![
        json!({"type": "message_start", "message": {"id": "msg_th", "model": "claude", "usage": {"input_tokens": 5, "output_tokens": 1}}}),
        json!({"type": "content_block_start", "index": 0, "content_block": {"type": "thinking", "thinking": ""}}),
        json!({"type": "content_block_delta", "index": 0, "delta": {"type": "thinking_delta", "thinking": "先分析"}}),
        json!({"type": "content_block_delta", "index": 0, "delta": {"type": "thinking_delta", "thinking": "问题"}}),
        json!({"type": "content_block_delta", "index": 0, "delta": {"type": "signature_delta", "signature": "sig-abc"}}),
        json!({"type": "content_block_stop", "index": 0}),
        json!({"type": "content_block_start", "index": 1, "content_block": {"type": "text", "text": ""}}),
        json!({"type": "content_block_delta", "index": 1, "delta": {"type": "text_delta", "text": "答案是"}}),
        json!({"type": "content_block_delta", "index": 1, "delta": {"type": "text_delta", "text": "42"}}),
        json!({"type": "content_block_stop", "index": 1}),
        json!({"type": "message_delta", "delta": {"stop_reason": "end_turn"}, "usage": {"output_tokens": 8}}),
        json!({"type": "message_stop"}),
    ];
    let (evs, err) = run(build_anthropic_body(&events)).await;

    assert!(err.is_none(), "thinking 流不应报错：{err:?}");
    // thinking_delta → ReasoningContentDelta（content_index=0）；signature_delta 不产事件。
    let reasoning: Vec<&str> = evs
        .iter()
        .filter_map(|ev| match ev {
            UnifiedEvent::ReasoningContentDelta { delta, content_index } => {
                assert_eq!(*content_index, 0, "thinking 块 index=0");
                Some(delta.as_str())
            }
            _ => None,
        })
        .collect();
    assert_eq!(reasoning, vec!["先分析", "问题"], "thinking 增量序列: {evs:?}");
    // text 增量仍正常（与思考共用 active item）。
    let text: Vec<&str> = evs
        .iter()
        .filter_map(|ev| match ev {
            UnifiedEvent::OutputTextDelta(s) => Some(s.as_str()),
            _ => None,
        })
        .collect();
    assert_eq!(text, vec!["答案是", "42"], "文本增量序列: {evs:?}");
    // 装配出 assistant 文本（thinking 不进 message content）。
    expect_assistant_text(&evs, "答案是42");

    // signature 不再丢弃：收口出 Reasoning item，continuity_token=signature、content=思考文本。
    let reasoning_idx = evs.iter().position(|ev| {
        matches!(ev, UnifiedEvent::OutputItemDone(ResponseItem::Reasoning { .. }))
    });
    let reasoning_idx = reasoning_idx.expect("应收口出 Reasoning item");
    let (content, token) = match &evs[reasoning_idx] {
        UnifiedEvent::OutputItemDone(ResponseItem::Reasoning {
            content, continuity_token, ..
        }) => (content.clone(), continuity_token.clone()),
        _ => unreachable!(),
    };
    assert_eq!(token.as_deref(), Some("sig-abc"), "continuity_token 应为 signature");
    let reasoning_text = content
        .expect("Reasoning content 应非空")
        .into_iter()
        .map(|c| match c {
            ReasoningItemContent::ReasoningText { text }
            | ReasoningItemContent::Text { text } => text,
        })
        .collect::<String>();
    assert_eq!(reasoning_text, "先分析问题", "思考文本应为累积全文");

    // 顺序约束：Reasoning 的 Done 必须早于 assistant 文本 Done（wire `[thinking, text]`）。
    let message_idx = evs
        .iter()
        .position(|ev| {
            matches!(
                ev,
                UnifiedEvent::OutputItemDone(ResponseItem::Message { role, .. }) if role == "assistant"
            )
        })
        .expect("应有 assistant Message Done");
    assert!(
        reasoning_idx < message_idx,
        "Reasoning Done 必须早于 Message Done：{evs:?}"
    );
}

/// redacted_thinking 整组守卫：本 turn 出现 redacted_thinking 时，即便有 regular thinking
/// 文本与签名，也不收口出 Reasoning item（避免部分回放被服务端拒 400）。live 思考增量与
/// 文本仍正常（守卫只抑制持久化，不抑制流式显示）。
#[tokio::test]
async fn fixture_redacted_thinking_suppresses_reasoning_item() {
    let events = vec![
        json!({"type": "message_start", "message": {"id": "msg_r", "model": "claude", "usage": {"input_tokens": 3, "output_tokens": 1}}}),
        json!({"type": "content_block_start", "index": 0, "content_block": {"type": "thinking", "thinking": ""}}),
        json!({"type": "content_block_delta", "index": 0, "delta": {"type": "thinking_delta", "thinking": "明文思考"}}),
        json!({"type": "content_block_delta", "index": 0, "delta": {"type": "signature_delta", "signature": "sig-plain"}}),
        json!({"type": "content_block_stop", "index": 0}),
        json!({"type": "content_block_start", "index": 1, "content_block": {"type": "redacted_thinking", "data": "encrypted-blob"}}),
        json!({"type": "content_block_stop", "index": 1}),
        json!({"type": "content_block_start", "index": 2, "content_block": {"type": "text", "text": ""}}),
        json!({"type": "content_block_delta", "index": 2, "delta": {"type": "text_delta", "text": "回答"}}),
        json!({"type": "content_block_stop", "index": 2}),
        json!({"type": "message_delta", "delta": {"stop_reason": "end_turn"}, "usage": {"output_tokens": 4}}),
        json!({"type": "message_stop"}),
    ];
    let (evs, err) = run(build_anthropic_body(&events)).await;
    assert!(err.is_none(), "redacted 流不应报错：{err:?}");
    // 关键：不收口出 Reasoning item（整组不回喂）。
    let has_reasoning_done = evs
        .iter()
        .any(|ev| matches!(ev, UnifiedEvent::OutputItemDone(ResponseItem::Reasoning { .. })));
    assert!(!has_reasoning_done, "redacted turn 不应收口出 Reasoning：{evs:?}");
    // live 思考增量仍流式（守卫只抑制持久化，不抑制显示）。
    assert!(
        evs.iter().any(|ev| matches!(
            ev,
            UnifiedEvent::ReasoningContentDelta { delta, .. } if delta == "明文思考"
        )),
        "明文思考增量仍应流式：{evs:?}"
    );
    expect_assistant_text(&evs, "回答");
}

/// 回放：历史 Reasoning（continuity_token=signature）+ 同 turn assistant Message → 请求 wire
/// 末条 assistant 消息 content 为 `[{thinking, signature}, {text}]`（序对齐 Anthropic 协议）。
/// 无签名的 Reasoning 不回喂（不产 thinking 块）。
#[tokio::test]
async fn replay_reasoning_with_signature_emits_thinking_block() {
    let body = build_anthropic_body(&[
        json!({"type": "message_start", "message": {"id": "msg_x", "model": "claude", "usage": {"input_tokens": 1, "output_tokens": 0}}}),
        json!({"type": "message_stop"}),
    ]);
    let captured = Arc::new(Mutex::new(None::<Value>));
    let adapter = AnthropicAdapter::new(
        CapturingTransport::new(body.clone(), captured.clone()),
        provider(),
        Arc::new(NoAuth),
    );
    let mut req = sample_unified_request();
    // 历史：user → assistant(Reasoning 思考 + 文本)。Reasoning 与 assistant Message 同 turn。
    req.input = vec![
        ResponseItem::Message {
            id: None,
            role: "user".to_string(),
            content: vec![ContentItem::InputText {
                text: "继续".to_string(),
            }],
            phase: None,
            internal_chat_message_metadata_passthrough: None,
        },
        ResponseItem::Reasoning {
            id: None,
            summary: vec![],
            content: Some(vec![ReasoningItemContent::ReasoningText {
                text: "上一轮思考".to_string(),
            }]),
            encrypted_content: None,
            continuity_token: Some("sig-prev".to_string()),
            internal_chat_message_metadata_passthrough: None,
        },
        ResponseItem::Message {
            id: None,
            role: "assistant".to_string(),
            content: vec![ContentItem::OutputText {
                text: "上一轮回答".to_string(),
            }],
            phase: None,
            internal_chat_message_metadata_passthrough: None,
        },
    ];
    let _ = LanguageModel::stream(&adapter, req, sample_unified_options())
        .await
        .expect("adapter stream 不应失败");
    let wire = captured
        .lock()
        .expect("capture mutex poisoned")
        .clone()
        .expect("应已捕获请求体");
    let messages = wire
        .get("messages")
        .and_then(|m| m.as_array())
        .expect("应有 messages");
    let last_assistant = messages
        .iter()
        .rev()
        .find(|m| m.get("role").and_then(|v| v.as_str()) == Some("assistant"))
        .expect("应有 assistant 消息");
    let blocks = last_assistant
        .get("content")
        .and_then(|c| c.as_array())
        .expect("assistant 应有 content 数组");
    assert_eq!(
        blocks[0].get("type").and_then(|v| v.as_str()),
        Some("thinking"),
        "首块应为 thinking：{blocks:?}"
    );
    assert_eq!(
        blocks[0].get("thinking").and_then(|v| v.as_str()),
        Some("上一轮思考")
    );
    assert_eq!(
        blocks[0].get("signature").and_then(|v| v.as_str()),
        Some("sig-prev")
    );
    assert_eq!(
        blocks[1].get("type").and_then(|v| v.as_str()),
        Some("text"),
        "次块应为 text：{blocks:?}"
    );
    assert_eq!(
        blocks[1].get("text").and_then(|v| v.as_str()),
        Some("上一轮回答")
    );

    // 反例：无 continuity_token 的 Reasoning 不回喂（不产 thinking 块）。
    let captured2 = Arc::new(Mutex::new(None::<Value>));
    let adapter2 = AnthropicAdapter::new(
        CapturingTransport::new(body, captured2.clone()),
        provider(),
        Arc::new(NoAuth),
    );
    let mut req2 = sample_unified_request();
    req2.input = vec![
        ResponseItem::Reasoning {
            id: None,
            summary: vec![],
            content: Some(vec![ReasoningItemContent::ReasoningText {
                text: "无签名思考".to_string(),
            }]),
            encrypted_content: None,
            continuity_token: None,
            internal_chat_message_metadata_passthrough: None,
        },
        ResponseItem::Message {
            id: None,
            role: "assistant".to_string(),
            content: vec![ContentItem::OutputText {
                text: "回答".to_string(),
            }],
            phase: None,
            internal_chat_message_metadata_passthrough: None,
        },
    ];
    let _ = LanguageModel::stream(&adapter2, req2, sample_unified_options())
        .await
        .expect("adapter stream 不应失败");
    let wire2 = captured2
        .lock()
        .expect("capture mutex poisoned")
        .clone()
        .expect("应已捕获请求体");
    let has_thinking = wire2
        .get("messages")
        .and_then(|m| m.as_array())
        .into_iter()
        .flatten()
        .flat_map(|m| {
            m.get("content")
                .and_then(|c| c.as_array())
                .cloned()
                .unwrap_or_default()
        })
        .any(|b| b.get("type").and_then(|v| v.as_str()) == Some("thinking"));
    assert!(!has_thinking, "无签名的 Reasoning 不应回喂 thinking 块：{wire2:?}");
}

/// provider.max_output_tokens 调小 → thinking budget 夹断到 `max_tokens - 1024`（留文本预算）；
/// max_tokens 过小（不足 1024 思考 + 1024 文本）→ 放弃 thinking。
#[tokio::test]
async fn reasoning_effort_thinking_clamps_to_max_tokens() {
    let body = build_anthropic_body(&[
        json!({"type": "message_start", "message": {"id": "msg_c", "model": "claude", "usage": {"input_tokens": 1, "output_tokens": 0}}}),
        json!({"type": "message_stop"}),
    ]);

    // max_tokens=4096，High 档位值 8192 越界 → 夹断到 4096-1024=3072。
    let captured = Arc::new(Mutex::new(None::<Value>));
    let mut small_provider = provider();
    small_provider.max_output_tokens = Some(4096);
    let mut req = sample_unified_request();
    req.reasoning = Some(UnifiedReasoning {
        effort: Some(ReasoningEffort::High),
        summary: None,
        context: None,
    });
    let adapter = AnthropicAdapter::new(
        CapturingTransport::new(body.clone(), captured.clone()),
        small_provider,
        Arc::new(NoAuth),
    );
    let _ = LanguageModel::stream(&adapter, req, sample_unified_options())
        .await
        .expect("adapter stream 不应失败");
    let wire = captured
        .lock()
        .expect("capture mutex poisoned")
        .clone()
        .expect("应已捕获请求体");
    let t = wire
        .get("thinking")
        .expect("High 档夹断后仍应发 thinking");
    assert_eq!(
        t.get("budget_tokens").and_then(|v| v.as_u64()),
        Some(3072),
        "max_tokens=4096 时 High(8192) 应夹断到 3072"
    );

    // max_tokens=1024（过小）→ High 夹断到 0（< 1024）→ 放弃 thinking。
    let captured2 = Arc::new(Mutex::new(None::<Value>));
    let mut provider2 = provider();
    provider2.max_output_tokens = Some(1024);
    let mut req2 = sample_unified_request();
    req2.reasoning = Some(UnifiedReasoning {
        effort: Some(ReasoningEffort::High),
        summary: None,
        context: None,
    });
    let adapter2 = AnthropicAdapter::new(
        CapturingTransport::new(body, captured2.clone()),
        provider2,
        Arc::new(NoAuth),
    );
    let _ = LanguageModel::stream(&adapter2, req2, sample_unified_options())
        .await
        .expect("adapter stream 不应失败");
    let wire2 = captured2
        .lock()
        .expect("capture mutex poisoned")
        .clone()
        .expect("应已捕获请求体");
    assert!(
        wire2.get("thinking").is_none(),
        "max_tokens=1024 过小应放弃 thinking（无法满足 1024 思考 + 1024 文本）"
    );
}

// === 结构化输出（text.format → tool-mode）==================================

/// text.format → adapter 内部注入虚拟工具 `respond_structured`（承载 schema）+ 强制
/// tool_choice 指向它（即便原本无 tools）。Anthropic 无原生 JSON-schema 约束，借此
/// 迫使模型以 tool_use.input 回吐结构化 JSON。
#[test]
fn request_translation_injects_structured_output_tool() {
    let schema = json!({"type": "object", "properties": {"x": {"type": "integer"}}, "required": ["x"]});
    let mut req = sample_unified_request();
    // 原本无 tools（结构化输出子流程的典型形态）。
    req.text = Some(UnifiedTextControls {
        verbosity: None,
        format: Some(UnifiedTextFormat {
            r#type: UnifiedTextFormatType::JsonSchema,
            strict: true,
            schema: schema.clone(),
            name: "point".to_string(),
        }),
    });

    let api: AnthropicApiRequest = req.into();
    let json = serde_json::to_value(&api).expect("可序列化");

    // 注入虚拟工具 respond_structured：tools 数组含之，input_schema 为原 schema。
    let tools = json
        .get("tools")
        .and_then(|t| t.as_array())
        .expect("结构化输出应注入 tools（虚拟工具）");
    let virtual_tool = tools
        .iter()
        .find(|t| t.get("name").and_then(|v| v.as_str()) == Some("respond_structured"))
        .expect("应有 respond_structured 虚拟工具");
    assert_eq!(
        virtual_tool.get("input_schema"),
        Some(&schema),
        "虚拟工具 input_schema 应承载原 schema"
    );
    assert!(
        virtual_tool.get("description").is_none(),
        "虚拟工具不应带 description（skip_serializing_if None）"
    );

    // tool_choice 强制指向 respond_structured（{type:tool, name}）。
    let tc = json
        .get("tool_choice")
        .expect("结构化输出应强制 tool_choice");
    assert_eq!(tc.get("type").and_then(|v| v.as_str()), Some("tool"));
    assert_eq!(
        tc.get("name").and_then(|v| v.as_str()),
        Some("respond_structured")
    );
}

/// 模型回 tool_use(name=respond_structured, input_json 累积 '{"x":1}') → parser 还原成
/// `Message(OutputText='{"x":1}')`，**不**产出 FunctionCall（不派发虚拟工具）；
/// 且 end_turn=Some(true)（结构化输出是终态，避免 stop_reason=tool_use 触发重采样）。
#[tokio::test]
async fn fixture_structured_output_restores_message() {
    let events = vec![
        json!({
            "type": "message_start",
            "message": {"id": "msg_s", "model": "claude-3-5-sonnet", "usage": {"input_tokens": 10, "output_tokens": 1}}
        }),
        json!({
            "type": "content_block_start",
            "index": 0,
            "content_block": {"type": "tool_use", "id": "toolu_s", "name": "respond_structured", "input": {}}
        }),
        json!({"type": "content_block_delta", "index": 0, "delta": {"type": "input_json_delta", "partial_json": "{\"x\":"}}),
        json!({"type": "content_block_delta", "index": 0, "delta": {"type": "input_json_delta", "partial_json": "1}"}}),
        json!({"type": "content_block_stop", "index": 0}),
        // 强制 tool_choice 下 Anthropic 通常回 stop_reason="tool_use"（验证：即便如此，
        // 结构化输出仍应被映射成 end_turn=Some(true) 终态，不触发 core 重采样）。
        json!({
            "type": "message_delta",
            "delta": {"stop_reason": "tool_use", "stop_sequence": null},
            "usage": {"output_tokens": 3}
        }),
        json!({"type": "message_stop"}),
    ];

    // 带上 text.format（忠实复现结构化输出子流程的请求形态）。
    let mut req = sample_unified_request();
    req.text = Some(UnifiedTextControls {
        verbosity: None,
        format: Some(UnifiedTextFormat {
            r#type: UnifiedTextFormatType::JsonSchema,
            strict: true,
            schema: json!({"type": "object"}),
            name: "point".to_string(),
        }),
    });
    let adapter = AnthropicAdapter::new(
        FixtureSseTransport::new(build_anthropic_body(&events)),
        provider(),
        Arc::new(NoAuth),
    );
    let stream = LanguageModel::stream(&adapter, req, sample_unified_options())
        .await
        .expect("adapter stream 不应失败");
    let (evs, err) = drain(stream).await;

    assert!(err.is_none(), "结构化输出流不应有错误：{err:?}");
    // input_json_delta 累积为 '{"x":1}'，还原成 assistant 文本 Message。
    expect_assistant_text(&evs, r#"{"x":1}"#);
    // **不**产出 FunctionCall 的 Done（虚拟工具不经派发）。
    let fc_done = evs.iter().any(|ev| matches!(
        ev,
        UnifiedEvent::OutputItemDone(ResponseItem::FunctionCall { name, .. }) if name == "respond_structured"
    ));
    assert!(!fc_done, "结构化输出不应产出 FunctionCall Done（不派发）：{evs:?}");
    // 终态：end_turn=Some(true)（即便 stop_reason=tool_use）。
    match evs.iter().find(|ev| matches!(ev, UnifiedEvent::Completed { .. })) {
        Some(UnifiedEvent::Completed { end_turn, .. }) => {
            assert_eq!(*end_turn, Some(true), "结构化输出应为终态 end_turn=Some(true)");
        }
        other => panic!("应存在 Completed 事件，实际: {other:?}"),
    }
}
