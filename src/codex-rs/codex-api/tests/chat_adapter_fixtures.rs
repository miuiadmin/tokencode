//! Fixture 回放测试：验证 `OpenaiChatAdapter` 把 OpenAI Chat Completions 的 SSE chunk
//! 流正确翻译为 core 已消费的 `UnifiedEvent` 序列。
//!
//! 三份捕获形态的 Chat SSE：① 纯文本完成；② 工具调用累积装配；③ 流内错误终止。
//! 驱动 `OpenaiChatAdapter`（经 `LanguageModel` trait），断言产出的 `UnifiedEvent` 序列。
//! 脚手架（fixture 传输 / 鉴权 / provider / body 构造）与 `responses_adapter_fixtures.rs`
//! 同构。

#![allow(clippy::expect_used)]

use std::sync::Arc;
use std::sync::Mutex;
use std::time::Duration;

use bytes::Bytes;
use codex_api::AuthProvider;
use codex_api::ChatApiRequest;
use codex_api::OpenaiChatAdapter;
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
use codex_language_model::UnifiedRequest;
use codex_language_model::UnifiedRequestOptions;
use codex_protocol::models::ContentItem;
use codex_protocol::models::ResponseItem;
use futures::StreamExt;
use http::HeaderMap;
use http::StatusCode;
use serde_json::Value;
use serde_json::json;

// --- 与 responses_adapter_fixtures.rs 同构的 fixture 传输 / 鉴权 / provider -----

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
        name: "openai".to_string(),
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

/// 把一组 Chat chunk（JSON）拼成 SSE body，末尾按需追加 `[DONE]`。
fn build_chat_body(chunks: &[Value], done: bool) -> String {
    let mut body = String::new();
    for chunk in chunks {
        body.push_str(&format!("data: {chunk}\n\n"));
    }
    if done {
        body.push_str("data: [DONE]\n\n");
    }
    body
}

// --- 中立请求样本 ----------------------------------------------------------

fn sample_unified_request() -> UnifiedRequest {
    UnifiedRequest {
        model: "glm-5.2".to_string(),
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

// --- 三份 fixture -----------------------------------------------------------

/// 驱动 adapter 跑给定 Chat SSE body，返回事件序列 + 错误。
async fn run(body: String) -> (Vec<UnifiedEvent>, Option<String>) {
    let adapter = OpenaiChatAdapter::new(
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

#[tokio::test]
async fn fixture_text_completion() {
    // 标准纯文本完成：role 帧 → content 分片 → finish_reason:stop → usage → [DONE]。
    let chunks = vec![
        json!({
            "id": "chatcmpl-1",
            "object": "chat.completion.chunk",
            "model": "glm-5.2",
            "choices": [{"index": 0, "delta": {"role": "assistant", "content": ""}, "finish_reason": null}]
        }),
        json!({
            "id": "chatcmpl-1",
            "choices": [{"index": 0, "delta": {"content": "Hello"}, "finish_reason": null}]
        }),
        json!({
            "id": "chatcmpl-1",
            "choices": [{"index": 0, "delta": {"content": ", world"}, "finish_reason": null}]
        }),
        json!({
            "id": "chatcmpl-1",
            "choices": [{"index": 0, "delta": {}, "finish_reason": "stop"}]
        }),
        json!({
            "id": "chatcmpl-1",
            "choices": [],
            "usage": {"prompt_tokens": 10, "completion_tokens": 2, "total_tokens": 12}
        }),
    ];
    let (evs, err) = run(build_chat_body(&chunks, true)).await;

    assert!(err.is_none(), "纯文本完成不应有错误：{err:?}");
    // 首事件 Created；末事件 Completed(end_turn=Some(true), token_usage 非空)。
    assert!(matches!(evs.first(), Some(UnifiedEvent::Created)), "首事件应为 Created: {evs:?}");
    let completed = evs
        .iter()
        .find(|ev| matches!(ev, UnifiedEvent::Completed { .. }));
    match completed {
        Some(UnifiedEvent::Completed { end_turn, token_usage, .. }) => {
            assert_eq!(*end_turn, Some(true), "finish_reason=stop → end_turn=Some(true)");
            let usage = token_usage
                .as_ref()
                .expect("Completed 应携带 token_usage（include_usage）");
            assert_eq!(usage.input_tokens, 10);
            assert_eq!(usage.output_tokens, 2);
            assert_eq!(usage.total_tokens, 12);
        }
        other => panic!("应存在 Completed 事件，实际: {other:?}"),
    }
    // 文本增量逐条到达。
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
    // 工具调用：首帧 id+name → arguments 分片 → finish_reason:tool_calls → usage → [DONE]。
    let chunks = vec![
        json!({
            "id": "chatcmpl-2",
            "model": "glm-5.2",
            "choices": [{"index": 0, "delta": {"role": "assistant", "tool_calls": [
                {"index": 0, "id": "call_1", "type": "function", "function": {"name": "get_weather", "arguments": ""}}
            ]}, "finish_reason": null}]
        }),
        json!({
            "id": "chatcmpl-2",
            "choices": [{"index": 0, "delta": {"tool_calls": [
                {"index": 0, "function": {"arguments": "{\"city\":"}}
            ]}, "finish_reason": null}]
        }),
        json!({
            "id": "chatcmpl-2",
            "choices": [{"index": 0, "delta": {"tool_calls": [
                {"index": 0, "function": {"arguments": "\"SF\"}"}}
            ]}, "finish_reason": null}]
        }),
        json!({
            "id": "chatcmpl-2",
            "choices": [{"index": 0, "delta": {}, "finish_reason": "tool_calls"}]
        }),
        json!({
            "id": "chatcmpl-2",
            "choices": [],
            "usage": {"prompt_tokens": 20, "completion_tokens": 5, "total_tokens": 25}
        }),
    ];
    let (evs, err) = run(build_chat_body(&chunks, true)).await;

    assert!(err.is_none(), "工具调用完成不应有错误：{err:?}");
    // 新 tool index 触发 OutputItemAdded(FunctionCall) 占位。
    assert!(
        evs.iter().any(|ev| matches!(
            ev,
            UnifiedEvent::OutputItemAdded(ResponseItem::FunctionCall { name, call_id, .. })
                if name == "get_weather" && call_id == "call_1"
        )),
        "应有 FunctionCall 的 OutputItemAdded 占位: {evs:?}"
    );
    // 装配出完整 FunctionCall（参数分片已累积）。
    expect_function_call(&evs, "get_weather", "call_1", r#"{"city":"SF"}"#);
    // finish_reason=tool_calls → end_turn=Some(false)（turn 待工具结果继续）。
    match evs.iter().find(|ev| matches!(ev, UnifiedEvent::Completed { .. })) {
        Some(UnifiedEvent::Completed { end_turn, .. }) => {
            assert_eq!(*end_turn, Some(false), "finish_reason=tool_calls → end_turn=Some(false)");
        }
        other => panic!("应存在 Completed 事件，实际: {other:?}"),
    }
}

#[tokio::test]
async fn fixture_in_stream_error_terminates() {
    // 流内错误帧（HTTP 2xx，body 为 {error:{message}}）：应终止并产出错误。
    let chunks = vec![json!({"error": {"message": "rate limited"}})];
    let (evs, err) = run(build_chat_body(&chunks, false)).await;

    assert!(evs.is_empty(), "错误帧不应产出任何 UnifiedEvent: {evs:?}");
    let err = err.expect("应终止于错误");
    assert!(
        err.contains("rate limited"),
        "错误信息应透传 'rate limited'，实际: {err}"
    );
}

#[test]
fn request_translation_normalizes_developer_role_to_system() {
    // 真实冒烟发现：OpenAI 兼容网关 / 第三方后端（GLM 等）只识别 system 角色，发 developer
    // 会被拒（422）。Chat adapter 必须把 developer 归一为 system（instructions 与 Message 均然）。
    let request = UnifiedRequest {
        model: "glm-5.2".to_string(),
        instructions: "系统级指令".to_string(),
        input: vec![
            ResponseItem::Message {
                id: None,
                role: "developer".to_string(),
                content: vec![ContentItem::InputText {
                    text: "额外的开发者指令".to_string(),
                }],
                phase: None,
                internal_chat_message_metadata_passthrough: None,
            },
            ResponseItem::Message {
                id: None,
                role: "user".to_string(),
                content: vec![ContentItem::InputText {
                    text: "你好".to_string(),
                }],
                phase: None,
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

    let api_request: ChatApiRequest = request.into();
    let json = serde_json::to_value(&api_request).expect("ChatApiRequest 应可序列化");
    let messages = json
        .get("messages")
        .and_then(|m| m.as_array())
        .expect("应有 messages 数组");

    // 不应出现任何 developer 角色。
    let roles: Vec<&str> = messages
        .iter()
        .map(|m| m.get("role").and_then(|r| r.as_str()).unwrap_or(""))
        .collect();
    assert!(
        !roles.iter().any(|r| *r == "developer"),
        "不应有 developer 角色（应归一为 system），实际 roles: {roles:?}"
    );
    // instructions → 首条 system；原 developer Message → system；user 保持 user。
    assert_eq!(
        roles,
        vec!["system", "system", "user"],
        "角色归一后序列应为 system(instructions), system(原developer), user: {roles:?}"
    );
}

#[tokio::test]
async fn provider_max_output_tokens_drives_wire_max_tokens() {
    // 经 CapturingTransport 捕获 adapter 发出的 wire 请求体，断言可选 max_tokens：
    //   provider.max_output_tokens = None  → 字段不发（skip_serializing_if）；
    //   provider.max_output_tokens = Some  → 字段值为配置值。
    for (configured, expected) in [(None, None::<u64>), (Some(8_192u32), Some(8_192u64))] {
        let captured = Arc::new(Mutex::new(None::<Value>));
        // 最小可完成的 Chat chunk + [DONE]，仅供 stream() 跑通；本用例只断言请求体。
        let chunk = json!({
            "id": "chatcmpl-cap",
            "object": "chat.completion.chunk",
            "choices": [{ "index": 0, "delta": { "content": "" }, "finish_reason": "stop" }]
        });
        let body = build_chat_body(&[chunk], true);
        let mut provider = provider();
        provider.max_output_tokens = configured;
        let adapter = OpenaiChatAdapter::new(
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
            .expect("应已捕获 Chat 请求体");
        assert_eq!(
            wire.get("max_tokens").and_then(|v| v.as_u64()),
            expected,
            "provider.max_output_tokens={configured:?} 时 wire max_tokens 应为 {expected:?}"
        );
    }
}
