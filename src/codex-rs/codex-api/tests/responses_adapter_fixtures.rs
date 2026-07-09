//! Fixture 回放测试：证明 `OpenaiResponsesAdapter` 是「行为零变化」的包装。
//!
//! 同一份 SSE fixture，分别经「`ResponsesClient` 直连」（baseline）与
//! 「`OpenaiResponsesAdapter`」（P0 路径）驱动，断言两者产出的事件序列逐事件
//! deep-equal、终止错误一致。覆盖 `Created` / 文本增量 / reasoning 三类增量 /
//! 工具调用增量 / `OutputItem(Added|Done)` / `Completed` / 失败错误路径。
//!
//! `ResponseEvent` 仅 derive(`Debug`)、无 `PartialEq`，故用 Debug 串做 deep-equal
//! （`pretty_assertions` 失败时给出 diff）。fixture 传输 / 鉴权 / provider / SSE body
//! 构造与 `tests/sse_end_to_end.rs` 同构。

#![allow(clippy::expect_used)]

use std::sync::Arc;
use std::time::Duration;

use bytes::Bytes;
use codex_api::AuthProvider;
use codex_api::OpenaiResponsesAdapter;
use codex_api::Provider;
use codex_api::ResponseEvent;
use codex_api::ResponsesApiRequest;
use codex_api::ResponsesClient;
use codex_api::ResponsesOptions;
use codex_api::RetryConfig;
use codex_client::HttpTransport;
use codex_client::Request;
use codex_client::Response;
use codex_client::StreamResponse;
use codex_client::TransportError;
use codex_language_model::LanguageModel;
use codex_language_model::UnifiedError;
use codex_language_model::UnifiedEvent;
use codex_language_model::UnifiedEventStream;
use codex_language_model::UnifiedRequest;
use codex_language_model::UnifiedRequestOptions;
use codex_api::ApiError;
use futures::StreamExt;
use http::HeaderMap;
use http::StatusCode;
use pretty_assertions::assert_eq;
use serde_json::Value;

// --- 与 sse_end_to_end.rs 同构的 fixture 传输 / 鉴权 / provider / body -----

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

fn build_responses_body(events: Vec<Value>) -> String {
    let mut body = String::new();
    for e in events {
        let kind = e
            .get("type")
            .and_then(|v| v.as_str())
            .expect("SSE fixture event should have a type");
        if e.as_object().map(|o| o.len() == 1).unwrap_or(false) {
            body.push_str(&format!("event: {kind}\n\n"));
        } else {
            body.push_str(&format!("event: {kind}\ndata: {e}\n\n"));
        }
    }
    body
}

// --- 中立请求样本 --------------------------------------------------
// fixture 传输忽略请求体、只回放 SSE，故字段值不影响事件序列；此处仅构造合法值。

fn sample_unified_request() -> UnifiedRequest {
    UnifiedRequest {
        model: "test-model".to_string(),
        instructions: String::new(),
        input: vec![],
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

// --- 流收集：返回 (事件序列, 终止错误 Debug) -------------------------
// RateLimits 源自 HTTP 头（fixture 无相关头），两边一致过滤，保证可比。

async fn drain_baseline(
    mut s: codex_api::ResponseStream,
) -> (Vec<ResponseEvent>, Option<String>) {
    let mut evs = Vec::new();
    let mut err = None;
    while let Some(item) = s.next().await {
        match item {
            Ok(ev) => {
                if !matches!(ev, ResponseEvent::RateLimits(_)) {
                    evs.push(ev);
                }
            }
            Err(e) => {
                err = Some(format!("{e:?}"));
                break;
            }
        }
    }
    (evs, err)
}

async fn drain_unified(
    mut s: UnifiedEventStream,
) -> (Vec<UnifiedEvent>, Option<String>) {
    let mut evs = Vec::new();
    let mut err = None;
    while let Some(item) = s.next().await {
        match item {
            Ok(ev) => {
                if !matches!(ev, UnifiedEvent::RateLimits(_)) {
                    evs.push(ev);
                }
            }
            Err(e) => {
                // adapter 把具体协议错误装箱为 Passthrough；downcast 回 ApiError 后比较，
                // 与 baseline 的 ApiError Debug 串对齐（装箱 / 拆箱不改变 Debug 表征）。
                err = Some(match e {
                    UnifiedError::Passthrough(boxed) => match boxed.downcast::<ApiError>() {
                        Ok(api) => format!("{api:?}"),
                        Err(_) => "UnifiedError::Passthrough(non-ApiError)".to_string(),
                    },
                    UnifiedError::Mapping(m) => format!("UnifiedError::Mapping({m})"),
                });
                break;
            }
        }
    }
    (evs, err)
}

/// 断言 baseline 与 adapter 产出等价：事件序列逐事件 deep-equal + 终止错误一致。
fn assert_equivalent(
    baseline: (Vec<ResponseEvent>, Option<String>),
    adapter: (Vec<UnifiedEvent>, Option<String>),
) {
    let (base_evs, base_err) = baseline;
    let (uni_evs, uni_err) = adapter;

    // unified 经反向 From 映射回 ResponseEvent（正是 core 在 agent 循环顶部做的事）。
    let adapter_evs: Vec<ResponseEvent> = uni_evs.into_iter().map(ResponseEvent::from).collect();

    let base_dbg: Vec<String> = base_evs.iter().map(|e| format!("{e:?}")).collect();
    let adapter_dbg: Vec<String> = adapter_evs.iter().map(|e| format!("{e:?}")).collect();
    assert_eq!(base_dbg, adapter_dbg, "事件序列不一致：baseline ≠ adapter");
    assert_eq!(base_err, uni_err, "终止错误不一致：baseline ≠ adapter");
}

/// 对一份 fixture 同时跑 baseline（ResponsesClient 直连）与 adapter（P0），断言等价。
async fn run_fixture(events: Vec<Value>) {
    let body = build_responses_body(events);

    // baseline：ResponsesClient 直连（与 adapter 内部调用的是同一个 stream_request）。
    let baseline_client = ResponsesClient::new(
        FixtureSseTransport::new(body.clone()),
        provider(),
        Arc::new(NoAuth),
    );
    let baseline_stream = baseline_client
        .stream_request(
            ResponsesApiRequest::from(sample_unified_request()),
            ResponsesOptions::from(sample_unified_options()),
        )
        .await
        .expect("baseline stream_request 不应失败");
    let baseline = drain_baseline(baseline_stream).await;

    // adapter：OpenaiResponsesAdapter（P0 路径，经 LanguageModel trait）。
    let adapter = OpenaiResponsesAdapter::new(
        FixtureSseTransport::new(body),
        provider(),
        Arc::new(NoAuth),
    );
    let unified_stream = LanguageModel::stream(
        &adapter,
        sample_unified_request(),
        sample_unified_options(),
    )
    .await
    .expect("adapter stream 不应失败");
    let adapter_result = drain_unified(unified_stream).await;

    assert_equivalent(baseline, adapter_result);
}

// --- 三份 fixture：正常 turn / reasoning+工具增量 / 失败错误 ----------

#[tokio::test]
async fn fixture_normal_completion() {
    let events = vec![
        serde_json::json!({"type":"response.created","response":{}}),
        serde_json::json!({"type":"response.output_text.delta","delta":"Hello"}),
        serde_json::json!({"type":"response.output_text.delta","delta":", world"}),
        serde_json::json!({"type":"response.output_item.done","item":{"type":"message","role":"assistant","content":[{"type":"output_text","text":"Hello, world"}]}}),
        serde_json::json!({"type":"response.completed","response":{"id":"resp_A","end_turn":true}}),
    ];
    run_fixture(events).await;
}

#[tokio::test]
async fn fixture_reasoning_and_tool_deltas() {
    let events = vec![
        serde_json::json!({"type":"response.reasoning_summary_part.added","summary_index":0}),
        serde_json::json!({"type":"response.reasoning_summary_text.delta","delta":"Thinking...","summary_index":0}),
        serde_json::json!({"type":"response.reasoning_text.delta","delta":"inner thought","content_index":0}),
        serde_json::json!({"type":"response.custom_tool_call_input.delta","delta":"{\"x\":1}","item_id":"call_1","call_id":"call_1"}),
        serde_json::json!({"type":"response.output_item.added","item":{"type":"message","role":"assistant","content":[{"type":"output_text","text":"partial"}]}}),
        serde_json::json!({"type":"response.output_item.done","item":{"type":"message","role":"assistant","content":[{"type":"output_text","text":"done"}]}}),
        serde_json::json!({"type":"response.completed","response":{"id":"resp_B"}}),
    ];
    run_fixture(events).await;
}

#[tokio::test]
async fn fixture_error_terminates_stream() {
    let events = vec![
        serde_json::json!({"type":"response.created","response":{}}),
        serde_json::json!({"type":"response.failed","response":{"id":"resp_C","error":{"code":"rate_limit_exceeded","message":"Rate limit reached"}}}),
    ];
    run_fixture(events).await;
}
