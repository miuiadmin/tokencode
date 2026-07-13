//! Fixture 回放测试：验证 `GeminiAdapter` 把 Gemini generateContent 的 SSE 事件流
//! 正确翻译为 core 已消费的 `UnifiedEvent` 序列。
//!
//! 覆盖：① 纯文本完成（多 chunk 累积 + STOP + usageMetadata）；② functionCall 工具调用装配；
//! ③ thinking（`thought:true` → `ReasoningContentDelta`）；④ MAX_TOKENS 截断（end_turn=false）；
//! ⑤ 流内错误终止；⑥ usage 含 thoughtsTokenCount/cachedContentTokenCount。
//! 再附 wire 翻译断言：instructions→systemInstruction、role 映射（assistant→model）、
//! functionCall 在 model 消息、functionResponse 在 user 消息、toolConfig、maxOutputTokens 注入、
//! thinkingConfig 双模态（gemini-2.5 thinkingBudget vs gemini-3 thinkingLevel）、sanitize（integer enum→string）。
//! 脚手架（fixture 传输 / 鉴权 / provider / body 构造）与 `anthropic_adapter_fixtures.rs` 同构。

#![allow(clippy::expect_used)]

use std::sync::Arc;
use std::sync::Mutex;
use std::time::Duration;

use bytes::Bytes;
use codex_api::AuthProvider;
use codex_api::GeminiAdapter;
use codex_api::GeminiApiRequest;
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
use codex_protocol::models::ContentItem;
use codex_protocol::models::ResponseItem;
use codex_protocol::openai_models::ReasoningEffort;
use futures::StreamExt;
use http::HeaderMap;
use http::StatusCode;
use serde_json::Value;
use serde_json::json;

// --- 与 anthropic_adapter_fixtures.rs 同构的 fixture 传输 / 鉴权 / provider --------

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

/// 捕获发往传输的请求体（wire JSON），用于断言 adapter 注入的字段（thinkingConfig / maxOutputTokens）。
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
        name: "gemini".to_string(),
        base_url: "https://generativelanguage.googleapis.com".to_string(),
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

/// 把一组 Gemini 事件帧（JSON）拼成 SSE body（每帧仅 `data:` 行；解析侧按帧 JSON 结构分派）。
fn build_gemini_body(events: &[Value]) -> String {
    let mut body = String::new();
    for event in events {
        body.push_str(&format!("data: {event}\n\n"));
    }
    body
}

// --- 中立请求样本 ----------------------------------------------------------

fn sample_unified_request() -> UnifiedRequest {
    UnifiedRequest {
        model: "gemini-2.5-pro".to_string(),
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

// --- 驱动 ------------------------------------------------------------------

/// 驱动 adapter 跑给定 Gemini SSE body，返回事件序列 + 错误。
async fn run(body: String) -> (Vec<UnifiedEvent>, Option<String>) {
    let adapter = GeminiAdapter::new(
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

// === fixture 回放 ==========================================================

#[tokio::test]
async fn fixture_text_completion() {
    // 纯文本完成：两帧 text 片段累积，末帧带 finishReason=STOP + usageMetadata。
    let events = vec![
        json!({
            "candidates": [{
                "content": {"role": "model", "parts": [{"text": "你好"}]},
                "index": 0
            }]
        }),
        json!({
            "candidates": [{
                "content": {"role": "model", "parts": [{"text": "，世界"}]},
                "index": 0,
                "finishReason": "STOP"
            }],
            "usageMetadata": {"promptTokenCount": 10, "candidatesTokenCount": 3, "totalTokenCount": 13}
        }),
    ];
    let (evs, err) = run(build_gemini_body(&events)).await;

    assert!(err.is_none(), "纯文本完成不应有错误：{err:?}");
    // 首事件 Created（Gemini 首帧即发，无 message_start 等价）。
    assert!(matches!(evs.first(), Some(UnifiedEvent::Created)), "首事件应为 Created: {evs:?}");
    // 文本增量逐条到达。
    let deltas: Vec<&str> = evs
        .iter()
        .filter_map(|ev| match ev {
            UnifiedEvent::OutputTextDelta(s) => Some(s.as_str()),
            _ => None,
        })
        .collect();
    assert_eq!(deltas, vec!["你好", "，世界"], "文本增量序列: {evs:?}");
    // 装配出完整 assistant 文本。
    expect_assistant_text(&evs, "你好，世界");
    // Completed：STOP → end_turn=Some(true)；usage input=prompt(含cached超集)/output=candidates。
    match evs.iter().find(|ev| matches!(ev, UnifiedEvent::Completed { .. })) {
        Some(UnifiedEvent::Completed { end_turn, token_usage, .. }) => {
            assert_eq!(*end_turn, Some(true), "STOP → end_turn=Some(true)");
            let usage = token_usage
                .as_ref()
                .expect("Completed 应携带 token_usage");
            assert_eq!(usage.input_tokens, 10);
            assert_eq!(usage.output_tokens, 3);
            assert_eq!(usage.total_tokens, 13);
        }
        other => panic!("应存在 Completed 事件，实际: {other:?}"),
    }
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
async fn fixture_tool_call() {
    // functionCall part：工具调用（通常一帧完整）。call_id 由 adapter 按序生成 call_{index}。
    let events = vec![json!({
        "candidates": [{
            "content": {
                "role": "model",
                "parts": [{"functionCall": {"name": "get_weather", "args": {"city": "北京"}}}]
            },
            "index": 0,
            "finishReason": "STOP"
        }],
        "usageMetadata": {"promptTokenCount": 20, "candidatesTokenCount": 5, "totalTokenCount": 25}
    })];
    let (evs, err) = run(build_gemini_body(&events)).await;

    assert!(err.is_none(), "工具调用完成不应有错误：{err:?}");
    // functionCall part 触发 OutputItemAdded(FunctionCall) 占位（call_id = call_0）。
    assert!(
        evs.iter().any(|ev| matches!(
            ev,
            UnifiedEvent::OutputItemAdded(ResponseItem::FunctionCall { name, call_id, .. })
                if name == "get_weather" && call_id == "call_0"
        )),
        "应有 FunctionCall 的 OutputItemAdded 占位: {evs:?}"
    );
    // 装配出完整 FunctionCall（args 序列化为 JSON 串）。
    expect_function_call(&evs, "get_weather", "call_0", r#"{"city":"北京"}"#);
    // STOP → end_turn=Some(true)（Gemini 工具调用收尾仍为 STOP；turn 是否继续由 core 据 FunctionCall 决定）。
    match evs.iter().find(|ev| matches!(ev, UnifiedEvent::Completed { .. })) {
        Some(UnifiedEvent::Completed { end_turn, token_usage, .. }) => {
            assert_eq!(*end_turn, Some(true), "STOP → end_turn=Some(true)");
            let usage = token_usage.as_ref().expect("应有 usage");
            assert_eq!(usage.input_tokens, 20);
            assert_eq!(usage.output_tokens, 5);
        }
        other => panic!("应存在 Completed 事件，实际: {other:?}"),
    }
}

#[tokio::test]
async fn fixture_thought_part_becomes_reasoning() {
    // thinking：thought:true part → ReasoningContentDelta；普通 text part → OutputTextDelta。
    // 思考文本不进 assistant Message（仅归一为 reasoning 增量）。
    let events = vec![json!({
        "candidates": [{
            "content": {
                "role": "model",
                "parts": [
                    {"thought": true, "text": "先思考"},
                    {"text": "答案"}
                ]
            },
            "index": 0,
            "finishReason": "STOP"
        }],
        "usageMetadata": {
            "promptTokenCount": 5,
            "candidatesTokenCount": 2,
            "thoughtsTokenCount": 8,
            "totalTokenCount": 15
        }
    })];
    let (evs, err) = run(build_gemini_body(&events)).await;

    assert!(err.is_none(), "thinking 流不应有错误：{err:?}");
    // thought part → ReasoningContentDelta（delta = 思考文本）。
    let reasoning: Vec<&str> = evs
        .iter()
        .filter_map(|ev| match ev {
            UnifiedEvent::ReasoningContentDelta { delta, .. } => Some(delta.as_str()),
            _ => None,
        })
        .collect();
    assert_eq!(reasoning, vec!["先思考"], "思考增量序列: {evs:?}");
    // 普通 text part → OutputTextDelta（思考文本不混入）。
    let text_deltas: Vec<&str> = evs
        .iter()
        .filter_map(|ev| match ev {
            UnifiedEvent::OutputTextDelta(s) => Some(s.as_str()),
            _ => None,
        })
        .collect();
    assert_eq!(text_deltas, vec!["答案"], "文本增量序列（不含思考）: {evs:?}");
    // assistant Message 只含普通文本（思考不进 message）。
    expect_assistant_text(&evs, "答案");
    // usage：thoughtsTokenCount → reasoning_output_tokens（单列）。
    match evs.iter().find(|ev| matches!(ev, UnifiedEvent::Completed { .. })) {
        Some(UnifiedEvent::Completed { token_usage, .. }) => {
            let usage = token_usage.as_ref().expect("应有 usage");
            assert_eq!(usage.input_tokens, 5);
            assert_eq!(usage.output_tokens, 2);
            assert_eq!(usage.reasoning_output_tokens, 8, "thoughtsTokenCount → reasoning_output_tokens");
            assert_eq!(usage.total_tokens, 15);
        }
        other => panic!("应存在 Completed 事件，实际: {other:?}"),
    }
}

#[tokio::test]
async fn fixture_stop_reason_max_tokens_is_not_end_turn() {
    // MAX_TOKENS 截断：end_turn=Some(false)（非自然结束）。
    let events = vec![json!({
        "candidates": [{
            "content": {"role": "model", "parts": [{"text": "截断的"}]},
            "index": 0,
            "finishReason": "MAX_TOKENS"
        }],
        "usageMetadata": {"promptTokenCount": 10, "candidatesTokenCount": 100, "totalTokenCount": 110}
    })];
    let (evs, err) = run(build_gemini_body(&events)).await;

    assert!(err.is_none(), "MAX_TOKENS 不应产出错误：{err:?}");
    match evs.iter().find(|ev| matches!(ev, UnifiedEvent::Completed { .. })) {
        Some(UnifiedEvent::Completed { end_turn, .. }) => {
            assert_eq!(*end_turn, Some(false), "MAX_TOKENS → end_turn=Some(false)");
        }
        other => panic!("应存在 Completed 事件，实际: {other:?}"),
    }
}

#[tokio::test]
async fn fixture_in_stream_error_terminates() {
    // 流内错误帧（HTTP 2xx，body 为 {error:{message}}）：应终止并产出错误，不发任何 UnifiedEvent。
    let events = vec![json!({"error": {"message": "quota exceeded"}})];
    let (evs, err) = run(build_gemini_body(&events)).await;

    assert!(evs.is_empty(), "错误帧不应产出任何 UnifiedEvent（含 Created）: {evs:?}");
    let err = err.expect("应终止于错误");
    assert!(
        err.contains("quota exceeded"),
        "错误信息应透传 'quota exceeded'，实际: {err}"
    );
}

#[tokio::test]
async fn fixture_usage_reasoning_and_cached() {
    // usageMetadata 含 thoughtsTokenCount（reasoning）+ cachedContentTokenCount（cached 子集）。
    // promptTokenCount 已含 cached（超集），满足 non_cached = input - cached 不变量。
    let events = vec![json!({
        "candidates": [{
            "content": {"role": "model", "parts": [{"text": "ok"}]},
            "index": 0,
            "finishReason": "STOP"
        }],
        "usageMetadata": {
            "promptTokenCount": 100,
            "candidatesTokenCount": 10,
            "thoughtsTokenCount": 50,
            "cachedContentTokenCount": 80,
            "totalTokenCount": 160
        }
    })];
    let (evs, err) = run(build_gemini_body(&events)).await;
    assert!(err.is_none(), "usage 流不应有错误：{err:?}");

    match evs.iter().find(|ev| matches!(ev, UnifiedEvent::Completed { .. })) {
        Some(UnifiedEvent::Completed { token_usage, .. }) => {
            let usage = token_usage.as_ref().expect("应有 usage");
            assert_eq!(usage.input_tokens, 100, "input = promptTokenCount（含 cached 超集）");
            assert_eq!(usage.cached_input_tokens, 80, "cached = cachedContentTokenCount（子集）");
            assert_eq!(usage.output_tokens, 10);
            assert_eq!(usage.reasoning_output_tokens, 50, "reasoning = thoughtsTokenCount");
            assert_eq!(usage.total_tokens, 160);
        }
        other => panic!("应存在 Completed 事件，实际: {other:?}"),
    }
}

// === 请求翻译（wire 形态）=================================================

#[test]
fn request_translation_shapes_gemini_wire() {
    // instructions → 顶层 systemInstruction；user 文本 → contents[].parts[].text；
    // assistant FunctionCall → model 消息 functionCall；user FunctionCallOutput → user 消息 functionResponse。
    let request = UnifiedRequest {
        model: "gemini-2.5-pro".to_string(),
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

    let api_request: GeminiApiRequest = request.into();
    let json = serde_json::to_value(&api_request).expect("GeminiApiRequest 应可序列化");

    // 无 model 字段（Gemini model 在 URL path，不在 body）。
    assert!(json.get("model").is_none(), "Gemini body 不应含 model 字段");
    // 无 stream 字段（流式由端点决定）。
    assert!(json.get("stream").is_none(), "Gemini body 不应含 stream 字段");
    // 无 tools 时不发 toolConfig（Gemini 无工具时拒绝 toolConfig）。
    assert!(json.get("toolConfig").is_none(), "无 tools 时不应发 toolConfig");

    // instructions → 顶层 systemInstruction（{parts:[{text}]}，不进 contents）。
    let system = json
        .get("systemInstruction")
        .expect("应有顶层 systemInstruction");
    assert_eq!(
        system.get("parts").and_then(|p| p.as_array()).unwrap()[0]
            .get("text")
            .and_then(|v| v.as_str()),
        Some("系统级指令")
    );

    // contents：user(文本) → model(functionCall) → user(functionResponse)，三条。
    let contents = json
        .get("contents")
        .and_then(|c| c.as_array())
        .expect("应有 contents 数组");
    assert_eq!(contents.len(), 3, "消息序列: {contents:?}");

    // ① user 文本 part。
    assert_eq!(contents[0].get("role").and_then(|v| v.as_str()), Some("user"));
    assert_eq!(
        contents[0].get("parts").and_then(|p| p.as_array()).unwrap()[0]
            .get("text")
            .and_then(|v| v.as_str()),
        Some("查天气")
    );

    // ② model functionCall part（assistant→model；name + args 对象）。
    assert_eq!(contents[1].get("role").and_then(|v| v.as_str()), Some("model"));
    let fc = &contents[1].get("parts").and_then(|p| p.as_array()).unwrap()[0]
        .get("functionCall")
        .expect("model 消息应有 functionCall part");
    assert_eq!(fc.get("name").and_then(|v| v.as_str()), Some("get_weather"));
    assert_eq!(
        fc.get("args").and_then(|v| v.get("city")).and_then(|v| v.as_str()),
        Some("SF")
    );

    // ③ user functionResponse part（role=user，与 OpenAI tool role 不同）；
    //    name 来自 call_id→name 映射；response 为 {output: text}（非 JSON 对象回落）。
    assert_eq!(contents[2].get("role").and_then(|v| v.as_str()), Some("user"));
    let fr = &contents[2].get("parts").and_then(|p| p.as_array()).unwrap()[0]
        .get("functionResponse")
        .expect("user 消息应有 functionResponse part");
    assert_eq!(fr.get("name").and_then(|v| v.as_str()), Some("get_weather"));
    assert_eq!(
        fr.get("response")
            .and_then(|v| v.get("output"))
            .and_then(|v| v.as_str()),
        Some("晴天")
    );
}

#[test]
fn request_translation_sanitizes_integer_enum_and_emits_tool_config() {
    // tools 带 integer enum schema → sanitize 后 type=string/enum=["1","2","3"]；
    // tools 非空 + tool_choice="auto" → toolConfig.functionCallingConfig.mode="AUTO"。
    let request = UnifiedRequest {
        model: "gemini-2.5-pro".to_string(),
        instructions: String::new(),
        input: vec![ResponseItem::Message {
            id: None,
            role: "user".to_string(),
            content: vec![ContentItem::InputText {
                text: "选尺寸".to_string(),
            }],
            phase: None,
            internal_chat_message_metadata_passthrough: None,
        }],
        tools: Some(vec![json!({
            "type": "function",
            "name": "pick_size",
            "description": "选尺寸",
            "parameters": {
                "type": "object",
                "properties": {
                    "size": {"type": "integer", "enum": [1, 2, 3]}
                },
                "required": ["size"]
            }
        })]),
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

    let api_request: GeminiApiRequest = request.into();
    let json = serde_json::to_value(&api_request).expect("应可序列化");

    // tools → [{functionDeclarations:[{name, parameters}]}]。
    let decls = json
        .get("tools")
        .and_then(|t| t.as_array())
        .expect("应有 tools 数组")[0]
        .get("functionDeclarations")
        .and_then(|f| f.as_array())
        .expect("应有 functionDeclarations");
    assert_eq!(decls.len(), 1);
    assert_eq!(decls[0].get("name").and_then(|v| v.as_str()), Some("pick_size"));

    // sanitize：integer enum → string enum（type 改写为 string，enum 值 stringify）。
    let size_schema = decls[0]
        .get("parameters")
        .and_then(|p| p.get("properties"))
        .and_then(|p| p.get("size"))
        .expect("应有 size schema");
    assert_eq!(
        size_schema.get("type").and_then(|v| v.as_str()),
        Some("string"),
        "integer enum 经 sanitize 后 type 应为 string"
    );
    let enum_vals: Vec<&str> = size_schema
        .get("enum")
        .and_then(|v| v.as_array())
        .expect("应有 enum")
        .iter()
        .filter_map(|v| v.as_str())
        .collect();
    assert_eq!(enum_vals, vec!["1", "2", "3"], "enum 值应 stringify");

    // tool_config（tools 非空 + tool_choice=auto）→ mode=AUTO。
    let mode = json
        .get("toolConfig")
        .and_then(|t| t.get("functionCallingConfig"))
        .and_then(|f| f.get("mode"))
        .and_then(|v| v.as_str());
    assert_eq!(mode, Some("AUTO"), "tool_choice=auto + tools 非空 → mode=AUTO");
}

// === adapter stream 注入（thinkingConfig / maxOutputTokens）================

/// 经 CapturingTransport 捕获 adapter stream 发出的 wire 请求体，覆盖 thinking_config_from_effort
/// 的 gemini-2.5 档位→thinkingBudget 映射 + None/Minimal/Custom→0（显式禁用）。
#[tokio::test]
async fn reasoning_effort_drives_wire_thinking_2_5() {
    let body = build_gemini_body(&[json!({
        "candidates": [{"content": {"role": "model", "parts": [{"text": "x"}]}, "finishReason": "STOP"}]
    })]);
    // (effort, 期望 thinkingBudget, 期望 includeThoughts 是否出现)
    let cases: Vec<(Option<ReasoningEffort>, Option<i64>, bool)> = vec![
        (Some(ReasoningEffort::Low), Some(2_048), true),
        (Some(ReasoningEffort::Medium), Some(8_192), true),
        (Some(ReasoningEffort::High), Some(16_384), true),
        (Some(ReasoningEffort::XHigh), Some(20_480), true),
        (Some(ReasoningEffort::Max), Some(24_576), true),
        (Some(ReasoningEffort::Ultra), Some(24_576), true),
        // None/Minimal/Custom → thinkingBudget=0（显式禁用），不发 includeThoughts。
        (Some(ReasoningEffort::None), Some(0), false),
        (Some(ReasoningEffort::Minimal), Some(0), false),
        (Some(ReasoningEffort::Custom("future".to_string())), Some(0), false),
        (None, Some(0), false),
    ];
    for (effort, expect_budget, expect_include) in cases {
        let captured = Arc::new(Mutex::new(None::<Value>));
        let mut req = sample_unified_request(); // model = gemini-2.5-pro
        req.reasoning = effort.clone().map(|e| UnifiedReasoning {
            effort: Some(e),
            summary: None,
            context: None,
        });
        let adapter = GeminiAdapter::new(
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
        let thinking = wire
            .get("generationConfig")
            .and_then(|g| g.get("thinkingConfig"))
            .unwrap_or_else(|| panic!("effort={effort:?} 应产出 generationConfig.thinkingConfig"));
        assert_eq!(
            thinking.get("thinkingBudget").and_then(|v| v.as_i64()),
            expect_budget,
            "effort={effort:?} thinkingBudget 应为 {expect_budget:?}"
        );
        assert_eq!(
            thinking.get("thinkingLevel").and_then(|v| v.as_str()),
            None,
            "gemini-2.5 不应发 thinkingLevel"
        );
        let has_include = thinking.get("includeThoughts").is_some();
        assert_eq!(
            has_include, expect_include,
            "effort={effort:?} includeThoughts 出现应为 {expect_include}"
        );
    }
}

/// gemini-3.x：effort → thinkingLevel（minimal/low/medium/high），始终带 includeThoughts:true。
#[tokio::test]
async fn reasoning_effort_drives_wire_thinking_3() {
    let body = build_gemini_body(&[json!({
        "candidates": [{"content": {"role": "model", "parts": [{"text": "x"}]}, "finishReason": "STOP"}]
    })]);
    let cases: Vec<(Option<ReasoningEffort>, &str)> = vec![
        (Some(ReasoningEffort::Low), "low"),
        (Some(ReasoningEffort::Medium), "medium"),
        (Some(ReasoningEffort::High), "high"),
        (Some(ReasoningEffort::XHigh), "high"),
        (Some(ReasoningEffort::Max), "high"),
        (Some(ReasoningEffort::Ultra), "high"),
        // None/Minimal/Custom → "minimal"（3.x 无法显式禁用，minimal 是最低档）。
        (Some(ReasoningEffort::None), "minimal"),
        (Some(ReasoningEffort::Minimal), "minimal"),
        (None, "minimal"),
    ];
    for (effort, expect_level) in cases {
        let captured = Arc::new(Mutex::new(None::<Value>));
        let mut req = sample_unified_request();
        req.model = "gemini-3-pro".to_string();
        req.reasoning = effort.clone().map(|e| UnifiedReasoning {
            effort: Some(e),
            summary: None,
            context: None,
        });
        let adapter = GeminiAdapter::new(
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
        let thinking = wire
            .get("generationConfig")
            .and_then(|g| g.get("thinkingConfig"))
            .expect("gemini-3 应产出 thinkingConfig");
        assert_eq!(
            thinking.get("thinkingLevel").and_then(|v| v.as_str()),
            Some(expect_level),
            "effort={effort:?} thinkingLevel 应为 {expect_level}"
        );
        assert_eq!(
            thinking.get("thinkingBudget").and_then(|v| v.as_i64()),
            None,
            "gemini-3 不应发 thinkingBudget"
        );
        assert_eq!(
            thinking.get("includeThoughts").and_then(|v| v.as_bool()),
            Some(true),
            "gemini-3 始终带 includeThoughts:true"
        );
    }
}

/// provider.max_output_tokens → generationConfig.maxOutputTokens：None 不发，Some 注入。
#[tokio::test]
async fn provider_max_output_tokens_drives_wire() {
    let body = build_gemini_body(&[json!({
        "candidates": [{"content": {"role": "model", "parts": [{"text": "x"}]}, "finishReason": "STOP"}]
    })]);
    for (configured, expected) in [(None, None), (Some(8_192u32), Some(8_192u64))] {
        let captured = Arc::new(Mutex::new(None::<Value>));
        let mut provider = provider();
        provider.max_output_tokens = configured;
        let adapter = GeminiAdapter::new(
            CapturingTransport::new(body.clone(), captured.clone()),
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
            .expect("应已捕获请求体");
        let max = wire
            .get("generationConfig")
            .and_then(|g| g.get("maxOutputTokens"))
            .and_then(|v| v.as_u64());
        assert_eq!(
            max, expected,
            "provider.max_output_tokens={configured:?} 时 wire maxOutputTokens 应为 {expected:?}"
        );
    }
}
