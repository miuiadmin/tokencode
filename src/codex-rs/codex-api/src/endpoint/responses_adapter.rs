//! OpenAI Responses 协议 adapter —— 把中立 `LanguageModel` 契约桥接到现有 `ResponsesClient`。
//!
//! P0 行为零变化：inner 复用 `ResponsesClient`（其源码零改动），本文件只做中立格式 ↔
//! codex_api 内部格式的双向机械映射 + SSE 事件流归一。所有 `From` 实现由编译期保证
//! 字段 / 变体映射穷尽——被映射的协议增删字段或事件变体时，本文件会编译失败而非静默漏映射。

use crate::auth::SharedAuthProvider;
use crate::common::OpenAiVerbosity;
use crate::common::Reasoning;
use crate::common::ReasoningContext;
use crate::common::ResponseEvent;
use crate::common::ResponseStream;
use crate::common::ResponsesApiRequest;
use crate::common::SafetyBuffering;
use crate::common::TextControls;
use crate::common::TextFormat;
use crate::common::TextFormatType;
use crate::endpoint::responses::ResponsesClient;
use crate::endpoint::responses::ResponsesOptions;
use crate::provider::Provider;
use crate::requests::Compression;
use crate::telemetry::SseTelemetry;
use codex_client::HttpTransport;
use codex_client::RequestTelemetry;
use codex_language_model::LanguageModel;
use codex_language_model::UnifiedCompression;
use codex_language_model::UnifiedError;
use codex_language_model::UnifiedEvent;
use codex_language_model::UnifiedEventStream;
use codex_language_model::UnifiedOpenAiVerbosity;
use codex_language_model::UnifiedReasoning;
use codex_language_model::UnifiedReasoningContext;
use codex_language_model::UnifiedRequest;
use codex_language_model::UnifiedRequestOptions;
use codex_language_model::UnifiedSafetyBuffering;
use codex_language_model::UnifiedTextControls;
use codex_language_model::UnifiedTextFormat;
use codex_language_model::UnifiedTextFormatType;
use futures::Stream;
use futures::future::BoxFuture;
use std::pin::Pin;
use std::sync::Arc;
use std::task::Context;
use std::task::Poll;

/// OpenAI Responses 协议 adapter：实现中立 `LanguageModel` trait，内部委托 `ResponsesClient`。
///
/// 泛型参数 `T: HttpTransport` 与 `ResponsesClient` 一致；P0 仅编译期内置此 adapter，
/// 不做运行时热插拔。
pub struct OpenaiResponsesAdapter<T: HttpTransport> {
    inner: ResponsesClient<T>,
}

impl<T: HttpTransport> OpenaiResponsesAdapter<T> {
    /// 构造 adapter（参数与 `ResponsesClient::new` 透传对齐）。
    pub fn new(transport: T, provider: Provider, auth: SharedAuthProvider) -> Self {
        Self {
            inner: ResponsesClient::new(transport, provider, auth),
        }
    }

    /// 注入 telemetry（与 `ResponsesClient::with_telemetry` 透传对齐）。
    pub fn with_telemetry(
        self,
        request: Option<Arc<dyn RequestTelemetry>>,
        sse: Option<Arc<dyn SseTelemetry>>,
    ) -> Self {
        Self {
            inner: self.inner.with_telemetry(request, sse),
        }
    }
}

impl<T: HttpTransport> LanguageModel for OpenaiResponsesAdapter<T> {
    fn stream(
        &self,
        request: UnifiedRequest,
        options: UnifiedRequestOptions,
    ) -> BoxFuture<'_, Result<UnifiedEventStream, UnifiedError>> {
        Box::pin(async move {
            // 中立请求 / 选项 → Responses wire 请求 / 选项（机械映射，编译期穷尽）。
            let api_request: ResponsesApiRequest = request.into();
            let api_options: ResponsesOptions = options.into();
            // 委托现有 client；具体协议错误原样透传为 Passthrough，调用方可 downcast 回 ApiError。
            let api_stream = self
                .inner
                .stream_request(api_request, api_options)
                .await
                .map_err(UnifiedError::passthrough)?;
            Ok(response_stream_to_unified(api_stream))
        })
    }
}

/// 将 `ResponseStream`（Item = `Result<ResponseEvent, ApiError>`）归一为中立事件流。
///
/// 采用零拷贝包装：不 spawn 任何 task，仅包一层 `MappedResponseStream` 在 poll 时
/// 逐事件做 `ResponseEvent → UnifiedEvent` / `ApiError → UnifiedError` 映射。
///
/// 对外暴露：WebSocket 等「不经 `LanguageModel` trait」的路径可在边界处调用本函数，
/// 把 Responses 事件流归一为中立流，再复用 core 侧统一的事件消费逻辑。
pub fn response_stream_to_unified(stream: ResponseStream) -> UnifiedEventStream {
    let upstream_request_id = stream.upstream_request_id.clone();
    UnifiedEventStream::new(MappedResponseStream { inner: stream }, upstream_request_id)
}

/// `ResponseStream` 的事件归一包装。`ResponseStream` 仅含 mpsc::Receiver + Option<String>，
/// 自动满足 `Unpin + Send + Sync + 'static`，故本包装同样满足。
struct MappedResponseStream {
    inner: ResponseStream,
}

impl Stream for MappedResponseStream {
    type Item = Result<UnifiedEvent, UnifiedError>;

    fn poll_next(self: Pin<&mut Self>, cx: &mut Context<'_>) -> Poll<Option<Self::Item>> {
        let inner = Pin::new(&mut self.get_mut().inner);
        Stream::poll_next(inner, cx).map(|opt| {
            opt.map(|res| {
                res.map(UnifiedEvent::from)
                    .map_err(UnifiedError::passthrough)
            })
        })
    }
}

// ---------------------------------------------------------------------------
// 中立格式 ↔ Responses wire 格式 的机械映射（From 实现，编译期穷尽保证）
// ---------------------------------------------------------------------------

impl From<UnifiedRequest> for ResponsesApiRequest {
    fn from(u: UnifiedRequest) -> Self {
        Self {
            model: u.model,
            instructions: u.instructions,
            input: u.input,
            tools: u.tools,
            tool_choice: u.tool_choice,
            parallel_tool_calls: u.parallel_tool_calls,
            reasoning: u.reasoning.map(Reasoning::from),
            store: u.store,
            stream: u.stream,
            include: u.include,
            service_tier: u.service_tier,
            prompt_cache_key: u.prompt_cache_key,
            text: u.text.map(TextControls::from),
            client_metadata: u.client_metadata,
        }
    }
}

impl From<UnifiedReasoning> for Reasoning {
    fn from(u: UnifiedReasoning) -> Self {
        Self {
            effort: u.effort,
            summary: u.summary,
            context: u.context.map(ReasoningContext::from),
        }
    }
}

impl From<UnifiedReasoningContext> for ReasoningContext {
    fn from(u: UnifiedReasoningContext) -> Self {
        match u {
            UnifiedReasoningContext::Auto => Self::Auto,
            UnifiedReasoningContext::CurrentTurn => Self::CurrentTurn,
            UnifiedReasoningContext::AllTurns => Self::AllTurns,
        }
    }
}

impl From<UnifiedTextControls> for TextControls {
    fn from(u: UnifiedTextControls) -> Self {
        Self {
            verbosity: u.verbosity.map(OpenAiVerbosity::from),
            format: u.format.map(TextFormat::from),
        }
    }
}

impl From<UnifiedOpenAiVerbosity> for OpenAiVerbosity {
    fn from(u: UnifiedOpenAiVerbosity) -> Self {
        match u {
            UnifiedOpenAiVerbosity::Low => Self::Low,
            UnifiedOpenAiVerbosity::Medium => Self::Medium,
            UnifiedOpenAiVerbosity::High => Self::High,
        }
    }
}

impl From<UnifiedTextFormat> for TextFormat {
    fn from(u: UnifiedTextFormat) -> Self {
        Self {
            r#type: u.r#type.into(),
            strict: u.strict,
            schema: u.schema,
            name: u.name,
        }
    }
}

impl From<UnifiedTextFormatType> for TextFormatType {
    fn from(u: UnifiedTextFormatType) -> Self {
        match u {
            UnifiedTextFormatType::JsonSchema => Self::JsonSchema,
        }
    }
}

impl From<UnifiedRequestOptions> for ResponsesOptions {
    fn from(u: UnifiedRequestOptions) -> Self {
        Self {
            session_id: u.session_id,
            thread_id: u.thread_id,
            session_source: u.session_source,
            extra_headers: u.extra_headers,
            compression: u.compression.into(),
            turn_state: u.turn_state,
        }
    }
}

impl From<UnifiedCompression> for Compression {
    fn from(u: UnifiedCompression) -> Self {
        match u {
            UnifiedCompression::None => Self::None,
            UnifiedCompression::Zstd => Self::Zstd,
        }
    }
}

impl From<ResponseEvent> for UnifiedEvent {
    fn from(e: ResponseEvent) -> Self {
        match e {
            ResponseEvent::Created => Self::Created,
            ResponseEvent::SafetyBuffering(s) => Self::SafetyBuffering(s.into()),
            ResponseEvent::OutputItemDone(i) => Self::OutputItemDone(i),
            ResponseEvent::OutputItemAdded(i) => Self::OutputItemAdded(i),
            ResponseEvent::ServerModel(m) => Self::ServerModel(m),
            ResponseEvent::ModelVerifications(v) => Self::ModelVerifications(v),
            ResponseEvent::TurnModerationMetadata(m) => Self::TurnModerationMetadata(m),
            ResponseEvent::ServerReasoningIncluded(b) => Self::ServerReasoningIncluded(b),
            ResponseEvent::Completed {
                response_id,
                token_usage,
                end_turn,
            } => Self::Completed {
                response_id,
                token_usage,
                end_turn,
            },
            ResponseEvent::OutputTextDelta(s) => Self::OutputTextDelta(s),
            ResponseEvent::ToolCallInputDelta {
                item_id,
                call_id,
                delta,
            } => Self::ToolCallInputDelta {
                item_id,
                call_id,
                delta,
            },
            ResponseEvent::ReasoningSummaryDelta {
                delta,
                summary_index,
            } => Self::ReasoningSummaryDelta {
                delta,
                summary_index,
            },
            ResponseEvent::ReasoningContentDelta {
                delta,
                content_index,
            } => Self::ReasoningContentDelta {
                delta,
                content_index,
            },
            ResponseEvent::ReasoningSummaryPartAdded { summary_index } => {
                Self::ReasoningSummaryPartAdded { summary_index }
            }
            ResponseEvent::RateLimits(r) => Self::RateLimits(r),
            ResponseEvent::ModelsEtag(s) => Self::ModelsEtag(s),
        }
    }
}

impl From<SafetyBuffering> for UnifiedSafetyBuffering {
    fn from(s: SafetyBuffering) -> Self {
        Self {
            use_cases: s.use_cases,
            reasons: s.reasons,
            show_buffering_ui: s.show_buffering_ui,
            faster_model: s.faster_model,
        }
    }
}

// ---------------------------------------------------------------------------
// 反向映射（Responses wire → 中立）：供 core 在请求侧把既有 `ResponsesApiRequest` /
// `ResponsesOptions` 转为中立类型后交给 adapter；事件侧由 core 在循环顶部把
// `UnifiedEvent` → `ResponseEvent`（`From<UnifiedEvent> for ResponseEvent`）。
// 与上方正向 From 一一对应，同样由编译期保证穷尽。
// ---------------------------------------------------------------------------

impl From<ResponsesApiRequest> for UnifiedRequest {
    fn from(r: ResponsesApiRequest) -> Self {
        Self {
            model: r.model,
            instructions: r.instructions,
            input: r.input,
            tools: r.tools,
            tool_choice: r.tool_choice,
            parallel_tool_calls: r.parallel_tool_calls,
            reasoning: r.reasoning.map(UnifiedReasoning::from),
            store: r.store,
            stream: r.stream,
            include: r.include,
            service_tier: r.service_tier,
            prompt_cache_key: r.prompt_cache_key,
            text: r.text.map(UnifiedTextControls::from),
            client_metadata: r.client_metadata,
        }
    }
}

impl From<Reasoning> for UnifiedReasoning {
    fn from(r: Reasoning) -> Self {
        Self {
            effort: r.effort,
            summary: r.summary,
            context: r.context.map(UnifiedReasoningContext::from),
        }
    }
}

impl From<ReasoningContext> for UnifiedReasoningContext {
    fn from(r: ReasoningContext) -> Self {
        match r {
            ReasoningContext::Auto => Self::Auto,
            ReasoningContext::CurrentTurn => Self::CurrentTurn,
            ReasoningContext::AllTurns => Self::AllTurns,
        }
    }
}

impl From<TextControls> for UnifiedTextControls {
    fn from(t: TextControls) -> Self {
        Self {
            verbosity: t.verbosity.map(UnifiedOpenAiVerbosity::from),
            format: t.format.map(UnifiedTextFormat::from),
        }
    }
}

impl From<OpenAiVerbosity> for UnifiedOpenAiVerbosity {
    fn from(v: OpenAiVerbosity) -> Self {
        match v {
            OpenAiVerbosity::Low => Self::Low,
            OpenAiVerbosity::Medium => Self::Medium,
            OpenAiVerbosity::High => Self::High,
        }
    }
}

impl From<TextFormat> for UnifiedTextFormat {
    fn from(t: TextFormat) -> Self {
        Self {
            r#type: t.r#type.into(),
            strict: t.strict,
            schema: t.schema,
            name: t.name,
        }
    }
}

impl From<TextFormatType> for UnifiedTextFormatType {
    fn from(t: TextFormatType) -> Self {
        match t {
            TextFormatType::JsonSchema => Self::JsonSchema,
        }
    }
}

impl From<ResponsesOptions> for UnifiedRequestOptions {
    fn from(o: ResponsesOptions) -> Self {
        Self {
            session_id: o.session_id,
            thread_id: o.thread_id,
            session_source: o.session_source,
            extra_headers: o.extra_headers,
            compression: o.compression.into(),
            turn_state: o.turn_state,
        }
    }
}

impl From<Compression> for UnifiedCompression {
    fn from(c: Compression) -> Self {
        match c {
            Compression::None => Self::None,
            Compression::Zstd => Self::Zstd,
        }
    }
}

impl From<UnifiedEvent> for ResponseEvent {
    fn from(e: UnifiedEvent) -> Self {
        match e {
            UnifiedEvent::Created => Self::Created,
            UnifiedEvent::SafetyBuffering(s) => Self::SafetyBuffering(s.into()),
            UnifiedEvent::OutputItemDone(i) => Self::OutputItemDone(i),
            UnifiedEvent::OutputItemAdded(i) => Self::OutputItemAdded(i),
            UnifiedEvent::ServerModel(m) => Self::ServerModel(m),
            UnifiedEvent::ModelVerifications(v) => Self::ModelVerifications(v),
            UnifiedEvent::TurnModerationMetadata(m) => Self::TurnModerationMetadata(m),
            UnifiedEvent::ServerReasoningIncluded(b) => Self::ServerReasoningIncluded(b),
            UnifiedEvent::Completed {
                response_id,
                token_usage,
                end_turn,
            } => Self::Completed {
                response_id,
                token_usage,
                end_turn,
            },
            UnifiedEvent::OutputTextDelta(s) => Self::OutputTextDelta(s),
            UnifiedEvent::ToolCallInputDelta {
                item_id,
                call_id,
                delta,
            } => Self::ToolCallInputDelta {
                item_id,
                call_id,
                delta,
            },
            UnifiedEvent::ReasoningSummaryDelta {
                delta,
                summary_index,
            } => Self::ReasoningSummaryDelta {
                delta,
                summary_index,
            },
            UnifiedEvent::ReasoningContentDelta {
                delta,
                content_index,
            } => Self::ReasoningContentDelta {
                delta,
                content_index,
            },
            UnifiedEvent::ReasoningSummaryPartAdded { summary_index } => {
                Self::ReasoningSummaryPartAdded { summary_index }
            }
            UnifiedEvent::RateLimits(r) => Self::RateLimits(r),
            UnifiedEvent::ModelsEtag(s) => Self::ModelsEtag(s),
        }
    }
}

impl From<UnifiedSafetyBuffering> for SafetyBuffering {
    fn from(s: UnifiedSafetyBuffering) -> Self {
        Self {
            use_cases: s.use_cases,
            reasons: s.reasons,
            show_buffering_ui: s.show_buffering_ui,
            faster_model: s.faster_model,
        }
    }
}

#[cfg(test)]
mod tests {
    //! 运行期 spot check：编译期穷尽性由 From 实现保证，这里只校验「取值正确」，
    //! 防止机械映射里把字段 / 变体贴错（如 reasoning.context、text.format.r#type）。

    use super::*;

    #[test]
    fn maps_request_with_reasoning_and_text() {
        let u = UnifiedRequest {
            model: "m".to_string(),
            instructions: String::new(),
            input: vec![],
            tools: None,
            tool_choice: "auto".to_string(),
            parallel_tool_calls: false,
            reasoning: Some(UnifiedReasoning {
                effort: None,
                summary: None,
                context: Some(UnifiedReasoningContext::CurrentTurn),
            }),
            store: false,
            stream: true,
            include: vec![],
            service_tier: None,
            prompt_cache_key: None,
            text: Some(UnifiedTextControls {
                verbosity: Some(UnifiedOpenAiVerbosity::Low),
                format: Some(UnifiedTextFormat {
                    r#type: UnifiedTextFormatType::JsonSchema,
                    strict: true,
                    schema: serde_json::json!({"type": "object"}),
                    name: "n".to_string(),
                }),
            }),
            client_metadata: None,
        };
        let r: ResponsesApiRequest = u.into();
        assert_eq!(r.model.as_str(), "m");
        assert!(r.stream);

        let Some(reasoning) = r.reasoning else {
            panic!("reasoning 未映射");
        };
        let Some(context) = reasoning.context else {
            panic!("context 未映射");
        };
        assert!(matches!(context, ReasoningContext::CurrentTurn));

        let Some(text) = r.text else {
            panic!("text 未映射");
        };
        let Some(verbosity) = text.verbosity else {
            panic!("verbosity 未映射");
        };
        assert!(matches!(verbosity, OpenAiVerbosity::Low));
        let Some(fmt) = text.format else {
            panic!("format 未映射");
        };
        assert!(fmt.strict);
        assert!(matches!(fmt.r#type, TextFormatType::JsonSchema));
        assert_eq!(fmt.name.as_str(), "n");
    }

    #[test]
    fn maps_compression() {
        assert!(matches!(
            Compression::from(UnifiedCompression::None),
            Compression::None
        ));
        assert!(matches!(
            Compression::from(UnifiedCompression::Zstd),
            Compression::Zstd
        ));
    }

    #[test]
    fn maps_events() {
        // 简单变体
        assert!(matches!(
            UnifiedEvent::from(ResponseEvent::Created),
            UnifiedEvent::Created
        ));
        assert!(matches!(
            UnifiedEvent::from(ResponseEvent::ServerReasoningIncluded(true)),
            UnifiedEvent::ServerReasoningIncluded(true)
        ));

        // 结构体变体 Completed：字段逐个断言
        let ev = UnifiedEvent::from(ResponseEvent::Completed {
            response_id: "r1".to_string(),
            token_usage: None,
            end_turn: Some(true),
        });
        match ev {
            UnifiedEvent::Completed {
                response_id,
                token_usage,
                end_turn,
            } => {
                assert_eq!(response_id.as_str(), "r1");
                assert!(token_usage.is_none());
                assert_eq!(end_turn, Some(true));
            }
            _ => panic!("期望 Completed 变体"),
        }

        // SafetyBuffering 子结构字段映射
        let ev = UnifiedEvent::from(ResponseEvent::SafetyBuffering(SafetyBuffering {
            use_cases: vec!["a".to_string()],
            reasons: vec![],
            show_buffering_ui: true,
            faster_model: Some("f".to_string()),
        }));
        match ev {
            UnifiedEvent::SafetyBuffering(s) => {
                assert!(s.show_buffering_ui);
                assert_eq!(s.faster_model.as_deref(), Some("f"));
                assert_eq!(s.use_cases.len(), 1);
            }
            _ => panic!("期望 SafetyBuffering 变体"),
        }
    }
}
