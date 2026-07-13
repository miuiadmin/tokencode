//! Gemini generateContent 协议 client —— 与 `AnthropicClient` 同构。
//!
//! 差异仅两处：① 路径动态含 model（`v1beta/models/{model}:streamGenerateContent?alt=sse`，
//!   Gemini 的 model 在 URL path 而非 body）；② SSE 解析改用 `spawn_gemini_stream`。
//! `EndpointSession`（URL 拼接 / auth / 重试）、`ResponsesOptions`（传输层选项，协议无关）原样复用。
//!
//! 认证形态与 Responses / Chat / Anthropic 不同：Gemini AI Studio 用 header `x-goog-api-key`
//! 而非 `Authorization: Bearer`。该改写由 [`GeminiAuth`] 承担（adapter 构造时包一层内层 provider）。

use crate::auth::AuthProvider;
use crate::auth::SharedAuthProvider;
use crate::common::GeminiApiRequest;
use crate::common::ResponseStream;
use crate::endpoint::ResponsesOptions;
use crate::endpoint::session::EndpointSession;
use crate::error::ApiError;
use crate::provider::Provider;
use crate::requests::Compression;
use crate::requests::headers::build_session_headers;
use crate::requests::headers::insert_header;
use crate::requests::headers::subagent_header;
use crate::sse::spawn_gemini_stream;
use crate::telemetry::SseTelemetry;
use codex_client::EncodedJsonBody;
use codex_client::HttpTransport;
use codex_client::RequestCompression;
use codex_client::RequestTelemetry;
use http::HeaderValue;
use http::HeaderMap;
use http::Method;
use std::sync::Arc;
use tracing::instrument;
use tracing::warn;

/// Gemini 认证包装：把内层 provider 产出的 `Authorization: Bearer <key>` 改写为
/// Gemini 的 `x-goog-api-key: <key>`。
///
/// Gemini AI Studio 的 API key 不走 OAuth Bearer（Bearer 只认 OAuth token，API key 走 Bearer 会被
/// 服务端 401）。本包装满足架构文档「认证零改动 / 只新增不改既有」原则：内层 provider 与 core
/// 的 Bearer auth 流不动，仅在构造 Gemini client 时外面包一层。key 走 header 而非 query `?key=`——
/// 避免凭证进 URL / 访问日志。
///
/// 实现与 `AnthropicAuth` 同构（复写 `add_auth_headers`）：先让内层 provider 填充头，再抽出 Bearer
/// token、删除 `Authorization`、改写为 `x-goog-api-key`。Gemini 无版本头（版本在 URL path 的 `v1beta`）。
pub struct GeminiAuth {
    inner: SharedAuthProvider,
}

impl GeminiAuth {
    pub fn new(inner: SharedAuthProvider) -> Self {
        Self { inner }
    }
}

/// 大小写不敏感地剥离 `Bearer ` 前缀；非该前缀则原样返回（视作裸 key，兼容非 Bearer 形态）。
///
/// 用 `get(..PREFIX.len())` 而非 `raw[..PREFIX.len()]`：前者在字节边界不在字符边界时返回
/// None（安全），后者会 panic。与 `anthropic::strip_bearer_prefix` 同构（两协议共用 Bearer 改写逻辑）。
fn strip_bearer_prefix(raw: &str) -> &str {
    const PREFIX: &str = "bearer ";
    match raw.get(..PREFIX.len()) {
        Some(prefix) if prefix.eq_ignore_ascii_case(PREFIX) => &raw[PREFIX.len()..],
        _ => raw,
    }
}

impl AuthProvider for GeminiAuth {
    fn add_auth_headers(&self, headers: &mut HeaderMap) {
        // 1. 先让内层 provider 填充（通常产出 `Authorization: Bearer <key>`）。
        self.inner.add_auth_headers(headers);

        // 2. 抽出 Bearer token，改写为 x-goog-api-key。任一异常 warn 告警，避免静默发出无 key 请求。
        match headers.remove(http::header::AUTHORIZATION) {
            Some(auth_value) => match auth_value.to_str() {
                Ok(raw) => {
                    let token = strip_bearer_prefix(raw).trim();
                    if token.is_empty() {
                        warn!(
                            "Gemini 认证：内层 Authorization 头值为空，x-goog-api-key 未设置（请求将被服务端 401）"
                        );
                    } else if let Ok(value) = HeaderValue::from_str(token) {
                        headers.insert("x-goog-api-key", value);
                    } else {
                        // 仅记长度，不记 token 本身——避免把凭证片段写进日志。
                        warn!(
                            token_len = token.len(),
                            "Gemini 认证：token 含非法 header 字节，x-goog-api-key 未设置"
                        );
                    }
                }
                Err(_) => {
                    warn!(
                        value_len = auth_value.as_bytes().len(),
                        "Gemini 认证：Authorization 头值含非 ASCII 字节，x-goog-api-key 未设置（请求将被服务端 401）"
                    );
                }
            },
            None => {
                warn!(
                    "Gemini 认证：内层未产出 Authorization 头，x-goog-api-key 未设置（请求将被服务端 401）"
                );
            }
        }

        // 3. 剥离内层 provider 可能附带的 OpenAI 专属头（账户标识 / FedRAMP 标记）——对 Gemini
        //    无意义，且 ChatGPT-Account-ID 会把账户标识泄露给 Google。仅清 auth 层产出的这两项；
        //    用户自配的 provider http_headers 不经此层（由 EndpointSession 另加），不受影响。
        headers.remove("ChatGPT-Account-ID");
        headers.remove("X-OpenAI-Fedramp");
    }
}

/// Gemini generateContent client（`POST {base_url}/v1beta/models/{model}:streamGenerateContent?alt=sse`）。
///
/// 与 `AnthropicClient` 同构；差异：path 动态含 model（model 单独传参，api_request 不含 model 字段），
/// 且 path 末尾固定带 `?alt=sse`（强制 SSE 流式——缺省会返回 NDJSON 数组）。
pub struct GeminiClient<T: HttpTransport> {
    session: EndpointSession<T>,
    sse_telemetry: Option<Arc<dyn SseTelemetry>>,
}

impl<T: HttpTransport> GeminiClient<T> {
    pub fn new(transport: T, provider: Provider, auth: SharedAuthProvider) -> Self {
        Self {
            session: EndpointSession::new(transport, provider, auth),
            sse_telemetry: None,
        }
    }

    pub fn with_telemetry(
        self,
        request: Option<Arc<dyn RequestTelemetry>>,
        sse: Option<Arc<dyn SseTelemetry>>,
    ) -> Self {
        Self {
            session: self.session.with_request_telemetry(request),
            sse_telemetry: sse,
        }
    }

    #[instrument(
        name = "gemini.stream_request",
        level = "info",
        skip_all,
        fields(
            transport = "gemini_http",
            http.method = "POST",
            api.path = "streamGenerateContent"
        )
    )]
    pub async fn stream_request(
        &self,
        model: &str,
        request: GeminiApiRequest,
        options: ResponsesOptions,
    ) -> Result<ResponseStream, ApiError> {
        // 守卫：Gemini 协议要求 contents 至少一条；空 contents 会被服务端 400。守卫下沉到 client 层
        // （而非只在 adapter），使 pub stream_request 的任意直接调用方都受保护。
        if request.contents.is_empty() {
            return Err(ApiError::Stream(
                "Gemini 请求翻译后无任何 content（Gemini 协议要求至少一条 content）".to_string(),
            ));
        }

        let ResponsesOptions {
            session_id,
            thread_id,
            session_source,
            extra_headers,
            compression,
            turn_state,
        } = options;

        let body = EncodedJsonBody::encode(&request)
            .map_err(|e| ApiError::Stream(format!("failed to encode gemini request: {e}")))?;

        let mut headers = extra_headers;
        if let Some(ref thread_id) = thread_id {
            insert_header(&mut headers, "x-client-request-id", thread_id);
        }
        headers.extend(build_session_headers(session_id, thread_id));
        if let Some(subagent) = subagent_header(&session_source) {
            insert_header(&mut headers, "x-openai-subagent", &subagent);
        }

        let request_compression = match compression {
            Compression::None => RequestCompression::None,
            Compression::Zstd => RequestCompression::Zstd,
        };

        // path 动态含 model（Gemini model 在 URL path 而非 body）；`alt=sse` 必须带（否则返回
        // NDJSON 数组而非 SSE 流）。base_url 约定为根（如 `https://generativelanguage.googleapis.com`），
        // 故 path 用 `v1beta/models/...`（⚠️ 配置 base_url 时切勿带 `/v1beta`，否则拼成双段）。
        let path = format!("v1beta/models/{model}:streamGenerateContent?alt=sse");

        let stream_response = self
            .session
            .stream_encoded_json_with(
                Method::POST,
                &path,
                headers,
                Some(body),
                |req| {
                    req.headers.insert(
                        http::header::ACCEPT,
                        HeaderValue::from_static("text/event-stream"),
                    );
                    req.compression = request_compression;
                },
            )
            .await?;

        Ok(spawn_gemini_stream(
            stream_response,
            self.session.provider().stream_idle_timeout,
            self.sse_telemetry.clone(),
            turn_state,
        ))
    }
}
