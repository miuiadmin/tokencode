//! Anthropic Messages 协议 client —— 与 `ChatClient` 同构。
//!
//! 差异仅两处：① 路径改为 `messages`；② SSE 解析改用 `spawn_anthropic_stream`
//! （Anthropic 的 `message_stop` 收尾 + content block 累积装配）。`EndpointSession`
//! （URL 拼接 / auth / 重试）、`ResponsesOptions`（传输层选项，协议无关）原样复用。
//!
//! 认证形态与 Responses / Chat 不同：Anthropic 用 `x-api-key` + `anthropic-version`
//! 而非 `Authorization: Bearer`。该改写由 [`AnthropicAuth`] 承担（adapter 构造时包一层内层
//! provider），client 本身协议无关地消费任意 `SharedAuthProvider`。

use crate::auth::AuthProvider;
use crate::auth::SharedAuthProvider;
use crate::common::AnthropicApiRequest;
use crate::common::ResponseStream;
use crate::endpoint::session::EndpointSession;
use crate::endpoint::ResponsesOptions;
use crate::error::ApiError;
use crate::provider::Provider;
use crate::requests::Compression;
use crate::requests::headers::build_session_headers;
use crate::requests::headers::insert_header;
use crate::requests::headers::subagent_header;
use crate::sse::spawn_anthropic_stream;
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

/// Anthropic Messages API 版本头值（固定 `2023-06-01`，对应稳定的 Messages 协议）。
const ANTHROPIC_VERSION: &str = "2023-06-01";

/// Anthropic 认证包装：把内层 provider 产出的 `Authorization: Bearer <key>` 改写为
/// Anthropic 的 `x-api-key: <key>` + `anthropic-version`。
///
/// Anthropic 的 API key 不走 OAuth Bearer（Bearer 只认 OAuth token，API key 走 Bearer 会被
/// 服务端 401）。本包装满足架构文档「认证零改动 / 只新增不改既有」原则：内层 provider 与 core
/// 的 Bearer auth 流不动，仅在构造 Anthropic client 时外面包一层。
///
/// 实现：复写 `add_auth_headers`（`apply_auth` 默认实现会调用它）——先让内层 provider 填充
/// 头，再抽出 Bearer token、删除 `Authorization`、改写为 `x-api-key`，并追加 `anthropic-version`。
pub struct AnthropicAuth {
    inner: SharedAuthProvider,
}

impl AnthropicAuth {
    pub fn new(inner: SharedAuthProvider) -> Self {
        Self { inner }
    }
}

/// 大小写不敏感地剥离 `Bearer ` 前缀；非该前缀则原样返回（视作裸 key，兼容非 Bearer 形态）。
///
/// 用 `get(..PREFIX.len())` 而非 `raw[..PREFIX.len()]`：前者在字节边界不在字符边界时返回
/// None（安全），后者会 panic。现网调用点经 `HeaderValue::to_str()` 只产单字节 ASCII（可见
/// 字符与制表符 0x09，恒在字节边界），但本函数签名接受任意 `&str`，`get` 让任意复用都
/// panic-safe。
fn strip_bearer_prefix(raw: &str) -> &str {
    const PREFIX: &str = "bearer ";
    match raw.get(..PREFIX.len()) {
        Some(prefix) if prefix.eq_ignore_ascii_case(PREFIX) => &raw[PREFIX.len()..],
        _ => raw,
    }
}

impl AuthProvider for AnthropicAuth {
    fn add_auth_headers(&self, headers: &mut HeaderMap) {
        // 1. 先让内层 provider 填充（通常产出 `Authorization: Bearer <key>`）。
        self.inner.add_auth_headers(headers);

        // 2. 抽出 Bearer token，改写为 x-api-key（大小写不敏感地剥离 Bearer 前缀，兼容
        //    `Bearer`/`bearer`/`BEARER` 等大小写；非 Bearer 形态的裸值原样作为 key）。
        //    任一异常（无 Authorization 头 / 值为空 / 含非法 header 字节）都 warn 告警，避免
        //    静默发出无 x-api-key 的请求导致难以排查的 401。
        match headers.remove(http::header::AUTHORIZATION) {
            Some(auth_value) => match auth_value.to_str() {
                Ok(raw) => {
                    let token = strip_bearer_prefix(raw).trim();
                    if token.is_empty() {
                        warn!(
                            "Anthropic 认证：内层 Authorization 头值为空，x-api-key 未设置（请求将被服务端 401）"
                        );
                    } else if let Ok(value) = HeaderValue::from_str(token) {
                        headers.insert("x-api-key", value);
                    } else {
                        // 仅记长度，不记 token 本身——避免把（可能含可识别片段的）凭证写进日志。
                        warn!(
                            token_len = token.len(),
                            "Anthropic 认证：token 含非法 header 字节，x-api-key 未设置"
                        );
                    }
                }
                Err(_) => {
                    // 值含非 ASCII / 控制字节，to_str() 失败 → 既无法诊断内容也无法构造合法 x-api-key。
                    // 仅记字节长度，不碰值本身（可能含凭证片段）。
                    warn!(
                        value_len = auth_value.as_bytes().len(),
                        "Anthropic 认证：Authorization 头值含非 ASCII 字节，x-api-key 未设置（请求将被服务端 401）"
                    );
                }
            },
            None => {
                warn!(
                    "Anthropic 认证：内层未产出 Authorization 头，x-api-key 未设置（请求将被服务端 401）"
                );
            }
        }

        // 3. 剥离内层 provider 可能附带的 OpenAI 专属头（账户标识 / FedRAMP 标记）——它们对
        //    Anthropic 无意义，且 ChatGPT-Account-ID 会把账户标识泄露给 Anthropic。仅清 auth 层
        //    产出的这两项；用户自配的 provider http_headers 不经此层（由 EndpointSession 另加），不受影响。
        headers.remove("ChatGPT-Account-ID");
        headers.remove("X-OpenAI-Fedramp");

        // 4. 固定追加 anthropic-version（Messages 协议必填）。
        headers.insert(
            "anthropic-version",
            HeaderValue::from_static(ANTHROPIC_VERSION),
        );
    }
}

/// Anthropic Messages client（`POST {base_url}/v1/messages`，流式）。
pub struct AnthropicClient<T: HttpTransport> {
    session: EndpointSession<T>,
    sse_telemetry: Option<Arc<dyn SseTelemetry>>,
}

impl<T: HttpTransport> AnthropicClient<T> {
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
        name = "anthropic.stream_request",
        level = "info",
        skip_all,
        fields(
            transport = "anthropic_http",
            http.method = "POST",
            api.path = "v1/messages"
        )
    )]
    pub async fn stream_request(
        &self,
        request: AnthropicApiRequest,
        options: ResponsesOptions,
    ) -> Result<ResponseStream, ApiError> {
        // 守卫：Anthropic 协议要求 messages 至少一条；空 messages 会被服务端 400。守卫下沉到
        // client 层（而非只在 adapter），使 pub stream_request 的任意直接调用方都受保护。
        if request.messages.is_empty() {
            return Err(ApiError::Stream(
                "Anthropic 请求翻译后无任何 message（Anthropic 协议要求至少一条 message）".to_string(),
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
            .map_err(|e| ApiError::Stream(format!("failed to encode anthropic request: {e}")))?;

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

        let stream_response = self
            .session
            .stream_encoded_json_with(
                Method::POST,
                Self::path(),
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

        Ok(spawn_anthropic_stream(
            stream_response,
            self.session.provider().stream_idle_timeout,
            self.sse_telemetry.clone(),
            turn_state,
        ))
    }

    fn path() -> &'static str {
        // Anthropic 约定 base_url 为根（如 `https://api.anthropic.com`），endpoint 含 `v1/`，
        // 故 path 用 `v1/messages`（与 OpenAI 约定 base 含 `/v1`、path 用 `chat/completions` 不同）。
        // ⚠️ 配置 base_url 时切勿带 `/v1`，否则会拼成 `/v1/v1/messages` 导致 404。
        "v1/messages"
    }
}
