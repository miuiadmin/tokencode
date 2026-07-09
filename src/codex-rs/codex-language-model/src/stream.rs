use crate::UnifiedError;
use crate::UnifiedEvent;
use futures::Stream;
use std::pin::Pin;
use std::task::Context;
use std::task::Poll;

/// 中立事件流。包装一个 `Stream<Item = Result<UnifiedEvent, UnifiedError>>`。
///
/// 不绑定具体 runtime（trait crate 不依赖 tokio）；adapter 在各自 crate 内用任意
/// runtime 构造底层 stream，再 Box::pin 注入。
pub struct UnifiedEventStream {
    inner: Pin<Box<dyn Stream<Item = Result<UnifiedEvent, UnifiedError>> + Send + Sync>>,
    /// 远端请求标识（如 x-request-id），用于追踪 / telemetry。
    pub upstream_request_id: Option<String>,
}

impl UnifiedEventStream {
    /// 由任意满足约束的 stream 构造。
    pub fn new<S>(stream: S, upstream_request_id: Option<String>) -> Self
    where
        S: Stream<Item = Result<UnifiedEvent, UnifiedError>> + Send + Sync + 'static,
    {
        Self {
            inner: Box::pin(stream),
            upstream_request_id,
        }
    }
}

impl Stream for UnifiedEventStream {
    type Item = Result<UnifiedEvent, UnifiedError>;

    fn poll_next(mut self: Pin<&mut Self>, cx: &mut Context<'_>) -> Poll<Option<Self::Item>> {
        self.inner.as_mut().poll_next(cx)
    }
}
