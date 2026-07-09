use crate::UnifiedError;
use crate::UnifiedEventStream;
use crate::UnifiedRequest;
use crate::UnifiedRequestOptions;
use futures::future::BoxFuture;

/// 模型接入层抽象。一个实现代表一种「说话方式」的模型后端。
///
/// trait 不持有 transport / auth —— 这些由 adapter 内部封装（如 OpenaiResponsesAdapter
/// 包装 ResponsesClient）。调用方只需提供中立请求，adapter 返回中立事件流。
///
/// P0 用泛型分派（impl LanguageModel for 具体 adapter），不强制 dyn-safe。
/// P1 若需运行时 registry，可将 stream 的返回生命周期收紧为 'static 以支持 `Arc<dyn>`。
pub trait LanguageModel: Send + Sync {
    /// 发起一次流式推理，返回中立事件流。
    ///
    /// 401 重试 / telemetry 注入等顶层策略由调用方（core）在外层 loop 处理，
    /// adapter 只负责单次请求的协议转换与 SSE 归一。
    fn stream(
        &self,
        request: UnifiedRequest,
        options: UnifiedRequestOptions,
    ) -> BoxFuture<'_, Result<UnifiedEventStream, UnifiedError>>;
}
