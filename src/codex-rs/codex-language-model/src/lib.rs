//! 模型接入层抽象：中立 `LanguageModel` trait 与统一请求 / 事件 schema。
//!
//! 本 crate 处于 Layer 0.5 —— 仅依赖 codex-protocol，不依赖任何具体协议实现，
//! 以保证 trait 契约对所有模型后端中立。各协议 adapter（OpenaiResponses / Anthropic /
//! Gemini 等）在更上层 crate 实现本 trait，做「中立格式 ↔ 厂商 wire format」双向转换。

mod adapter_type;
mod error;
mod event;
mod request;
mod stream;
mod traits;

pub use adapter_type::AdapterType;
pub use error::UnifiedError;
pub use event::{UnifiedEvent, UnifiedSafetyBuffering};
pub use request::{
    UnifiedCompression, UnifiedOpenAiVerbosity, UnifiedReasoning, UnifiedReasoningContext,
    UnifiedRequest, UnifiedRequestOptions, UnifiedTextControls, UnifiedTextFormat,
    UnifiedTextFormatType,
};
pub use stream::UnifiedEventStream;
pub use traits::LanguageModel;
