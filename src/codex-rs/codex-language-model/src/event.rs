use codex_protocol::models::ResponseItem;
use codex_protocol::protocol::ModelVerification;
use codex_protocol::protocol::RateLimitSnapshot;
use codex_protocol::protocol::TokenUsage;
use codex_protocol::protocol::TurnModerationMetadataEvent;

/// 中立事件流的一个事件。
///
/// P0 变体与 codex_api::ResponseEvent 1:1 对齐。各 adapter 把自家 SSE 事件归一成本
/// 类型，agent 循环只消费 UnifiedEvent（经 core 的 map_response_events 转成 TurnItem 等）。
#[derive(Debug)]
pub enum UnifiedEvent {
    Created,
    SafetyBuffering(UnifiedSafetyBuffering),
    OutputItemDone(ResponseItem),
    OutputItemAdded(ResponseItem),
    /// 服务端实际使用的模型（可能与请求模型不同，如安全路由）。
    ServerModel(String),
    /// 服务端建议的额外账户验证。
    ModelVerifications(Vec<ModelVerification>),
    /// 服务端返回的 turn moderation 元数据。
    TurnModerationMetadata(TurnModerationMetadataEvent),
    /// 服务端已计入历史 reasoning token，客户端无需再估算。
    ServerReasoningIncluded(bool),
    Completed {
        response_id: String,
        token_usage: Option<TokenUsage>,
        /// 模型是否明确结束 turn（None 表示未声明，由调用方回退判断）。
        end_turn: Option<bool>,
    },
    OutputTextDelta(String),
    ToolCallInputDelta {
        item_id: String,
        call_id: Option<String>,
        delta: String,
    },
    ReasoningSummaryDelta {
        delta: String,
        summary_index: i64,
    },
    ReasoningContentDelta {
        delta: String,
        content_index: i64,
    },
    ReasoningSummaryPartAdded {
        summary_index: i64,
    },
    RateLimits(RateLimitSnapshot),
    ModelsEtag(String),
}

/// 安全缓冲提示（对齐 codex_api::SafetyBuffering 语义）。
#[derive(Debug, Clone)]
pub struct UnifiedSafetyBuffering {
    pub use_cases: Vec<String>,
    pub reasons: Vec<String>,
    pub show_buffering_ui: bool,
    pub faster_model: Option<String>,
}
