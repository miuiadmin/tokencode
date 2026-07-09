use thiserror::Error;

/// 模型接入层统一错误。
///
/// 设计要点：`Passthrough` 变体以 `Box<dyn Error>` 承载 adapter 产出的具体协议错误，
/// 使调用方（core）能 downcast 回原始错误类型（如 codex_api::ApiError），从而保留
/// 既有的错误分类处理路径（如 401 重试），实现「行为零变化」。
#[derive(Debug, Error)]
pub enum UnifiedError {
    /// 内部 adapter 的具体错误原样透传。调用方可 downcast 取回原始类型。
    #[error("inner adapter error: {0}")]
    Passthrough(Box<dyn std::error::Error + Send + Sync>),

    /// 中立格式与 wire 格式之间的映射失败（字段缺失、变体未知等）。
    #[error("unified mapping error: {0}")]
    Mapping(String),
}

impl UnifiedError {
    /// 将一个具体协议错误包装为 Passthrough。
    pub fn passthrough<E>(err: E) -> Self
    where
        E: std::error::Error + Send + Sync + 'static,
    {
        Self::Passthrough(Box::new(err))
    }

    /// 尝试取回内部具体协议错误的引用，供调用方 downcast 回原始类型。
    pub fn as_passthrough(&self) -> Option<&(dyn std::error::Error + Send + Sync)> {
        match self {
            Self::Passthrough(b) => Some(b.as_ref()),
            _ => None,
        }
    }
}
