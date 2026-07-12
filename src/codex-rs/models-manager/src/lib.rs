pub(crate) mod cache;
pub mod collaboration_mode_presets;
pub(crate) mod config;
pub mod manager;
pub mod model_info;
pub mod model_presets;
pub mod test_support;

pub use codex_protocol::auth::AuthMode;
pub use config::ModelsManagerConfig;

/// Load the bundled model catalog shipped with `codex-models-manager`.
pub fn bundled_models_response()
-> std::result::Result<codex_protocol::openai_models::ModelsResponse, serde_json::Error> {
    serde_json::from_str(include_str!("../models.json"))
}

/// 按 model slug 查询其绑定的 `provider_id`（来自内置 models.json）。
///
/// 供 config 加载阶段在不持有 `ModelsManagerConfig` 时推导 provider：model 声明了
/// `provider_id` 则返回它；未声明或 slug 不在内置目录则返回 None（由调用方回落到
/// 显式 `model_provider` 或默认 openai）。同步、无 IO，避开 config 加载的鸡蛋问题。
pub fn provider_id_for_model(slug: &str) -> Option<String> {
    bundled_models_response()
        .ok()?
        .models
        .into_iter()
        .find(|model| model.slug == slug)
        .and_then(|model| model.provider_id)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn provider_id_for_known_cross_vendor_models() {
        assert_eq!(provider_id_for_model("glm-5.2"), Some("glm".to_string()));
        assert_eq!(
            provider_id_for_model("claude-opus-4-8"),
            Some("anthropic".to_string())
        );
        assert_eq!(
            provider_id_for_model("minimax-m3"),
            Some("minimax".to_string())
        );
        assert_eq!(
            provider_id_for_model("kimi-k2.6"),
            Some("moonshot".to_string())
        );
        assert_eq!(
            provider_id_for_model("gpt-5.6-sol"),
            Some("openai".to_string())
        );
    }

    #[test]
    fn provider_id_for_legacy_or_unknown_model_is_none() {
        // 旧 GPT 模型未声明 provider_id —— 回落调用方默认逻辑
        assert_eq!(provider_id_for_model("gpt-5.5"), None);
        // 未知 slug
        assert_eq!(provider_id_for_model("does-not-exist"), None);
    }
}

/// Convert the client version string to a whole version string (e.g. "1.2.3-alpha.4" -> "1.2.3").
pub fn client_version_to_whole() -> String {
    format!(
        "{}.{}.{}",
        env!("CARGO_PKG_VERSION_MAJOR"),
        env!("CARGO_PKG_VERSION_MINOR"),
        env!("CARGO_PKG_VERSION_PATCH")
    )
}
