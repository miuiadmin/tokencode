//! 按模型解析其绑定的 provider。
//!
//! provider 选择跟随 model：每条模型元数据可声明 `provider_id`，指向 config 的
//! `model_providers` map；未声明或未命中时回落到会话默认 provider，保持向后兼容。
//! 这样 TUI 里切换模型时，wire 协议与 base_url 会自动切到该模型对应的厂商。

use std::collections::HashMap;

use codex_model_provider_info::ModelProviderInfo;
use codex_protocol::openai_models::ModelInfo;

/// 依据模型的 `provider_id` 解析它应使用的 provider。
///
/// 解析顺序：
/// 1. `model_info.provider_id` 命中 `model_providers` 中已配置的 provider → 返回它；
/// 2. 否则回落到会话默认 provider（`default_provider_id`，即今天的 `config.model_provider`）；
/// 3. 默认 provider 也缺失 → 返回 `ModelProviderInfo::default()`（调用方应保证不走到这里）。
///
/// 第 2 步保证老配置（模型无 `provider_id` 概念）行为与现状逐字节一致。
#[allow(dead_code)] // PR1 仅落地数据层，PR2 的 with_model/make_turn_context 会接入。
pub fn resolve_provider_for_model(
    model_info: &ModelInfo,
    model_providers: &HashMap<String, ModelProviderInfo>,
    default_provider_id: &str,
) -> ModelProviderInfo {
    match model_info.provider_id.as_deref() {
        Some(id) if model_providers.contains_key(id) => model_providers[id].clone(),
        _ => model_providers
            .get(default_provider_id)
            .cloned()
            .unwrap_or_default(),
    }
}

/// 与 [`resolve_provider_for_model`] 同款选择逻辑，但只返回 provider id（字符串）。
///
/// 供需要同时拿到 provider 信息与 id 的调用方使用：`ModelProviderInfo` 不携带自身 id，
/// 故 id 须单独解析。选择顺序与 [`resolve_provider_for_model`] 完全一致，避免两者漂移。
pub fn resolve_provider_id_for_model(
    model_info: &ModelInfo,
    model_providers: &HashMap<String, ModelProviderInfo>,
    default_provider_id: &str,
) -> String {
    match model_info.provider_id.as_deref() {
        Some(id) if model_providers.contains_key(id) => id.to_string(),
        _ => default_provider_id.to_string(),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use codex_model_provider_info::{WireApi, built_in_model_providers};
    use codex_protocol::openai_models::{
        ConfigShellToolType, ModelInfo, ModelVisibility, TruncationPolicyConfig, WebSearchToolType,
    };

    /// 构造一个只关心 `provider_id` 的最小 ModelInfo（其余字段走兜底值）。
    fn model_with_provider(slug: &str, provider_id: Option<&str>) -> ModelInfo {
        ModelInfo {
            slug: slug.to_string(),
            display_name: slug.to_string(),
            description: None,
            default_reasoning_level: None,
            supported_reasoning_levels: Vec::new(),
            shell_type: ConfigShellToolType::ShellCommand,
            visibility: ModelVisibility::List,
            supported_in_api: true,
            priority: 1,
            additional_speed_tiers: Vec::new(),
            service_tiers: Vec::new(),
            default_service_tier: None,
            availability_nux: None,
            upgrade: None,
            base_instructions: String::new(),
            model_messages: None,
            include_skills_usage_instructions: false,
            supports_reasoning_summaries: false,
            default_reasoning_summary: Default::default(),
            support_verbosity: false,
            default_verbosity: None,
            apply_patch_tool_type: None,
            web_search_tool_type: WebSearchToolType::Text,
            truncation_policy: TruncationPolicyConfig::bytes(10_000),
            supports_parallel_tool_calls: false,
            supports_image_detail_original: false,
            context_window: None,
            max_context_window: None,
            auto_compact_token_limit: None,
            comp_hash: None,
            effective_context_window_percent: 95,
            experimental_supported_tools: Vec::new(),
            input_modalities: Vec::new(),
            used_fallback_model_metadata: false,
            supports_search_tool: false,
            use_responses_lite: false,
            auto_review_model_override: None,
            tool_mode: None,
            multi_agent_version: None,
            provider_id: provider_id.map(String::from),
        }
    }

    #[test]
    fn hits_declared_provider() {
        let providers = built_in_model_providers(None);
        let info = model_with_provider("glm-5.2", Some("glm"));
        let resolved = resolve_provider_for_model(&info, &providers, "openai");
        assert_eq!(resolved.wire_api, WireApi::Chat);
        assert_eq!(
            resolved.base_url.as_deref(),
            Some("https://open.bigmodel.cn/api/paas/v4")
        );
    }

    #[test]
    fn anthropic_provider_wires_anthropic_api() {
        let providers = built_in_model_providers(None);
        let info = model_with_provider("claude-opus-4-8", Some("anthropic"));
        let resolved = resolve_provider_for_model(&info, &providers, "openai");
        assert_eq!(resolved.wire_api, WireApi::Anthropic);
        // Anthropic Messages 的 max_tokens 必填，构造时应给了兜底上限。
        assert_eq!(resolved.max_output_tokens, Some(4096));
    }

    #[test]
    fn falls_back_to_default_when_provider_id_missing() {
        let providers = built_in_model_providers(None);
        let info = model_with_provider("legacy-model", None);
        let resolved = resolve_provider_for_model(&info, &providers, "openai");
        // 回落到默认 openai provider（Responses 协议）。
        assert_eq!(resolved.wire_api, WireApi::Responses);
    }

    #[test]
    fn falls_back_to_default_when_provider_id_unknown() {
        let providers = built_in_model_providers(None);
        let info = model_with_provider("x", Some("no-such-provider"));
        let resolved = resolve_provider_for_model(&info, &providers, "openai");
        assert_eq!(resolved.wire_api, WireApi::Responses);
    }

    #[test]
    fn id_helper_matches_provider_helper_selection() {
        let providers = built_in_model_providers(None);

        // 命中声明 provider：id 与 provider 一致（glm）。
        let info = model_with_provider("glm-5.2", Some("glm"));
        assert_eq!(
            resolve_provider_id_for_model(&info, &providers, "openai"),
            "glm"
        );

        // 缺省 provider_id：回落默认 openai。
        let info = model_with_provider("legacy", None);
        assert_eq!(
            resolve_provider_id_for_model(&info, &providers, "openai"),
            "openai"
        );

        // 未知 provider_id：同样回落默认。
        let info = model_with_provider("x", Some("no-such-provider"));
        assert_eq!(
            resolve_provider_id_for_model(&info, &providers, "openai"),
            "openai"
        );
    }
}
