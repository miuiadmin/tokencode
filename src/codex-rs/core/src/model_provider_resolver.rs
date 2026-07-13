//! 按模型解析其绑定的 provider。
//!
//! provider 选择跟随 model：每条模型元数据可声明 `provider_id`，指向 config 的
//! `model_providers` map。但若用户**显式**指定了 `model_provider`（CLI `-c` 或
//! config.toml `model_provider =`），显式 provider 压制模型自带的 `provider_id`——
//! 这让用户能用自己的网关/协议接管某个内置模型（如把 `glm-5.2` 走自建 Anthropic
//! 网关，而非内置 glm/bigmodel.cn）。未显式时模型 provider_id 优先（向后兼容）。

use std::collections::HashMap;

use codex_model_provider_info::{OPENAI_PROVIDER_ID, ModelProviderInfo};
use codex_protocol::openai_models::ModelInfo;

/// 依据模型的 `provider_id` 与会话默认 provider 解析它应使用的 provider。
///
/// 解析顺序：
/// 1. 若 `default_is_explicit`（用户经 CLI/config.toml 显式选了 provider）→ 用
///    `default_provider_id`，**压制**模型自带的 `provider_id`；命中缺失时回落
///    内置 `openai`，再回落 `ModelProviderInfo::default()`（空壳）。
/// 2. 否则（派生/默认）：`model_info.provider_id` 命中 `model_providers` → 用它；
/// 3. 未命中 → 回落 `default_provider_id` → 内置 `openai` → 空壳。
///
/// 第 1 档保证「用户显式选的 provider 真正生效」（修运行期被模型 provider_id 覆盖、
/// 请求打到非预期厂商的 bug）；第 2-3 档保持向后兼容（无显式 provider 时，TUI 切
/// 模型自动切厂商）。`default_is_explicit` 由加载期派生（`config/mod.rs`），切模型
/// 不改动它，故用户 config 显式选的 provider 在会话生命期内稳定。
pub fn resolve_provider_for_model(
    model_info: &ModelInfo,
    model_providers: &HashMap<String, ModelProviderInfo>,
    default_provider_id: &str,
    default_is_explicit: bool,
) -> ModelProviderInfo {
    // ① 用户显式指定 provider → 压制模型自带 provider_id
    if default_is_explicit {
        return model_providers
            .get(default_provider_id)
            .or_else(|| model_providers.get(OPENAI_PROVIDER_ID))
            .cloned()
            .unwrap_or_default();
    }
    // ② 非显式：模型 provider_id 优先（派生语义，向后兼容），未命中回落 default→openai
    match model_info.provider_id.as_deref() {
        Some(id) if model_providers.contains_key(id) => model_providers[id].clone(),
        _ => model_providers
            .get(default_provider_id)
            .or_else(|| model_providers.get(OPENAI_PROVIDER_ID))
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
    default_is_explicit: bool,
) -> String {
    // ① 用户显式指定 → 压制模型 provider_id
    if default_is_explicit {
        if model_providers.contains_key(default_provider_id) {
            return default_provider_id.to_string();
        }
        return OPENAI_PROVIDER_ID.to_string();
    }
    // ② 非显式：模型 provider_id 优先，回落 default→openai
    match model_info.provider_id.as_deref() {
        Some(id) if model_providers.contains_key(id) => id.to_string(),
        _ => {
            if model_providers.contains_key(default_provider_id) {
                default_provider_id.to_string()
            } else {
                OPENAI_PROVIDER_ID.to_string()
            }
        }
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
            supports_reasoning_effort: None,
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
        let resolved = resolve_provider_for_model(&info, &providers, "openai", false);
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
        let resolved = resolve_provider_for_model(&info, &providers, "openai", false);
        assert_eq!(resolved.wire_api, WireApi::Anthropic);
        // Anthropic 内置 provider 不硬编码 max_output_tokens，留 None 由 adapter 常量兜底。
        assert_eq!(resolved.max_output_tokens, None);
    }

    #[test]
    fn falls_back_to_default_when_provider_id_missing() {
        let providers = built_in_model_providers(None);
        let info = model_with_provider("legacy-model", None);
        let resolved = resolve_provider_for_model(&info, &providers, "openai", false);
        // 回落到默认 openai provider（Responses 协议）。
        assert_eq!(resolved.wire_api, WireApi::Responses);
    }

    #[test]
    fn falls_back_to_default_when_provider_id_unknown() {
        let providers = built_in_model_providers(None);
        let info = model_with_provider("x", Some("no-such-provider"));
        let resolved = resolve_provider_for_model(&info, &providers, "openai", false);
        assert_eq!(resolved.wire_api, WireApi::Responses);
    }

    #[test]
    fn falls_back_to_openai_when_default_provider_missing() {
        // default_provider_id 指向不存在的 provider：应回落到内置 openai，而非空壳
        // （空壳 name 为空、base_url 为空，请求会打到无效端点）。
        let providers = built_in_model_providers(None);
        let info = model_with_provider("legacy-model", None);

        let resolved = resolve_provider_for_model(&info, &providers, "no-such-default", false);
        assert_eq!(resolved.name, "OpenAI", "应回落到内置 openai provider，而非空壳");
        assert!(
            !resolved.name.is_empty(),
            "回落结果不得是空壳 provider（name 为空）"
        );

        // id 变体选择顺序与 provider 变体一致：默认缺失时回落 openai。
        assert_eq!(
            resolve_provider_id_for_model(&info, &providers, "no-such-default", false),
            "openai"
        );
    }

    #[test]
    fn id_helper_matches_provider_helper_selection() {
        let providers = built_in_model_providers(None);

        // 命中声明 provider：id 与 provider 一致（glm）。
        let info = model_with_provider("glm-5.2", Some("glm"));
        assert_eq!(
            resolve_provider_id_for_model(&info, &providers, "openai", false),
            "glm"
        );

        // 缺省 provider_id：回落默认 openai。
        let info = model_with_provider("legacy", None);
        assert_eq!(
            resolve_provider_id_for_model(&info, &providers, "openai", false),
            "openai"
        );

        // 未知 provider_id：同样回落默认。
        let info = model_with_provider("x", Some("no-such-provider"));
        assert_eq!(
            resolve_provider_id_for_model(&info, &providers, "openai", false),
            "openai"
        );
    }

    #[test]
    fn explicit_default_overrides_model_provider_id() {
        // 用户显式选了 provider（default_is_explicit=true）：即便模型自带 provider_id
        // 命中内置，也用显式 provider——本修复的核心，让显式选择压制模型绑定。
        let providers = built_in_model_providers(None);
        let info = model_with_provider("glm-5.2", Some("glm")); // 模型绑定 glm
        let resolved = resolve_provider_for_model(&info, &providers, "anthropic", true);
        // 显式选 anthropic，压制 glm → 返回 anthropic（Anthropic 协议）
        assert_eq!(resolved.wire_api, WireApi::Anthropic);
        assert_ne!(
            resolved.base_url.as_deref(),
            Some("https://open.bigmodel.cn/api/paas/v4"),
            "显式 provider 不得被模型自带 provider_id 覆盖"
        );

        // id 变体同理：返回显式 provider id
        assert_eq!(
            resolve_provider_id_for_model(&info, &providers, "anthropic", true),
            "anthropic"
        );
    }

    #[test]
    fn explicit_default_missing_falls_back_to_openai() {
        // 显式 provider 指向不存在的 id：回落内置 openai（而非空壳），与派生路径回落一致。
        let providers = built_in_model_providers(None);
        let info = model_with_provider("glm-5.2", Some("glm"));
        let resolved = resolve_provider_for_model(&info, &providers, "no-such-explicit", true);
        assert_eq!(
            resolved.name, "OpenAI",
            "显式 provider 缺失应回落内置 openai，而非空壳"
        );

        assert_eq!(
            resolve_provider_id_for_model(&info, &providers, "no-such-explicit", true),
            "openai"
        );
    }
}
