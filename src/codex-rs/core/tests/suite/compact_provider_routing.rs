//! 回归：切模型后手动 `Op::Compact` 的压缩请求走 **turn 级 provider**（新模型经
//! `provider_id` 绑定的 provider），而非会话默认 provider。
//!
//! `compact_conversation_history` 收 `&turn_context.provider`（`compact_remote.rs:250`），
//! turn 级 provider 在每个 turn 构建时由 `resolve_provider_for_model` 按当前模型的
//! `provider_id` 重新解析（`turn_context.rs:755`）。本测试用两个物理隔离的 mock server
//! 证明：会话默认 provider 指向 server_a，切到 `provider_id="openai-b"` 的模型后压缩请求
//! 落在 server_b，server_a 收不到压缩请求。若该修复被破坏（压缩改走会话默认 provider），
//! server_a 的压缩 mock 会收到请求而 server_b 收不到 → 断言失败。
//!
//! 注意两点易错项（见 `supports_remote_compaction` 门与 v2 分支）：
//! - 两 provider 的 `name` 必须是 `"OpenAI"`（大小写敏感），否则 `supports_remote_compaction`
//!   返回 false、压缩静默回落本地路径、两个 mock 都收不到请求。
//! - 必须 `disable(Feature::RemoteCompactionV2)`，否则压缩走 v2 分支、不经过本修复的调用点。

#![allow(clippy::unwrap_used)]

use anyhow::Result;
use codex_features::Feature;
use codex_login::CodexAuth;
use codex_model_provider_info::built_in_model_providers;
use codex_models_manager::bundled_models_response;
use codex_protocol::models::ResponseItem;
use codex_protocol::openai_models::ModelsResponse;
use codex_protocol::protocol::EventMsg;
use codex_protocol::protocol::Op;
use codex_protocol::protocol::ThreadSettingsOverrides;
use codex_protocol::user_input::UserInput;
use core_test_support::responses::ev_assistant_message;
use core_test_support::responses::ev_completed;
use core_test_support::responses::mount_compact_json_once;
use core_test_support::responses::mount_sse_sequence;
use core_test_support::responses::sse;
use core_test_support::skip_if_no_network;
use core_test_support::submit_thread_settings;
use core_test_support::test_codex::test_codex;
use core_test_support::wait_for_event;
use serde_json::json;
use wiremock::MockServer;

/// 在自带 8MB 栈的线程 + 手动 multi_thread runtime 里跑给定的 async 测试体。
///
/// 切模型 + 压缩的完整 turn 循环调用栈较深，`#[tokio::test]` 的 worker 线程默认 2MB 栈在
/// `cargo test` 下会溢出（与 `chat_anthropic_agent_loop.rs` 同因）。worker 栈也显式放大到 16MB。
/// 传「构造 future 的闭包」而非 future 本身：`*_impl` 的 future 因 tracing span 跨 await 持
/// 非 Send 状态，须在线程内部创建。
fn run_with_deep_stack<F, Fut>(name: &'static str, make_fut: F) -> Result<()>
where
    F: FnOnce() -> Fut + Send + 'static,
    Fut: std::future::Future<Output = Result<()>>,
{
    const TEST_STACK_SIZE_BYTES: usize = 8 * 1024 * 1024;
    const WORKER_STACK_SIZE_BYTES: usize = 16 * 1024 * 1024;
    let handle = std::thread::Builder::new()
        .name(name.to_string())
        .stack_size(TEST_STACK_SIZE_BYTES)
        .spawn(|| {
            let runtime = tokio::runtime::Builder::new_multi_thread()
                .worker_threads(2)
                .thread_stack_size(WORKER_STACK_SIZE_BYTES)
                .enable_all()
                .build()?;
            runtime.block_on(make_fut())
        })?;
    match handle.join() {
        Ok(result) => result,
        Err(_) => Err(anyhow::anyhow!("{name} thread panicked")),
    }
}

/// 切模型后压缩请求打到 turn 级 provider（server_b），而非会话默认 provider（server_a）。
#[test]
fn compact_routes_to_turn_level_provider_after_model_switch() -> Result<()> {
    run_with_deep_stack(
        "compact_routes_to_turn_level_provider_after_model_switch",
        compact_routes_to_turn_level_provider_after_model_switch_impl,
    )
}

async fn compact_routes_to_turn_level_provider_after_model_switch_impl() -> Result<()> {
    skip_if_no_network!(Ok(()));

    // 两个物理隔离的 mock server：server_a=会话默认 provider，server_b=切模型后的 turn 级 provider。
    let server_a = MockServer::start().await;
    let server_b = MockServer::start().await;
    let server_a_uri = server_a.uri();
    let server_b_uri = server_b.uri();

    let mut builder = test_codex().with_auth(CodexAuth::from_api_key("dummy-compact-routing"));
    builder = builder.with_config(move |config| {
        // 以内置 openai provider 为模板构造两个 name="OpenAI" 的 provider（过 supports_remote_compaction 门），
        // 仅 base_url 不同 → 分别指向两个 mock server。复用 built_in_model_providers 的 HashMap 类型。
        let mut providers = built_in_model_providers(/*openai_base_url*/ None);
        let template = providers
            .get("openai")
            .expect("内置 openai provider 应存在")
            .clone();
        // 过 supports_remote_compaction 门要求 name=="OpenAI"（大小写敏感）；内置模板若改名，
        // 压缩会静默回落本地、两个 mock 都收不到请求。这里早失败给出明确信号，避免迷惑性的 0!=1。
        assert!(
            template.name == "OpenAI",
            "内置 openai provider 的 name 应为 \"OpenAI\"（实际 {:?}），否则过不了 supports_remote_compaction 门",
            template.name,
        );
        let mut provider_a = template.clone();
        provider_a.base_url = Some(format!("{server_a_uri}/v1"));
        provider_a.env_key = None;
        let mut provider_b = template.clone();
        provider_b.base_url = Some(format!("{server_b_uri}/v1"));
        provider_b.env_key = None;
        providers.clear();
        providers.insert("openai-a".to_string(), provider_a.clone());
        providers.insert("openai-b".to_string(), provider_b);

        // 会话默认 provider 指向 server_a。
        config.model_provider = providers
            .get("openai-a")
            .expect("openai-a provider 应存在")
            .clone();
        config.model_providers = providers;

        // 构造两个仅 slug + provider_id 不同的模型（其余字段沿用内置模板，wire_api 由 provider 决定）。
        let template_model = bundled_models_response()
            .expect("bundled models.json should parse")
            .models
            .into_iter()
            .next()
            .expect("bundled catalog 应非空");
        let mut model_a = template_model.clone();
        model_a.slug = "compact-route-a".into();
        model_a.display_name = "compact-route-a".into();
        model_a.provider_id = Some("openai-a".into());
        let mut model_b = template_model.clone();
        model_b.slug = "compact-route-b".into();
        model_b.display_name = "compact-route-b".into();
        model_b.provider_id = Some("openai-b".into());
        config.model_catalog = Some(ModelsResponse {
            models: vec![model_a, model_b],
        });

        config.model = Some("compact-route-a".into());
        config.model_provider_id = "openai-a".into();

        // 走 compact_remote.rs（v1）路径——本修复 `compact_conversation_history(&turn_context.provider, …)` 所在。
        let _ = config.features.disable(Feature::RemoteCompactionV2);
    });
    let test = builder.build(&server_a).await?;
    let codex = test.codex.clone();

    // server_a：初始 turn 的 SSE（恰好 1 次请求建立 history）。
    let _initial_mock = mount_sse_sequence(
        &server_a,
        vec![sse(vec![
            ev_assistant_message("msg-1", "FIRST_REPLY"),
            ev_completed("resp-1"),
        ])],
    )
    .await;

    // server_b：压缩请求的 JSON 应答（恰好 1 次）。server_a 不挂压缩 mock → 若压缩误走 server_a 会 404。
    let compacted_history = vec![ResponseItem::Compaction {
        id: None,
        encrypted_content: "ENCRYPTED_COMPACTION_SUMMARY".to_string(),
        internal_chat_message_metadata_passthrough: None,
    }];
    let compact_mock_b = mount_compact_json_once(
        &server_b,
        json!({ "output": compacted_history.clone() }),
    )
    .await;
    // server_a 也挂一份压缩 mock 用于反向断言（应保持空）。
    let compact_mock_a = mount_compact_json_once(&server_a, json!({ "output": compacted_history })).await;

    // 1) 初始 turn 建 history（走 server_a）。
    codex
        .submit(Op::UserInput {
            items: vec![UserInput::Text {
                text: "hello before compact".into(),
                text_elements: Vec::new(),
            }],
            final_output_json_schema: None,
            responsesapi_client_metadata: None,
            additional_context: Default::default(),
            thread_settings: Default::default(),
        })
        .await?;
    wait_for_event(&codex, |ev| matches!(ev, EventMsg::TurnComplete(_))).await;

    // 2) 切到 provider_id="openai-b" 的模型 → session 预解析 provider 为 server_b。
    submit_thread_settings(
        &codex,
        ThreadSettingsOverrides {
            model: Some("compact-route-b".to_string()),
            ..Default::default()
        },
    )
    .await?;

    // 3) 手动压缩 → 应走 turn 级 provider（server_b）。
    codex.submit(Op::Compact).await?;
    wait_for_event(&codex, |ev| matches!(ev, EventMsg::TurnComplete(_))).await;

    // 压缩请求落在 turn 级 provider（server_b），路径为 /v1/responses/compact。
    assert_eq!(
        compact_mock_b.requests().len(),
        1,
        "压缩应走 turn 级 provider（server_b）"
    );
    assert_eq!(
        compact_mock_b.single_request().path(),
        "/v1/responses/compact"
    );
    // 会话默认 provider（server_a）不应收到压缩请求。
    assert!(
        compact_mock_a.requests().is_empty(),
        "压缩不应走会话默认 provider（server_a），实际收到 {} 条压缩请求",
        compact_mock_a.requests().len()
    );

    Ok(())
}
