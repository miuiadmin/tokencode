use super::HandleOutputCtx;
use super::TurnItemContributorPolicy;
use super::completed_item_defers_mailbox_delivery_to_next_turn;
use super::finalize_non_tool_response_item;
use super::handle_non_tool_response_item;
use super::handle_output_item_done;
use super::synthesize_call_from_action;
use super::build_tool_spec_map;
use super::validate_progress_channel_tool_call;
use super::image_generation_artifact_path;
use super::last_assistant_message_from_item;
use super::response_item_may_include_external_context;
use super::save_image_generation_result;
use crate::session::step_context::StepContext;
use crate::session::tests::make_session_and_context;
use crate::tools::ToolRouter;
use crate::tools::parallel::ToolCallRuntime;
use crate::tools::router::ToolCall;
use crate::turn_diff_tracker::TurnDiffTracker;
use codex_extension_api::ExtensionData;
use codex_extension_api::TurnItemContributor;
use codex_protocol::error::CodexErr;
use codex_protocol::items::AgentMessageContent;
use codex_protocol::items::TurnItem;
use codex_protocol::memory_citation::MemoryCitation;
use codex_protocol::models::ContentItem;
use codex_protocol::models::FunctionCallOutputPayload;
use codex_protocol::models::LocalShellAction;
use codex_protocol::models::LocalShellExecAction;
use codex_protocol::models::LocalShellStatus;
use codex_protocol::models::MessagePhase;
use codex_protocol::models::ResponseItem;
use codex_protocol::ToolName;
use codex_tools::FreeformTool;
use codex_tools::FreeformToolFormat;
use codex_tools::JsonSchema;
use codex_tools::ResponsesApiTool;
use codex_tools::ToolPayload;
use codex_tools::ToolSpec;
use codex_utils_absolute_path::test_support::PathExt;
use pretty_assertions::assert_eq;
use std::sync::Arc;
use tokio_util::sync::CancellationToken;

fn assistant_output_text(text: &str) -> ResponseItem {
    assistant_output_text_with_phase(text, /*phase*/ None)
}

fn assistant_output_text_with_phase(text: &str, phase: Option<MessagePhase>) -> ResponseItem {
    ResponseItem::Message {
        id: Some("msg-1".to_string()),
        role: "assistant".to_string(),
        content: vec![ContentItem::OutputText {
            text: text.to_string(),
        }],
        phase,
        internal_chat_message_metadata_passthrough: None,
    }
}

#[test]
fn external_context_pollution_items_include_web_search_and_tool_search() {
    let polluting_items = [
        ResponseItem::WebSearchCall {
            id: None,
            status: Some("completed".to_string()),
            action: None,
            internal_chat_message_metadata_passthrough: None,
        },
        ResponseItem::ToolSearchCall {
            id: None,
            call_id: Some("search-1".to_string()),
            status: None,
            execution: "client".to_string(),
            arguments: serde_json::json!({"query": "calendar"}),
            internal_chat_message_metadata_passthrough: None,
        },
        ResponseItem::ToolSearchOutput {
            id: None,
            call_id: Some("search-1".to_string()),
            status: "completed".to_string(),
            execution: "client".to_string(),
            tools: Vec::new(),
            internal_chat_message_metadata_passthrough: None,
        },
    ];

    assert!(
        polluting_items
            .iter()
            .all(response_item_may_include_external_context)
    );
}

#[test]
fn external_context_pollution_items_exclude_local_tool_calls() {
    let non_polluting_items = [
        ResponseItem::LocalShellCall {
            id: None,
            call_id: Some("shell-1".to_string()),
            status: LocalShellStatus::Completed,
            action: LocalShellAction::Exec(LocalShellExecAction {
                command: vec!["cat".to_string(), "README.md".to_string()],
                timeout_ms: None,
                working_directory: None,
                env: None,
                user: None,
            }),
            internal_chat_message_metadata_passthrough: None,
        },
        ResponseItem::FunctionCall {
            id: None,
            name: "shell".to_string(),
            namespace: None,
            arguments: "{}".to_string(),
            call_id: "call-1".to_string(),
            internal_chat_message_metadata_passthrough: None,
        },
        ResponseItem::FunctionCallOutput {
            id: None,
            call_id: "call-1".to_string(),
            output: FunctionCallOutputPayload::from_text("ok".to_string()),
            internal_chat_message_metadata_passthrough: None,
        },
        ResponseItem::CustomToolCall {
            id: None,
            status: None,
            call_id: "custom-1".to_string(),
            name: "apply_patch".to_string(),
            namespace: None,
            input: "*** Begin Patch\n*** End Patch\n".to_string(),
            internal_chat_message_metadata_passthrough: None,
        },
        ResponseItem::CustomToolCallOutput {
            id: None,
            call_id: "custom-1".to_string(),
            name: Some("apply_patch".to_string()),
            output: FunctionCallOutputPayload::from_text("ok".to_string()),
            internal_chat_message_metadata_passthrough: None,
        },
        assistant_output_text("plain assistant text"),
    ];

    assert!(
        !non_polluting_items
            .iter()
            .any(response_item_may_include_external_context)
    );
}

#[tokio::test]
async fn handle_non_tool_response_item_strips_citations_from_assistant_message() {
    let (session, turn_context) = make_session_and_context().await;
    let item = assistant_output_text(
        "hello<oai-mem-citation><citation_entries>\nMEMORY.md:1-2|note=[x]\n</citation_entries>\n<rollout_ids>\n019cc2ea-1dff-7902-8d40-c8f6e5d83cc4\n</rollout_ids></oai-mem-citation> world",
    );

    let turn_item = handle_non_tool_response_item(
        &session,
        &turn_context,
        TurnItemContributorPolicy::Skip,
        &item,
        /*plan_mode*/ false,
    )
    .await
    .expect("assistant message should parse");

    let TurnItem::AgentMessage(agent_message) = turn_item else {
        panic!("expected agent message");
    };
    let text = agent_message
        .content
        .iter()
        .map(|entry| match entry {
            codex_protocol::items::AgentMessageContent::Text { text } => text.as_str(),
        })
        .collect::<String>();
    assert_eq!(text, "hello world");
    let memory_citation = agent_message
        .memory_citation
        .expect("memory citation should be parsed");
    assert_eq!(memory_citation.entries.len(), 1);
    assert_eq!(memory_citation.entries[0].path, "MEMORY.md");
    assert_eq!(
        memory_citation.rollout_ids,
        vec!["019cc2ea-1dff-7902-8d40-c8f6e5d83cc4".to_string()]
    );
}

struct TestTurnItemContributor;

#[derive(Debug)]
struct TurnItemContributorRan;

impl TurnItemContributor for TestTurnItemContributor {
    fn contribute<'a>(
        &'a self,
        _thread_store: &'a ExtensionData,
        turn_store: &'a ExtensionData,
        item: &'a mut TurnItem,
    ) -> codex_extension_api::ExtensionFuture<'a, Result<(), String>> {
        Box::pin(async move {
            turn_store.insert(TurnItemContributorRan);
            if let TurnItem::AgentMessage(agent_message) = item {
                agent_message.memory_citation = Some(MemoryCitation {
                    entries: Vec::new(),
                    rollout_ids: Vec::new(),
                });
            }
            Ok(())
        })
    }
}

struct RewriteAgentMessageContributor;

impl TurnItemContributor for RewriteAgentMessageContributor {
    fn contribute<'a>(
        &'a self,
        _thread_store: &'a ExtensionData,
        _turn_store: &'a ExtensionData,
        item: &'a mut TurnItem,
    ) -> codex_extension_api::ExtensionFuture<'a, Result<(), String>> {
        Box::pin(async move {
            if let TurnItem::AgentMessage(agent_message) = item {
                agent_message.content = vec![AgentMessageContent::Text {
                    text: "contributed assistant text".to_string(),
                }];
            }
            Ok(())
        })
    }
}

#[tokio::test]
async fn handle_non_tool_response_item_runs_turn_item_contributors_only_when_requested() {
    let (mut session, turn_context) = make_session_and_context().await;
    let mut builder = codex_extension_api::ExtensionRegistryBuilder::new();
    builder.turn_item_contributor(Arc::new(TestTurnItemContributor));
    session.services.extensions = Arc::new(builder.build());
    let turn_store = ExtensionData::new(turn_context.sub_id.clone());
    let item = assistant_output_text(
        "hello<oai-mem-citation>ignored by memory parser</oai-mem-citation> world",
    );

    let provisional_turn_item = handle_non_tool_response_item(
        &session,
        &turn_context,
        TurnItemContributorPolicy::Skip,
        &item,
        /*plan_mode*/ false,
    )
    .await
    .expect("assistant message should parse");

    assert!(turn_store.get::<TurnItemContributorRan>().is_none());
    let TurnItem::AgentMessage(provisional_agent_message) = provisional_turn_item else {
        panic!("expected agent message");
    };
    assert_eq!(provisional_agent_message.memory_citation, None);

    let turn_item = handle_non_tool_response_item(
        &session,
        &turn_context,
        TurnItemContributorPolicy::Run(&turn_store),
        &item,
        /*plan_mode*/ false,
    )
    .await
    .expect("assistant message should parse");

    assert!(turn_store.get::<TurnItemContributorRan>().is_some());
    let TurnItem::AgentMessage(agent_message) = turn_item else {
        panic!("expected agent message");
    };
    assert!(agent_message.memory_citation.is_some());
    let text = agent_message
        .content
        .iter()
        .map(|entry| match entry {
            codex_protocol::items::AgentMessageContent::Text { text } => text.as_str(),
        })
        .collect::<String>();
    assert_eq!(text, "hello world");
}

#[tokio::test]
async fn handle_output_item_done_returns_contributed_last_agent_message() {
    let (mut session, turn_context) = make_session_and_context().await;
    let mut builder = codex_extension_api::ExtensionRegistryBuilder::new();
    builder.turn_item_contributor(Arc::new(RewriteAgentMessageContributor));
    session.services.extensions = Arc::new(builder.build());
    let session = Arc::new(session);
    let turn_context = Arc::new(turn_context);
    let step_context = StepContext::for_test(Arc::clone(&turn_context));
    let router = Arc::new(ToolRouter::from_context(
        step_context.as_ref(),
        crate::tools::router::ToolRouterParams {
            mcp_tools: None,
            deferred_mcp_tools: None,
            extension_tool_executors: Vec::new(),
            dynamic_tools: turn_context.dynamic_tools.as_slice(),
        },
        &Default::default(),
    ));
    let tracker = Arc::new(tokio::sync::Mutex::new(TurnDiffTracker::new()));
    let tool_runtime = ToolCallRuntime::new(router, Arc::clone(&session), step_context, tracker);
    let item = assistant_output_text("original assistant text");
    let mut ctx = HandleOutputCtx {
        sess: session,
        turn_context: Arc::clone(&turn_context),
        turn_store: Arc::new(ExtensionData::new(turn_context.sub_id.clone())),
        tool_runtime,
        cancellation_token: CancellationToken::new(),
    };

    let output = handle_output_item_done(&mut ctx, item, /*previously_active_item*/ None)
        .await
        .expect("assistant message should complete");

    assert_eq!(
        output.last_agent_message.as_deref(),
        Some("contributed assistant text")
    );
}

#[tokio::test]
async fn finalized_turn_item_defers_mailbox_for_contributed_visible_text() {
    let (mut session, turn_context) = make_session_and_context().await;
    let mut builder = codex_extension_api::ExtensionRegistryBuilder::new();
    builder.turn_item_contributor(Arc::new(RewriteAgentMessageContributor));
    session.services.extensions = Arc::new(builder.build());
    let turn_store = ExtensionData::new(turn_context.sub_id.clone());
    let item = assistant_output_text("<oai-mem-citation>hidden only</oai-mem-citation>");

    let finalized = finalize_non_tool_response_item(
        &session,
        &turn_context,
        TurnItemContributorPolicy::Run(&turn_store),
        &item,
        /*plan_mode*/ false,
    )
    .await
    .expect("assistant message should parse");

    assert_eq!(
        finalized.facts.last_agent_message.as_deref(),
        Some("contributed assistant text")
    );
    assert!(finalized.facts.defers_mailbox_delivery_to_next_turn);
}

#[tokio::test]
async fn finalized_turn_item_keeps_mailbox_open_for_commentary_text() {
    let (mut session, turn_context) = make_session_and_context().await;
    let mut builder = codex_extension_api::ExtensionRegistryBuilder::new();
    builder.turn_item_contributor(Arc::new(RewriteAgentMessageContributor));
    session.services.extensions = Arc::new(builder.build());
    let turn_store = ExtensionData::new(turn_context.sub_id.clone());
    let item = assistant_output_text_with_phase("still working", Some(MessagePhase::Commentary));

    let finalized = finalize_non_tool_response_item(
        &session,
        &turn_context,
        TurnItemContributorPolicy::Run(&turn_store),
        &item,
        /*plan_mode*/ false,
    )
    .await
    .expect("assistant message should parse");

    assert_eq!(
        finalized.facts.last_agent_message.as_deref(),
        Some("contributed assistant text")
    );
    assert!(!finalized.facts.defers_mailbox_delivery_to_next_turn);
}

#[test]
fn last_assistant_message_from_item_strips_citations_and_plan_blocks() {
    let item = assistant_output_text(
        "before<oai-mem-citation>doc1</oai-mem-citation>\n<proposed_plan>\n- x\n</proposed_plan>\nafter",
    );

    let message = last_assistant_message_from_item(&item, /*plan_mode*/ true)
        .expect("assistant text should remain after stripping");

    assert_eq!(message, "before\nafter");
}

#[test]
fn last_assistant_message_from_item_returns_none_for_citation_only_message() {
    let item = assistant_output_text("<oai-mem-citation>doc1</oai-mem-citation>");

    assert_eq!(
        last_assistant_message_from_item(&item, /*plan_mode*/ false),
        None
    );
}

#[test]
fn last_assistant_message_from_item_returns_none_for_plan_only_hidden_message() {
    let item = assistant_output_text("<proposed_plan>\n- x\n</proposed_plan>");

    assert_eq!(
        last_assistant_message_from_item(&item, /*plan_mode*/ true),
        None
    );
}

#[test]
fn completed_item_defers_mailbox_delivery_for_unknown_phase_messages() {
    let item = assistant_output_text("final answer");

    assert!(completed_item_defers_mailbox_delivery_to_next_turn(
        &item, /*plan_mode*/ false,
    ));
}

#[test]
fn completed_item_keeps_mailbox_delivery_open_for_commentary_messages() {
    let item = assistant_output_text_with_phase("still working", Some(MessagePhase::Commentary));

    assert!(!completed_item_defers_mailbox_delivery_to_next_turn(
        &item, /*plan_mode*/ false,
    ));
}

#[test]
fn completed_item_defers_mailbox_delivery_for_image_generation_calls() {
    let item = ResponseItem::ImageGenerationCall {
        id: Some("ig-1".to_string()),
        status: "completed".to_string(),
        revised_prompt: None,
        result: "Zm9v".to_string(),
        internal_chat_message_metadata_passthrough: None,
    };

    assert!(completed_item_defers_mailbox_delivery_to_next_turn(
        &item, /*plan_mode*/ false,
    ));
}

#[tokio::test]
async fn save_image_generation_result_saves_base64_to_png_in_codex_home() {
    let codex_home = tempfile::tempdir().expect("create tokencode home");
    let codex_home = codex_home.path().abs();
    let expected_path = image_generation_artifact_path(&codex_home, "session-1", "ig_save_base64");
    let _ = std::fs::remove_file(&expected_path);

    let saved_path =
        save_image_generation_result(&codex_home, "session-1", "ig_save_base64", "Zm9v")
            .await
            .expect("image should be saved");

    assert_eq!(saved_path, expected_path);
    assert_eq!(std::fs::read(&saved_path).expect("saved file"), b"foo");
    let _ = std::fs::remove_file(&saved_path);
}

#[tokio::test]
async fn save_image_generation_result_rejects_data_url_payload() {
    let result = "data:image/jpeg;base64,Zm9v";
    let codex_home = tempfile::tempdir().expect("create tokencode home");
    let codex_home = codex_home.path().abs();

    let err = save_image_generation_result(&codex_home, "session-1", "ig_456", result)
        .await
        .expect_err("data url payload should error");
    assert!(matches!(err, CodexErr::InvalidRequest(_)));
}

#[tokio::test]
async fn save_image_generation_result_overwrites_existing_file() {
    let codex_home = tempfile::tempdir().expect("create tokencode home");
    let codex_home = codex_home.path().abs();
    let existing_path = image_generation_artifact_path(&codex_home, "session-1", "ig_overwrite");
    std::fs::create_dir_all(
        existing_path
            .parent()
            .expect("generated image path should have a parent"),
    )
    .expect("create image output dir");
    std::fs::write(&existing_path, b"existing").expect("seed existing image");

    let saved_path = save_image_generation_result(&codex_home, "session-1", "ig_overwrite", "Zm9v")
        .await
        .expect("image should be saved");

    assert_eq!(saved_path, existing_path);
    assert_eq!(std::fs::read(&saved_path).expect("saved file"), b"foo");
    let _ = std::fs::remove_file(&saved_path);
}

#[tokio::test]
async fn save_image_generation_result_sanitizes_call_id_for_codex_home_output_path() {
    let codex_home = tempfile::tempdir().expect("create tokencode home");
    let codex_home = codex_home.path().abs();
    let expected_path = image_generation_artifact_path(&codex_home, "session-1", "../ig/..");
    let _ = std::fs::remove_file(&expected_path);

    let saved_path = save_image_generation_result(&codex_home, "session-1", "../ig/..", "Zm9v")
        .await
        .expect("image should be saved");

    assert_eq!(saved_path, expected_path);
    assert_eq!(std::fs::read(&saved_path).expect("saved file"), b"foo");
    let _ = std::fs::remove_file(&saved_path);
}

#[tokio::test]
async fn save_image_generation_result_rejects_non_standard_base64() {
    let codex_home = tempfile::tempdir().expect("create tokencode home");
    let codex_home = codex_home.path().abs();
    let err = save_image_generation_result(&codex_home, "session-1", "ig_urlsafe", "_-8")
        .await
        .expect_err("non-standard base64 should error");
    assert!(matches!(err, CodexErr::InvalidRequest(_)));
}

#[tokio::test]
async fn save_image_generation_result_rejects_non_base64_data_urls() {
    let codex_home = tempfile::tempdir().expect("create tokencode home");
    let codex_home = codex_home.path().abs();
    let err = save_image_generation_result(
        &codex_home,
        "session-1",
        "ig_svg",
        "data:image/svg+xml,<svg/>",
    )
    .await
    .expect_err("non-base64 data url should error");
    assert!(matches!(err, CodexErr::InvalidRequest(_)));
}

// ===== 进度副信道 P3b：合成纯函数单测（自 client_tests 平移，decorator 退场后宿主迁移）=====

#[test]
fn synthesize_call_action_to_function_call() {
    let kinds = std::collections::HashMap::from([("shell".to_string(), false)]);
    let action = serde_json::json!({"tool": "shell", "arguments": {"command": ["ls", "-la"]}});
    let (item, tool, _args) =
        synthesize_call_from_action(&action, &kinds).expect("known tool synthesizes");
    assert_eq!(tool, "shell");
    match item {
        ResponseItem::FunctionCall {
            name,
            namespace,
            arguments,
            call_id,
            ..
        } => {
            assert_eq!(name, "shell");
            assert_eq!(namespace, None);
            let args: serde_json::Value = serde_json::from_str(&arguments).unwrap();
            assert_eq!(args["command"][0], "ls");
            assert_eq!(args["command"][1], "-la");
            assert!(
                call_id.starts_with("shell-"),
                "call_id 形如 {{tool}}-{{uuid}}: {call_id}"
            );
        }
        other => panic!("expected FunctionCall, got {other:?}"),
    }
}

#[test]
fn synthesize_call_action_to_custom_tool_call_for_freeform() {
    let kinds = std::collections::HashMap::from([("apply_patch".to_string(), true)]);
    let lark = "*** Begin Patch\n*** End Patch";
    let action = serde_json::json!({"tool": "apply_patch", "arguments": {"input": lark}});
    let (item, tool, _args) =
        synthesize_call_from_action(&action, &kinds).expect("freeform tool synthesizes");
    assert_eq!(tool, "apply_patch");
    match item {
        ResponseItem::CustomToolCall {
            name,
            input,
            status,
            call_id,
            ..
        } => {
            assert_eq!(name, "apply_patch");
            assert_eq!(input, lark);
            assert_eq!(status.as_deref(), Some("completed"));
            assert!(
                call_id.starts_with("apply_patch-"),
                "call_id 形如 {{tool}}-{{uuid}}: {call_id}"
            );
        }
        other => panic!("expected CustomToolCall, got {other:?}"),
    }
}

#[test]
fn synthesize_call_restores_namespace_from_flat_name() {
    // flat 名 "mcp__demo__shell"：from_flat_wire_name 仅在首个 __ 切分 →
    // namespace="mcp", name="demo__shell"（与 to_flat_wire_name 严格互逆）。
    let kinds = std::collections::HashMap::from([("mcp__demo__shell".to_string(), false)]);
    let action = serde_json::json!({"tool": "mcp__demo__shell", "arguments": {}});
    let (item, _tool, _args) =
        synthesize_call_from_action(&action, &kinds).expect("namespaced tool synthesizes");
    match item {
        ResponseItem::FunctionCall { name, namespace, .. } => {
            assert_eq!(namespace.as_deref(), Some("mcp"));
            assert_eq!(name, "demo__shell");
        }
        other => panic!("expected FunctionCall, got {other:?}"),
    }
}

#[test]
fn synthesize_call_returns_none_for_unknown_or_missing_tool() {
    let kinds = std::collections::HashMap::from([("shell".to_string(), false)]);
    // 未知工具（不在 kinds）→ None，透传不执行。
    let action = serde_json::json!({"tool": "nonexistent", "arguments": {}});
    assert!(synthesize_call_from_action(&action, &kinds).is_none());
    // 缺 tool 字段 → None。
    let action = serde_json::json!({"arguments": {}});
    assert!(synthesize_call_from_action(&action, &kinds).is_none());
}

// ===== P3b §12.8 执行前二次校验（纯函数 validate_progress_channel_tool_call）=====

fn fn_spec(name: &str, required: &[&str]) -> ToolSpec {
    ToolSpec::Function(ResponsesApiTool {
        name: name.to_string(),
        description: String::new(),
        strict: false,
        defer_loading: None,
        parameters: JsonSchema {
            required: Some(required.iter().map(|s| s.to_string()).collect()),
            ..Default::default()
        },
        output_schema: None,
    })
}

fn freeform_spec(name: &str) -> ToolSpec {
    ToolSpec::Freeform(FreeformTool {
        name: name.to_string(),
        description: String::new(),
        format: FreeformToolFormat {
            r#type: "text".to_string(),
            syntax: "lark".to_string(),
            definition: String::new(),
        },
    })
}

fn fn_call(tool: &str, args_json: &str) -> ToolCall {
    ToolCall {
        tool_name: ToolName::from_flat_wire_name(tool),
        call_id: format!("{tool}-test-call-id"),
        payload: ToolPayload::Function {
            arguments: args_json.to_string(),
        },
    }
}

fn custom_call(tool: &str, input: &str) -> ToolCall {
    ToolCall {
        tool_name: ToolName::from_flat_wire_name(tool),
        call_id: format!("{tool}-test-call-id"),
        payload: ToolPayload::Custom {
            input: input.to_string(),
        },
    }
}

#[test]
fn validate_passes_function_call_with_all_required_fields() {
    let specs = build_tool_spec_map(&[fn_spec("read_file", &["path"])]);
    let call = fn_call("read_file", r#"{"path":"/x"}"#);
    assert!(validate_progress_channel_tool_call(&specs, &call).is_none());
}

#[test]
fn validate_rejects_function_call_missing_required_field() {
    let specs = build_tool_spec_map(&[fn_spec("read_file", &["path"])]);
    let call = fn_call("read_file", r#"{"content":"x"}"#);
    let err = validate_progress_channel_tool_call(&specs, &call).expect("缺必填字段应校验失败");
    assert!(err.contains("read_file"), "报错应含工具名：{err}");
    assert!(err.contains("path"), "报错应含缺失字段：{err}");
}

#[test]
fn validate_passes_apply_patch_with_valid_lark() {
    let specs = build_tool_spec_map(&[freeform_spec("apply_patch")]);
    let input = "*** Begin Patch\n*** Add File: /tmp/a.rs\n+hi\n*** End Patch";
    let call = custom_call("apply_patch", input);
    assert!(validate_progress_channel_tool_call(&specs, &call).is_none());
}

#[test]
fn validate_rejects_apply_patch_with_invalid_lark() {
    let specs = build_tool_spec_map(&[freeform_spec("apply_patch")]);
    let call = custom_call("apply_patch", "这根本不是合法 patch");
    let err = validate_progress_channel_tool_call(&specs, &call).expect("非法 lark 应校验失败");
    assert!(err.contains("apply_patch"), "{err}");
}

#[test]
fn validate_rejects_freeform_empty_input() {
    // 非 apply_patch 的 Freeform（如 code_mode exec）：input 为空应回流。
    let specs = build_tool_spec_map(&[freeform_spec("code_mode")]);
    let call = custom_call("code_mode", "   ");
    let err = validate_progress_channel_tool_call(&specs, &call).expect("空 input 应校验失败");
    assert!(err.contains("code_mode"), "{err}");
}
