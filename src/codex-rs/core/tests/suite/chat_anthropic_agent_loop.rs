//! Chat / Anthropic adapter 端到端 agent-loop 回归。
//!
//! 驱动完整 turn 循环经非 Responses 协议跑通：模型发工具调用 → agent 在无沙箱下执行
//! `shell_command` → 工具结果回灌 → 收尾文本。断言 turn-2 的请求体按各自 adapter 形态
//! 携带了工具输出，证明 `ResponseItem`↔wire 双向转换、工具名透传、调用 id 配对、SSE
//! 工具块累加整条链路工作。`codex-api` 的 adapter 单测只覆盖单轮机制；这里覆盖端到端。
//!
//! 完整 agent 循环调用栈较深，`#[tokio::test]` 的 worker 线程默认栈在 `cargo test` 下会
//! 溢出，故仿 `suite/rmcp_client.rs` / `src/guardian/tests.rs` 的写法：在自带 8MB 栈的
//! std 线程里起手动 runtime（worker 栈也放大到 16MB）跑 `*_impl`。
//!
//! 注意：`TestCodex::submit_turn` 内部已等待 `TurnComplete`（见 `test_codex.rs` 的
//! `SUBMIT_TURN_COMPLETE_TIMEOUT`），故提交返回即表示 turn 已完成，无需再等。

#![allow(clippy::unwrap_used)]

use anyhow::Result;
use codex_model_provider_info::built_in_model_providers;
use codex_models_manager::bundled_models_response;
use core_test_support::anthropic_fixtures::mount_anthropic_function_call_agent_response;
use core_test_support::chat_fixtures::mount_chat_function_call_agent_response;
use core_test_support::responses::start_mock_server;
use core_test_support::test_codex::test_codex;
use serde_json::Value;
use serde_json::json;

/// 在自带 8MB 栈的线程 + 手动 multi_thread runtime 里跑给定的 async 测试体。
///
/// 传「构造 future 的闭包」而非 future 本身：`*_impl` 的 future 非 Send（tracing span 跨
/// await 持非 Send 状态），须在线程内部创建，spawn 的闭包才 Send。
/// worker 线程也要显式放大栈：agent 循环的 spawned 子任务（SSE 流、工具执行）会在 worker
/// 上跑深层递归，tokio 默认 2MB 会溢出。
fn run_with_deep_stack<F, Fut>(name: &str, make_fut: F) -> Result<()>
where
    F: FnOnce() -> Fut + Send + 'static,
    Fut: std::future::Future<Output = Result<()>>,
{
    const TEST_STACK_SIZE_BYTES: usize = 8 * 1024 * 1024;
    const WORKER_STACK_SIZE_BYTES: usize = 16 * 1024 * 1024;
    let name_handle = name.to_string();
    let handle = std::thread::Builder::new()
        .name(name_handle)
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

/// 构造 `shell_command` 工具的入参 JSON 串（与 skill_approval 用法一致）。
fn shell_command_arguments(command: &str) -> String {
    serde_json::to_string(&json!({
        "command": command,
        "timeout_ms": 5_000,
    }))
    .expect("serialize shell args")
}

/// 在 Chat messages[] 里找 role==tool 且 tool_call_id 匹配的消息，返回其 content。
fn chat_tool_output<'a>(messages: &'a [Value], tool_call_id: &str) -> Option<&'a str> {
    messages.iter().find_map(|m| {
        if m.get("role").and_then(|v| v.as_str()) != Some("tool") {
            return None;
        }
        if m.get("tool_call_id").and_then(|v| v.as_str()) != Some(tool_call_id) {
            return None;
        }
        m.get("content").and_then(|v| v.as_str())
    })
}

/// 在 Anthropic messages[].content[] 里找 tool_use_id 匹配的 tool_result 块，返回其 content。
fn anthropic_tool_result<'a>(messages: &'a [Value], tool_use_id: &str) -> Option<&'a str> {
    messages
        .iter()
        .flat_map(|m| {
            m.get("content")
                .and_then(|c| c.as_array())
                .map(|arr| arr.iter())
                .into_iter()
                .flatten()
        })
        .find_map(|b| {
            if b.get("type").and_then(|v| v.as_str()) != Some("tool_result") {
                return None;
            }
            if b.get("tool_use_id").and_then(|v| v.as_str()) != Some(tool_use_id) {
                return None;
            }
            b.get("content").and_then(|v| v.as_str())
        })
}

/// 在 Chat messages[] 里找 role==assistant 且 tool_calls 含 id==call_id 的项，返回其 function.name。
/// 用于验证 ResponseItem→wire 反向序列化保留了工具名（plan 回归重点：工具名透传）。
fn chat_assistant_tool_name<'a>(messages: &'a [Value], call_id: &str) -> Option<&'a str> {
    messages.iter().find_map(|m| {
        if m.get("role").and_then(|v| v.as_str()) != Some("assistant") {
            return None;
        }
        m.get("tool_calls")
            .and_then(|tcs| tcs.as_array())?
            .iter()
            .find_map(|tc| {
                if tc.get("id").and_then(|v| v.as_str()) != Some(call_id) {
                    return None;
                }
                tc.get("function")
                    .and_then(|f| f.get("name"))
                    .and_then(|v| v.as_str())
            })
    })
}

/// 在 Anthropic messages[].content[] 里找 role==assistant 下 id==call_id 的 tool_use 块，返回其 name。
/// 用于验证 ResponseItem→wire 反向序列化保留了工具名（plan 回归重点：工具名透传）。
fn anthropic_assistant_tool_use_name<'a>(messages: &'a [Value], call_id: &str) -> Option<&'a str> {
    messages
        .iter()
        .filter_map(|m| {
            if m.get("role").and_then(|v| v.as_str()) != Some("assistant") {
                return None;
            }
            m.get("content").and_then(|c| c.as_array()).map(|arr| arr.iter())
        })
        .flatten()
        .find_map(|b| {
            if b.get("type").and_then(|v| v.as_str()) != Some("tool_use") {
                return None;
            }
            if b.get("id").and_then(|v| v.as_str()) != Some(call_id) {
                return None;
            }
            b.get("name").and_then(|v| v.as_str())
        })
}

#[test]
fn chat_glm_agent_loop_round_trips_tool_call() -> Result<()> {
    run_with_deep_stack(
        "chat_glm_agent_loop_round_trips_tool_call",
        || chat_glm_agent_loop_round_trips_tool_call_impl(),
    )
}

#[test]
fn anthropic_claude_agent_loop_round_trips_tool_call() -> Result<()> {
    run_with_deep_stack(
        "anthropic_claude_agent_loop_round_trips_tool_call",
        || anthropic_claude_agent_loop_round_trips_tool_call_impl(),
    )
}

async fn chat_glm_agent_loop_round_trips_tool_call_impl() -> Result<()> {
    let server = start_mock_server().await;
    let base_url = format!("{}/v1", server.uri());

    let mut builder = test_codex().with_config(move |c| {
        // 重建内置 provider 表，把 glm 指向 mock、清 env_key（回落 dummy Bearer 鉴权）。
        let mut providers = built_in_model_providers(None);
        if let Some(p) = providers.get_mut("glm") {
            p.base_url = Some(base_url.clone());
            p.env_key = None;
        }
        c.model_provider = providers
            .get("glm")
            .expect("glm provider 应存在")
            .clone();
        c.model_providers = providers;
        c.model = Some("glm-5.2".to_string());
        c.model_provider_id = "glm".to_string();
        c.model_catalog = Some(bundled_models_response().expect("bundled models.json should parse"));
    });
    let test = builder.build(&server).await?;

    let call_id = "chat-glm-agent-loop";
    let arguments = shell_command_arguments("echo chat-round-trip-ok");
    let mocks =
        mount_chat_function_call_agent_response(&server, call_id, &arguments, "shell_command")
            .await;

    // submit_turn 内部已等 TurnComplete；返回即代表两轮 SSE 都消费完、工具已执行。
    test.submit_turn("run the echo command").await?;

    // turn-1 请求按 Chat 形态序列化（messages[] 含 user）—— 证明走了 Chat adapter。
    let first_body = mocks.function_call.single_request().body_json();
    let first_messages = first_body
        .get("messages")
        .and_then(|m| m.as_array())
        .expect("turn-1 请求应含 messages 数组");
    assert!(
        first_messages
            .iter()
            .any(|m| m.get("role").and_then(|v| v.as_str()) == Some("user")),
        "turn-1 请求应经 Chat adapter 序列化为 messages[user]: {first_messages:?}",
    );

    // turn-2 请求携带工具输出（role:tool、tool_call_id 对齐）—— 证明完整往返。
    let second_body = mocks.completion.single_request().body_json();
    let second_messages = second_body
        .get("messages")
        .and_then(|m| m.as_array())
        .expect("turn-2 请求应含 messages 数组");
    let tool_output = chat_tool_output(second_messages, call_id)
        .expect("turn-2 请求应含对齐 call_id 的 tool 消息");
    assert!(
        tool_output.contains("chat-round-trip-ok"),
        "工具输出应含 echo 的字符串，实际: {tool_output:?}",
    );

    // turn-2 请求历史里的 assistant tool_call 名仍为 shell_command —— 反向序列化保留了工具名。
    let tool_name = chat_assistant_tool_name(second_messages, call_id)
        .expect("turn-2 请求历史应含对齐 call_id 的 assistant tool_call");
    assert_eq!(
        tool_name, "shell_command",
        "工具名应经 ResponseItem→wire 反向序列化保留",
    );

    Ok(())
}

async fn anthropic_claude_agent_loop_round_trips_tool_call_impl() -> Result<()> {
    let server = start_mock_server().await;
    // Anthropic adapter 自带 v1/messages 段，base_url 不得带 /v1（否则拼成 /v1/v1/messages）。
    let base_url = server.uri().to_string();

    let mut builder = test_codex().with_config(move |c| {
        let mut providers = built_in_model_providers(None);
        if let Some(p) = providers.get_mut("anthropic") {
            p.base_url = Some(base_url.clone());
            p.env_key = None;
            // max_output_tokens 已由 create_anthropic_provider 设为 Some(4096)（Anthropic 必填），保留。
        }
        c.model_provider = providers
            .get("anthropic")
            .expect("anthropic provider 应存在")
            .clone();
        c.model_providers = providers;
        c.model = Some("claude-opus-4-8".to_string());
        c.model_provider_id = "anthropic".to_string();
        c.model_catalog = Some(bundled_models_response().expect("bundled models.json should parse"));
    });
    let test = builder.build(&server).await?;

    let call_id = "anthropic-claude-agent-loop";
    let arguments = shell_command_arguments("echo anthropic-round-trip-ok");
    let mocks = mount_anthropic_function_call_agent_response(
        &server,
        call_id,
        &arguments,
        "shell_command",
    )
    .await;

    test.submit_turn("run the echo command").await?;

    // turn-1 请求按 Anthropic 形态序列化（顶层 system 数组）—— 证明走了 Anthropic adapter。
    let first = mocks.function_call.single_request().body_json();
    assert!(
        first.get("system").and_then(|s| s.as_array()).is_some(),
        "turn-1 请求应经 Anthropic adapter 序列化出顶层 system 数组: {first}",
    );

    // turn-2 请求携带 tool_result 块（tool_use_id 对齐）—— 证明完整往返。
    let second_body = mocks.completion.single_request().body_json();
    let second_messages = second_body
        .get("messages")
        .and_then(|m| m.as_array())
        .expect("turn-2 请求应含 messages 数组");
    let tool_result = anthropic_tool_result(second_messages, call_id)
        .expect("turn-2 请求应含对齐 call_id 的 tool_result 块");
    assert!(
        tool_result.contains("anthropic-round-trip-ok"),
        "工具输出应含 echo 的字符串，实际: {tool_result:?}",
    );

    // turn-2 请求历史里的 assistant tool_use 块名仍为 shell_command —— 反向序列化保留了工具名。
    let tool_name = anthropic_assistant_tool_use_name(second_messages, call_id)
        .expect("turn-2 请求历史应含对齐 call_id 的 assistant tool_use 块");
    assert_eq!(
        tool_name, "shell_command",
        "工具名应经 ResponseItem→wire 反向序列化保留",
    );

    Ok(())
}
