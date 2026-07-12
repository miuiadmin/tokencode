//! Anthropic Messages SSE fixture —— 给 core 端到端 agent-loop 回归用。
//!
//! `codex-api/tests/anthropic_adapter_fixtures.rs` 只测 adapter 机制（单轮 UnifiedRequest→wire、
//! SSE→UnifiedEvent）。本模块把同样的 SSE 事件形态搬到 core 的 wiremock 桩上，挂到
//! `v1/messages` 路径，让完整 turn 循环（模型发 tool_use → 执行 → 结果回灌 → 收尾）能经
//! Anthropic adapter 跑通，复用 `responses::ResponseMock` 做请求捕获。

#![allow(clippy::unwrap_used)]

use serde_json::Value;
use serde_json::json;
use wiremock::Mock;
use wiremock::MockServer;
use wiremock::matchers::method;
use wiremock::matchers::path_regex;

use crate::responses::FunctionCallResponseMocks;
use crate::responses::ResponseMock;
use crate::responses::sse_response;

/// 把一组 Anthropic 事件（JSON）拼成 SSE body（每帧仅 `data:` 行，解析侧按 JSON `type` 分派）。
pub fn build_anthropic_body(events: &[Value]) -> String {
    let mut body = String::new();
    for event in events {
        body.push_str(&format!("data: {event}\n\n"));
    }
    body
}

/// turn-1：模型发一个工具调用（message_start → tool_use 块 → input_json_delta 分片 →
/// content_block_stop → message_delta(stop_reason:tool_use) → message_stop）。
///
/// arguments 为 JSON 字符串，拆成两段 `partial_json` 由 adapter 累加器拼回，覆盖 SSE 工具块
/// 累加这条易错路径。ASCII JSON 在中点 split_at 是字符边界安全的。
pub fn anthropic_tool_use_events(call_id: &str, tool_name: &str, arguments: &str) -> Vec<Value> {
    let mid = arguments.len() / 2;
    let (first_half, second_half) = arguments.split_at(mid);
    vec![
        json!({
            "type": "message_start",
            "message": {
                "id": "msg-agent-1",
                "model": "anthropic-fixture",
                "usage": {"input_tokens": 20, "output_tokens": 1}
            }
        }),
        json!({
            "type": "content_block_start",
            "index": 0,
            "content_block": {"type": "tool_use", "id": call_id, "name": tool_name, "input": {}}
        }),
        json!({"type": "content_block_delta", "index": 0, "delta": {"type": "input_json_delta", "partial_json": first_half}}),
        json!({"type": "content_block_delta", "index": 0, "delta": {"type": "input_json_delta", "partial_json": second_half}}),
        json!({"type": "content_block_stop", "index": 0}),
        json!({
            "type": "message_delta",
            "delta": {"stop_reason": "tool_use", "stop_sequence": null},
            "usage": {"output_tokens": 5}
        }),
        json!({"type": "message_stop"}),
    ]
}

/// turn-2：模型发收尾文本（message_start → text 块 → text_delta → content_block_stop →
/// message_delta(stop_reason:end_turn) → message_stop）。
pub fn anthropic_completion_events(text: &str) -> Vec<Value> {
    vec![
        json!({
            "type": "message_start",
            "message": {
                "id": "msg-agent-2",
                "model": "anthropic-fixture",
                "usage": {"input_tokens": 30, "output_tokens": 1}
            }
        }),
        json!({"type": "content_block_start", "index": 0, "content_block": {"type": "text", "text": ""}}),
        json!({"type": "content_block_delta", "index": 0, "delta": {"type": "text_delta", "text": text}}),
        json!({"type": "content_block_stop", "index": 0}),
        json!({
            "type": "message_delta",
            "delta": {"stop_reason": "end_turn", "stop_sequence": null},
            "usage": {"output_tokens": 3}
        }),
        json!({"type": "message_stop"}),
    ]
}

/// 在 `POST .*/v1/messages$` 上挂一份一次性 SSE 应答，返回捕获器。
pub async fn mount_anthropic_sse_once(server: &MockServer, body: String) -> ResponseMock {
    let response_mock = ResponseMock::new();
    Mock::given(method("POST"))
        .and(path_regex(".*/v1/messages$"))
        .and(response_mock.clone())
        .respond_with(sse_response(body))
        .up_to_n_times(1)
        .mount(server)
        .await;
    response_mock
}

/// 挂两轮 Anthropic SSE（turn-1 工具调用、turn-2 收尾文本），结构与
/// `responses::mount_function_call_agent_response` 同构。两个桩同路径、各 `up_to_n_times(1)`，
/// wiremock 按注册顺序依次消费。
pub async fn mount_anthropic_function_call_agent_response(
    server: &MockServer,
    call_id: &str,
    arguments: &str,
    tool_name: &str,
) -> FunctionCallResponseMocks {
    let first = build_anthropic_body(&anthropic_tool_use_events(call_id, tool_name, arguments));
    let function_call = mount_anthropic_sse_once(server, first).await;

    let second = build_anthropic_body(&anthropic_completion_events("done"));
    let completion = mount_anthropic_sse_once(server, second).await;

    FunctionCallResponseMocks {
        function_call,
        completion,
    }
}
