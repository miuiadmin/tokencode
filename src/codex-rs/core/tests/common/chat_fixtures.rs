//! Chat Completions（OpenAI 兼容网关）SSE fixture —— 给 core 端到端 agent-loop 回归用。
//!
//! `codex-api/tests/chat_adapter_fixtures.rs` 只测 adapter 机制（单轮 UnifiedRequest→wire、
//! SSE→UnifiedEvent）。本模块把同样的 SSE chunk 形态搬到 core 的 wiremock 桩上，挂到
//! `chat/completions` 路径，让完整 turn 循环（模型发 tool_call → 执行 → 结果回灌 → 收尾）能
//! 经 Chat adapter 跑通，复用 `responses::ResponseMock` 做请求捕获。

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

/// 把一组 Chat chunk（JSON）拼成 SSE body，末尾按需追加 `[DONE]`。
pub fn build_chat_body(chunks: &[Value], done: bool) -> String {
    let mut body = String::new();
    for chunk in chunks {
        body.push_str(&format!("data: {chunk}\n\n"));
    }
    if done {
        body.push_str("data: [DONE]\n\n");
    }
    body
}

/// turn-1：模型发一个工具调用（首帧 role+id+name+完整 arguments → finish_reason:tool_calls → usage）。
///
/// arguments 为 JSON 字符串（如 `{"command":"...","timeout_ms":500}`），整段放进首帧的
/// `function.arguments`，由 adapter 累加器合并为完整 FunctionCall。
pub fn chat_tool_call_chunks(call_id: &str, tool_name: &str, arguments: &str) -> Vec<Value> {
    vec![
        json!({
            "id": "chatcmpl-agent-1",
            "object": "chat.completion.chunk",
            "model": "chat-fixture",
            "choices": [{"index": 0, "delta": {"role": "assistant", "tool_calls": [
                {"index": 0, "id": call_id, "type": "function", "function": {"name": tool_name, "arguments": arguments}}
            ]}, "finish_reason": null}]
        }),
        json!({
            "id": "chatcmpl-agent-1",
            "choices": [{"index": 0, "delta": {}, "finish_reason": "tool_calls"}]
        }),
        json!({
            "id": "chatcmpl-agent-1",
            "choices": [],
            "usage": {"prompt_tokens": 20, "completion_tokens": 5, "total_tokens": 25}
        }),
    ]
}

/// turn-2：模型发收尾文本（role+content 帧 → finish_reason:stop → usage）。
pub fn chat_completion_chunks(text: &str) -> Vec<Value> {
    vec![
        json!({
            "id": "chatcmpl-agent-2",
            "object": "chat.completion.chunk",
            "model": "chat-fixture",
            "choices": [{"index": 0, "delta": {"role": "assistant", "content": text}, "finish_reason": null}]
        }),
        json!({
            "id": "chatcmpl-agent-2",
            "choices": [{"index": 0, "delta": {}, "finish_reason": "stop"}]
        }),
        json!({
            "id": "chatcmpl-agent-2",
            "choices": [],
            "usage": {"prompt_tokens": 30, "completion_tokens": 2, "total_tokens": 32}
        }),
    ]
}

/// 在 `POST .*/chat/completions$` 上挂一份一次性 SSE 应答，返回捕获器。
pub async fn mount_chat_sse_once(server: &MockServer, body: String) -> ResponseMock {
    let response_mock = ResponseMock::new();
    Mock::given(method("POST"))
        .and(path_regex(".*/chat/completions$"))
        .and(response_mock.clone())
        .respond_with(sse_response(body))
        .up_to_n_times(1)
        .mount(server)
        .await;
    response_mock
}

/// 挂两轮 Chat SSE（turn-1 工具调用、turn-2 收尾文本），结构与
/// `responses::mount_function_call_agent_response` 同构。两个桩同路径、各 `up_to_n_times(1)`，
/// wiremock 按注册顺序依次消费。
pub async fn mount_chat_function_call_agent_response(
    server: &MockServer,
    call_id: &str,
    arguments: &str,
    tool_name: &str,
) -> FunctionCallResponseMocks {
    let first = build_chat_body(&chat_tool_call_chunks(call_id, tool_name, arguments), true);
    let function_call = mount_chat_sse_once(server, first).await;

    let second = build_chat_body(&chat_completion_chunks("done"), true);
    let completion = mount_chat_sse_once(server, second).await;

    FunctionCallResponseMocks {
        function_call,
        completion,
    }
}
