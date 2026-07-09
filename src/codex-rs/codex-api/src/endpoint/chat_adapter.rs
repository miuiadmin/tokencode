//! OpenAI Chat Completions 协议 adapter —— 把中立 `LanguageModel` 契约桥接到 `ChatClient`。
//!
//! 与 `OpenaiResponsesAdapter` 同构；差异仅在请求翻译：Responses 形态的 `UnifiedRequest`
//! （`input: Vec<ResponseItem>` 等）→ Chat 形态的 `ChatApiRequest`（扁平 `messages[]` +
//! Chat tools schema）。SSE 解析已在 `ChatClient` / `spawn_chat_stream` 内产出 core 已消费
//! 的 `ResponseEvent`，故事件流归一复用 `response_stream_to_unified`（与 Responses
//! adapter 共用同一份，零改动）。
//!
//! 对 `ResponseItem` 的翻译为编译期穷尽 match：上游增删变体时本文件编译失败而非静默漏映射。

use crate::auth::SharedAuthProvider;
use crate::common::ChatApiRequest;
use crate::common::ChatMessage;
use crate::common::ChatStreamOptions;
use crate::common::ChatToolCall;
use crate::common::ChatToolCallFunction;
use crate::endpoint::chat::ChatClient;
use crate::endpoint::response_stream_to_unified;
use crate::endpoint::ResponsesOptions;
use crate::provider::Provider;
use crate::telemetry::SseTelemetry;
use codex_client::HttpTransport;
use codex_client::RequestTelemetry;
use codex_language_model::LanguageModel;
use codex_language_model::UnifiedError;
use codex_language_model::UnifiedEventStream;
use codex_language_model::UnifiedRequest;
use codex_language_model::UnifiedRequestOptions;
use codex_protocol::models::ContentItem;
use codex_protocol::models::ImageDetail;
use codex_protocol::models::ResponseItem;
use futures::future::BoxFuture;
use serde_json::Map;
use serde_json::Value;
use serde_json::json;
use std::sync::Arc;
use tracing::debug;

/// OpenAI Chat Completions 协议 adapter：实现中立 `LanguageModel` trait，内部委托 `ChatClient`。
pub struct OpenaiChatAdapter<T: HttpTransport> {
    inner: ChatClient<T>,
    /// 可选的最大输出 token 上限（来自 `Provider.max_output_tokens`）。
    /// Some 时注入 `ChatApiRequest.max_tokens`；None 则不发该字段（由服务端决定）。
    max_output_tokens: Option<u32>,
}

impl<T: HttpTransport> OpenaiChatAdapter<T> {
    pub fn new(transport: T, provider: Provider, auth: SharedAuthProvider) -> Self {
        // provider 即将 move 进 ChatClient，先拷出 max_output_tokens（Option<u32>: Copy）。
        let max_output_tokens = provider.max_output_tokens;
        Self {
            inner: ChatClient::new(transport, provider, auth),
            max_output_tokens,
        }
    }

    pub fn with_telemetry(
        self,
        request: Option<Arc<dyn RequestTelemetry>>,
        sse: Option<Arc<dyn SseTelemetry>>,
    ) -> Self {
        Self {
            inner: self.inner.with_telemetry(request, sse),
            max_output_tokens: self.max_output_tokens,
        }
    }
}

impl<T: HttpTransport> LanguageModel for OpenaiChatAdapter<T> {
    fn stream(
        &self,
        request: UnifiedRequest,
        options: UnifiedRequestOptions,
    ) -> BoxFuture<'_, Result<UnifiedEventStream, UnifiedError>> {
        Box::pin(async move {
            // 中立请求 / 选项 → Chat wire 请求 / 选项（翻译表 + 既有 From）。
            let mut api_request: ChatApiRequest = request.into();
            // provider 显式配置的 max_output_tokens 注入 max_tokens；None 时保持不发（From 写 None）。
            api_request.max_tokens = self.max_output_tokens;
            let api_options: ResponsesOptions = options.into();
            // 委托 ChatClient；具体协议错误原样透传为 Passthrough，调用方可 downcast 回 ApiError。
            let api_stream = self
                .inner
                .stream_request(api_request, api_options)
                .await
                .map_err(UnifiedError::passthrough)?;
            // Chat parser 已产出 ResponseEvent，事件归一复用 Responses 同一份逻辑。
            Ok(response_stream_to_unified(api_stream))
        })
    }
}

// ===========================================================================
// UnifiedRequest → ChatApiRequest 翻译（编译期对 ResponseItem 穷尽 match）
// ===========================================================================

impl From<UnifiedRequest> for ChatApiRequest {
    fn from(req: UnifiedRequest) -> Self {
        let mut messages: Vec<ChatMessage> = Vec::new();

        // instructions（非空）→ 前置 system 消息。
        // 用 system 而非 developer：developer 角色为 OpenAI 专属，大量 OpenAI 兼容网关 /
        // 第三方后端（含本测试网关的 GLM 后端）只识别 system，发 developer 会被拒（422）。
        // 经 `normalize_chat_role` 统一归一，保证 Chat adapter 对「非 Responses 后端」可用。
        if !req.instructions.is_empty() {
            messages.push(ChatMessage {
                role: "system".to_string(),
                content: Some(Value::String(req.instructions.clone())),
                name: None,
                tool_call_id: None,
                tool_calls: None,
            });
        }

        // input: Vec<ResponseItem> → messages[]（穷尽 match，逐变体翻译）。
        for item in req.input {
            translate_item(item, &mut messages);
        }

        // tools：Responses 形态 {type:"function", name, ...} → Chat {type:"function", function:{...}}。
        let tools = req
            .tools
            .map(|tools| tools.into_iter().filter_map(responses_tool_to_chat).collect::<Vec<_>>())
            .filter(|v| !v.is_empty());

        ChatApiRequest {
            model: req.model,
            messages,
            tools,
            tool_choice: tool_choice_to_value(&req.tool_choice),
            parallel_tool_calls: req.parallel_tool_calls,
            reasoning_effort: req
                .reasoning
                .as_ref()
                .and_then(|r| r.effort.as_ref().map(|e| e.to_string())),
            service_tier: req.service_tier,
            response_format: req
                .text
                .as_ref()
                .and_then(|t| t.format.as_ref())
                .map(|f| {
                    json!({
                        "type": "json_schema",
                        "json_schema": {
                            "name": f.name,
                            "strict": f.strict,
                            "schema": f.schema,
                        }
                    })
                }),
            // max_tokens 默认不发（由服务端决定）；OpenaiChatAdapter::stream 注入 provider 配置值。
            max_tokens: None,
            // 流式开启时，让服务端在末尾追加一帧 usage（Completed 携带 token 统计）。
            stream_options: req.stream.then_some(ChatStreamOptions { include_usage: true }),
            stream: req.stream,
        }
    }
}

/// 逐 `ResponseItem` 变体翻译为 Chat message（或跳过）。
fn translate_item(item: ResponseItem, messages: &mut Vec<ChatMessage>) {
    match item {
        ResponseItem::Message { role, content, .. } => {
            // 纯文本 → String；含图 → 多模态数组；全空 → 跳过。
            let chat_content = message_content_to_value(&content);
            if chat_content.is_none() {
                return;
            }
            messages.push(ChatMessage {
                role: normalize_chat_role(role),
                content: chat_content,
                name: None,
                tool_call_id: None,
                tool_calls: None,
            });
        }
        ResponseItem::FunctionCall { name, arguments, call_id, .. } => {
            push_tool_call(messages, call_id, name, arguments);
        }
        ResponseItem::CustomToolCall { call_id, name, input, .. } => {
            push_tool_call(messages, call_id, name, input);
        }
        ResponseItem::FunctionCallOutput { call_id, output, .. } => {
            push_tool_output(messages, call_id, output.body.to_text());
        }
        ResponseItem::CustomToolCallOutput { call_id, output, .. } => {
            push_tool_output(messages, call_id, output.body.to_text());
        }
        // 以下变体 Chat 无等价或不可解码：一律跳过。
        // - Reasoning：encrypted_content 不可回喂，Chat 无 reasoning item。
        // - AgentMessage / LocalShellCall / ToolSearchCall / ToolSearchOutput / WebSearchCall /
        //   ImageGenerationCall / Compaction / ContextCompaction / CompactionTrigger /
        //   AdditionalTools：Responses 专属 / 多 agent / 压缩 / 控制项。
        // - Other：未知透传项。
        ResponseItem::Reasoning { .. }
        | ResponseItem::AgentMessage { .. }
        | ResponseItem::LocalShellCall { .. }
        | ResponseItem::ToolSearchCall { .. }
        | ResponseItem::ToolSearchOutput { .. }
        | ResponseItem::WebSearchCall { .. }
        | ResponseItem::ImageGenerationCall { .. }
        | ResponseItem::Compaction { .. }
        | ResponseItem::ContextCompaction { .. }
        | ResponseItem::CompactionTrigger {}
        | ResponseItem::AdditionalTools { .. }
        | ResponseItem::Other => {}
    }
}

/// 把一条工具调用并入 messages：连续的 FunctionCall/CustomToolCall 合并为同一条 assistant
/// 消息的 `tool_calls[]`（Chat wire 要求同 turn 多 tool_calls 在一条消息）；遇非工具项自然 flush。
fn push_tool_call(
    messages: &mut Vec<ChatMessage>,
    call_id: String,
    name: String,
    arguments: String,
) {
    if let Some(last) = messages.last_mut()
        && last.role == "assistant"
        && last.tool_calls.is_some()
    {
        last.tool_calls.as_mut().expect("已检查 is_some").push(ChatToolCall {
            id: call_id,
            r#type: "function".to_string(),
            function: ChatToolCallFunction { name, arguments },
        });
        return;
    }
    messages.push(ChatMessage {
        role: "assistant".to_string(),
        content: None,
        name: None,
        tool_call_id: None,
        tool_calls: Some(vec![ChatToolCall {
            id: call_id,
            r#type: "function".to_string(),
            function: ChatToolCallFunction { name, arguments },
        }]),
    });
}

/// 工具调用输出 → `{role:"tool", tool_call_id, content}`。content 取 output 的纯文本表征。
fn push_tool_output(
    messages: &mut Vec<ChatMessage>,
    call_id: String,
    text: Option<String>,
) {
    messages.push(ChatMessage {
        role: "tool".to_string(),
        content: Some(Value::String(text.unwrap_or_default())),
        name: None,
        tool_call_id: Some(call_id),
        tool_calls: None,
    });
}

/// `ContentItem` 列表 → Chat message content。全文本 → `Value::String`；含图 → 多模态
/// `Value::Array([{type:"text"|"image_url",...}])`；全空 → `None`（跳过）。
fn message_content_to_value(content: &[ContentItem]) -> Option<Value> {
    let has_image = content
        .iter()
        .any(|c| matches!(c, ContentItem::InputImage { .. }));

    if !has_image {
        let text: String = content
            .iter()
            .filter_map(|c| match c {
                ContentItem::InputText { text } | ContentItem::OutputText { text } => Some(text.as_str()),
                _ => None,
            })
            .collect::<Vec<_>>()
            .join("");
        return if text.is_empty() { None } else { Some(Value::String(text)) };
    }

    let parts: Vec<Value> = content
        .iter()
        .filter_map(|c| match c {
            ContentItem::InputText { text } | ContentItem::OutputText { text } => {
                Some(json!({ "type": "text", "text": text }))
            }
            ContentItem::InputImage { image_url, detail } => {
                let detail_str = detail.as_ref().map(image_detail_str);
                let mut obj = json!({ "type": "image_url", "image_url": { "url": image_url } });
                if let Some(d) = detail_str {
                    obj["image_url"]["detail"] = Value::String(d.to_string());
                }
                Some(obj)
            }
        })
        .collect();
    if parts.is_empty() {
        return None;
    }
    Some(Value::Array(parts))
}

/// 归一化 message role 以适配 Chat Completions 的「非 Responses 后端」生态。
///
/// `developer` 是 OpenAI 专属角色（Responses / 新版 Chat）；大量 OpenAI 兼容网关与第三方
/// 后端（GLM 等）只识别 `system`，发 `developer` 会被拒（实测本测试网关 422）。故统一回落到
/// `system`。其余角色（`system`/`user`/`assistant`/`tool`）原样保留。
fn normalize_chat_role(role: String) -> String {
    if role == "developer" {
        "system".to_string()
    } else {
        role
    }
}

/// `ImageDetail` → Chat image_url.detail 字符串（Original 无直接对应，回落 auto）。
fn image_detail_str(detail: &ImageDetail) -> &'static str {
    match detail {
        ImageDetail::Auto => "auto",
        ImageDetail::Low => "low",
        ImageDetail::High => "high",
        ImageDetail::Original => "auto",
    }
}

/// Responses 工具描述（`{type:"function", name, description, parameters, strict}`）
/// → Chat 工具描述（`{type:"function", function:{name, description, parameters, strict}}`）。
///
/// 非 function 类型（如 `web_search` / `local_shell` / `mcp` 等 Responses 内置工具）在 Chat
/// 无等价，跳过并记录（上层若需在 Chat 下暴露这些能力，应改以 function 工具形态提供）。
fn responses_tool_to_chat(tool: Value) -> Option<Value> {
    let obj = tool.as_object()?;
    let kind = obj.get("type").and_then(|v| v.as_str()).unwrap_or("");
    if kind != "function" {
        debug!(tool_type = kind, "跳过非 function 工具：Chat 协议无等价");
        return None;
    }
    let mut function = Map::new();
    if let Some(name) = obj.get("name") {
        function.insert("name".to_string(), name.clone());
    }
    if let Some(description) = obj.get("description") {
        function.insert("description".to_string(), description.clone());
    }
    if let Some(parameters) = obj.get("parameters") {
        function.insert("parameters".to_string(), parameters.clone());
    }
    if let Some(strict) = obj.get("strict") {
        function.insert("strict".to_string(), strict.clone());
    }
    Some(json!({ "type": "function", "function": Value::Object(function) }))
}

/// tool_choice（中立 String）→ Chat tool_choice（`Value::String` 或 None）。
///
/// 空串 → None（用服务端默认，等价 auto）；其余原样透传（`auto` / `none` / `required` /
/// 自定义函数名等）。中立 schema 当前仅承载字符串形态，故不产生对象形态的 tool_choice。
fn tool_choice_to_value(tool_choice: &str) -> Option<Value> {
    if tool_choice.is_empty() {
        None
    } else {
        Some(Value::String(tool_choice.to_string()))
    }
}
