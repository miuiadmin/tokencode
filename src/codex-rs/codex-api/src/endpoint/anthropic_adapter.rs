//! Anthropic Messages 协议 adapter —— 把中立 `LanguageModel` 契约桥接到 `AnthropicClient`。
//!
//! 与 `OpenaiChatAdapter` 同构；差异仅在请求翻译与 SSE 解析：
//! - 请求：Responses 形态的 `UnifiedRequest`（`input: Vec<ResponseItem>` 等）→ Anthropic 形态的
//!   `AnthropicApiRequest`（顶层 `system` + `messages[].content[]` 内容块 + `tool_use`/`tool_result`）。
//! - 事件：`AnthropicClient` / `spawn_anthropic_stream` 已产出 core 消费的 `ResponseEvent`，
//!   事件归一复用 `response_stream_to_unified`（与 Responses / Chat adapter 共用同一份）。
//!
//! 对 `ResponseItem` 的翻译为编译期穷尽 match：`ResponseItem` 新增/删除变体时本文件编译失败而非静默漏映射。

use crate::auth::SharedAuthProvider;
use crate::common::AnthropicApiRequest;
use crate::common::AnthropicContentBlock;
use crate::common::AnthropicImageSource;
use crate::common::AnthropicMessage;
use crate::common::AnthropicSystemTextBlock;
use crate::common::AnthropicTool;
use crate::common::AnthropicToolChoice;
use crate::endpoint::anthropic::AnthropicAuth;
use crate::endpoint::anthropic::AnthropicClient;
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
use codex_protocol::models::ResponseItem;
use futures::future::BoxFuture;
use serde_json::Value;
use serde_json::json;
use std::sync::Arc;
use tracing::debug;
use tracing::warn;

/// Anthropic 必填 `max_tokens` 的默认值（`Provider.max_output_tokens` 为 None 时回落到此）。
///
/// 取 16384：本仓库内置的 Anthropic 模型为 claude-opus-4-8 / claude-sonnet-5，二者输出上限
/// 均 ≥ 16384，该值不会触发 Anthropic 的 400（`max_tokens` 超过模型上限时服务端直接 400）。
/// 旧的 4096 是为兼容 Claude 3 整代（上限 4096）而取的最小公分母，但本仓库已不内置 Claude 3
/// 模型，该保守值只会把现代模型的长输出截断在 4096。若经自定义 config 接入 Claude 3 等低上限
/// 模型，应在 [model_providers] 里显式下调 `max_output_tokens`。Responses / Chat 路径不传
/// `max_tokens`；Anthropic 路径在 `Provider.max_output_tokens` 为 Some 时改用配置值覆盖（见
/// `AnthropicAdapter::stream`）。
pub const ANTHROPIC_DEFAULT_MAX_TOKENS: u32 = 16_384;

// ===========================================================================
// UnifiedRequest → AnthropicApiRequest 翻译（编译期对 ResponseItem 穷尽 match）
// ===========================================================================

impl From<UnifiedRequest> for AnthropicApiRequest {
    fn from(req: UnifiedRequest) -> Self {
        // instructions（非空）→ 顶层 system（数组形态，便于后续追加 cache_control）。
        let system = if req.instructions.is_empty() {
            None
        } else {
            Some(vec![AnthropicSystemTextBlock::new(req.instructions.clone())])
        };

        // input: Vec<ResponseItem> → messages[]（穷尽 match，逐变体翻译；连续同角色项合并为一条消息）。
        let mut messages: Vec<AnthropicMessage> = Vec::new();
        for item in req.input {
            translate_item(item, &mut messages);
        }

        // tools：Responses 形态 {type:"function", name, parameters, ...} → Anthropic {name, description?, input_schema}。
        let tools = req
            .tools
            .map(|tools| {
                tools
                    .into_iter()
                    .filter_map(responses_tool_to_anthropic)
                    .collect::<Vec<_>>()
            })
            .filter(|v| !v.is_empty());

        // tool_choice：仅当 tools 非空时才发——Anthropic 在无 tools 时拒绝 tool_choice（直接 400）。
        // parallel_tool_calls=false → disable_parallel_tool_use=true（该字段挂在 tool_choice 上，
        // 仅 tool_choice 存在时可表达；无 tools 时串行意图无承载点，一并省略）。
        let tool_choice = tools
            .as_ref()
            .and_then(|_| tool_choice_to_anthropic(&req.tool_choice, req.parallel_tool_calls));

        AnthropicApiRequest {
            model: req.model,
            max_tokens: ANTHROPIC_DEFAULT_MAX_TOKENS,
            messages,
            system,
            tools,
            tool_choice,
            stream: req.stream,
        }
    }
}

/// 逐 `ResponseItem` 变体翻译为 Anthropic 消息内容块（或跳过）。
///
/// 连续同角色项（如 assistant 文本 + tool_use，或多个 tool_result）合并为同一条消息，
/// 以满足 Anthropic 的 user/assistant 交替约束并减少消息数。
fn translate_item(item: ResponseItem, messages: &mut Vec<AnthropicMessage>) {
    match item {
        ResponseItem::Message { role, content, .. } => {
            // 仅 user/assistant 角色；system/developer 应经顶层 instructions 承载，此处跳过避免非法角色。
            let Some(role) = normalize_role(role) else {
                return;
            };
            let blocks = message_content_to_blocks(&content);
            append_blocks(messages, &role, blocks);
        }
        ResponseItem::FunctionCall {
            name, arguments, call_id, ..
        } => {
            push_tool_use(messages, call_id, name, arguments);
        }
        ResponseItem::CustomToolCall {
            call_id, name, input, ..
        } => {
            push_tool_use(messages, call_id, name, input);
        }
        ResponseItem::FunctionCallOutput { call_id, output, .. } => {
            push_tool_result(messages, call_id, output.body.to_text(), output.success);
        }
        ResponseItem::CustomToolCallOutput { call_id, output, .. } => {
            push_tool_result(messages, call_id, output.body.to_text(), output.success);
        }
        // 以下变体 Anthropic 无等价或不可解码：一律跳过。
        // - Reasoning：encrypted_content 不可回喂，Anthropic v1 不发 thinking 字段。
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

/// 把若干内容块并入消息：末条同角色则追加，否则新建一条。空块集不操作。
fn append_blocks(
    messages: &mut Vec<AnthropicMessage>,
    role: &str,
    new_blocks: Vec<AnthropicContentBlock>,
) {
    if new_blocks.is_empty() {
        return;
    }
    if let Some(last) = messages.last_mut()
        && last.role == role
    {
        last.content.extend(new_blocks);
    } else {
        messages.push(AnthropicMessage {
            role: role.to_string(),
            content: new_blocks,
        });
    }
}

/// assistant 发起的工具调用 → `tool_use` 块，并入 assistant 消息（与同 turn 的文本 / 其他 tool_use 合并）。
fn push_tool_use(
    messages: &mut Vec<AnthropicMessage>,
    id: String,
    name: String,
    arguments: String,
) {
    let block = AnthropicContentBlock::ToolUse {
        id,
        name,
        input: parse_tool_input(&arguments),
    };
    append_blocks(messages, "assistant", vec![block]);
}

/// 工具调用输出 → user 消息的 `tool_result` 块（`tool_use_id` 对齐被回应的 `tool_use.id`）。
///
/// `success`（中立语义：`Some(true)` 成功 / `Some(false)` 失败）取反映射为 Anthropic 的 `is_error`
/// （`Some(true)` 失败）——两者布尔语义相反，不可直接透传。让模型区分成功与失败的输出，避免按
/// 失败结果继续错误推理或反复重试同一失败工具。`None` 表示成败未表态，不序列化 `is_error`（默认成功）。
fn push_tool_result(
    messages: &mut Vec<AnthropicMessage>,
    call_id: String,
    text: Option<String>,
    success: Option<bool>,
) {
    let block = AnthropicContentBlock::ToolResult {
        tool_use_id: call_id,
        content: text.unwrap_or_default(),
        is_error: success.map(|s| !s),
    };
    append_blocks(messages, "user", vec![block]);
}

/// `ContentItem` 列表 → Anthropic 内容块。空文本项过滤（Anthropic 拒绝空 content）。
fn message_content_to_blocks(content: &[ContentItem]) -> Vec<AnthropicContentBlock> {
    content
        .iter()
        .filter_map(|c| match c {
            ContentItem::InputText { text } | ContentItem::OutputText { text } => {
                if text.is_empty() {
                    None
                } else {
                    Some(AnthropicContentBlock::Text { text: text.clone() })
                }
            }
            ContentItem::InputImage { image_url, .. } => Some(AnthropicContentBlock::Image {
                source: anthropic_image_source(image_url),
            }),
        })
        .collect()
}

/// 解析工具参数 JSON 串为 Anthropic `input`（须为对象）。
///
/// 解析失败或非对象时回落空对象（Anthropic 要求 input 为对象，不可缺省），并记录 warn——
/// 静默替换 `{}` 会丢弃真实参数、误导后续推理，故至少要可观测。
fn parse_tool_input(arguments: &str) -> Value {
    serde_json::from_str::<Value>(arguments)
        .ok()
        .filter(|v| v.is_object())
        .unwrap_or_else(|| {
            warn!(
                arguments,
                "工具参数非合法 JSON 对象，回落为空对象 input={{}}（真实参数被丢弃）"
            );
            json!({})
        })
}

/// 图片 URL → Anthropic 图片来源：`data:<mime>;base64,<data>` 拆为 base64；其余视为外链 url。
///
/// 缺逗号的 `data:` 串（被截断 / 非法 data URL）无法拆出 base64 数据，落回 url 分支会把
/// 整段 `data:` 串当外链发出（服务端无法抓取），记录 warn 以便排查。
fn anthropic_image_source(image_url: &str) -> AnthropicImageSource {
    if let Some(rest) = image_url.strip_prefix("data:") {
        if let Some((meta, data)) = rest.split_once(',') {
            let media_type = meta.split(';').next().unwrap_or("image/png");
            return AnthropicImageSource::Base64 {
                media_type: media_type.to_string(),
                data: data.to_string(),
            };
        }
        warn!(
            image_url,
            "data: 图片 URL 缺逗号（无法拆出 base64 数据），将以外链 url 发出，服务端可能无法获取"
        );
    }
    AnthropicImageSource::Url {
        url: image_url.to_string(),
    }
}

/// 归一化 message role：仅保留 `user` / `assistant`；其余（system / developer / 未知）返回 None。
fn normalize_role(role: String) -> Option<String> {
    match role.as_str() {
        "user" | "assistant" => Some(role),
        _ => None,
    }
}

/// Responses 工具描述（`{type:"function", name, description, parameters, strict}`）
/// → Anthropic 工具描述（`{name, description?, input_schema}`）。
///
/// 非 function 类型（如 `web_search` / `local_shell` / `mcp` 等 Responses 内置工具）在 Anthropic
/// 无等价，跳过并记录。
fn responses_tool_to_anthropic(tool: Value) -> Option<AnthropicTool> {
    let obj = tool.as_object()?;
    let kind = obj.get("type").and_then(|v| v.as_str()).unwrap_or("");
    if kind != "function" {
        debug!(tool_type = kind, "跳过非 function 工具：Anthropic 协议无等价");
        return None;
    }
    let name = obj.get("name")?.as_str()?.to_string();
    let description = obj
        .get("description")
        .and_then(|v| v.as_str())
        .map(String::from);
    // input_schema 缺省空对象（Anthropic 要求 input_schema 为对象）。
    let input_schema = obj
        .get("parameters")
        .cloned()
        .filter(|v| v.is_object())
        .unwrap_or_else(|| json!({}));
    Some(AnthropicTool {
        name,
        description,
        input_schema,
    })
}

/// tool_choice（中立 String）+ parallel_tool_calls → Anthropic tool_choice。
///
/// 映射：``（空）→ 不传（服务端默认 auto）；`none` → `{type:none}`（显式禁用工具调用）；
/// `auto` → `{type:auto}`；`required` → `{type:any}`；其余视为指定函数名 → `{type:tool, name}`。
///
/// `parallel_tool_calls=false` → 追加 `disable_parallel_tool_use:true`（Anthropic 该字段仅在
/// tool_choice 存在时有效；`none` 已禁用工具无并行可言，故仅对 auto/any/tool 追加）。
fn tool_choice_to_anthropic(
    tool_choice: &str,
    parallel_tool_calls: bool,
) -> Option<AnthropicToolChoice> {
    // 仅在允许并行的多工具场景下，串行禁用才有意义：none 禁用工具、空串不发 tool_choice。
    let disable_parallel = (!parallel_tool_calls).then_some(true);
    match tool_choice {
        "" => None,
        "none" => Some(AnthropicToolChoice {
            type_: "none".to_string(),
            name: None,
            disable_parallel_tool_use: None,
        }),
        "auto" => Some(AnthropicToolChoice {
            type_: "auto".to_string(),
            name: None,
            disable_parallel_tool_use: disable_parallel,
        }),
        "required" => Some(AnthropicToolChoice {
            type_: "any".to_string(),
            name: None,
            disable_parallel_tool_use: disable_parallel,
        }),
        name => Some(AnthropicToolChoice {
            type_: "tool".to_string(),
            name: Some(name.to_string()),
            disable_parallel_tool_use: disable_parallel,
        }),
    }
}

// ===========================================================================
// AnthropicAdapter：中立 LanguageModel 契约 → AnthropicClient 桥接
// ===========================================================================

/// Anthropic Messages 协议 adapter：实现中立 `LanguageModel` trait，内部委托 `AnthropicClient`。
///
/// 与 `OpenaiChatAdapter` 同构；差异仅在请求翻译（`From<UnifiedRequest> for AnthropicApiRequest`）
/// 与认证改写（构造时把内层 provider 包一层 `AnthropicAuth`，把 `Authorization: Bearer` 改写为
/// `x-api-key` + `anthropic-version`）。事件流归一复用 `response_stream_to_unified`（与
/// Responses / Chat adapter 共用同一份，零改动）。
pub struct AnthropicAdapter<T: HttpTransport> {
    inner: AnthropicClient<T>,
    /// 可选的最大输出 token 上限（来自 `Provider.max_output_tokens`）。
    /// Some 时覆盖 `From<UnifiedRequest>` 写入的 `ANTHROPIC_DEFAULT_MAX_TOKENS` 缺省。
    max_output_tokens: Option<u32>,
}

impl<T: HttpTransport> AnthropicAdapter<T> {
    pub fn new(transport: T, provider: Provider, auth: SharedAuthProvider) -> Self {
        // 内层 provider（Bearer 形态）→ AnthropicAuth（x-api-key + anthropic-version 形态）。
        let anthropic_auth: SharedAuthProvider = Arc::new(AnthropicAuth::new(auth));
        // provider 即将 move 进 AnthropicClient，先拷出 max_output_tokens（Option<u32>: Copy）。
        let max_output_tokens = provider.max_output_tokens;
        Self {
            inner: AnthropicClient::new(transport, provider, anthropic_auth),
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

impl<T: HttpTransport> LanguageModel for AnthropicAdapter<T> {
    fn stream(
        &self,
        request: UnifiedRequest,
        options: UnifiedRequestOptions,
    ) -> BoxFuture<'_, Result<UnifiedEventStream, UnifiedError>> {
        Box::pin(async move {
            // 中立请求 / 选项 → Anthropic wire 请求 / 选项（翻译表 + 既有 From）。
            let mut api_request: AnthropicApiRequest = request.into();
            // provider 显式配置的 max_output_tokens 覆盖 From 写入的 ANTHROPIC_DEFAULT_MAX_TOKENS
            // 缺省；None 时保持 16384（内置 Claude 模型均支持，见常量注释）。
            if let Some(max) = self.max_output_tokens {
                api_request.max_tokens = max;
            }
            let api_options: ResponsesOptions = options.into();
            // 委托 AnthropicClient；空-messages 守卫由 client 层 stream_request 承担（保护所有
            // 调用方），具体协议错误原样透传为 Passthrough，调用方可 downcast 回 ApiError。
            let api_stream = self
                .inner
                .stream_request(api_request, api_options)
                .await
                .map_err(UnifiedError::passthrough)?;
            // Anthropic parser 已产出 ResponseEvent，事件归一复用 Responses 同一份逻辑。
            Ok(response_stream_to_unified(api_stream))
        })
    }
}
