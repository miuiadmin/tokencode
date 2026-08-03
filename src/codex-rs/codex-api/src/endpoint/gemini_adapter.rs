//! Gemini generateContent 协议 adapter —— 把中立 `LanguageModel` 契约桥接到 `GeminiClient`。
//!
//! 与 `AnthropicAdapter` 同构；差异仅在请求翻译与 SSE 解析：
//! - 请求：Responses 形态的 `UnifiedRequest`（`input: Vec<ResponseItem>` 等）→ Gemini 形态的
//!   `GeminiApiRequest`（`contents[].parts[]` + 顶层 `systemInstruction` + `functionCall`/
//!   `functionResponse` + `tools[].functionDeclarations[]` + `toolConfig` + `generationConfig`）。
//! - 事件：`GeminiClient` / `spawn_gemini_stream` 已产出 core 消费的 `ResponseEvent`，
//!   事件归一复用 `response_stream_to_unified`（与 Responses / Chat / Anthropic adapter 共用）。
//!
//! 对 `ResponseItem` 的翻译为编译期穷尽 match：`ResponseItem` 新增/删除变体时本文件编译失败而非静默漏映射。
//!
//! 本文件分两段：① `From<UnifiedRequest>` 翻译段（请求侧）；② `GeminiAdapter` impl 段
//! （`LanguageModel` 桥接，依赖 `GeminiClient` + `spawn_gemini_stream`）。

use crate::auth::SharedAuthProvider;
use crate::common::GeminiApiRequest;
use crate::common::GeminiContent;
use crate::common::GeminiFunctionCall;
use crate::common::GeminiFunctionCallingConfig;
use crate::common::GeminiFunctionResponse;
use crate::common::GeminiInlineData;
use crate::common::GeminiPart;
use crate::common::GeminiSystemInstruction;
use crate::common::GeminiThinkingConfig;
use crate::common::GeminiTool;
use crate::common::GeminiToolConfig;
use crate::common::GeminiToolDeclaration;
use codex_language_model::UnifiedRequest;
use codex_protocol::models::ContentItem;
use codex_protocol::models::ReasoningItemContent;
use codex_protocol::models::ResponseItem;
use codex_protocol::ToolName;
use serde_json::Value;
use serde_json::json;
use std::collections::HashMap;
use tracing::debug;
use tracing::warn;

// ----- GeminiAdapter 桥接段额外导入 -----
use crate::endpoint::ResponsesOptions;
use crate::endpoint::gemini::GeminiAuth;
use crate::endpoint::gemini::GeminiClient;
use crate::endpoint::response_stream_to_unified;
use crate::provider::Provider;
use crate::telemetry::SseTelemetry;
use codex_client::HttpTransport;
use codex_client::RequestTelemetry;
use codex_language_model::LanguageModel;
use codex_language_model::UnifiedError;
use codex_language_model::UnifiedEventStream;
use codex_language_model::UnifiedRequestOptions;
use codex_protocol::openai_models::ReasoningEffort;
use futures::future::BoxFuture;
use std::sync::Arc;

// ===========================================================================
// UnifiedRequest → GeminiApiRequest 翻译（编译期对 ResponseItem 穷尽 match）
// ===========================================================================

impl From<UnifiedRequest> for GeminiApiRequest {
    fn from(req: UnifiedRequest) -> Self {
        // instructions（非空）→ 顶层 systemInstruction（Gemini 不允许 contents 出现 system 角色）。
        let system_instruction = if req.instructions.is_empty() {
            None
        } else {
            Some(GeminiSystemInstruction {
                role: None,
                parts: vec![GeminiPart::Text {
                    text: req.instructions.clone(),
                }],
            })
        };

        // call_id → 工具名映射：Gemini functionResponse 必须带 name，而 Responses
        // FunctionCallOutput 只有 call_id。遍历时 FunctionCall 记录映射、FunctionCallOutput 查映射。
        let mut tool_name_by_call_id: HashMap<String, String> = HashMap::new();

        // input: Vec<ResponseItem> → contents[]（穷尽 match，逐变体翻译；连续同角色项合并为一条 content）。
        let mut contents: Vec<GeminiContent> = Vec::new();
        for item in req.input {
            translate_item(item, &mut contents, &mut tool_name_by_call_id);
        }

        // tools → functionDeclarations（经 sanitize_gemini_schema 清洗，否则 Gemini 严格校验 400）。
        // 形态 `tools:[{functionDeclarations:[...]}]`（函数声明包一层 functionDeclarations 数组）。
        let tools = req.tools.and_then(|tools| {
            let decls: Vec<GeminiToolDeclaration> =
                tools.into_iter().filter_map(responses_tool_to_gemini).collect();
            if decls.is_empty() {
                None
            } else {
                Some(vec![GeminiTool {
                    function_declarations: decls,
                }])
            }
        });

        // tool_choice → toolConfig（仅当 tools 非空时才发——无 tools 时 Gemini 拒绝 toolConfig）。
        let tool_config = tools
            .as_ref()
            .and_then(|_| tool_choice_to_gemini(&req.tool_choice));

        GeminiApiRequest {
            contents,
            system_instruction,
            tools,
            tool_config,
            // generation_config（max_output_tokens / thinking_config）由 adapter stream 推导注入。
            generation_config: None,
        }
    }
}

/// 剥除 Gemini responseSchema 不认的 JSON Schema 键（用于结构化输出，区别于工具参数）。
///
/// Gemini `responseSchema` 只认 OpenAPI 3.0 子集，下列键会被服务端拒（400）：
/// - `$schema`：JSON Schema draft 声明，OpenAPI 无此字段。
/// - `title`：OpenAPI Schema 不带 title。
/// - `$defs` / `$ref`：OpenAPI 用 `components.schemas` + `$ref` 语义不同，Gemini responseSchema
///   不做引用解析，含 `$ref` 直接拒；`$defs` 是纯定义容器，Gemini 不消费。
/// - `default` / `examples`：responseSchema 不支持默认值与样例。
///
/// 递归剥除后，再交给 [`sanitize_gemini_schema`] 做工具参数同款的 enum stringify / array
/// items 补全 / 非容器剥 properties 等清洗。两者顺序固定：先 strip（移除禁键）再 sanitize
/// （规整 Gemini 专属形态）。
///
/// 仅用于结构化输出（`generationConfig.responseSchema`）；工具参数（`functionDeclarations`）
/// 仅需 `sanitize_gemini_schema`，不走本函数。
fn strip_gemini_response_schema_keys(schema: &mut Value) {
    match schema {
        Value::Object(obj) => {
            // 先剥自身禁键，再递归 properties / items / array items（schema 嵌套点）。
            obj.remove("$schema");
            obj.remove("title");
            obj.remove("$defs");
            obj.remove("$ref");
            obj.remove("default");
            obj.remove("examples");
            if let Some(props) = obj.get_mut("properties").and_then(|v| v.as_object_mut()) {
                for v in props.values_mut() {
                    strip_gemini_response_schema_keys(v);
                }
            }
            if let Some(items) = obj.get_mut("items") {
                strip_gemini_response_schema_keys(items);
            }
            // enum / required / allOf 等数组型容器内的元素多为字面量，不递归（非 schema 节点）。
            // definitions（draft 旧名）若残留一并剥。
            obj.remove("definitions");
        }
        Value::Array(arr) => {
            for v in arr.iter_mut() {
                strip_gemini_response_schema_keys(v);
            }
        }
        _ => {}
    }
}

/// 清洗 JSON Schema 使其符合 Gemini 工具参数校验（Gemini 比 OpenAI/Anthropic 严格，不做会 400）。
///
/// 四件套（递归）：
/// ① **enum stringify**：`enum:[1,2,3]` → `enum:["1","2","3"]`，并把 `type` 改写为 `"string"`
///   （Gemini 的 enum 只允许字符串值；原 integer/number/boolean enum 须 stringify）。
/// ② **array items 补全**：`type:"array"` 缺 `items` 补 `{type:"string"}`（Gemini 要求 array 必有 items）。
/// ③ **非 object/array 剥 properties/required**：`type:"string"` 等带 properties/required 会被拒，剥掉。
/// ④ **required 过滤**：`required` 只留 `properties` 里真实存在的键（指向不存在字段的 required 非法），清空则移除。
fn sanitize_gemini_schema(schema: &mut Value) {
    match schema {
        Value::Object(obj) => {
            // ⓪ type 数组（如 ["object","null"]）归一为单 type + nullable:true。Gemini
            //    responseSchema 只认 OpenAPI 3.0 的 `nullable`，不认 JSON Schema 的 type 数组表 null。
            if let Some(arr) = obj.get_mut("type").and_then(|v| v.as_array_mut()) {
                // 取首个非 null 类型；全 null 退化为 "string"。
                let primary = arr
                    .iter()
                    .find_map(|v| v.as_str().filter(|s| *s != "null"))
                    .unwrap_or("string")
                    .to_string();
                let nullable = arr.iter().any(|v| v.as_str() == Some("null"));
                obj.insert("type".to_string(), Value::String(primary));
                if nullable {
                    obj.insert("nullable".to_string(), Value::Bool(true));
                }
            }
            // ① enum stringify + type 改写为 string（须在取 type 之前做，让后续分支按新 type 走）。
            if let Some(enum_arr) = obj.get_mut("enum").and_then(|v| v.as_array_mut()) {
                for item in enum_arr.iter_mut() {
                    if !item.is_string() {
                        *item = Value::String(item.to_string());
                    }
                }
                obj.insert("type".to_string(), Value::String("string".to_string()));
            }

            let ty = obj.get("type").and_then(|v| v.as_str()).map(|s| s.to_string());
            match ty.as_deref() {
                Some("array") => {
                    // ② array 必须有 items；缺则补 string。
                    if !obj.contains_key("items") {
                        obj.insert("items".to_string(), json!({"type":"string"}));
                    }
                    if let Some(items) = obj.get_mut("items") {
                        sanitize_gemini_schema(items);
                    }
                }
                Some("object") => {
                    // 递归 properties 各值。
                    if let Some(props) = obj.get_mut("properties").and_then(|v| v.as_object_mut()) {
                        for v in props.values_mut() {
                            sanitize_gemini_schema(v);
                        }
                    }
                    // ④ required 只留 properties 里真实存在的键；清空则移除整个 required。
                    let required_keys: Vec<String> = obj
                        .get("required")
                        .and_then(|v| v.as_array())
                        .map(|arr| arr.iter().filter_map(|v| v.as_str().map(String::from)).collect())
                        .unwrap_or_default();
                    if !required_keys.is_empty() {
                        let prop_keys: std::collections::HashSet<&str> = obj
                            .get("properties")
                            .and_then(|v| v.as_object())
                            .map(|m| m.keys().map(String::as_str).collect())
                            .unwrap_or_default();
                        let kept: Vec<Value> = required_keys
                            .iter()
                            .filter(|k| prop_keys.contains(k.as_str()))
                            .map(|k| Value::String(k.clone()))
                            .collect();
                        if kept.is_empty() {
                            obj.remove("required");
                        } else {
                            obj.insert("required".to_string(), Value::Array(kept));
                        }
                    }
                }
                Some(_) => {
                    // ③ 非容器类型（string/integer/number/boolean/null）：剥掉非法的 properties/required。
                    obj.remove("properties");
                    obj.remove("required");
                }
                None => {
                    // type 缺省：宽容地仍递归 properties（嵌套 schema 中可能出现无 type 的对象）。
                    if let Some(props) = obj.get_mut("properties").and_then(|v| v.as_object_mut()) {
                        for v in props.values_mut() {
                            sanitize_gemini_schema(v);
                        }
                    }
                }
            }
        }
        Value::Array(arr) => {
            for v in arr.iter_mut() {
                sanitize_gemini_schema(v);
            }
        }
        _ => {}
    }
}

/// 逐 `ResponseItem` 变体翻译为 Gemini 内容 parts（或跳过）。
///
/// 连续同角色项合并为同一条 content（Gemini 要求 user/model 交替，合并减少条数）。
/// `tool_name_by_call_id` 在遍历中填充（FunctionCall/CustomToolCall）与查询（FunctionCallOutput）。
fn translate_item(
    item: ResponseItem,
    contents: &mut Vec<GeminiContent>,
    tool_name_by_call_id: &mut HashMap<String, String>,
) {
    match item {
        ResponseItem::Message { role, content, .. } => {
            // 仅 user/assistant 角色；system/developer 应经顶层 systemInstruction 承载，此处跳过避免非法角色。
            let Some(role) = normalize_role(role) else {
                return;
            };
            let parts = message_content_to_parts(&content);
            append_parts(contents, &role, parts);
        }
        ResponseItem::FunctionCall {
            name, namespace, arguments, call_id, ..
        } => {
            // 历史 functionCall 的 name 必须与 functionDeclarations（展平后的 flat 名）一致，
            // 否则 Gemini 服务端校验失败；flat 名同时入映射供 functionResponse 取用。
            let flat = ToolName::new(namespace, name).to_flat_wire_name();
            tool_name_by_call_id.insert(call_id, flat.clone());
            push_function_call(contents, flat, arguments);
        }
        ResponseItem::CustomToolCall {
            call_id, name, namespace, input, ..
        } => {
            let flat = ToolName::new(namespace, name).to_flat_wire_name();
            tool_name_by_call_id.insert(call_id, flat.clone());
            push_function_call(contents, flat, input);
        }
        ResponseItem::FunctionCallOutput { call_id, output, .. } => {
            let name = tool_name_by_call_id.get(&call_id).cloned();
            push_function_response(contents, call_id, output.body.to_text(), output.success, name);
        }
        ResponseItem::CustomToolCallOutput { call_id, output, .. } => {
            let name = tool_name_by_call_id.get(&call_id).cloned();
            push_function_response(contents, call_id, output.body.to_text(), output.success, name);
        }
        ResponseItem::Reasoning { content, continuity_token, .. } => {
            // 仅带 Gemini thoughtSignature（continuity_token 承载）的思考才可回喂——无签名的推理项
            // 无 Gemini 连续凭证，跳过。文本取 content 的 ReasoningText/Text 拼接（OpenAI
            // encrypted_content 走自家字段，与此无关）。回喂为 model 角色的 Thought part（thought:true
            // + thoughtSignature），随后同 turn 的 Message/FunctionCall 因同角色并入同一 model 消息。
            let Some(signature) = continuity_token else {
                return;
            };
            let thought_text = content
                .map(|cs| {
                    cs.into_iter()
                        .filter_map(|c| match c {
                            ReasoningItemContent::ReasoningText { text }
                            | ReasoningItemContent::Text { text } => Some(text),
                        })
                        .collect::<String>()
                })
                .unwrap_or_default();
            if thought_text.is_empty() {
                return;
            }
            append_parts(
                contents,
                "model",
                vec![GeminiPart::Thought {
                    thought: true,
                    text: thought_text,
                    thought_signature: Some(signature),
                }],
            );
        }
        // 以下变体 Gemini 无等价或不可解码：一律跳过（与 Anthropic adapter 对齐）。
        // - AgentMessage / LocalShellCall / ToolSearchCall / ToolSearchOutput / WebSearchCall /
        //   ImageGenerationCall / Compaction / ContextCompaction / CompactionTrigger /
        //   AdditionalTools：Responses 专属 / 多 agent / 压缩 / 控制项。
        // - Other：未知透传项。
        ResponseItem::AgentMessage { .. }
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

/// 若干 parts 并入 contents：末条同角色则追加 parts，否则新建一条。空 parts 不操作。
fn append_parts(contents: &mut Vec<GeminiContent>, role: &str, new_parts: Vec<GeminiPart>) {
    if new_parts.is_empty() {
        return;
    }
    if let Some(last) = contents.last_mut()
        && last.role == role
    {
        last.parts.extend(new_parts);
    } else {
        contents.push(GeminiContent {
            role: role.to_string(),
            parts: new_parts,
        });
    }
}

/// 模型发起的工具调用 → `functionCall` part，并入 model 消息（Gemini 工具调用固定在 model 角色）。
fn push_function_call(contents: &mut Vec<GeminiContent>, name: String, arguments: String) {
    let part = GeminiPart::FunctionCall {
        function_call: GeminiFunctionCall {
            name,
            args: parse_tool_args(&arguments),
        },
    };
    append_parts(contents, "model", vec![part]);
}

/// 工具调用输出 → user 消息的 `functionResponse` part。
///
/// Gemini `functionResponse` 形态 `{name, response}`：`name` 必填（服务端校验）。`name_opt` 来自
/// call_id→name 映射；映射缺失（call 被压缩/历史截断）时用 call_id 兜底并 warn——保证不 400
/// （name 仅用于回显对齐，call_id 作占位不影响模型推理语义）。
///
/// `response`：成功时把 output 文本解析为 JSON 对象（非对象/解析失败则包 `{output: text}`）；
/// 失败（`success == Some(false)`）时塞 `{error: text}`，让模型区分成败（Gemini 无独立 is_error 字段）。
fn push_function_response(
    contents: &mut Vec<GeminiContent>,
    call_id: String,
    text: Option<String>,
    success: Option<bool>,
    name_opt: Option<String>,
) {
    let name = name_opt.unwrap_or_else(|| {
        warn!(
            call_id,
            "Gemini functionResponse 缺工具名（call_id→name 映射未命中，可能 call 被压缩），用 call_id 兜底"
        );
        call_id.clone()
    });
    let body = text.unwrap_or_default();
    let response = if success == Some(false) {
        json!({ "error": body })
    } else {
        serde_json::from_str::<Value>(&body)
            .ok()
            .filter(|v| v.is_object())
            .unwrap_or_else(|| json!({ "output": body }))
    };
    let part = GeminiPart::FunctionResponse {
        function_response: GeminiFunctionResponse { name, response },
    };
    append_parts(contents, "user", vec![part]);
}

/// `ContentItem` 列表 → Gemini parts。空文本项过滤（Gemini 拒绝空 text part）。
fn message_content_to_parts(content: &[ContentItem]) -> Vec<GeminiPart> {
    content
        .iter()
        .filter_map(|c| match c {
            ContentItem::InputText { text } | ContentItem::OutputText { text } => {
                if text.is_empty() {
                    None
                } else {
                    Some(GeminiPart::Text { text: text.clone() })
                }
            }
            ContentItem::InputImage { image_url, .. } => gemini_inline_data(image_url),
        })
        .collect()
}

/// 图片 URL → Gemini `inlineData`（base64）。
///
/// Gemini 只支持 `inlineData`（base64 内嵌），**不支持外链 url**（与 Anthropic 不同——后者可发 url）。
/// `data:<mime>;base64,<data>` 拆为 inlineData；缺逗号的残缺 data URL 与外链均无法内嵌，跳过并 warn。
fn gemini_inline_data(image_url: &str) -> Option<GeminiPart> {
    if let Some(rest) = image_url.strip_prefix("data:") {
        if let Some((meta, data)) = rest.split_once(',') {
            let mime_type = meta.split(';').next().unwrap_or("image/png").to_string();
            return Some(GeminiPart::InlineData {
                inline_data: GeminiInlineData {
                    mime_type,
                    data: data.to_string(),
                },
            });
        }
        warn!(image_url, "data: 图片 URL 缺逗号（无法拆出 base64），跳过该图片");
        return None;
    }
    warn!(
        image_url,
        "Gemini 仅支持 base64 内嵌图片，外链 URL 无法转换，跳过"
    );
    None
}

/// 解析工具参数 JSON 串为 Gemini `args`（须为对象）。
///
/// 解析失败或非对象时回落空对象（Gemini 要求 args 为对象），并 warn——静默替换 `{}` 会丢弃真实参数。
fn parse_tool_args(arguments: &str) -> Value {
    serde_json::from_str::<Value>(arguments)
        .ok()
        .filter(|v| v.is_object())
        .unwrap_or_else(|| {
            warn!(arguments, "工具参数非合法 JSON 对象，回落为空对象 args={{}}");
            json!({})
        })
}

/// 归一化 message role：`user`→`user`、`assistant`→`model`；其余（system/developer/未知）返回 None。
///
/// Gemini contents[] 只认 user/model 两角色（与 OpenAI assistant / Anthropic assistant 不同）。
fn normalize_role(role: String) -> Option<String> {
    match role.as_str() {
        "user" => Some("user".to_string()),
        "assistant" => Some("model".to_string()),
        _ => None,
    }
}

/// Responses 工具描述（`{type:"function", name, description, parameters}`）→ Gemini 函数声明。
///
/// `parameters` 经 `sanitize_gemini_schema` 清洗；缺省 `{"type":"object"}`（无参工具的合法空 schema）。
/// 非 function 类型（web_search / local_shell / mcp 等 Responses 内置工具）在 Gemini 无等价，跳过。
fn responses_tool_to_gemini(tool: Value) -> Option<GeminiToolDeclaration> {
    let obj = tool.as_object()?;
    let kind = obj.get("type").and_then(|v| v.as_str()).unwrap_or("");
    if kind != "function" {
        debug!(tool_type = kind, "跳过非 function 工具：Gemini 协议无等价");
        return None;
    }
    let name = obj.get("name")?.as_str()?.to_string();
    let description = obj
        .get("description")
        .and_then(|v| v.as_str())
        .map(String::from);
    let mut parameters = obj
        .get("parameters")
        .cloned()
        .filter(|v| v.is_object())
        .unwrap_or_else(|| json!({"type":"object"}));
    sanitize_gemini_schema(&mut parameters);
    Some(GeminiToolDeclaration {
        name,
        description,
        parameters: Some(parameters),
    })
}

/// tool_choice（中立 String）→ Gemini `toolConfig`。
///
/// 映射：空串 → 不传（服务端默认 AUTO）；`none` → `NONE`（禁用工具）；`auto` → `AUTO`；
/// `required` → `ANY`（强制调用任一）；其余视为指定函数名 → `ANY` + `allowedFunctionNames:[name]`。
///
/// Gemini 无「禁用并行工具」字段（`parallel_tool_calls=false` 不表达——与 Anthropic 的
/// `disable_parallel_tool_use` 不同），故串行意图在 Gemini 协议无承载点，忽略。
fn tool_choice_to_gemini(tool_choice: &str) -> Option<GeminiToolConfig> {
    let config = match tool_choice {
        "" => return None,
        "none" => GeminiFunctionCallingConfig {
            mode: "NONE".to_string(),
            allowed_function_names: None,
        },
        "auto" => GeminiFunctionCallingConfig {
            mode: "AUTO".to_string(),
            allowed_function_names: None,
        },
        "required" => GeminiFunctionCallingConfig {
            mode: "ANY".to_string(),
            allowed_function_names: None,
        },
        name => GeminiFunctionCallingConfig {
            mode: "ANY".to_string(),
            allowed_function_names: Some(vec![name.to_string()]),
        },
    };
    Some(GeminiToolConfig {
        function_calling_config: config,
    })
}

// ===========================================================================
// GeminiAdapter：中立 LanguageModel 契约 → GeminiClient 桥接
// ===========================================================================

/// 从中立 `ReasoningEffort` 档位 + model 名推导 Gemini thinking 配置（双模态）。
///
/// 按 model 名分叉（Gemini 不同代次的思考配置形态不同）：
/// - **gemini-2.5**：发 `thinking_budget`（int；0=显式禁用，>0 为思考 token 预算）。
///   `None`/`Minimal`/`Custom` → 0（显式禁用，覆盖 2.5 默认开启的思考）；其余档位 → 量级预算。
///   Gemini 2.5 的思考预算**独立于** `max_output_tokens`（不占用输出预算，与 Anthropic 的
///   `budget_tokens` 占用 `max_tokens` 不同），故无需夹断。
/// - **gemini-3.x**：发 `thinking_level`（enum：minimal/low/medium/high）。`None`/`Minimal`/`Custom`
///   → `minimal`（3.x 无法显式禁用，minimal 是最低档）；`High`+ → `high`。
/// - **其他 model**：返回 None（不发 thinkingConfig，用服务端默认）——保守，避免给未知 model
///   发它不支持的思考字段。
///
/// 启用思考时（2.5 budget>0 / 3.x 任意 level）均带 `include_thoughts:true`，让思考流以
/// `{thought:true, text}` part 返回，adapter 据此归一为 `ReasoningContentDelta`。
fn thinking_config_from_effort(
    effort: Option<ReasoningEffort>,
    model: &str,
) -> Option<GeminiThinkingConfig> {
    if model.contains("gemini-2.5") {
        let (budget, include) = match effort {
            None
            | Some(ReasoningEffort::None)
            | Some(ReasoningEffort::Minimal)
            | Some(ReasoningEffort::Custom(_)) => (0, None),
            Some(ReasoningEffort::Low) => (2_048, Some(true)),
            Some(ReasoningEffort::Medium) => (8_192, Some(true)),
            Some(ReasoningEffort::High) => (16_384, Some(true)),
            Some(ReasoningEffort::XHigh) => (20_480, Some(true)),
            Some(ReasoningEffort::Max) | Some(ReasoningEffort::Ultra) => (24_576, Some(true)),
        };
        Some(GeminiThinkingConfig {
            thinking_budget: Some(budget),
            thinking_level: None,
            include_thoughts: include,
        })
    } else if model.contains("gemini-3") {
        let level = match effort {
            None
            | Some(ReasoningEffort::None)
            | Some(ReasoningEffort::Minimal)
            | Some(ReasoningEffort::Custom(_)) => "minimal",
            Some(ReasoningEffort::Low) => "low",
            Some(ReasoningEffort::Medium) => "medium",
            Some(ReasoningEffort::High)
            | Some(ReasoningEffort::XHigh)
            | Some(ReasoningEffort::Max)
            | Some(ReasoningEffort::Ultra) => "high",
        };
        Some(GeminiThinkingConfig {
            thinking_budget: None,
            thinking_level: Some(level.to_string()),
            include_thoughts: Some(true),
        })
    } else {
        None
    }
}

/// Gemini generateContent 协议 adapter：实现中立 `LanguageModel` trait，内部委托 `GeminiClient`。
///
/// 与 `AnthropicAdapter` 同构；差异仅在请求翻译（`From<UnifiedRequest> for GeminiApiRequest`）、
/// 认证改写（构造时把内层 provider 包一层 `GeminiAuth`，把 `Authorization: Bearer` 改写为
/// `x-goog-api-key`）、思考配置（双模态，按 model 名分叉）。事件流归一复用
/// `response_stream_to_unified`（与 Responses / Chat / Anthropic adapter 共用同一份，零改动）。
pub struct GeminiAdapter<T: HttpTransport> {
    inner: GeminiClient<T>,
    /// 可选的最大输出 token 上限（来自 `Provider.max_output_tokens`）。
    /// Some 时注入 `generationConfig.maxOutputTokens`；None 时不发（走服务端默认）。
    max_output_tokens: Option<u32>,
}

impl<T: HttpTransport> GeminiAdapter<T> {
    pub fn new(transport: T, provider: Provider, auth: SharedAuthProvider) -> Self {
        // 内层 provider（Bearer 形态）→ GeminiAuth（x-goog-api-key 形态）。
        let gemini_auth: SharedAuthProvider = Arc::new(GeminiAuth::new(auth));
        // provider 即将 move 进 GeminiClient，先拷出 max_output_tokens（Option<u32>: Copy）。
        let max_output_tokens = provider.max_output_tokens;
        Self {
            inner: GeminiClient::new(transport, provider, gemini_auth),
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

impl<T: HttpTransport> LanguageModel for GeminiAdapter<T> {
    fn stream(
        &self,
        request: UnifiedRequest,
        options: UnifiedRequestOptions,
    ) -> BoxFuture<'_, Result<UnifiedEventStream, UnifiedError>> {
        Box::pin(async move {
            // 先取出 effort + model + text_format（request.into() 会消费 request）：model 用于推导
            // thinking 双模态分叉 + 构造 URL path（Gemini model 在 path 不在 body）；text_format
            // 用于结构化输出（responseMimeType + responseSchema）。
            let effort = request.reasoning.as_ref().and_then(|r| r.effort.clone());
            let model = request.model.clone();
            let text_format = request.text.as_ref().and_then(|t| t.format.clone());
            // 中立请求 → Gemini wire 请求（翻译表 + 既有 From；generation_config 此时为 None）。
            let mut api_request: GeminiApiRequest = request.into();
            // 注入 generation_config（thinking_config 双模态 + max_output_tokens + 结构化输出）。
            let generation_cfg = api_request
                .generation_config
                .get_or_insert_with(Default::default);
            generation_cfg.thinking_config = thinking_config_from_effort(effort, &model);
            if let Some(max) = self.max_output_tokens {
                generation_cfg.max_output_tokens = Some(max);
            }
            // 结构化输出：带 text.format → responseMimeType=application/json + responseSchema。
            // 先 strip 禁键（$schema/title/$defs/$ref/default/examples，Gemini responseSchema 不认）
            // 再过 sanitize_gemini_schema（enum stringify / array items / 剥非容器 properties 等）。
            if let Some(fmt) = text_format {
                let mut schema = fmt.schema;
                strip_gemini_response_schema_keys(&mut schema);
                sanitize_gemini_schema(&mut schema);
                generation_cfg.response_mime_type = Some("application/json".to_string());
                generation_cfg.response_schema = Some(schema);
            }
            let api_options: ResponsesOptions = options.into();
            // 委托 GeminiClient；空-contents 守卫由 client 层 stream_request 承担（保护所有调用方），
            // 具体协议错误原样透传为 Passthrough，调用方可 downcast 回 ApiError。
            let api_stream = self
                .inner
                .stream_request(&model, api_request, api_options)
                .await
                .map_err(UnifiedError::passthrough)?;
            // Gemini parser 已产出 ResponseEvent，事件归一复用 Responses 同一份逻辑。
            Ok(response_stream_to_unified(api_stream))
        })
    }
}

#[cfg(test)]
mod tests {
    use super::sanitize_gemini_schema;
    use serde_json::json;

    #[test]
    fn sanitize_normalizes_nullable_type_array() {
        let mut schema = json!({
            "type": ["object", "null"],
            "properties": {
                "action": {"type": ["object", "null"]}
            }
        });
        sanitize_gemini_schema(&mut schema);
        // 顶层：type 数组归一为单 type + nullable:true（Gemini 只认 OpenAPI 3.0 的 nullable）。
        assert_eq!(schema["type"], "object");
        assert_eq!(schema["nullable"], true);
        // 嵌套 properties 里的同款 type 数组也被递归归一。
        assert_eq!(schema["properties"]["action"]["type"], "object");
        assert_eq!(schema["properties"]["action"]["nullable"], true);
    }

    #[test]
    fn sanitize_keeps_non_nullable_type_untouched() {
        let mut schema = json!({"type": "string", "enum": ["a", "b"]});
        sanitize_gemini_schema(&mut schema);
        assert_eq!(schema["type"], "string");
        assert!(schema.get("nullable").is_none());
        assert_eq!(schema["enum"], json!(["a", "b"]));
    }
}
