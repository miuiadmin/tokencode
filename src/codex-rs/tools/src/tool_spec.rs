use crate::FreeformTool;
use crate::JsonSchema;
use crate::LoadableToolSpec;
use crate::ResponsesApiNamespace;
use crate::ResponsesApiNamespaceTool;
use crate::ResponsesApiTool;
use codex_protocol::ToolName;
use codex_protocol::config_types::WebSearchContextSize;
use codex_protocol::config_types::WebSearchFilters as ConfigWebSearchFilters;
use codex_protocol::config_types::WebSearchUserLocation as ConfigWebSearchUserLocation;
use codex_protocol::config_types::WebSearchUserLocationType;
use serde::Serialize;
use serde_json::Value;

/// When serialized as JSON, this produces a valid "Tool" in the OpenAI
/// Responses API.
#[derive(Debug, Clone, Serialize, PartialEq)]
#[serde(tag = "type")]
pub enum ToolSpec {
    #[serde(rename = "function")]
    Function(ResponsesApiTool),
    #[serde(rename = "namespace")]
    Namespace(ResponsesApiNamespace),
    #[serde(rename = "tool_search")]
    ToolSearch {
        execution: String,
        description: String,
        parameters: JsonSchema,
    },
    #[serde(rename = "image_generation")]
    ImageGeneration { output_format: String },
    // TODO: Understand why we get an error on web_search although the API docs
    // say it's supported.
    // https://platform.openai.com/docs/guides/tools-web-search?api-mode=responses#:~:text=%7B%20type%3A%20%22web_search%22%20%7D%2C
    // The `external_web_access` field determines whether the web search is over
    // cached or live content.
    // https://platform.openai.com/docs/guides/tools-web-search#live-internet-access
    #[serde(rename = "web_search")]
    WebSearch {
        #[serde(skip_serializing_if = "Option::is_none")]
        external_web_access: Option<bool>,
        #[serde(skip_serializing_if = "Option::is_none")]
        index_gated_web_access: Option<bool>,
        #[serde(skip_serializing_if = "Option::is_none")]
        filters: Option<ResponsesApiWebSearchFilters>,
        #[serde(skip_serializing_if = "Option::is_none")]
        user_location: Option<ResponsesApiWebSearchUserLocation>,
        #[serde(skip_serializing_if = "Option::is_none")]
        search_context_size: Option<WebSearchContextSize>,
        #[serde(skip_serializing_if = "Option::is_none")]
        search_content_types: Option<Vec<String>>,
    },
    #[serde(rename = "custom")]
    Freeform(FreeformTool),
}

impl ToolSpec {
    pub fn name(&self) -> &str {
        match self {
            ToolSpec::Function(tool) => tool.name.as_str(),
            ToolSpec::Namespace(namespace) => namespace.name.as_str(),
            ToolSpec::ToolSearch { .. } => "tool_search",
            ToolSpec::ImageGeneration { .. } => "image_generation",
            ToolSpec::WebSearch { .. } => "web_search",
            ToolSpec::Freeform(tool) => tool.name.as_str(),
        }
    }
}

impl From<LoadableToolSpec> for ToolSpec {
    fn from(value: LoadableToolSpec) -> Self {
        match value {
            LoadableToolSpec::Function(tool) => ToolSpec::Function(tool),
            LoadableToolSpec::Namespace(namespace) => ToolSpec::Namespace(namespace),
        }
    }
}

/// Returns JSON values that are compatible with Function Calling in the
/// Responses API:
/// https://platform.openai.com/docs/guides/function-calling?api-mode=responses
pub fn create_tools_json_for_responses_api(
    tools: &[ToolSpec],
) -> Result<Vec<Value>, serde_json::Error> {
    let mut tools_json = Vec::new();

    for tool in tools {
        let json = serde_json::to_value(tool)?;
        tools_json.push(json);
    }

    Ok(tools_json)
}

/// 构造「结构化进度副信道」的统一输出 JSON Schema（增量协议）。
///
/// 模型每步须按此 schema 输出 `{action, progress_patch, digest_override}`：
/// - `action.tool` 为 enum（工具名列表），规避 Gemini responseSchema 不支持 oneOf 的硬伤，
///   schema 体积恒定、不随工具数膨胀；
/// - `action.arguments` 为自由 object（各工具参数不同），约束挪到 P3 执行前二次校验；
/// - 顶层与固定结构加 `additionalProperties:false` 约束外壳，`arguments` 不加（strict=false，
///   因自由 arguments 与 OpenAI strict 的 `additionalProperties:false` 硬冲突）；
/// - `action` / `progress_patch` 用 `type:["object","null"]` 表 nullable（Gemini sanitize
///   在 `sanitize_gemini_schema` 起始处转成 OpenAPI 3.0 的 `nullable:true`）。
///
/// 设计依据见 `设计文档/结构化输出与进度副信道-设计方案.md` §4。
pub fn build_progress_channel_schema(tool_names: &[String]) -> Value {
    use serde_json::json;
    let enum_vals: Vec<Value> = tool_names.iter().cloned().map(Value::String).collect();
    json!({
        "type": "object",
        "additionalProperties": false,
        "properties": {
            "action": {
                "type": ["object", "null"],
                "additionalProperties": false,
                "properties": {
                    "tool": { "type": "string", "enum": enum_vals },
                    "arguments": { "type": "object" }
                },
                "required": ["tool", "arguments"]
            },
            "progress_patch": {
                "type": ["object", "null"],
                "additionalProperties": false,
                "properties": {
                    "intent": { "type": "string" },
                    "todo_ops": {
                        "type": "array",
                        "items": { "type": "object", "additionalProperties": true }
                    },
                    "assumptions": {
                        "type": "array",
                        "items": { "type": "object", "additionalProperties": true }
                    },
                    "next_step": { "type": "string" }
                }
            },
            "digest_override": {}
        },
        "required": ["action", "progress_patch", "digest_override"]
    })
}

/// 把模型可见工具渲染成中文工具目录文本，编进 instructions（关闭原生 function calling 后，
/// 模型从 instructions 读工具目录，而非 wire 的 `tools` 字段）。
///
/// 与 [`create_tools_json_for_responses_api`] 对偶：后者产出机器可读 wire JSON，
/// 本函数产出模型可读文本。namespace 工具应在调用前先用
/// [`flatten_namespaces_for_flat_wire`] 展平；本函数仍兜底处理未展平的 namespace。
pub fn render_tool_catalog(tools: &[ToolSpec]) -> String {
    use std::fmt::Write as _;
    let mut out = String::new();
    out.push_str("\n\n# 可用工具目录\n");
    out.push_str(
        "（以下工具通过统一输出 schema 的 action.tool 调用，参数填 action.arguments；勿使用原生 function calling）\n\n",
    );
    if tools.is_empty() {
        out.push_str("（当前无可用工具）\n");
        return out;
    }
    for tool in tools {
        match tool {
            ToolSpec::Function(t) => {
                let _ = writeln!(out, "## {}", t.name);
                if !t.description.is_empty() {
                    let _ = writeln!(out, "{}", t.description);
                }
                if let Ok(params) = serde_json::to_string_pretty(&t.parameters) {
                    let _ = writeln!(out, "参数 schema:\n```\n{params}\n```");
                }
            }
            ToolSpec::Freeform(t) => {
                let _ = writeln!(out, "## {}（自由格式工具）", t.name);
                if !t.description.is_empty() {
                    let _ = writeln!(out, "{}", t.description);
                }
                let _ = writeln!(out, "格式: {} ({})", t.format.syntax, t.format.r#type);
                let _ = writeln!(
                    out,
                    "调用形态: action.arguments 填 `{{\"input\": <上方格式文本>}}`（raw 文本经 input 键承载，系统据此提取）"
                );
            }
            ToolSpec::Namespace(ns) => {
                let _ = writeln!(out, "## namespace: {}", ns.name);
                for inner in &ns.tools {
                    let ResponsesApiNamespaceTool::Function(t) = inner;
                    let first_line = t.description.lines().next().unwrap_or("");
                    let _ = writeln!(out, "  - {}: {}", t.name, first_line);
                }
            }
            other => {
                let _ = writeln!(out, "## {}", other.name());
            }
        }
    }
    out
}

/// 把 `ToolSpec::Namespace` 展平成若干 `ToolSpec::Function`，工具名按
/// `{namespace}__{name}` 拼接（见 [`ToolName::to_flat_wire_name`]），其余变体原样保留。
///
/// 用途：非 Responses 协议（OpenAI Chat / Anthropic / Gemini）的 wire 只携带单个工具名
/// 字符串，没有 namespace 概念——`{type:"namespace",...}` 会被这三家 adapter 直接丢弃，
/// 导致 MCP / 多智能体等 namespace 工具对这些协议静默消失。展平后 adapter 只见 function
/// 工具，namespace 工具得以暴露给模型；模型回传的 flat 名再由各家 SSE parser 用
/// [`ToolName::from_flat_wire_name`] 还原出 namespace，从而命中按 namespaced key 注册的
/// 派发表（router 不变）。
///
/// Responses 协议无需展平：其 wire 原生支持 namespace 结构，模型回传时 namespace 是独立字段。
pub fn flatten_namespaces_for_flat_wire(specs: &[ToolSpec]) -> Vec<ToolSpec> {
    let mut flattened = Vec::with_capacity(specs.len());
    for spec in specs {
        match spec {
            ToolSpec::Namespace(namespace) => {
                for inner in &namespace.tools {
                    match inner {
                        ResponsesApiNamespaceTool::Function(tool) => {
                            let mut flat = tool.clone();
                            flat.name = ToolName::namespaced(&namespace.name, &tool.name)
                                .to_flat_wire_name();
                            flattened.push(ToolSpec::Function(flat));
                        }
                    }
                }
            }
            other => flattened.push(other.clone()),
        }
    }
    flattened
}

#[derive(Debug, Clone, Serialize, PartialEq)]
pub struct ResponsesApiWebSearchFilters {
    #[serde(skip_serializing_if = "Option::is_none")]
    pub allowed_domains: Option<Vec<String>>,
}

impl From<ConfigWebSearchFilters> for ResponsesApiWebSearchFilters {
    fn from(filters: ConfigWebSearchFilters) -> Self {
        Self {
            allowed_domains: filters.allowed_domains,
        }
    }
}

#[derive(Debug, Clone, Serialize, PartialEq)]
pub struct ResponsesApiWebSearchUserLocation {
    #[serde(rename = "type")]
    pub r#type: WebSearchUserLocationType,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub country: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub region: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub city: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub timezone: Option<String>,
}

impl From<ConfigWebSearchUserLocation> for ResponsesApiWebSearchUserLocation {
    fn from(user_location: ConfigWebSearchUserLocation) -> Self {
        Self {
            r#type: user_location.r#type,
            country: user_location.country,
            region: user_location.region,
            city: user_location.city,
            timezone: user_location.timezone,
        }
    }
}

#[cfg(test)]
#[path = "tool_spec_tests.rs"]
mod tests;
