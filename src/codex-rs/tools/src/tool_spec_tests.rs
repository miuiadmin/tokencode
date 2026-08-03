use super::ResponsesApiNamespace;
use super::ResponsesApiWebSearchFilters;
use super::ResponsesApiWebSearchUserLocation;
use super::ToolSpec;
use crate::AdditionalProperties;
use crate::FreeformTool;
use crate::FreeformToolFormat;
use crate::JsonSchema;
use crate::ResponsesApiNamespaceTool;
use crate::ResponsesApiTool;
use crate::create_tools_json_for_responses_api;
use crate::build_progress_channel_schema;
use crate::flatten_namespaces_for_flat_wire;
use crate::render_tool_catalog;
use codex_protocol::config_types::WebSearchContextSize;
use codex_protocol::config_types::WebSearchFilters as ConfigWebSearchFilters;
use codex_protocol::config_types::WebSearchUserLocation as ConfigWebSearchUserLocation;
use codex_protocol::config_types::WebSearchUserLocationType;
use pretty_assertions::assert_eq;
use serde_json::json;
use std::collections::BTreeMap;

#[test]
fn tool_spec_name_covers_all_variants() {
    assert_eq!(
        ToolSpec::Function(ResponsesApiTool {
            name: "lookup_order".to_string(),
            description: "Look up an order".to_string(),
            strict: false,
            defer_loading: None,
            parameters: JsonSchema::object(
                BTreeMap::new(),
                /*required*/ None,
                /*additional_properties*/ None
            ),
            output_schema: None,
        })
        .name(),
        "lookup_order"
    );
    assert_eq!(
        ToolSpec::Namespace(ResponsesApiNamespace {
            name: "mcp__demo__".to_string(),
            description: "Demo tools".to_string(),
            tools: Vec::new(),
        })
        .name(),
        "mcp__demo__"
    );
    assert_eq!(
        ToolSpec::ToolSearch {
            execution: "sync".to_string(),
            description: "Search for tools".to_string(),
            parameters: JsonSchema::object(
                BTreeMap::new(),
                /*required*/ None,
                /*additional_properties*/ None
            ),
        }
        .name(),
        "tool_search"
    );
    assert_eq!(
        ToolSpec::ImageGeneration {
            output_format: "png".to_string(),
        }
        .name(),
        "image_generation"
    );
    assert_eq!(
        ToolSpec::WebSearch {
            external_web_access: Some(true),
            index_gated_web_access: None,
            filters: None,
            user_location: None,
            search_context_size: None,
            search_content_types: None,
        }
        .name(),
        "web_search"
    );
    assert_eq!(
        ToolSpec::Freeform(FreeformTool {
            name: "exec".to_string(),
            description: "Run a command".to_string(),
            format: FreeformToolFormat {
                r#type: "grammar".to_string(),
                syntax: "lark".to_string(),
                definition: "start: \"exec\"".to_string(),
            },
        })
        .name(),
        "exec"
    );
}

#[test]
fn web_search_config_converts_to_responses_api_types() {
    assert_eq!(
        ResponsesApiWebSearchFilters::from(ConfigWebSearchFilters {
            allowed_domains: Some(vec!["example.com".to_string()]),
        }),
        ResponsesApiWebSearchFilters {
            allowed_domains: Some(vec!["example.com".to_string()]),
        }
    );
    assert_eq!(
        ResponsesApiWebSearchUserLocation::from(ConfigWebSearchUserLocation {
            r#type: WebSearchUserLocationType::Approximate,
            country: Some("US".to_string()),
            region: Some("California".to_string()),
            city: Some("San Francisco".to_string()),
            timezone: Some("America/Los_Angeles".to_string()),
        }),
        ResponsesApiWebSearchUserLocation {
            r#type: WebSearchUserLocationType::Approximate,
            country: Some("US".to_string()),
            region: Some("California".to_string()),
            city: Some("San Francisco".to_string()),
            timezone: Some("America/Los_Angeles".to_string()),
        }
    );
}

#[test]
fn create_tools_json_for_responses_api_includes_top_level_name() {
    assert_eq!(
        create_tools_json_for_responses_api(&[ToolSpec::Function(ResponsesApiTool {
            name: "demo".to_string(),
            description: "A demo tool".to_string(),
            strict: false,
            defer_loading: None,
            parameters: JsonSchema::object(
                BTreeMap::from([("foo".to_string(), JsonSchema::string(/*description*/ None),)]),
                /*required*/ None,
                /*additional_properties*/ None
            ),
            output_schema: None,
        })])
        .expect("serialize tools"),
        vec![json!({
            "type": "function",
            "name": "demo",
            "description": "A demo tool",
            "strict": false,
            "parameters": {
                "type": "object",
                "properties": {
                    "foo": { "type": "string" }
                },
            },
        })]
    );
}

#[test]
fn namespace_tool_spec_serializes_expected_wire_shape() {
    assert_eq!(
        serde_json::to_value(ToolSpec::Namespace(ResponsesApiNamespace {
            name: "mcp__demo__".to_string(),
            description: "Demo tools".to_string(),
            tools: vec![ResponsesApiNamespaceTool::Function(ResponsesApiTool {
                name: "lookup_order".to_string(),
                description: "Look up an order".to_string(),
                strict: false,
                defer_loading: None,
                parameters: JsonSchema::object(
                    BTreeMap::from([(
                        "order_id".to_string(),
                        JsonSchema::string(/*description*/ None),
                    )]),
                    /*required*/ None,
                    /*additional_properties*/ None,
                ),
                output_schema: None,
            })],
        }))
        .expect("serialize namespace tool"),
        json!({
            "type": "namespace",
            "name": "mcp__demo__",
            "description": "Demo tools",
            "tools": [
                {
                    "type": "function",
                    "name": "lookup_order",
                    "description": "Look up an order",
                    "strict": false,
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "order_id": { "type": "string" },
                        },
                    },
                },
            ],
        })
    );
}

#[test]
fn web_search_tool_spec_serializes_expected_wire_shape() {
    assert_eq!(
        serde_json::to_value(ToolSpec::WebSearch {
            external_web_access: Some(true),
            index_gated_web_access: None,
            filters: Some(ResponsesApiWebSearchFilters {
                allowed_domains: Some(vec!["example.com".to_string()]),
            }),
            user_location: Some(ResponsesApiWebSearchUserLocation {
                r#type: WebSearchUserLocationType::Approximate,
                country: Some("US".to_string()),
                region: Some("California".to_string()),
                city: Some("San Francisco".to_string()),
                timezone: Some("America/Los_Angeles".to_string()),
            }),
            search_context_size: Some(WebSearchContextSize::High),
            search_content_types: Some(vec!["text".to_string(), "image".to_string()]),
        })
        .expect("serialize web_search"),
        json!({
            "type": "web_search",
            "external_web_access": true,
            "filters": {
                "allowed_domains": ["example.com"],
            },
            "user_location": {
                "type": "approximate",
                "country": "US",
                "region": "California",
                "city": "San Francisco",
                "timezone": "America/Los_Angeles",
            },
            "search_context_size": "high",
            "search_content_types": ["text", "image"],
        })
    );
}

#[test]
fn tool_search_tool_spec_serializes_expected_wire_shape() {
    assert_eq!(
        serde_json::to_value(ToolSpec::ToolSearch {
            execution: "sync".to_string(),
            description: "Search app tools".to_string(),
            parameters: JsonSchema::object(
                BTreeMap::from([(
                    "query".to_string(),
                    JsonSchema::string(Some("Tool search query".to_string()),),
                )]),
                Some(vec!["query".to_string()]),
                Some(AdditionalProperties::Boolean(false))
            ),
        })
        .expect("serialize tool_search"),
        json!({
            "type": "tool_search",
            "execution": "sync",
            "description": "Search app tools",
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {
                        "type": "string",
                        "description": "Tool search query",
                    }
                },
                "required": ["query"],
                "additionalProperties": false,
            },
        })
    );
}

#[test]
fn flatten_namespaces_emits_flat_function_per_inner_tool() {
    // namespace 内两个 function + 一个独立 function + 一个非 function 变体（ToolSearch）。
    let specs = vec![
        ToolSpec::Namespace(ResponsesApiNamespace {
            name: "context7".to_string(),
            description: "Docs tools".to_string(),
            tools: vec![
                ResponsesApiNamespaceTool::Function(ResponsesApiTool {
                    name: "get-docs".to_string(),
                    description: "Get docs".to_string(),
                    strict: false,
                    defer_loading: None,
                    parameters: JsonSchema::object(
                        BTreeMap::new(),
                        /*required*/ None,
                        /*additional_properties*/ None,
                    ),
                    output_schema: None,
                }),
                ResponsesApiNamespaceTool::Function(ResponsesApiTool {
                    name: "search".to_string(),
                    description: "Search docs".to_string(),
                    strict: false,
                    defer_loading: None,
                    parameters: JsonSchema::object(
                        BTreeMap::new(),
                        /*required*/ None,
                        /*additional_properties*/ None,
                    ),
                    output_schema: None,
                }),
            ],
        }),
        ToolSpec::Function(ResponsesApiTool {
            name: "apply_patch".to_string(),
            description: "Apply a patch".to_string(),
            strict: false,
            defer_loading: None,
            parameters: JsonSchema::object(
                BTreeMap::new(),
                /*required*/ None,
                /*additional_properties*/ None,
            ),
            output_schema: None,
        }),
    ];

    let flattened = flatten_namespaces_for_flat_wire(&specs);

    // namespace 展平成两个 flat function，名字按 `{ns}__{name}` 拼接；独立 function 原样保留。
    assert_eq!(flattened.len(), 3);
    assert_eq!(flattened[0].name(), "context7__get-docs");
    assert_eq!(flattened[1].name(), "context7__search");
    assert_eq!(flattened[2].name(), "apply_patch");

    // 展平出的 flat 名能被 ToolName 无损还原（派发契约）。
    assert_eq!(
        codex_protocol::ToolName::from_flat_wire_name(flattened[0].name()),
        codex_protocol::ToolName::namespaced("context7", "get-docs")
    );
}

#[test]
fn progress_channel_schema_shape() {
    let names = vec!["shell".to_string(), "apply_patch".to_string()];
    let schema = build_progress_channel_schema(&names);
    // 顶层是 object + additionalProperties:false
    assert_eq!(schema["type"], "object");
    assert_eq!(schema["additionalProperties"], false);
    // required 三键
    let required = schema["required"].as_array().unwrap();
    assert!(required.contains(&json!("action")));
    assert!(required.contains(&json!("progress_patch")));
    assert!(required.contains(&json!("digest_override")));
    // action.tool 是 enum（工具名列表）
    let tool_enum = schema["properties"]["action"]["properties"]["tool"]["enum"]
        .as_array()
        .unwrap();
    assert!(tool_enum.contains(&json!("shell")));
    assert!(tool_enum.contains(&json!("apply_patch")));
    // arguments 自由 object，无 additionalProperties:false（strict=false 允许）
    let arguments = &schema["properties"]["action"]["properties"]["arguments"];
    assert_eq!(arguments["type"], "object");
    assert!(arguments.get("additionalProperties").is_none());
    // action nullable（type 数组含 object + null）
    let action_type = schema["properties"]["action"]["type"].as_array().unwrap();
    assert!(action_type.contains(&json!("object")));
    assert!(action_type.contains(&json!("null")));
    // digest_override 空 schema（any：null 或完整对象）
    assert_eq!(schema["properties"]["digest_override"].as_object().unwrap().len(), 0);
}

#[test]
fn render_tool_catalog_contains_tool_names() {
    let tools = vec![
        ToolSpec::Function(ResponsesApiTool {
            name: "lookup_order".to_string(),
            description: "Look up an order".to_string(),
            strict: false,
            defer_loading: None,
            parameters: JsonSchema::object(BTreeMap::new(), None, None),
            output_schema: None,
        }),
        ToolSpec::Freeform(FreeformTool {
            name: "apply_patch".to_string(),
            description: "Edit files".to_string(),
            format: FreeformToolFormat {
                r#type: "grammar".to_string(),
                syntax: "lark".to_string(),
                definition: "".to_string(),
            },
        }),
    ];
    let catalog = render_tool_catalog(&tools);
    assert!(catalog.contains("可用工具目录"));
    assert!(catalog.contains("lookup_order"));
    assert!(catalog.contains("apply_patch"));
    assert!(catalog.contains("自由格式工具"));
}
