use std::collections::HashMap;
use std::pin::Pin;
use std::sync::Arc;

use base64::Engine;
use base64::engine::general_purpose::STANDARD as BASE64_STANDARD;
use codex_extension_api::ExtensionData;
use codex_protocol::config_types::ModeKind;
use codex_protocol::items::ImageGenerationItem;
use codex_protocol::items::TurnItem;
use codex_utils_stream_parser::strip_citations;
use tokio_util::sync::CancellationToken;

use crate::context::ContextualUserFragment;
use crate::context::ImageGenerationInstructions;
use crate::function_tool::FunctionCallError;
use crate::parse_turn_item;
use crate::session::session::Session;
use crate::session::turn_context::TurnContext;
use crate::tools::parallel::ToolCallRuntime;
use crate::tools::router::{ToolCall, ToolRouter};
use codex_memories_read::citations::parse_memory_citation;
use codex_memories_read::citations::thread_ids_from_memory_citation;
use codex_protocol::error::CodexErr;
use codex_protocol::error::Result;
use codex_protocol::memory_citation::MemoryCitation;
use codex_protocol::models::FunctionCallOutputBody;
use codex_protocol::models::FunctionCallOutputPayload;
use codex_protocol::models::MessagePhase;
use codex_protocol::models::ResponseInputItem;
use codex_protocol::models::ResponseItem;
use codex_protocol::models::ContentItem;
use codex_protocol::protocol::{DigestEntry, DigestKind, TurnRef};
use codex_protocol::ToolName;
use codex_tools::flatten_namespaces_for_flat_wire;
use codex_tools::{ToolPayload, ToolSpec};
use serde_json::Value;
use codex_rollout::state_db;
use codex_utils_absolute_path::AbsolutePathBuf;
use codex_utils_stream_parser::strip_proposed_plan_blocks;
use futures::Future;
use tracing::debug;
use tracing::instrument;
use tracing::warn;

const GENERATED_IMAGE_ARTIFACTS_DIR: &str = "generated_images";

/// Returns the host-owned default artifact path for a generated image.
pub fn image_generation_artifact_path(
    codex_home: &AbsolutePathBuf,
    session_id: &str,
    call_id: &str,
) -> AbsolutePathBuf {
    let sanitize = |value: &str| {
        let mut sanitized: String = value
            .chars()
            .map(|ch| {
                if ch.is_ascii_alphanumeric() || ch == '-' || ch == '_' {
                    ch
                } else {
                    '_'
                }
            })
            .collect();
        if sanitized.is_empty() {
            sanitized = "generated_image".to_string();
        }
        sanitized
    };

    codex_home
        .join(GENERATED_IMAGE_ARTIFACTS_DIR)
        .join(sanitize(session_id))
        .join(format!("{}.png", sanitize(call_id)))
}

fn strip_hidden_assistant_markup(text: &str, plan_mode: bool) -> String {
    let (without_citations, _) = strip_citations(text);
    if plan_mode {
        strip_proposed_plan_blocks(&without_citations)
    } else {
        without_citations
    }
}

fn strip_hidden_assistant_markup_and_parse_memory_citation(
    text: &str,
    plan_mode: bool,
) -> (
    String,
    Option<codex_protocol::memory_citation::MemoryCitation>,
) {
    let (without_citations, citations) = strip_citations(text);
    let visible_text = if plan_mode {
        strip_proposed_plan_blocks(&without_citations)
    } else {
        without_citations
    };
    (visible_text, parse_memory_citation(citations))
}

pub(crate) fn raw_assistant_output_text_from_item(item: &ResponseItem) -> Option<String> {
    if let ResponseItem::Message { role, content, .. } = item
        && role == "assistant"
    {
        let combined = content
            .iter()
            .filter_map(|ci| match ci {
                codex_protocol::models::ContentItem::OutputText { text } => Some(text.as_str()),
                _ => None,
            })
            .collect::<String>();
        return Some(combined);
    }
    None
}

async fn save_image_generation_result(
    codex_home: &AbsolutePathBuf,
    session_id: &str,
    call_id: &str,
    result: &str,
) -> Result<AbsolutePathBuf> {
    let bytes = BASE64_STANDARD
        .decode(result.trim().as_bytes())
        .map_err(|err| {
            CodexErr::InvalidRequest(format!("invalid image generation payload: {err}"))
        })?;
    let path = image_generation_artifact_path(codex_home, session_id, call_id);
    if let Some(parent) = path.parent() {
        tokio::fs::create_dir_all(parent).await?;
    }
    tokio::fs::write(&path, bytes).await?;
    Ok(path)
}

pub(crate) async fn persist_image_generation_item(
    sess: &Session,
    turn_context: &TurnContext,
    image_item: &mut ImageGenerationItem,
) -> Option<AbsolutePathBuf> {
    image_item.saved_path = None;
    let session_id = sess.thread_id.to_string();
    match save_image_generation_result(
        &turn_context.config.codex_home,
        &session_id,
        &image_item.id,
        &image_item.result,
    )
    .await
    {
        Ok(path) => {
            image_item.saved_path = Some(path.clone());
            Some(path)
        }
        Err(err) => {
            let output_path = image_generation_artifact_path(
                &turn_context.config.codex_home,
                &session_id,
                &image_item.id,
            );
            let output_dir = output_path
                .parent()
                .unwrap_or_else(|| turn_context.config.codex_home.clone());
            tracing::warn!(
                call_id = %image_item.id,
                output_dir = %output_dir.display(),
                "failed to save generated image: {err}"
            );
            None
        }
    }
}

async fn record_image_generation_instructions(
    sess: &Session,
    turn_context: &TurnContext,
    image_item: &ImageGenerationItem,
) {
    if image_item.saved_path.is_none() {
        return;
    }
    let session_id = sess.thread_id.to_string();
    let image_output_path =
        image_generation_artifact_path(&turn_context.config.codex_home, &session_id, "<image_id>");
    let image_output_dir = image_output_path
        .parent()
        .unwrap_or_else(|| turn_context.config.codex_home.clone());
    let message: ResponseItem = ContextualUserFragment::into(ImageGenerationInstructions::new(
        image_output_dir.display(),
        image_output_path.display(),
    ));
    sess.record_conversation_items(turn_context, &[message])
        .await;
}

/// Persist a completed model response item and record any cited memory usage.
pub(crate) async fn record_completed_response_item(
    sess: &Session,
    turn_context: &TurnContext,
    item: &ResponseItem,
) {
    record_completed_response_item_with_finalized_facts(
        sess,
        turn_context,
        item,
        /*finalized_facts*/ None,
    )
    .await;
}

pub(crate) async fn record_completed_response_item_with_finalized_facts(
    sess: &Session,
    turn_context: &TurnContext,
    item: &ResponseItem,
    finalized_facts: Option<&FinalizedTurnItemFacts>,
) {
    sess.record_conversation_items(turn_context, std::slice::from_ref(item))
        .await;
    let defers_mailbox_delivery = finalized_facts.map_or_else(
        || {
            completed_item_defers_mailbox_delivery_to_next_turn(
                item,
                turn_context.collaboration_mode.mode == ModeKind::Plan,
            )
        },
        |facts| facts.defers_mailbox_delivery_to_next_turn,
    );
    if defers_mailbox_delivery {
        sess.input_queue
            .defer_mailbox_delivery_to_next_turn(&sess.active_turn, &turn_context.sub_id)
            .await;
    }
    mark_thread_memory_mode_polluted_if_external_context(sess, turn_context, item).await;
    let has_memory_citation = if let Some(memory_citation) =
        finalized_facts.and_then(|facts| facts.memory_citation.as_ref())
    {
        record_stage1_output_usage_for_memory_citation(
            sess.services.state_db.as_ref(),
            memory_citation,
        )
        .await
    } else {
        record_stage1_output_usage_and_detect_memory_citation(sess.services.state_db.as_ref(), item)
            .await
    };
    if has_memory_citation {
        sess.record_memory_citation_for_turn(&turn_context.sub_id)
            .await;
    }
}

fn response_item_may_include_external_context(item: &ResponseItem) -> bool {
    matches!(
        item,
        ResponseItem::ToolSearchCall { .. }
            | ResponseItem::ToolSearchOutput { .. }
            | ResponseItem::WebSearchCall { .. }
    )
}

pub(crate) async fn mark_thread_memory_mode_polluted_if_external_context(
    sess: &Session,
    turn_context: &TurnContext,
    item: &ResponseItem,
) {
    if !turn_context.config.memories.disable_on_external_context
        || !response_item_may_include_external_context(item)
    {
        return;
    }
    state_db::mark_thread_memory_mode_polluted(
        sess.services.state_db.as_deref(),
        sess.thread_id,
        "record_completed_response_item",
    )
    .await;
}

async fn record_stage1_output_usage_and_detect_memory_citation(
    state_db_ctx: Option<&state_db::StateDbHandle>,
    item: &ResponseItem,
) -> bool {
    let Some(raw_text) = raw_assistant_output_text_from_item(item) else {
        return false;
    };

    let (_, citations) = strip_citations(&raw_text);
    let Some(memory_citation) = parse_memory_citation(citations) else {
        return false;
    };
    record_stage1_output_usage_for_memory_citation(state_db_ctx, &memory_citation).await
}

async fn record_stage1_output_usage_for_memory_citation(
    state_db_ctx: Option<&state_db::StateDbHandle>,
    memory_citation: &MemoryCitation,
) -> bool {
    let thread_ids = thread_ids_from_memory_citation(memory_citation);
    if thread_ids.is_empty() {
        return true;
    }

    if let Some(db) = state_db_ctx {
        let _ = db.memories().record_stage1_output_usage(&thread_ids).await;
    }
    true
}

/// Handle a completed output item from the model stream, recording it and
/// queuing any tool execution futures. This records items immediately so
/// history and rollout stay in sync even if the turn is later cancelled.
pub(crate) type InFlightFuture<'f> =
    Pin<Box<dyn Future<Output = Result<ResponseInputItem>> + Send + 'f>>;

#[derive(Default)]
pub(crate) struct OutputItemResult {
    pub last_agent_message: Option<String>,
    pub needs_follow_up: bool,
    pub tool_future: Option<InFlightFuture<'static>>,
}

pub(crate) struct HandleOutputCtx {
    pub sess: Arc<Session>,
    pub turn_context: Arc<TurnContext>,
    pub turn_store: Arc<ExtensionData>,
    pub tool_runtime: ToolCallRuntime,
    pub cancellation_token: CancellationToken,
}

pub(crate) async fn apply_turn_item_contributors(
    sess: &Session,
    turn_store: &ExtensionData,
    item: &mut TurnItem,
) {
    let contributors = sess.services.extensions.turn_item_contributors().to_vec();
    for contributor in contributors {
        if let Err(err) = contributor
            .contribute(&sess.services.thread_extension_data, turn_store, item)
            .await
        {
            warn!("turn item contributor failed: {err}");
        }
    }
}

pub(crate) enum TurnItemContributorPolicy<'a> {
    Skip,
    Run(&'a ExtensionData),
}

pub(crate) struct FinalizedTurnItem {
    pub(crate) turn_item: TurnItem,
    pub(crate) facts: FinalizedTurnItemFacts,
}

#[derive(Clone, Default)]
pub(crate) struct FinalizedTurnItemFacts {
    pub(crate) memory_citation: Option<MemoryCitation>,
    pub(crate) last_agent_message: Option<String>,
    pub(crate) defers_mailbox_delivery_to_next_turn: bool,
}

pub(crate) async fn finalize_non_tool_response_item(
    sess: &Session,
    turn_context: &TurnContext,
    contributor_policy: TurnItemContributorPolicy<'_>,
    item: &ResponseItem,
    plan_mode: bool,
) -> Option<FinalizedTurnItem> {
    let turn_item =
        handle_non_tool_response_item(sess, turn_context, contributor_policy, item, plan_mode)
            .await?;
    let (memory_citation, last_agent_message, defers_mailbox_delivery_to_next_turn) =
        match &turn_item {
            TurnItem::AgentMessage(agent_message) => {
                let combined = agent_message
                    .content
                    .iter()
                    .map(|entry| match entry {
                        codex_protocol::items::AgentMessageContent::Text { text } => text.as_str(),
                    })
                    .collect::<String>();
                let last_agent_message = if combined.trim().is_empty() {
                    None
                } else {
                    Some(combined)
                };
                let defers_mailbox_delivery_to_next_turn =
                    !matches!(agent_message.phase, Some(MessagePhase::Commentary))
                        && last_agent_message.is_some();
                (
                    agent_message.memory_citation.clone(),
                    last_agent_message,
                    defers_mailbox_delivery_to_next_turn,
                )
            }
            TurnItem::ImageGeneration(_) => (None, None, true),
            _ => (None, None, false),
        };
    Some(FinalizedTurnItem {
        turn_item,
        facts: FinalizedTurnItemFacts {
            memory_citation,
            last_agent_message,
            defers_mailbox_delivery_to_next_turn,
        },
    })
}

// ===== 进度副信道 P3b：响应侧消费接管（替代 P3a decorator）=====
//
// decorator 退场后，turn loop 直接消费模型按统一 schema 输出的 envelope
// `{action, progress_patch, digest_override}`：一次性三路分流，避免先 rewrite 丢 patch。
// 合成物与原生 FunctionCall / CustomToolCall 字节级等价，故 turn 循环与工具执行子系统零改动。

/// 构建「工具 flat 名 → 是否自由格式（Freeform）」映射，供合成 FC 时按工具 kind 分流
/// （Freeform → CustomToolCall，否则 → FunctionCall）。平移自原 client.rs（decorator 退场）。
fn build_tool_kind_map(tools: &[ToolSpec]) -> HashMap<String, bool> {
    flatten_namespaces_for_flat_wire(tools)
        .into_iter()
        .map(|spec| {
            (
                spec.name().to_string(),
                matches!(spec, ToolSpec::Freeform(_)),
            )
        })
        .collect()
}

/// 解析 envelope 的 `action` 字段，按工具 kind 合成原生 FunctionCall / CustomToolCall。
/// 返回 `(合成项, tool flat 名, arguments)`：tool 名 + arguments 供 files_touched 推导（§12.7 step 4）。
/// `action` 缺 tool / 未知工具 → None（调用侧透传原 Message，不执行）。
fn synthesize_call_from_action(
    action: &Value,
    kinds: &HashMap<String, bool>,
) -> Option<(ResponseItem, String, Value)> {
    let tool = action.get("tool").and_then(|t| t.as_str())?;
    let is_freeform = *kinds.get(tool)?;
    let tool = tool.to_string();
    let arguments = action.get("arguments").cloned().unwrap_or(Value::Null);
    let ToolName { name, namespace } = ToolName::from_flat_wire_name(&tool);
    let call_id = format!("{tool}-{}", uuid::Uuid::new_v4());
    let item = if is_freeform {
        // apply_patch 等 Freeform 工具的 input 是 lark 文本而非 JSON：约定模型把 raw 文本塞进
        // arguments["input"]，系统据此提取；模型乱填时退化为整段 JSON 字符串。
        let input = arguments
            .get("input")
            .and_then(|v| v.as_str())
            .map(|s| s.to_string())
            .unwrap_or_else(|| arguments.to_string());
        ResponseItem::CustomToolCall {
            id: None,
            status: Some("completed".to_string()),
            call_id,
            name,
            namespace,
            input,
            internal_chat_message_metadata_passthrough: None,
        }
    } else {
        ResponseItem::FunctionCall {
            id: None,
            name,
            namespace,
            arguments: serde_json::to_string(&arguments).unwrap_or_default(),
            call_id,
            internal_chat_message_metadata_passthrough: None,
        }
    };
    Some((item, tool, arguments))
}

/// 构建「工具 flat 名 → ToolSpec」映射，供执行前二次校验按 spec 分流（§12.8）。
fn build_tool_spec_map(tools: &[ToolSpec]) -> HashMap<String, ToolSpec> {
    flatten_namespaces_for_flat_wire(tools)
        .into_iter()
        .map(|spec| (spec.name().to_string(), spec))
        .collect()
}

/// 执行前二次校验（P3b §12.8，strict=false 下唯一参数约束层 L7）。
///
/// progress_channel 下对合成的 FC 在 `build_tool_call`→`handle_tool_call` 派发前校验：
/// - **Function 工具**：`parameters.required` 字段齐备性（v1 轻量；type/enum 留后续）；
/// - **Freeform apply_patch**：顺 lark 语法解析（`parse_patch`），失败回流；
/// - **Freeform code_mode exec 等**：input 非空。
/// 失败返回回流报错 `Some(msg)`；通过返回 `None`。语义/危险模式（越界路径 / 越权命令）由
/// handler 内 `assess_patch_safety` / `exec_policy` 正交兜底（更外层），本层不重复。
fn validate_progress_channel_tool_call(
    specs: &HashMap<String, ToolSpec>,
    call: &ToolCall,
) -> Option<String> {
    let flat = call.tool_name.to_flat_wire_name();
    let spec = specs.get(&flat)?;
    match spec {
        ToolSpec::Function(tool) => {
            let required = tool.parameters.required.as_ref()?;
            let args: Value = match &call.payload {
                ToolPayload::Function { arguments } => {
                    serde_json::from_str(arguments).unwrap_or(Value::Null)
                }
                _ => return None,
            };
            for req in required {
                if args.get(req.as_str()).is_none() {
                    return Some(format!("进度副信道校验：工具 {flat} 缺必填字段「{req}」"));
                }
            }
            None
        }
        ToolSpec::Freeform(_) => {
            let input = match &call.payload {
                ToolPayload::Custom { input } => input.as_str(),
                _ => return None,
            };
            if flat == "apply_patch" {
                if let Err(err) = codex_apply_patch::parse_patch(input) {
                    return Some(format!("进度副信道校验：apply_patch 语法解析失败：{err}"));
                }
            } else if input.trim().is_empty() {
                return Some(format!("进度副信道校验：工具 {flat} 的 input 为空"));
            }
            None
        }
        // Namespace/ToolSearch/ImageGeneration/WebSearch：非合成主流路径，v1 不校验。
        _ => None,
    }
}

/// 进度副信道响应侧消费（§12.3 step 4 三路分流）。
///
/// 仅处理 assistant `Message{OutputText}` 且文本可解析为 JSON envelope 的项；其余原样返回。
/// 一次性解析 `{action, progress_patch, digest_override}`：
/// - `progress_patch` → `apply_progress_patch`（写 ProgressState + warnings 回流）；
/// - `digest_override` → `consume_digest_override`（延迟一步覆盖上一 step 草稿）；
/// - `action ≠ null` → 合成原生 FC + `derive_files_touched_from_args`，返回 FC（进 build_tool_call 派发）；
/// - `action == null`（对话轮 S2）→ 生成 kind=dialog 草稿落 L0，返回原 Message（→ Ok(None) 显示）。
///
/// 透传条件（与 P3a fallback 同行为）：非 Message / 非 JSON / action 缺 tool 或未知工具 → 原样。
async fn consume_progress_channel_output(
    ctx: &HandleOutputCtx,
    item: ResponseItem,
) -> ResponseItem {
    let text = match &item {
        ResponseItem::Message { role, content, .. } if role == "assistant" => match content.as_slice() {
            [ContentItem::OutputText { text }] => text.as_str(),
            _ => return item,
        },
        _ => return item,
    };
    let parsed = match serde_json::from_str::<Value>(text) {
        Ok(v) => v,
        Err(_) => return item, // 非 JSON → 透传（可见 JSON 文本，不崩）。
    };

    let turn_ref = TurnRef {
        turn_id: ctx.turn_context.sub_id.clone(),
        item_index: None,
    };

    // (b) progress_patch：object 才应用（null / 缺失跳过）。
    if let Some(patch) = parsed.get("progress_patch").filter(|v| v.is_object()) {
        ctx.sess
            .apply_progress_patch(patch.clone(), turn_ref.clone())
            .await;
    }
    // (c) digest_override：非 null 才消费（延迟一步覆盖，多数轮省略）。
    if let Some(override_val) = parsed.get("digest_override").filter(|v| !v.is_null()) {
        ctx.sess
            .consume_digest_override(override_val.clone(), turn_ref.clone())
            .await;
    }

    // (a)/(a') action 三态。
    match parsed.get("action").filter(|v| !v.is_null()) {
        Some(action_val) => {
            // action ≠ null → 合成 FC + 推导 files_touched。
            let kinds = build_tool_kind_map(&ctx.tool_runtime.model_visible_specs());
            match synthesize_call_from_action(action_val, &kinds) {
                Some((call, tool, args)) => {
                    let _confidence = ctx
                        .sess
                        .derive_files_touched_from_args(&tool, &args)
                        .await;
                    call
                }
                None => item, // 未知工具 → 透传（可见，不执行）。
            }
        }
        None => {
            // (a') action == null → 对话轮：生成 kind=dialog 草稿落 L0（§12.5）。
            // 摘要优先取 progress_patch.next_step / intent；缺则截断 envelope 文本兜底。
            let summary = parsed
                .get("progress_patch")
                .and_then(|p| p.get("next_step"))
                .and_then(|v| v.as_str())
                .or_else(|| {
                    parsed
                        .get("progress_patch")
                        .and_then(|p| p.get("intent"))
                        .and_then(|v| v.as_str())
                })
                .map(|s| s.to_string())
                .unwrap_or_else(|| text.chars().take(300).collect::<String>());
            ctx.sess
                .append_digest_to_l0(
                    DigestEntry {
                        kind: DigestKind::Dialog,
                        tool: None,
                        brief_result: None,
                        dialog_summary: Some(summary),
                        turn_ref: turn_ref.clone(),
                        sequence: 0, // append_digest_to_l0 回填
                        confidence: None,
                    },
                    Vec::new(),
                )
                .await;
            item // 原样继续（→ Ok(None)，作为 message 显示）
        }
    }
}

#[instrument(level = "trace", skip_all)]
pub(crate) async fn handle_output_item_done(
    ctx: &mut HandleOutputCtx,
    item: ResponseItem,
    previously_active_item: Option<TurnItem>,
) -> Result<OutputItemResult> {
    let mut output = OutputItemResult::default();
    let plan_mode = ctx.turn_context.collaboration_mode.mode == ModeKind::Plan;

    // 进度副信道（P3b §12.3 step 4）：decorator 退场后，turn loop 直接消费 envelope
    // {action, progress_patch, digest_override}。flag 关时原样透传，行为零变化。
    let item = if ctx.turn_context.progress_channel_enabled() {
        consume_progress_channel_output(ctx, item).await
    } else {
        item
    };

    match ToolRouter::build_tool_call(item.clone()) {
        // The model emitted a tool call; log it, persist the item immediately, and queue the tool execution.
        Ok(Some(call)) => {
            ctx.sess
                .input_queue
                .accept_mailbox_delivery_for_current_turn(
                    &ctx.sess.active_turn,
                    &ctx.turn_context.sub_id,
                )
                .await;

            let payload_preview = call.payload.log_payload().into_owned();
            tracing::info!(
                thread_id = %ctx.sess.thread_id,
                "ToolCall: {} {}",
                call.tool_name,
                payload_preview
            );

            record_completed_response_item(ctx.sess.as_ref(), ctx.turn_context.as_ref(), &item)
                .await;

            // 进度副信道（P3b §12.8）：执行前二次校验（strict=false 下唯一参数约束层 L7）。
            // 失败 → 合成 FCO（用合成 FC 的 call_id，S7）回模型自纠，不执行。
            if ctx.turn_context.progress_channel_enabled() {
                let specs = build_tool_spec_map(&ctx.tool_runtime.model_visible_specs());
                if let Some(err_msg) = validate_progress_channel_tool_call(&specs, &call) {
                    let response = ResponseInputItem::FunctionCallOutput {
                        call_id: call.call_id.clone(),
                        output: FunctionCallOutputPayload {
                            body: FunctionCallOutputBody::Text(err_msg),
                            success: Some(false),
                        },
                    };
                    if let Some(response_item) = response_input_to_response_item(&response) {
                        ctx.sess
                            .record_conversation_items(
                                &ctx.turn_context,
                                std::slice::from_ref(&response_item),
                            )
                            .await;
                        // 校验失败也走 FCO 回流（记 last_error + 草稿），与副路径一致。
                        ctx.sess
                            .progress_channel_fco_reflow_failure(
                                ctx.turn_context.as_ref(),
                                &response_item,
                            )
                            .await;
                    }
                    output.needs_follow_up = true;
                    return Ok(output);
                }
            }

            let cancellation_token = ctx.cancellation_token.child_token();
            let tool_future: InFlightFuture<'static> = Box::pin(
                ctx.tool_runtime
                    .clone()
                    .handle_tool_call(call, cancellation_token),
            );

            output.needs_follow_up = true;
            output.tool_future = Some(tool_future);
        }
        // No tool call: convert messages/reasoning into turn items and mark them as complete.
        Ok(None) => {
            let finalized_turn_item = finalize_non_tool_response_item(
                ctx.sess.as_ref(),
                ctx.turn_context.as_ref(),
                TurnItemContributorPolicy::Run(ctx.turn_store.as_ref()),
                &item,
                plan_mode,
            )
            .await;
            let finalized_facts = finalized_turn_item
                .as_ref()
                .map(|finalized| finalized.facts.clone());
            if let Some(finalized_turn_item) = finalized_turn_item {
                if previously_active_item.is_none() {
                    let mut started_item = finalized_turn_item.turn_item.clone();
                    if let TurnItem::ImageGeneration(item) = &mut started_item {
                        item.status = "in_progress".to_string();
                        item.revised_prompt = None;
                        item.result.clear();
                        item.saved_path = None;
                    }
                    ctx.sess
                        .emit_turn_item_started(&ctx.turn_context, &started_item)
                        .await;
                }

                ctx.sess
                    .emit_turn_item_completed(&ctx.turn_context, finalized_turn_item.turn_item)
                    .await;
            }
            record_completed_response_item_with_finalized_facts(
                ctx.sess.as_ref(),
                ctx.turn_context.as_ref(),
                &item,
                finalized_facts.as_ref(),
            )
            .await;

            output.last_agent_message = finalized_facts.and_then(|facts| facts.last_agent_message);
        }
        // The tool request should be answered directly (or was denied); push that response into the transcript.
        Err(FunctionCallError::RespondToModel(message)) => {
            let response = ResponseInputItem::FunctionCallOutput {
                call_id: String::new(),
                output: FunctionCallOutputPayload {
                    body: FunctionCallOutputBody::Text(message),
                    ..Default::default()
                },
            };
            record_completed_response_item(ctx.sess.as_ref(), ctx.turn_context.as_ref(), &item)
                .await;
            if let Some(response_item) = response_input_to_response_item(&response) {
                // 进度副信道（P3b §12.7 step 5 副路径）：RespondToModel = 结构化失败，
                // message 即 brief；必记 last_error + 草稿落 L0（call_id 空串，S7 副路径）。
                if ctx.turn_context.progress_channel_enabled() {
                    ctx.sess
                        .progress_channel_fco_reflow_failure(
                            ctx.turn_context.as_ref(),
                            &response_item,
                        )
                        .await;
                }
                ctx.sess
                    .record_conversation_items(
                        &ctx.turn_context,
                        std::slice::from_ref(&response_item),
                    )
                    .await;
            }

            output.needs_follow_up = true;
        }
        // A fatal error occurred; surface it back into history.
        Err(FunctionCallError::Fatal(message)) => {
            return Err(CodexErr::Fatal(message));
        }
    }

    Ok(output)
}

pub(crate) async fn handle_non_tool_response_item(
    sess: &Session,
    turn_context: &TurnContext,
    contributor_policy: TurnItemContributorPolicy<'_>,
    item: &ResponseItem,
    plan_mode: bool,
) -> Option<TurnItem> {
    debug!(?item, "Output item");

    match item {
        ResponseItem::Message { .. }
        | ResponseItem::Reasoning { .. }
        | ResponseItem::WebSearchCall { .. }
        | ResponseItem::ImageGenerationCall { .. } => {
            let mut turn_item = parse_turn_item(item)?;
            finalize_turn_item(
                sess,
                turn_context,
                contributor_policy,
                &mut turn_item,
                plan_mode,
            )
            .await;
            if let TurnItem::ImageGeneration(image_item) = &turn_item {
                record_image_generation_instructions(sess, turn_context, image_item).await;
            }
            Some(turn_item)
        }
        ResponseItem::FunctionCallOutput { .. }
        | ResponseItem::CustomToolCallOutput { .. }
        | ResponseItem::ToolSearchOutput { .. } => {
            debug!("unexpected tool output from stream");
            None
        }
        _ => None,
    }
}

pub(crate) async fn finalize_turn_item(
    sess: &Session,
    turn_context: &TurnContext,
    contributor_policy: TurnItemContributorPolicy<'_>,
    turn_item: &mut TurnItem,
    plan_mode: bool,
) {
    if let TurnItemContributorPolicy::Run(turn_store) = contributor_policy {
        apply_turn_item_contributors(sess, turn_store, turn_item).await;
    }
    if let TurnItem::AgentMessage(agent_message) = &mut *turn_item {
        let combined = agent_message
            .content
            .iter()
            .map(|entry| match entry {
                codex_protocol::items::AgentMessageContent::Text { text } => text.as_str(),
            })
            .collect::<String>();
        let (stripped, memory_citation) =
            strip_hidden_assistant_markup_and_parse_memory_citation(&combined, plan_mode);
        agent_message.content =
            vec![codex_protocol::items::AgentMessageContent::Text { text: stripped }];
        if agent_message.memory_citation.is_none() {
            agent_message.memory_citation = memory_citation;
        }
    }
    if let TurnItem::ImageGeneration(image_item) = &mut *turn_item
        && !image_item.result.is_empty()
    {
        persist_image_generation_item(sess, turn_context, image_item).await;
    }
}

pub(crate) fn last_assistant_message_from_item(
    item: &ResponseItem,
    plan_mode: bool,
) -> Option<String> {
    if let Some(combined) = raw_assistant_output_text_from_item(item) {
        if combined.is_empty() {
            return None;
        }
        let stripped = strip_hidden_assistant_markup(&combined, plan_mode);
        if stripped.trim().is_empty() {
            return None;
        }
        return Some(stripped);
    }
    None
}

fn completed_item_defers_mailbox_delivery_to_next_turn(
    item: &ResponseItem,
    plan_mode: bool,
) -> bool {
    match item {
        ResponseItem::Message { role, phase, .. } => {
            if role != "assistant" || matches!(phase, Some(MessagePhase::Commentary)) {
                return false;
            }
            // Treat `None` like final-answer text so untagged providers default
            // to the safer "defer mailbox mail" behavior.
            last_assistant_message_from_item(item, plan_mode).is_some()
        }
        ResponseItem::ImageGenerationCall { .. } => true,
        _ => false,
    }
}

pub(crate) fn response_input_to_response_item(input: &ResponseInputItem) -> Option<ResponseItem> {
    match input {
        ResponseInputItem::FunctionCallOutput { call_id, output } => {
            Some(ResponseItem::FunctionCallOutput {
                id: None,
                call_id: call_id.clone(),
                output: output.clone(),
                internal_chat_message_metadata_passthrough: None,
            })
        }
        ResponseInputItem::CustomToolCallOutput {
            call_id,
            name,
            output,
        } => Some(ResponseItem::CustomToolCallOutput {
            id: None,
            call_id: call_id.clone(),
            name: name.clone(),
            output: output.clone(),
            internal_chat_message_metadata_passthrough: None,
        }),
        ResponseInputItem::McpToolCallOutput { call_id, output } => {
            let output = output.as_function_call_output_payload();
            Some(ResponseItem::FunctionCallOutput {
                id: None,
                call_id: call_id.clone(),
                output,
                internal_chat_message_metadata_passthrough: None,
            })
        }
        ResponseInputItem::ToolSearchOutput {
            call_id,
            status,
            execution,
            tools,
        } => Some(ResponseItem::ToolSearchOutput {
            id: None,
            call_id: Some(call_id.clone()),
            status: status.clone(),
            execution: execution.clone(),
            tools: tools.clone(),
            internal_chat_message_metadata_passthrough: None,
        }),
        _ => None,
    }
}

#[cfg(test)]
#[path = "stream_events_utils_tests.rs"]
mod tests;
