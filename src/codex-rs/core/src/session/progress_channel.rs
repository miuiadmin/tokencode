//! 进度副信道的会话级消费方法(§12.3 turn 内状态机的 Session 封装,墙 B)。
//!
//! 这些 `pub(crate)` 方法封装 `ProgressState` 的读写,使 turn loop / handle_output_item_done
//! 等不在 `session::*` 子树内的调用方经 Session 方法间接操作(不裸 `state.lock`)。

use super::session::Session;
use super::turn_context::TurnContext;
use codex_protocol::models::{ContentItem, ResponseItem};
use codex_protocol::protocol::Confidence;
use codex_protocol::protocol::DigestEntry;
use codex_protocol::protocol::DigestKind;
use codex_protocol::protocol::IntentChange;
use codex_protocol::protocol::ProgressDigestItem;
use codex_protocol::protocol::RolloutItem;
use codex_protocol::protocol::TurnRef;
use serde_json::Value;

use crate::state::AssumptionStatus;
use crate::state::LastError;
use crate::state::ProgressState;
use crate::state::TodoStatus;

impl Session {
    /// 应用 `progress_patch` 到会话级 `ProgressState`(§12.4)。
    ///
    /// 不依赖 WorldState snapshot/diff(M9),直接增量更新内存态。校验失败(intent 演进、
    /// todo_ops、assumptions 的 op/id/字段合法性)写 `warnings`,下轮注入回流(§12.4 M7),
    /// 不中断。intent 演进旧值 append L0(防「演进」滑成「漂移」,§10.2 #9)。
    pub(crate) async fn apply_progress_patch(&self, patch: Value, turn_ref: TurnRef) {
        let mut to_persist: Vec<RolloutItem> = Vec::new();
        {
            let mut state = self.state.lock().await;
            let progress = &mut state.progress_state;

            // intent 演进:旧值 append L0(与 digest 同 carrier,统一不进 history)。
            if let Some(new_intent) = patch.get("intent").and_then(|v| v.as_str()) {
                let new_intent = new_intent.to_string();
                if let Some(old) = progress.intent.clone()
                    && old != new_intent
                {
                    let seq = progress.digest_seq;
                    progress.digest_seq += 1;
                    to_persist.push(RolloutItem::ProgressDigest(ProgressDigestItem {
                        entry: DigestEntry {
                            kind: DigestKind::Dialog,
                            tool: None,
                            brief_result: None,
                            dialog_summary: Some(format!("intent 演进旧值: {old}")),
                            turn_ref: turn_ref.clone(),
                            sequence: seq,
                            confidence: None,
                        },
                        intent_history: vec![IntentChange {
                            old,
                            turn_ref: turn_ref.clone(),
                        }],
                    }));
                }
                progress.intent = Some(new_intent);
            }

            // todo_ops:按数组顺序应用;非法 op/id 跳过 + warnings,不中断(§12.4 S9)。
            if let Some(ops) = patch.get("todo_ops").and_then(|v| v.as_array()) {
                for op_val in ops {
                    apply_todo_op(progress, op_val);
                }
            }

            // assumptions:每条 statement 非空、status 合法;非法条目 warnings 跳过(§12.4 M2)。
            if let Some(assumptions) = patch.get("assumptions").and_then(|v| v.as_array()) {
                for a_val in assumptions {
                    apply_assumption(progress, a_val);
                }
            }

            // next_step
            if let Some(next) = patch.get("next_step").and_then(|v| v.as_str()) {
                progress.next_step = Some(next.to_string());
            }
        }
        // 落 L0(intent 演进旧值)。锁已释放,避免持锁 persist。
        if !to_persist.is_empty() {
            self.persist_rollout_items(&to_persist).await;
        }
    }

    /// step 4 推导 files_touched(§12.7,B 类系统记账不决策)。
    ///
    /// 合成 FC 时 arguments 在手,直接从 apply_patch/shell 参数推导,免 call_id 配对(S6)。
    /// 返回 `Confidence`:apply_patch 命中路径 = High;shell/code_mode 静态难准 = Low(不写,
    /// backtrack 是 P4 回溯工具的事,files_touched 是「提示非真值」非「真值」,S3)。
    pub(crate) async fn derive_files_touched_from_args(
        &self,
        tool: &str,
        args: &Value,
    ) -> Confidence {
        let paths: Vec<String> = match tool {
            // apply_patch:解析 lark input 的 *** Add/Update/Delete File marker。
            "apply_patch" => {
                let Some(input) = args.get("input").and_then(|v| v.as_str()) else {
                    return Confidence::High;
                };
                parse_apply_patch_paths(input)
            }
            // shell:命令路径静态难解析(cd/重定向/管道),v1 不推导,标 Low。
            "shell" => return Confidence::Low,
            // code_mode exec 等 Freeform:改的文件难从 input 静态推导,v1 不推导(M11),标 Low。
            _ => return Confidence::Low,
        };
        if !paths.is_empty() {
            let mut state = self.state.lock().await;
            for p in &paths {
                state.progress_state.files_touched.insert(p.clone());
            }
            Confidence::High
        } else {
            Confidence::Low
        }
    }

    /// step 5 推导 last_error(§12.7,B 类)。失败判断 + brief 提取在调用侧(Layer 4/5
    /// 看 FCO output / 副路径 `Err(RespondToModel)`),本方法只负责截断 + 写 ProgressState。
    pub(crate) async fn record_last_error_from_fco(&self, brief: String, turn_ref: TurnRef) {
        let brief = truncate_brief(&brief);
        let mut state = self.state.lock().await;
        state.progress_state.last_error = Some(LastError { turn_ref, brief });
    }

    /// 消费 `digest_override`(§12.5,模型辅)。延迟一步语义:step N 的 override 覆盖
    /// step N-1 已落 L0 的草稿。**v1 缺口**:覆盖已落盘 `ProgressDigest` 需读改写 rollout
    /// 文件,P3b v1 不实现;仅校验 override 可解析(非 object / 解析失败 → 采用草稿,不崩),
    /// 真正回写留后续。多数轮模型省略 = null = 采用草稿。
    pub(crate) async fn consume_digest_override(&self, override_val: Value, _turn_ref: TurnRef) {
        if !override_val.is_object() {
            // 非 object → 采用草稿,不崩(§12.5 schema 自由兜底)。
            return;
        }
        tracing::debug!(
            target: "progress_channel",
            "digest_override 接收(v1 暂不回写 L0,采用草稿)"
        );
    }

    /// digest 草稿落 L0(§12.5)。分配 sequence,以 `RolloutItem::ProgressDigest` 落档
    /// (只存档、不进模型可见 history,绕开三 adapter 对 role=system 的静默丢弃)。
    /// 去重(M3/M8):v1 不做指纹/时间窗去重,每条都落,留后续。
    pub(crate) async fn append_digest_to_l0(
        &self,
        mut entry: DigestEntry,
        intent_history: Vec<IntentChange>,
    ) {
        // 分配 sequence(锁内自增)。
        {
            let mut state = self.state.lock().await;
            entry.sequence = state.progress_state.digest_seq;
            state.progress_state.digest_seq += 1;
        }
        self.persist_rollout_items(&[RolloutItem::ProgressDigest(ProgressDigestItem {
            entry,
            intent_history,
        })])
        .await;
    }

    /// 读 ProgressState 快照供 step 顶部注入全量态(§12.6)。克隆返回后清空 `warnings`
    /// (一次性回流,防累积,§12.4 M7)。
    pub(crate) async fn snapshot_progress_for_inject(&self) -> ProgressState {
        let mut state = self.state.lock().await;
        let snapshot = state.progress_state.clone();
        state.progress_state.warnings.clear();
        snapshot
    }

    /// FCO 回流(§12.7 step 5):主路径(drain_in_flight)拦截 FunctionCallOutput /
    /// CustomToolCallOutput。失败判定:M5 主路径 `success==Some(false)`。tool 名取
    /// CustomToolCallOutput.name;FunctionCallOutput 无 name 字段,tool 置 None。
    /// brief 取 output 文本截断(truncate_brief)。
    pub(crate) async fn progress_channel_fco_reflow(
        &self,
        turn_context: &TurnContext,
        item: &ResponseItem,
    ) {
        let (tool, output) = match item {
            ResponseItem::FunctionCallOutput { output, .. } => (None, output),
            ResponseItem::CustomToolCallOutput { name, output, .. } => (name.clone(), output),
            _ => return, // 非 FCO → 跳过(drain_in_flight 可能含其他项)。
        };
        let brief = output
            .body
            .to_text()
            .map(|t| truncate_brief(&t))
            .unwrap_or_default();
        let failed = output.success == Some(false);
        self.record_fco_progress(turn_context, tool, brief, failed)
            .await;
    }

    /// FCO 回流副路径(§12.7 step 5 副):`Err(RespondToModel)` 结构化失败——工具请求被
    /// 直接应答/拒绝,必为失败,message 即 brief(call_id 空串,S7 副路径)。无 tool 名。
    pub(crate) async fn progress_channel_fco_reflow_failure(
        &self,
        turn_context: &TurnContext,
        item: &ResponseItem,
    ) {
        let brief = match item {
            ResponseItem::FunctionCallOutput { output, .. }
            | ResponseItem::CustomToolCallOutput { output, .. } => output
                .body
                .to_text()
                .map(|t| truncate_brief(&t))
                .unwrap_or_default(),
            _ => return,
        };
        self.record_fco_progress(turn_context, None, brief, true)
            .await;
    }

    /// FCO 回流核心:生成 kind=tool 草稿落 L0;`failed` 时额外记 last_error。
    /// 主路径(failed 由 success 推断)与副路径(必失败)共用。
    async fn record_fco_progress(
        &self,
        turn_context: &TurnContext,
        tool: Option<String>,
        brief: String,
        failed: bool,
    ) {
        let turn_ref = TurnRef {
            turn_id: turn_context.sub_id.clone(),
            item_index: None,
        };
        if failed {
            self.record_last_error_from_fco(brief.clone(), turn_ref.clone())
                .await;
        }
        // 草稿落 L0(kind=tool,每条都落,去重留后续 M3/M8)。
        self.append_digest_to_l0(
            DigestEntry {
                kind: DigestKind::Tool,
                tool,
                brief_result: if brief.is_empty() { None } else { Some(brief) },
                dialog_summary: None,
                turn_ref,
                sequence: 0, // append_digest_to_l0 回填。
                confidence: None,
            },
            Vec::new(),
        )
        .await;
    }

    /// 下轮注入全量态(§12.6):step 顶部快照 ProgressState → 拼 assistant Message
    /// (S10:assistant role 避被三 adapter 静默丢 system)prepend。空态返回 None。
    /// warnings 段注入后由 snapshot_progress_for_inject 清空(一次性回流,§12.4 M7)。
    pub(crate) async fn render_progress_injection(&self) -> Option<ResponseItem> {
        let snap = self.snapshot_progress_for_inject().await;
        if snap.is_empty() {
            return None;
        }
        let mut sections: Vec<String> = Vec::new();
        sections.push("[progress-state] 系统进度态快照(只读;勿逐字重复本消息)".to_string());
        if let Some(intent) = &snap.intent {
            sections.push(format!("# 意图\n{intent}"));
        }
        if !snap.todo.is_empty() {
            let mut lines = String::from("# 待办");
            for t in &snap.todo {
                let mark = match t.status {
                    TodoStatus::Todo => "[ ]",
                    TodoStatus::Doing => "[~]",
                    TodoStatus::Done => "[x]",
                    TodoStatus::Blocked => "[!]",
                };
                let blocked = t
                    .blocked_reason
                    .as_deref()
                    .map(|r| format!("（阻塞：{r}）"))
                    .unwrap_or_default();
                lines.push_str(&format!("\n{mark} t{} {}{}", t.id, t.text, blocked));
            }
            sections.push(lines);
        }
        if let Some(next) = &snap.next_step {
            sections.push(format!("# 下一步\n{next}"));
        }
        if !snap.assumptions.is_empty() {
            let mut lines = String::from("# 关键假设");
            for a in &snap.assumptions {
                let label = match a.status {
                    AssumptionStatus::Assumed => "假设",
                    AssumptionStatus::Confirmed => "已确认",
                    AssumptionStatus::Stale => "已失效",
                };
                lines.push_str(&format!("\n- a{} {}（{label}）", a.id, a.statement));
            }
            sections.push(lines);
        }
        if !snap.files_touched.is_empty() {
            let files: Vec<&str> = snap.files_touched.iter().map(String::as_str).collect();
            sections.push(format!("# 已触及文件\n{}", files.join(", ")));
        }
        if let Some(err) = &snap.last_error {
            sections.push(format!(
                "# 上次错误（turn {}）\n{}",
                err.turn_ref.turn_id, err.brief
            ));
        }
        if !snap.warnings.is_empty() {
            let mut lines = String::from("# 校验回流");
            for w in &snap.warnings {
                lines.push_str(&format!("\n- {w}"));
            }
            sections.push(lines);
        }
        Some(ResponseItem::Message {
            id: None,
            role: "assistant".to_string(),
            content: vec![ContentItem::OutputText {
                text: sections.join("\n\n"),
            }],
            phase: None,
            internal_chat_message_metadata_passthrough: None,
        })
    }
}

// ===== 辅助函数(操作 &mut ProgressState,自由函数,非 Session 方法)=====

/// 应用单条 todo_op(§12.4 S9)。非法 op / id 不存在 / 字段不全 → 写 warnings 跳过,不中断。
fn apply_todo_op(progress: &mut ProgressState, op_val: &Value) {
    let Some(op) = op_val.get("op").and_then(|v| v.as_str()) else {
        progress.warnings.push("todo_op 缺 op 字段".to_string());
        return;
    };
    match op {
        "add" => {
            let Some(text) = op_val.get("text").and_then(|v| v.as_str()) else {
                progress.warnings.push("todo_op add 缺 text".to_string());
                return;
            };
            let id = progress.todo_id_seq;
            progress.todo_id_seq += 1;
            progress.todo.push(crate::state::TodoItem {
                id,
                text: text.to_string(),
                status: TodoStatus::Todo,
                blocked_reason: None,
            });
        }
        "done" => {
            if let Some(id) = parse_todo_id(op_val, progress) {
                if let Some(item) = progress.todo.iter_mut().find(|t| t.id == id) {
                    item.status = TodoStatus::Done;
                    item.blocked_reason = None;
                } else {
                    progress.warnings.push(format!("todo_op done: id t{id} 不存在"));
                }
            }
        }
        "drop" => {
            if let Some(id) = parse_todo_id(op_val, progress) {
                let before = progress.todo.len();
                progress.todo.retain(|t| t.id != id);
                if progress.todo.len() == before {
                    progress.warnings.push(format!("todo_op drop: id t{id} 不存在"));
                }
            }
        }
        "status" => {
            if let Some(id) = parse_todo_id(op_val, progress)
                && let Some(item) = progress.todo.iter_mut().find(|t| t.id == id)
            {
                let Some(status_str) = op_val.get("status").and_then(|v| v.as_str()) else {
                    progress.warnings.push(format!("todo_op status: t{id} 缺 status"));
                    return;
                };
                let Some(new_status) = parse_todo_status(status_str) else {
                    progress
                        .warnings
                        .push(format!("todo_op status: 非法状态 {status_str}"));
                    return;
                };
                let blocked_reason = if matches!(new_status, TodoStatus::Blocked) {
                    let reason = op_val.get("reason").and_then(|v| v.as_str()).map(String::from);
                    if reason.is_none() {
                        progress
                            .warnings
                            .push(format!("todo_op status: t{id} blocked 须带 reason"));
                        return;
                    }
                    reason
                } else {
                    None
                };
                item.status = new_status;
                item.blocked_reason = blocked_reason;
            } else if let Some(id) = parse_todo_id(op_val, progress) {
                // id 解析成功但 todo 项不存在(上面 find 失败)。
                let _ = id; // parse_todo_id 已对失败 id 写 warnings,此处静默。
            }
        }
        "reset" => {
            // M1:items 非空、每条字段完整(text 必填、status 合法);todo_id_seq 不归零。
            let Some(items) = op_val.get("items").and_then(|v| v.as_array()) else {
                progress.warnings.push("todo_op reset: 缺 items".to_string());
                return;
            };
            if items.is_empty() {
                progress.warnings.push("todo_op reset: items 为空".to_string());
                return;
            }
            let mut new_todo = Vec::with_capacity(items.len());
            for it in items {
                let Some(text) = it.get("text").and_then(|v| v.as_str()) else {
                    progress.warnings.push("todo_op reset: 某条缺 text".to_string());
                    return;
                };
                let Some(status_str) = it.get("status").and_then(|v| v.as_str()) else {
                    progress.warnings.push("todo_op reset: 某条缺 status".to_string());
                    return;
                };
                let Some(status) = parse_todo_status(status_str) else {
                    progress
                        .warnings
                        .push(format!("todo_op reset: 非法状态 {status_str}"));
                    return;
                };
                let blocked_reason = if matches!(status, TodoStatus::Blocked) {
                    let reason = it.get("reason").and_then(|v| v.as_str()).map(String::from);
                    if reason.is_none() {
                        progress
                            .warnings
                            .push("todo_op reset: blocked 须带 reason".to_string());
                        return;
                    }
                    reason
                } else {
                    None
                };
                let id = progress.todo_id_seq;
                progress.todo_id_seq += 1;
                new_todo.push(crate::state::TodoItem {
                    id,
                    text: text.to_string(),
                    status,
                    blocked_reason,
                });
            }
            progress.todo = new_todo;
        }
        other => {
            progress.warnings.push(format!("todo_op: 非法 op {other}"));
        }
    }
}

/// 解析模型引用的 todo id(模型可见形态 `t{N}`,§12.4 S9)。失败写 warnings 返回 None。
/// 兼容模型直接发数字 id 的情形(宽容)。
fn parse_todo_id(op_val: &Value, progress: &mut ProgressState) -> Option<u64> {
    let id = if let Some(s) = op_val.get("id").and_then(|v| v.as_str()) {
        s.strip_prefix('t').and_then(|n| n.parse::<u64>().ok())
    } else {
        op_val.get("id").and_then(|v| v.as_u64())
    };
    match id {
        Some(id) => Some(id),
        None => {
            progress
                .warnings
                .push("todo_op: id 缺失或格式错(期望 tN)".to_string());
            None
        }
    }
}

fn parse_todo_status(s: &str) -> Option<TodoStatus> {
    Some(match s {
        "todo" => TodoStatus::Todo,
        "doing" => TodoStatus::Doing,
        "done" => TodoStatus::Done,
        "blocked" => TodoStatus::Blocked,
        _ => return None,
    })
}

/// 应用单条 assumption(§12.4 M2)。statement 非空、status 合法;id 引用存在则更新,
/// 否则系统分配新 id(复用 todo_id_seq 作全局 id 源)。非法条目 warnings 跳过。
fn apply_assumption(progress: &mut ProgressState, a_val: &Value) {
    let Some(statement) = a_val.get("statement").and_then(|v| v.as_str()) else {
        progress.warnings.push("assumption 缺 statement".to_string());
        return;
    };
    if statement.trim().is_empty() {
        progress
            .warnings
            .push("assumption statement 为空".to_string());
        return;
    }
    let status = match a_val.get("status").and_then(|v| v.as_str()) {
        Some("assumed") => AssumptionStatus::Assumed,
        Some("confirmed") => AssumptionStatus::Confirmed,
        Some("stale") => AssumptionStatus::Stale,
        Some(other) => {
            progress
                .warnings
                .push(format!("assumption: 非法状态 {other}"));
            return;
        }
        None => AssumptionStatus::Assumed,
    };
    // id 引用存在 → 更新;否则系统分配新(忽略模型自造 id)。
    if let Some(id) = a_val.get("id").and_then(|v| v.as_u64())
        && let Some(a) = progress.assumptions.iter_mut().find(|a| a.id == id)
    {
        a.statement = statement.to_string();
        a.status = status;
        return;
    }
    let id = progress.todo_id_seq;
    progress.todo_id_seq += 1;
    progress.assumptions.push(crate::state::Assumption {
        id,
        statement: statement.to_string(),
        status,
    });
}

/// 解析 apply_patch lark 文本中的文件路径(*** Add/Update/Delete File marker)。
fn parse_apply_patch_paths(input: &str) -> Vec<String> {
    let mut paths = Vec::new();
    for line in input.lines() {
        let line = line.trim();
        for marker in &["*** Add File:", "*** Update File:", "*** Delete File:"] {
            if let Some(rest) = line.strip_prefix(marker) {
                let path = rest.trim();
                if !path.is_empty() {
                    paths.push(path.to_string());
                }
            }
        }
    }
    paths
}

/// 截断 FCO brief(首尾保留 + 中间省略,§12.5 kind=tool brief_result 长度控制)。
fn truncate_brief(s: &str) -> String {
    const MAX: usize = 500;
    const EDGE: usize = 200;
    let chars: Vec<char> = s.chars().collect();
    if chars.len() <= MAX {
        return s.to_string();
    }
    let head: String = chars.iter().take(EDGE).collect();
    let tail: String = chars
        .iter()
        .rev()
        .take(EDGE)
        .collect::<Vec<_>>()
        .into_iter()
        .rev()
        .collect();
    format!("{head}…{tail}")
}

#[cfg(test)]
mod tests {
    //! P3b §12.4 / §12.7 纯函数单测：progress_patch 校验（todo_ops / assumptions）+
    //! files_touched 推导辅助（parse_apply_patch_paths）。这些函数操作 &mut ProgressState，
    //! 无需 Session，直接测逻辑正确性 + warnings 回流（§12.4 M7）。
    use super::apply_assumption;
    use super::apply_todo_op;
    use super::parse_apply_patch_paths;
    use crate::state::AssumptionStatus;
    use crate::state::ProgressState;
    use crate::state::TodoStatus;
    use serde_json::json;

    #[test]
    fn todo_op_add_appends_item_with_system_id() {
        let mut p = ProgressState::default();
        apply_todo_op(&mut p, &json!({"op":"add","text":"写测试"}));
        assert_eq!(p.todo.len(), 1);
        assert_eq!(p.todo[0].id, 0);
        assert_eq!(p.todo[0].text, "写测试");
        assert!(matches!(p.todo[0].status, TodoStatus::Todo));
        // 重复 add 自增 id（§12.2 S13）。
        apply_todo_op(&mut p, &json!({"op":"add","text":"收尾"}));
        assert_eq!(p.todo[1].id, 1);
    }

    #[test]
    fn todo_op_done_marks_existing_item() {
        let mut p = ProgressState::default();
        apply_todo_op(&mut p, &json!({"op":"add","text":"a"}));
        apply_todo_op(&mut p, &json!({"op":"done","id":"t0"}));
        assert!(matches!(p.todo[0].status, TodoStatus::Done));
    }

    #[test]
    fn todo_op_done_unknown_id_warns_without_panic() {
        let mut p = ProgressState::default();
        apply_todo_op(&mut p, &json!({"op":"done","id":"t9"}));
        assert!(p.warnings.iter().any(|w| w.contains("t9")));
    }

    #[test]
    fn todo_op_drop_removes_item() {
        let mut p = ProgressState::default();
        apply_todo_op(&mut p, &json!({"op":"add","text":"a"}));
        apply_todo_op(&mut p, &json!({"op":"drop","id":"t0"}));
        assert!(p.todo.is_empty());
    }

    #[test]
    fn todo_op_status_blocked_requires_reason() {
        let mut p = ProgressState::default();
        apply_todo_op(&mut p, &json!({"op":"add","text":"a"}));
        // blocked 缺 reason → warning，状态不变。
        apply_todo_op(&mut p, &json!({"op":"status","id":"t0","status":"blocked"}));
        assert!(p.warnings.iter().any(|w| w.contains("reason")));
        assert!(matches!(p.todo[0].status, TodoStatus::Todo));
        // 带 reason → 生效。
        apply_todo_op(
            &mut p,
            &json!({"op":"status","id":"t0","status":"blocked","reason":"等依赖"}),
        );
        assert!(matches!(p.todo[0].status, TodoStatus::Blocked));
        assert_eq!(p.todo[0].blocked_reason.as_deref(), Some("等依赖"));
    }

    #[test]
    fn todo_op_reset_replaces_list() {
        let mut p = ProgressState::default();
        apply_todo_op(&mut p, &json!({"op":"add","text":"旧"}));
        apply_todo_op(
            &mut p,
            &json!({"op":"reset","items":[
                {"text":"新1","status":"todo"},
                {"text":"新2","status":"done"}
            ]}),
        );
        assert_eq!(p.todo.len(), 2);
        assert_eq!(p.todo[0].text, "新1");
        assert!(matches!(p.todo[1].status, TodoStatus::Done));
    }

    #[test]
    fn todo_op_illegal_op_warns() {
        let mut p = ProgressState::default();
        apply_todo_op(&mut p, &json!({"op":"frobnicate"}));
        assert!(p.warnings.iter().any(|w| w.contains("frobnicate")));
    }

    #[test]
    fn assumption_add_then_update_by_id() {
        let mut p = ProgressState::default();
        apply_assumption(&mut p, &json!({"statement":"用户用 Rust","status":"assumed"}));
        assert_eq!(p.assumptions.len(), 1);
        let id = p.assumptions[0].id;
        // 同 id → 更新（不新增）。
        apply_assumption(
            &mut p,
            &json!({"id":id,"statement":"用户用 Rust 1.95","status":"confirmed"}),
        );
        assert_eq!(p.assumptions.len(), 1);
        assert_eq!(p.assumptions[0].statement, "用户用 Rust 1.95");
        assert!(matches!(p.assumptions[0].status, AssumptionStatus::Confirmed));
    }

    #[test]
    fn assumption_illegal_status_warns_and_skips() {
        let mut p = ProgressState::default();
        apply_assumption(&mut p, &json!({"statement":"x","status":"bogus"}));
        assert!(p.warnings.iter().any(|w| w.contains("bogus")));
        assert!(p.assumptions.is_empty());
    }

    #[test]
    fn parse_apply_patch_paths_extracts_markers() {
        // §12.7 step 4 files_touched 推导辅助：识别 Add/Update/Delete marker 路径。
        let input = "*** Begin Patch\n\
                     *** Add File: /a.rs\n+x\n\
                     *** Update File: /b.rs\n@@\n+y\n\
                     *** Delete File: /c.rs\n\
                     *** End Patch";
        let paths = parse_apply_patch_paths(input);
        assert_eq!(paths, vec!["/a.rs".to_string(), "/b.rs".to_string(), "/c.rs".to_string()]);
    }

    #[test]
    fn progress_state_is_empty_default_only() {
        assert!(ProgressState::default().is_empty());
        let mut p = ProgressState::default();
        p.intent = Some("做 P3b".to_string());
        assert!(!p.is_empty());
    }
}
