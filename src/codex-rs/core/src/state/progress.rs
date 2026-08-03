//! 进度副信道会话级状态容器(§12.2)。
//!
//! 跨 step 累积、跨 turn 持久(同 Session 内);住 `SessionState`,不仿 `WorldState`
//! 每 step 重建。v1 不做 snapshot/diff(M9):每 step 全量读快照 + patch 增量更新内存态。

use std::collections::BTreeSet;

use codex_protocol::protocol::TurnRef;

/// 进度副信道的单一真值源:系统持有,模型只读全量 + 写增量 patch(§4.1)。
#[derive(Debug, Default, Clone)]
pub(crate) struct ProgressState {
    /// original_intent,当前活的 goal;演进旧值 append L0(载体 = ProgressDigestItem)。
    pub(crate) intent: Option<String>,
    /// 系统 id 稳定的待办;模型可见形态为 `t{id}`(§12.2 S9)。
    pub(crate) todo: Vec<TodoItem>,
    pub(crate) next_step: Option<String>,
    /// 关键假设累积(§4 ·四 ②)。
    pub(crate) assumptions: Vec<Assumption>,
    /// 从 apply_patch/shell 推导累加(B 类,§12.7 step 4 推导);提示非真值。
    pub(crate) files_touched: BTreeSet<String>,
    /// 从失败 FCO 提取(B 类,§12.7 step 5 推导)。
    pub(crate) last_error: Option<LastError>,
    /// 系统 id 分配器(add 时自增);新 Session 从 0 起,同 Session 跨 turn 不归零(§12.2 S13)。
    pub(crate) todo_id_seq: u64,
    /// digest 落 L0 顺序号分配器(append_digest_to_l0 自增,§12.5 sequence 字段)。
    pub(crate) digest_seq: u64,
    /// patch 校验回流提示(§12.4 M7);下轮注入附带后清空(一次性,防累积)。
    pub(crate) warnings: Vec<String>,
}

#[derive(Debug, Clone)]
pub(crate) struct TodoItem {
    /// 内部数值 id;模型可见形态为 `t{id}`(§12.2 S9)。
    pub(crate) id: u64,
    pub(crate) text: String,
    pub(crate) status: TodoStatus,
    pub(crate) blocked_reason: Option<String>,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum TodoStatus {
    Todo,
    Doing,
    Done,
    Blocked,
}

#[derive(Debug, Clone)]
pub(crate) struct Assumption {
    pub(crate) id: u64,
    pub(crate) statement: String,
    pub(crate) status: AssumptionStatus,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum AssumptionStatus {
    Assumed,
    Confirmed,
    Stale,
}

#[derive(Debug, Clone)]
pub(crate) struct LastError {
    pub(crate) turn_ref: TurnRef,
    pub(crate) brief: String,
}

impl ProgressState {
    /// 是否为「未发生任何进度」的空态（下轮注入跳过，§12.6）。
    /// `todo_id_seq` / `digest_seq` 是纯分配器，不代表有内容，不计入。
    pub(crate) fn is_empty(&self) -> bool {
        self.intent.is_none()
            && self.todo.is_empty()
            && self.next_step.is_none()
            && self.assumptions.is_empty()
            && self.files_touched.is_empty()
            && self.last_error.is_none()
            && self.warnings.is_empty()
    }
}
