use crate::config::RolloutBudgetConfig;
use codex_protocol::ThreadId;
use codex_protocol::protocol::TokenUsage;
use std::collections::HashMap;
use std::sync::Mutex;
use std::sync::MutexGuard;
use std::sync::OnceLock;

pub(crate) struct RolloutBudgetReminder {
    pub(crate) remaining_tokens: i64,
    reminder_index: i64,
}

/// Shared accounting and reminder state for one root-thread session tree.
#[derive(Default)]
pub(crate) struct RolloutBudget {
    state: OnceLock<Mutex<RolloutBudgetState>>,
}

struct RolloutBudgetState {
    config: RolloutBudgetConfig,
    weighted_tokens_used: f64,
    /// Last reminder delivered to each thread, so every thread observes crossed thresholds.
    deliveries: HashMap<ThreadId, ThreadBudgetDelivery>,
}

struct ThreadBudgetDelivery {
    window_id: String,
    reminder_index: i64,
}

impl RolloutBudget {
    pub(crate) fn configure(&self, config: RolloutBudgetConfig) {
        self.state.get_or_init(|| {
            Mutex::new(RolloutBudgetState {
                config,
                weighted_tokens_used: 0.0,
                deliveries: HashMap::new(),
            })
        });
    }

    /// Returns true once the configured budget is exhausted, including on later calls.
    pub(crate) fn record_usage(&self, usage: &TokenUsage) -> bool {
        let Some(mut state) = self.lock() else {
            return false;
        };
        // 推理 token 独立计项：`reasoning_output_tokens` 是思考专属输出（Gemini
        // thoughtsTokenCount 等），与正文采样输出分离。Anthropic 把 thinking 折进
        // `output_tokens`、其 `reasoning_output_tokens` 恒 0，该项对其无影响。
        // 不计该项会让 thinking 占比高的非 OpenAI 会话静默漏算、超支不报警。
        state.weighted_tokens_used += usage.output_tokens.max(0) as f64
            * state.config.sampling_token_weight
            + usage.non_cached_input() as f64 * state.config.prefill_token_weight
            + usage.reasoning_output_tokens.max(0) as f64 * state.config.reasoning_token_weight;
        state.weighted_tokens_used >= state.config.limit_tokens as f64
    }

    pub(crate) fn pending_reminder(
        &self,
        thread_id: ThreadId,
        window_id: &str,
    ) -> Option<RolloutBudgetReminder> {
        let state = self.lock()?;
        let remaining_tokens = (state.config.limit_tokens as f64 - state.weighted_tokens_used)
            .max(0.0)
            .floor() as i64;
        let reminder_index = state
            .config
            .reminder_at_remaining_tokens
            .iter()
            .filter(|&&threshold| remaining_tokens <= threshold)
            .count() as i64;
        if state.deliveries.get(&thread_id).is_some_and(|delivery| {
            delivery.window_id.as_str() == window_id && delivery.reminder_index >= reminder_index
        }) {
            return None;
        }
        Some(RolloutBudgetReminder {
            remaining_tokens,
            reminder_index,
        })
    }

    pub(crate) fn mark_reminder_delivered(
        &self,
        thread_id: ThreadId,
        window_id: &str,
        reminder: RolloutBudgetReminder,
    ) {
        // Mark delivery only after history insertion; cancellation before then should retry it.
        let Some(mut state) = self.lock() else {
            return;
        };
        state.deliveries.insert(
            thread_id,
            ThreadBudgetDelivery {
                window_id: window_id.to_string(),
                reminder_index: reminder.reminder_index,
            },
        );
    }

    /// Forces the next sampling request for `thread_id` to restate the current remainder.
    pub(crate) fn rearm_reminder(&self, thread_id: ThreadId) {
        let Some(mut state) = self.lock() else {
            return;
        };
        state.deliveries.remove(&thread_id);
    }

    fn lock(&self) -> Option<MutexGuard<'_, RolloutBudgetState>> {
        self.state.get().map(|state| {
            state
                .lock()
                .unwrap_or_else(std::sync::PoisonError::into_inner)
        })
    }
}

#[cfg(test)]
mod tests {
    use super::RolloutBudget;
    use crate::config::RolloutBudgetConfig;
    use codex_protocol::protocol::TokenUsage;

    fn config(limit_tokens: i64, reasoning_token_weight: f64) -> RolloutBudgetConfig {
        RolloutBudgetConfig {
            limit_tokens,
            reminder_at_remaining_tokens: vec![],
            sampling_token_weight: 1.0,
            prefill_token_weight: 1.0,
            reasoning_token_weight,
        }
    }

    /// 仅含推理 token（output/input 全 0）的 usage，用于隔离推理项对预算的贡献。
    fn reasoning_only_usage(reasoning_output_tokens: i64) -> TokenUsage {
        TokenUsage {
            input_tokens: 0,
            cached_input_tokens: 0,
            output_tokens: 0,
            reasoning_output_tokens,
            total_tokens: 0,
        }
    }

    #[test]
    fn reasoning_output_tokens_count_toward_budget() {
        // M5 修复：推理 token 须按 reasoning_token_weight 计入会话预算。
        // limit=100、weight=1.0：先记 60 不耗尽，再记 50（累计 110）耗尽。
        let budget = RolloutBudget::default();
        budget.configure(config(100, 1.0));
        assert!(!budget.record_usage(&reasoning_only_usage(60)));
        assert!(budget.record_usage(&reasoning_only_usage(50)));
    }

    #[test]
    fn reasoning_weight_zero_excludes_reasoning() {
        // reasoning_token_weight=0：推理 token 不计预算，纯推理 usage 永不觉醒。
        let budget = RolloutBudget::default();
        budget.configure(config(100, 0.0));
        assert!(!budget.record_usage(&reasoning_only_usage(1_000_000)));
    }
}
