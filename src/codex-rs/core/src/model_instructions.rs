//! 按模型与 wire 协议提供通用 coding-agent 基础指令。
//!
//! models.json 里未声明 base_instructions 的模型（如跨厂商直连的新增模型）在此回落到
//! 一份精简通用的 coding-agent 模板，使无内置指令的模型也有一致的角色与工作约定。
//! 已声明 base_instructions 的模型不受影响。

use codex_model_provider_info::WireApi;

/// 返回给定模型在指定 wire 协议下的通用 base_instructions。
///
/// 仅当 [`codex_protocol::openai_models::ModelInfo::base_instructions`] 为空时由
/// `Session::get_base_instructions` 调用。当前三套 wire 协议共用同一份核心指令
/// （协议差异由 adapter 层负责转换），`wire_api` 入参保留以便未来按协议微调措辞。
pub fn base_instructions_for(wire_api: WireApi, _slug: &str) -> String {
    let _ = wire_api;
    r#"你是 TokenCode，一个运行在用户终端的编程助手。

你的职责
- 理解用户的编程意图，通过读写文件、执行命令、调用工具完成任务
- 改动前先读懂相关上下文，做最小且正确的修改
- 修改后主动验证：编译、运行测试或执行命令确认结果

工作原则
- 优先复用目标代码库已有的实现、风格与约定，不引入多余依赖
- 遇到不确定或破坏性的操作，先向用户确认，而非擅自假设
- 报告结果时给出结论与关键依据，省略无关的过程细节

交互风格
- 默认使用中文回复
- 简洁直接，避免冗长的客套与重复
- 涉及代码时指明文件与位置，便于用户核对"#
        .to_string()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn returns_non_empty_template_for_each_wire_api() {
        for wire_api in [WireApi::Responses, WireApi::Chat, WireApi::Anthropic] {
            let text = base_instructions_for(wire_api, "glm-5.2");
            assert!(!text.is_empty());
            assert!(text.contains("TokenCode"));
        }
    }
}
