use serde::Deserialize;
use serde::Serialize;
use std::cmp::Ordering;
use std::fmt;

/// 命名空间工具名展平为单字符串时使用的分隔符。
///
/// 非 Responses 协议（OpenAI Chat / Anthropic / Gemini）在 wire 上只携带单个工具名字符串，
/// 没有 namespace 概念。为让这些协议也能调用 MCP / 多智能体等 namespace 工具，序列化侧把
/// `{namespace}__{name}` 拼成单字符串下发，解析侧再按此分隔符还原。
///
/// 约定：**plain 工具名不得包含该分隔符**（当前所有 builtin / namespace 内工具名均不含 `__`），
/// 否则 `from_flat_wire_name` 会误判为带 namespace。这与既有 MCP hook 命名约定一致。
pub const TOOL_NAMESPACE_DELIMITER: &str = "__";

/// Identifies a callable tool, preserving the namespace split when the model
/// provides one.
#[derive(Clone, Debug, Deserialize, Eq, Hash, PartialEq, Serialize)]
pub struct ToolName {
    pub name: String,
    pub namespace: Option<String>,
}

impl ToolName {
    pub fn new(namespace: Option<String>, name: impl Into<String>) -> Self {
        Self {
            name: name.into(),
            namespace,
        }
    }

    pub fn plain(name: impl Into<String>) -> Self {
        Self {
            name: name.into(),
            namespace: None,
        }
    }

    pub fn namespaced(namespace: impl Into<String>, name: impl Into<String>) -> Self {
        Self {
            name: name.into(),
            namespace: Some(namespace.into()),
        }
    }

    /// 展平成 `{namespace}__{name}` 单字符串（非 Responses 协议 wire 用）。
    ///
    /// 注意：这里是**纯净拼接**，不做任何 trim / 前缀处理——因为解析侧必须能无损还原成
    /// 与 registry key 完全一致的 `ToolName`（registry 用未归一化的 namespace/name 建键）。
    /// 不要与 `core::tools::handlers::mcp::join_tool_name` 混淆，后者带 MCP hook 专用的
    /// `trim_*matches('_')` 归一化与 `mcp__` 前缀，仅供 hook / telemetry 命名，不可逆。
    pub fn to_flat_wire_name(&self) -> String {
        match &self.namespace {
            Some(namespace) => {
                format!("{namespace}{TOOL_NAMESPACE_DELIMITER}{}", self.name)
            }
            None => self.name.clone(),
        }
    }

    /// `to_flat_wire_name` 的逆运算：从单字符串还原 `ToolName`。
    ///
    /// 含分隔符 → `namespaced(分隔符左侧, 右侧)`（仅在**首个**分隔符处切分，因此 name 自身
    /// 允许含分隔符）；不含 → `plain`。详见 [`TOOL_NAMESPACE_DELIMITER`] 的 plain 名约定。
    pub fn from_flat_wire_name(flat: &str) -> Self {
        match flat.split_once(TOOL_NAMESPACE_DELIMITER) {
            Some((namespace, name)) => Self::namespaced(namespace, name),
            None => Self::plain(flat),
        }
    }
}

impl fmt::Display for ToolName {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match &self.namespace {
            Some(namespace) => write!(f, "{namespace}{}", self.name),
            None => f.write_str(&self.name),
        }
    }
}

impl Ord for ToolName {
    fn cmp(&self, other: &Self) -> Ordering {
        let lhs = match &self.namespace {
            Some(namespace) => (namespace.as_str(), Some(self.name.as_str())),
            None => (self.name.as_str(), None),
        };
        let rhs = match &other.namespace {
            Some(namespace) => (namespace.as_str(), Some(other.name.as_str())),
            None => (other.name.as_str(), None),
        };
        lhs.cmp(&rhs)
    }
}

impl PartialOrd for ToolName {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        Some(self.cmp(other))
    }
}

impl From<String> for ToolName {
    fn from(name: String) -> Self {
        Self::plain(name)
    }
}

impl From<&str> for ToolName {
    fn from(name: &str) -> Self {
        Self::plain(name)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn flat_wire_name_round_trips_namespaced_and_plain() {
        // 带 namespace：往返无损
        let ns = ToolName::namespaced("context7", "get-docs");
        assert_eq!(ns.to_flat_wire_name(), "context7__get-docs");
        assert_eq!(ToolName::from_flat_wire_name("context7__get-docs"), ns);

        // plain：无分隔符，原样
        let plain = ToolName::plain("apply_patch");
        assert_eq!(plain.to_flat_wire_name(), "apply_patch");
        assert_eq!(ToolName::from_flat_wire_name("apply_patch"), plain);
    }

    #[test]
    fn from_flat_wire_name_splits_only_on_first_delimiter() {
        // name 自身含分隔符：仅首个分隔符切分，右侧整体作为 name
        let ns = ToolName::namespaced("ns", "do__something");
        assert_eq!(ns.to_flat_wire_name(), "ns__do__something");
        assert_eq!(ToolName::from_flat_wire_name("ns__do__something"), ns);
    }

    #[test]
    fn flat_wire_name_preserves_namespace_and_name_verbatim() {
        // 不做 trim / 前缀归一化（与 registry key 一致）
        let ns = ToolName::namespaced("mcp_server", "list_tools");
        assert_eq!(ns.to_flat_wire_name(), "mcp_server__list_tools");
        let back = ToolName::from_flat_wire_name("mcp_server__list_tools");
        assert_eq!(back.namespace.as_deref(), Some("mcp_server"));
        assert_eq!(back.name, "list_tools");
    }
}
