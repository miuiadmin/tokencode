//! 本地连接器辅助函数。
//!
//! 这两个函数负责把"全量目录连接器"与"MCP accessible 连接器"按需合并、
//! 以及从连接器列表中筛出插件声明的 app 连接器。底层复用 `codex_connectors::merge`
//! 中公开的合并实现，app-server 不再依赖云目录 crate。

use std::collections::HashMap;
use std::collections::HashSet;

use codex_connectors::AppInfo;
use codex_connectors::merge::merge_connectors;
use codex_connectors::merge::merge_plugin_connectors;
use codex_plugin::AppConnectorId;

/// 合并全量目录连接器与 MCP accessible 连接器。
///
/// - `all_connectors_loaded` 为 `true` 时（已拿到完整云目录），accessible 列表中
///   不在目录内的连接器会被过滤掉，避免显示目录之外的条目。
/// - 为 `false` 时（API key 鉴权模式，无云目录），保留全部 accessible 连接器，
///   只走 MCP 本地派生能力。
pub(crate) fn merge_connectors_with_accessible(
    connectors: Vec<AppInfo>,
    accessible_connectors: Vec<AppInfo>,
    all_connectors_loaded: bool,
) -> Vec<AppInfo> {
    let accessible_connectors = if all_connectors_loaded {
        let connector_ids: HashSet<&str> = connectors
            .iter()
            .map(|connector| connector.id.as_str())
            .collect();
        accessible_connectors
            .into_iter()
            .filter(|connector| connector_ids.contains(connector.id.as_str()))
            .collect()
    } else {
        accessible_connectors
    };
    merge_connectors(connectors, accessible_connectors)
}

/// 从连接器列表中筛出插件声明的 app 连接器。
///
/// 缺失的 app 连接器会通过 `merge_plugin_connectors` 以插件声明合成默认条目，
/// 再按 `plugin_apps` 顺序返回。
pub(crate) fn connectors_for_plugin_apps(
    connectors: Vec<AppInfo>,
    plugin_apps: &[AppConnectorId],
) -> Vec<AppInfo> {
    let connectors = merge_plugin_connectors(
        connectors,
        plugin_apps
            .iter()
            .map(|connector_id| connector_id.0.clone()),
    );
    let mut connectors_by_id = connectors
        .into_iter()
        .map(|connector| (connector.id.clone(), connector))
        .collect::<HashMap<_, _>>();

    plugin_apps
        .iter()
        .filter_map(|connector_id| connectors_by_id.remove(connector_id.0.as_str()))
        .collect()
}
