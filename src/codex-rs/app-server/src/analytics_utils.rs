use codex_analytics::AnalyticsEventsClient;
use codex_core::config::Config;

pub(crate) fn analytics_events_client_from_config(config: &Config) -> AnalyticsEventsClient {
    AnalyticsEventsClient::new(config.analytics_enabled)
}
