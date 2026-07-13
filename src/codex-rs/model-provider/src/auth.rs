use std::sync::Arc;

use codex_agent_identity::AgentIdentityKey;
use codex_agent_identity::authorization_header_for_agent_task;
use codex_api::AgentIdentityTelemetry;
use codex_api::AuthProvider;
use codex_api::SharedAuthProvider;
use codex_login::AuthManager;
use codex_login::CodexAuth;
use codex_login::auth::AgentIdentityAuth;
use codex_model_provider_info::ModelProviderInfo;
use codex_protocol::error::CodexErr;
use http::HeaderMap;
use http::HeaderValue;

use crate::bearer_auth_provider::BearerAuthProvider;

const BEDROCK_API_KEY_UNSUPPORTED_MESSAGE: &str =
    "Bedrock API key auth is only supported by the Amazon Bedrock model provider";

/// Provider auth resolved for a request, plus metadata describing the effective auth.
#[derive(Clone)]
pub struct ResolvedProviderAuth {
    pub auth: SharedAuthProvider,
    pub agent_identity_telemetry: Option<AgentIdentityTelemetry>,
}

impl ResolvedProviderAuth {
    pub(crate) fn new(auth: SharedAuthProvider) -> Self {
        Self {
            auth,
            agent_identity_telemetry: None,
        }
    }
}

pub(crate) fn agent_identity_telemetry(auth: &AgentIdentityAuth) -> AgentIdentityTelemetry {
    AgentIdentityTelemetry {
        agent_id: auth.record().agent_runtime_id.clone(),
        task_id: auth.run_task_id().to_string(),
    }
}

#[derive(Clone, Debug)]
struct AgentIdentityAuthProvider {
    auth: AgentIdentityAuth,
}

impl AuthProvider for AgentIdentityAuthProvider {
    fn add_auth_headers(&self, headers: &mut HeaderMap) {
        let record = self.auth.record();
        let header_value = authorization_header_for_agent_task(
            AgentIdentityKey {
                agent_runtime_id: &record.agent_runtime_id,
                private_key_pkcs8_base64: &record.agent_private_key,
            },
            self.auth.run_task_id(),
        )
        .map_err(std::io::Error::other);

        if let Ok(header_value) = header_value
            && let Ok(header) = HeaderValue::from_str(&header_value)
        {
            let _ = headers.insert(http::header::AUTHORIZATION, header);
        }

        if let Ok(header) = HeaderValue::from_str(self.auth.account_id()) {
            let _ = headers.insert("ChatGPT-Account-ID", header);
        }

        if self.auth.is_fedramp_account() {
            let _ = headers.insert("X-OpenAI-Fedramp", HeaderValue::from_static("true"));
        }
    }
}

// Some providers are meant to send no auth headers. Examples include local OSS
// providers and custom test providers with `requires_openai_auth = false`.
#[derive(Clone, Debug)]
struct UnauthenticatedAuthProvider;

impl AuthProvider for UnauthenticatedAuthProvider {
    fn add_auth_headers(&self, _headers: &mut HeaderMap) {}
}

pub fn unauthenticated_auth_provider() -> SharedAuthProvider {
    Arc::new(UnauthenticatedAuthProvider)
}

/// Returns the provider-scoped auth manager when this provider uses command-backed auth.
///
/// Providers without custom auth continue using the caller-supplied base manager, when present.
pub(crate) fn auth_manager_for_provider(
    auth_manager: Option<Arc<AuthManager>>,
    provider: &ModelProviderInfo,
) -> Option<Arc<AuthManager>> {
    match provider.auth.clone() {
        Some(config) => Some(AuthManager::external_bearer_only(config)),
        None => auth_manager,
    }
}

pub(crate) fn resolve_provider_auth(
    auth: Option<&CodexAuth>,
    provider: &ModelProviderInfo,
) -> codex_protocol::error::Result<SharedAuthProvider> {
    if matches!(auth, Some(CodexAuth::BedrockApiKey(_))) {
        return Err(CodexErr::UnsupportedOperation(
            BEDROCK_API_KEY_UNSUPPORTED_MESSAGE.to_string(),
        ));
    }

    if let Some(auth) = bearer_auth_for_provider(provider)? {
        return Ok(Arc::new(auth));
    }

    Ok(match auth {
        Some(auth) => auth_provider_from_auth(auth),
        None => unauthenticated_auth_provider(),
    })
}

fn bearer_auth_for_provider(
    provider: &ModelProviderInfo,
) -> codex_protocol::error::Result<Option<BearerAuthProvider>> {
    if let Some(api_key) = provider.api_key()? {
        return Ok(Some(BearerAuthProvider::new(api_key)));
    }

    if let Some(token) = provider.experimental_bearer_token.clone() {
        return Ok(Some(BearerAuthProvider::new(token)));
    }

    Ok(None)
}

/// Builds request-header auth for a first-party TokenCode auth snapshot.
pub fn auth_provider_from_auth(auth: &CodexAuth) -> SharedAuthProvider {
    match auth {
        CodexAuth::AgentIdentity(auth) => {
            Arc::new(AgentIdentityAuthProvider { auth: auth.clone() })
        }
        CodexAuth::BedrockApiKey(_) => unreachable!("{BEDROCK_API_KEY_UNSUPPORTED_MESSAGE}"),
        CodexAuth::ApiKey(_)
        | CodexAuth::Chatgpt(_)
        | CodexAuth::ChatgptAuthTokens(_)
        | CodexAuth::PersonalAccessToken(_) => Arc::new(BearerAuthProvider {
            token: auth.get_token().ok(),
        }),
    }
}

#[cfg(test)]
mod tests {
    use codex_agent_identity::generate_agent_key_material;
    use codex_login::auth::AgentIdentityAuthRecord;
    use codex_login::auth::BedrockApiKeyAuth;
    use codex_model_provider_info::WireApi;
    use codex_model_provider_info::create_oss_provider_with_base_url;
    use codex_protocol::account::PlanType;
    use pretty_assertions::assert_eq;

    use super::*;

    async fn agent_identity_auth(chatgpt_account_is_fedramp: bool) -> AgentIdentityAuth {
        let key_material = generate_agent_key_material().expect("generate key material");
        AgentIdentityAuth::from_record(
            AgentIdentityAuthRecord {
                agent_runtime_id: "agent-runtime-1".to_string(),
                agent_private_key: key_material.private_key_pkcs8_base64,
                account_id: "account-1".to_string(),
                chatgpt_user_id: "user-1".to_string(),
                email: Some("agent@example.com".to_string()),
                plan_type: PlanType::Plus,
                chatgpt_account_is_fedramp,
                task_id: Some("task-run-1".to_string()),
            },
            "https://auth.openai.com/api/accounts",
            /*auth_route_config*/ None,
        )
        .await
        .expect("agent identity auth record should include task id")
    }

    #[test]
    fn unauthenticated_auth_provider_adds_no_headers() {
        let provider =
            create_oss_provider_with_base_url("http://localhost:11434/v1", WireApi::Responses);
        let auth = resolve_provider_auth(/*auth*/ None, &provider).expect("auth should resolve");

        assert!(auth.to_auth_headers().is_empty());
    }

    #[test]
    fn openai_provider_rejects_bedrock_api_key_auth() {
        let provider = ModelProviderInfo::create_openai_provider(/*base_url*/ None);
        let auth = CodexAuth::BedrockApiKey(BedrockApiKeyAuth {
            api_key: "bedrock-api-key-test".to_string(),
            region: "us-east-1".to_string(),
        });

        match resolve_provider_auth(Some(&auth), &provider) {
            Err(CodexErr::UnsupportedOperation(message)) => {
                assert_eq!(message, BEDROCK_API_KEY_UNSUPPORTED_MESSAGE);
            }
            Err(err) => panic!("unexpected auth error: {err:?}"),
            Ok(_) => panic!("Bedrock API key auth should be rejected"),
        }
    }

    #[tokio::test]
    async fn agent_identity_auth_provider_preserves_account_routing_headers() {
        let auth = agent_identity_auth(/*chatgpt_account_is_fedramp*/ true).await;
        let provider = auth_provider_from_auth(&CodexAuth::AgentIdentity(auth));

        let headers = provider.to_auth_headers();

        assert!(
            headers
                .get(http::header::AUTHORIZATION)
                .and_then(|value| value.to_str().ok())
                .is_some_and(|value| value.starts_with("AgentAssertion "))
        );
        assert_eq!(
            headers
                .get("ChatGPT-Account-ID")
                .and_then(|value| value.to_str().ok()),
            Some("account-1")
        );
        assert_eq!(
            headers
                .get("X-OpenAI-Fedramp")
                .and_then(|value| value.to_str().ok()),
            Some("true")
        );
    }
}
