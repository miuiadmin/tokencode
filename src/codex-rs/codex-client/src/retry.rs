use crate::error::TransportError;
use crate::request::Request;
use http::HeaderMap;
use rand::Rng;
use std::future::Future;
use std::time::Duration;
use tokio::time::sleep;

#[derive(Debug, Clone)]
pub struct RetryPolicy {
    pub max_attempts: u64,
    pub base_delay: Duration,
    pub retry_on: RetryOn,
}

#[derive(Debug, Clone)]
pub struct RetryOn {
    pub retry_429: bool,
    pub retry_5xx: bool,
    pub retry_transport: bool,
}

impl RetryOn {
    pub fn should_retry(&self, err: &TransportError, attempt: u64, max_attempts: u64) -> bool {
        if attempt >= max_attempts {
            return false;
        }
        match err {
            TransportError::Http { status, .. } => {
                (self.retry_429 && status.as_u16() == 429)
                    || (self.retry_5xx && status.is_server_error())
            }
            TransportError::Timeout | TransportError::Network(_) => self.retry_transport,
            _ => false,
        }
    }
}

pub fn backoff(base: Duration, attempt: u64) -> Duration {
    if attempt == 0 {
        return base;
    }
    let exp = 2u64.saturating_pow(attempt as u32 - 1);
    let millis = base.as_millis() as u64;
    let raw = millis.saturating_mul(exp);
    let jitter: f64 = rand::rng().random_range(0.9..1.1);
    Duration::from_millis((raw as f64 * jitter) as u64)
}

/// 从 HTTP 响应头解析退避时长：优先 `retry-after-ms`（毫秒，部分兼容网关采用），
/// 回落 `Retry-After`（整数秒，RFC 7231 delta-seconds）。不支持 HTTP-date（设计决策）。
/// 非法或缺失返回 None，调用方据此回落本地指数 backoff。
pub fn parse_retry_after_header(headers: &HeaderMap) -> Option<Duration> {
    if let Some(ms) = headers
        .get("retry-after-ms")
        .and_then(|v| v.to_str().ok())
        .and_then(|s| s.parse::<u64>().ok())
    {
        return Some(Duration::from_millis(ms));
    }
    headers
        .get("retry-after")
        .and_then(|v| v.to_str().ok())
        .and_then(|s| s.parse::<u64>().ok())
        .map(Duration::from_secs)
}

pub async fn run_with_retry<T, F, Fut>(
    policy: RetryPolicy,
    mut make_req: impl FnMut() -> Request,
    op: F,
) -> Result<T, TransportError>
where
    F: Fn(Request, u64) -> Fut,
    Fut: Future<Output = Result<T, TransportError>>,
{
    for attempt in 0..=policy.max_attempts {
        let req = make_req();
        match op(req, attempt).await {
            Ok(resp) => return Ok(resp),
            Err(err)
                if policy
                    .retry_on
                    .should_retry(&err, attempt, policy.max_attempts) =>
            {
                // 服务端经 Retry-After / retry-after-ms 头明示退避时长时优先采用（尊重限流
                // 指示），否则回落本地指数 backoff。三个 adapter 共享此传输层，一处惠及全部
                // provider。429 默认 retry_429=false 不经此分支（其退避在应用层 map_api_error）。
                let delay = match &err {
                    TransportError::Http { headers: Some(h), .. } => parse_retry_after_header(h)
                        .unwrap_or_else(|| backoff(policy.base_delay, attempt + 1)),
                    _ => backoff(policy.base_delay, attempt + 1),
                };
                sleep(delay).await;
            }
            Err(err) => return Err(err),
        }
    }
    Err(TransportError::RetryLimit)
}

#[cfg(test)]
mod tests {
    use super::*;
    use http::HeaderValue;
    use std::str::FromStr;

    fn headers(pairs: &[(&str, &str)]) -> HeaderMap {
        let mut h = HeaderMap::new();
        for (k, v) in pairs {
            h.insert(
                http::HeaderName::from_str(k).unwrap(),
                HeaderValue::from_str(v).unwrap(),
            );
        }
        h
    }

    #[test]
    fn parse_retry_after_seconds() {
        let h = headers(&[("retry-after", "30")]);
        assert_eq!(parse_retry_after_header(&h), Some(Duration::from_secs(30)));
    }

    #[test]
    fn parse_retry_after_millis() {
        let h = headers(&[("retry-after-ms", "500")]);
        assert_eq!(
            parse_retry_after_header(&h),
            Some(Duration::from_millis(500))
        );
    }

    #[test]
    fn parse_retry_after_ms_takes_precedence() {
        // retry-after-ms 优先于 retry-after（毫秒粒度更精细，部分兼容网关采用）
        let h = headers(&[("retry-after", "30"), ("retry-after-ms", "500")]);
        assert_eq!(
            parse_retry_after_header(&h),
            Some(Duration::from_millis(500))
        );
    }

    #[test]
    fn parse_retry_after_missing_returns_none() {
        assert_eq!(parse_retry_after_header(&HeaderMap::new()), None);
    }

    #[test]
    fn parse_retry_after_invalid_value_returns_none() {
        // 非整数（含 HTTP-date 串，本设计不支持）→ None
        let h = headers(&[("retry-after", "abc")]);
        assert_eq!(parse_retry_after_header(&h), None);
    }

    #[test]
    fn parse_retry_after_case_insensitive() {
        // HeaderMap 名大小写不敏感（http crate 内部归一为小写）
        let h = headers(&[("Retry-After", "30")]);
        assert_eq!(parse_retry_after_header(&h), Some(Duration::from_secs(30)));
    }
}
