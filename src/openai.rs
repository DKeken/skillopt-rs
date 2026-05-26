use anyhow::{anyhow, bail, Context, Result};
use governor::{clock::DefaultClock, state::{InMemoryState, NotKeyed}, Quota, RateLimiter};
use nonzero_ext::nonzero;
use reqwest::Client;
use serde::{Deserialize, Serialize};
use std::sync::Arc;
use std::time::Duration;

use crate::types::ChatMessage;

type Limiter = RateLimiter<NotKeyed, InMemoryState, DefaultClock>;

#[derive(Clone)]
pub struct OpenAIClient {
    http: Client,
    base_url: String,
    api_key: String,
    pub target_model: String,
    pub optimizer_model: String,
    max_retries: u32,
    limiter: Option<Arc<Limiter>>,
}

#[derive(Serialize)]
struct ChatReq<'a> {
    model: &'a str,
    messages: &'a [ChatMessage],
    stream: bool,
    #[serde(skip_serializing_if = "Option::is_none")]
    temperature: Option<f32>,
    #[serde(skip_serializing_if = "Option::is_none")]
    max_tokens: Option<u32>,
    #[serde(skip_serializing_if = "Option::is_none")]
    response_format: Option<ResponseFormat>,
    #[serde(skip_serializing_if = "Option::is_none")]
    reasoning_effort: Option<&'a str>,
}

#[derive(Serialize)]
struct ResponseFormat {
    #[serde(rename = "type")]
    kind: String,
}

#[derive(Deserialize)]
struct ChatResp {
    choices: Vec<Choice>,
    #[serde(default)]
    usage: Option<Usage>,
}

#[derive(Deserialize)]
struct Choice {
    message: ChoiceMessage,
}

#[derive(Deserialize)]
struct ChoiceMessage {
    content: Option<String>,
}

#[derive(Deserialize, Default, Clone, Copy, Debug)]
pub struct Usage {
    #[serde(default)]
    pub prompt_tokens: u32,
    #[serde(default)]
    pub completion_tokens: u32,
    #[serde(default)]
    pub total_tokens: u32,
}

pub struct ChatOptions<'a> {
    pub temperature: Option<f32>,
    pub max_tokens: Option<u32>,
    pub json: bool,
    pub reasoning_effort: Option<&'a str>,
}

impl<'a> Default for ChatOptions<'a> {
    fn default() -> Self {
        Self { temperature: None, max_tokens: None, json: false, reasoning_effort: None }
    }
}

pub struct ChatOutcome {
    pub text: String,
    pub usage: Usage,
}

impl OpenAIClient {
    pub fn from_env() -> Result<Self> {
        let base_url = std::env::var("OPENAI_BASE_URL")
            .map(|s| s.trim_end_matches('/').to_string())
            .map_err(|_| anyhow!("OPENAI_BASE_URL must be set (e.g. http://localhost:8787/v1)"))?;
        let api_key = std::env::var("OPENAI_API_KEY")
            .unwrap_or_else(|_| "sk-noauth".into());
        let default_model = std::env::var("OPENAI_MODEL").unwrap_or_else(|_| "gpt-5.5".into());
        let target_model = std::env::var("OPENAI_TARGET_MODEL").unwrap_or_else(|_| default_model.clone());
        let optimizer_model = std::env::var("OPENAI_OPTIMIZER_MODEL").unwrap_or(default_model);
        let timeout = std::env::var("OPENAI_TIMEOUT_SECS").ok()
            .and_then(|s| s.parse().ok())
            .unwrap_or(180u64);
        let max_retries = std::env::var("OPENAI_MAX_RETRIES").ok()
            .and_then(|s| s.parse().ok())
            .unwrap_or(4u32);
        let rps = std::env::var("OPENAI_RPS").ok().and_then(|s| s.parse::<u32>().ok());
        let limiter = rps.and_then(|n| std::num::NonZeroU32::new(n))
            .map(|nz| Arc::new(RateLimiter::direct(Quota::per_second(nz))))
            .or_else(|| {
                // sensible default: cap at 30 req/s to be 9router-friendly under workers=24
                Some(Arc::new(RateLimiter::direct(Quota::per_second(nonzero!(30u32)))))
            });
        let http = Client::builder()
            .timeout(Duration::from_secs(timeout))
            .build()?;
        Ok(Self { http, base_url, api_key, target_model, optimizer_model, max_retries, limiter })
    }

    pub async fn chat(&self, model: &str, messages: &[ChatMessage], opts: ChatOptions<'_>) -> Result<ChatOutcome> {
        if let Some(lim) = &self.limiter {
            lim.until_ready().await;
        }
        let url = format!("{}/chat/completions", self.base_url);
        let body = ChatReq {
            model,
            messages,
            stream: false,
            temperature: opts.temperature,
            max_tokens: opts.max_tokens,
            response_format: opts.json.then(|| ResponseFormat { kind: "json_object".into() }),
            reasoning_effort: opts.reasoning_effort,
        };

        let mut last_err: Option<anyhow::Error> = None;
        for attempt in 0..=self.max_retries {
            let req = self.http.post(&url)
                .bearer_auth(&self.api_key)
                .json(&body);
            match req.send().await {
                Ok(r) if r.status().is_success() => {
                    let parsed: ChatResp = r.json().await.context("decode chat response")?;
                    let text = parsed.choices.into_iter().next()
                        .and_then(|c| c.message.content)
                        .unwrap_or_default();
                    return Ok(ChatOutcome { text, usage: parsed.usage.unwrap_or_default() });
                }
                Ok(r) => {
                    let status = r.status();
                    let body_text = r.text().await.unwrap_or_default();
                    last_err = Some(anyhow!("{} {}", status, body_text));
                    if !status.is_server_error() && status.as_u16() != 429 {
                        break;
                    }
                }
                Err(e) => { last_err = Some(e.into()); }
            }
            let backoff_ms = 250u64 * (1u64 << attempt.min(5));
            tokio::time::sleep(Duration::from_millis(backoff_ms)).await;
        }
        bail!(last_err.unwrap_or_else(|| anyhow!("chat failed")));
    }
}
