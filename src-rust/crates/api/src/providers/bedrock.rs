// providers/bedrock.rs — Amazon Bedrock provider adapter.
//
// Uses the Bedrock Converse Streaming API which accepts a unified message
// format similar to Anthropic's, making it straightforward to map from
// our internal ProviderRequest.
//
// Endpoint:
//   POST https://bedrock-runtime.{region}.amazonaws.com/model/{model_id}/converse-stream
//
// Auth:
//   - If AWS_ACCESS_KEY_ID and AWS_SECRET_ACCESS_KEY are set: AWS SigV4
//   - Otherwise, if AWS_BEARER_TOKEN_BEDROCK is set: Authorization: Bearer <token>
//
// Only Claude models on Bedrock are officially supported by this adapter.

use std::pin::Pin;

use async_stream::stream;
use async_trait::async_trait;
use claurst_core::provider_id::{ModelId, ProviderId};
use claurst_core::types::{ContentBlock, MessageContent, Role, ToolResultContent, UsageInfo};
use futures::Stream;
use serde_json::{json, Value};
use tracing::debug;

use crate::error_handling::parse_error_response;
use crate::provider::{LlmProvider, ModelInfo};
use crate::provider_error::ProviderError;
use crate::provider_types::{
    ProviderCapabilities, ProviderRequest, ProviderResponse, ProviderStatus, StopReason,
    StreamEvent, SystemPrompt, SystemPromptStyle,
};

use super::message_normalization::remove_empty_messages;
use super::request_options::merge_bedrock_options;

// ---------------------------------------------------------------------------
// BedrockProvider
// ---------------------------------------------------------------------------

pub struct BedrockProvider {
    id: ProviderId,
    region: String,
    http_client: reqwest::Client,
    access_key_id: Option<String>,
    secret_access_key: Option<String>,
    session_token: Option<String>,
    bearer_token: Option<String>,
}

impl BedrockProvider {
    pub fn from_env() -> Option<Self> {
        let region = std::env::var("AWS_REGION")
            .or_else(|_| std::env::var("AWS_DEFAULT_REGION"))
            .unwrap_or_else(|_| "us-east-1".to_string());

        let http_client = reqwest::Client::builder()
            .timeout(crate::request_timeout())
            .build()
            .expect("failed to build reqwest client");

        let auth = Self::auth_from_env()?;

        Some(Self {
            id: ProviderId::new(ProviderId::AMAZON_BEDROCK),
            region,
            http_client,
            access_key_id: auth.access_key_id,
            secret_access_key: auth.secret_access_key,
            session_token: auth.session_token,
            bearer_token: auth.bearer_token,
        })
    }

    /// Build a provider from explicit credentials and region.
    /// Used by `provider_from_config` when `provider_configs["amazon-bedrock"]`
    /// specifies a region and the credentials come from env vars.
    pub fn from_env_with_region(region: String) -> Option<Self> {
        let http_client = reqwest::Client::builder()
            .timeout(std::time::Duration::from_secs(600))
            .build()
            .expect("failed to build reqwest client");

        let auth = Self::auth_from_env()?;

        Some(Self {
            id: ProviderId::new(ProviderId::AMAZON_BEDROCK),
            region,
            http_client,
            access_key_id: auth.access_key_id,
            secret_access_key: auth.secret_access_key,
            session_token: auth.session_token,
            bearer_token: auth.bearer_token,
        })
    }

    fn auth_from_env() -> Option<BedrockAuth> {
        Self::auth_from_values(
            non_empty_env("AWS_ACCESS_KEY_ID"),
            non_empty_env("AWS_SECRET_ACCESS_KEY"),
            non_empty_env("AWS_SESSION_TOKEN"),
            non_empty_env("AWS_BEARER_TOKEN_BEDROCK"),
        )
    }

    fn auth_from_values(
        access_key_id: Option<String>,
        secret_access_key: Option<String>,
        session_token: Option<String>,
        bearer_token: Option<String>,
    ) -> Option<BedrockAuth> {
        if let (Some(access_key_id), Some(secret_access_key)) = (access_key_id, secret_access_key) {
            return Some(BedrockAuth {
                access_key_id: Some(access_key_id),
                secret_access_key: Some(secret_access_key),
                session_token,
                bearer_token: None,
            });
        }

        bearer_token.map(|token| BedrockAuth {
            access_key_id: None,
            secret_access_key: None,
            session_token: None,
            bearer_token: Some(token),
        })
    }

    /// Add a regional cross-inference prefix for models that support it.
    fn model_id_with_prefix(&self, model: &str) -> String {
        // Skip if already has a dot-separated prefix (e.g. "us.anthropic.claude-...")
        if model.contains('.') {
            return model.to_string();
        }
        let region = &self.region;
        if region.starts_with("us-") && !region.contains("gov") {
            if model.contains("claude") || model.contains("nova") {
                return format!("us.{}", model);
            }
        } else if region.starts_with("eu-") && model.contains("claude") {
            return format!("eu.{}", model);
        }
        model.to_string()
    }

    fn endpoint_url(&self, model_id: &str) -> String {
        format!(
            "https://bedrock-runtime.{}.amazonaws.com/model/{}/converse-stream",
            self.region,
            urlencoding::encode(model_id)
        )
    }

    // -----------------------------------------------------------------------
    // AWS SigV4 signing
    // -----------------------------------------------------------------------

    fn sign_request(
        &self,
        method: &str,
        url_str: &str,
        body: &str,
        date: &chrono::DateTime<chrono::Utc>,
    ) -> std::collections::HashMap<String, String> {
        use hmac::{Hmac, Mac};
        use sha2::{Digest, Sha256};

        type HmacSha256 = Hmac<Sha256>;

        let mut headers = std::collections::HashMap::new();

        // If we have a bearer token, skip SigV4.
        if let Some(ref token) = self.bearer_token {
            headers.insert("Authorization".to_string(), format!("Bearer {}", token));
            return headers;
        }

        let access_key = match &self.access_key_id {
            Some(k) => k.clone(),
            None => return headers,
        };
        let secret_key = match &self.secret_access_key {
            Some(s) => s.clone(),
            None => return headers,
        };

        let date_str = date.format("%Y%m%d").to_string();
        let datetime_str = date.format("%Y%m%dT%H%M%SZ").to_string();
        let service = "bedrock";
        let region = &self.region;

        // Parse path and query from URL.
        let parsed = url::Url::parse(url_str).unwrap_or_else(|_| {
            url::Url::parse("https://bedrock-runtime.us-east-1.amazonaws.com/").unwrap()
        });
        let canonical_uri = sigv4_canonical_uri(parsed.path());
        let canonical_query = parsed.query().unwrap_or("").to_string();

        // Body hash.
        let body_hash = hex::encode(Sha256::digest(body.as_bytes()));

        // Canonical headers (must be sorted, lowercased).
        let host = parsed.host_str().unwrap_or_default().to_string();
        let content_type = "application/json";

        // Build canonical headers string and signed headers list.
        // Include: content-type, host, x-amz-content-sha256, x-amz-date,
        // and optionally x-amz-security-token.
        let mut canonical_headers = format!(
            "content-type:{}\nhost:{}\nx-amz-content-sha256:{}\nx-amz-date:{}\n",
            content_type, host, body_hash, datetime_str
        );
        let mut signed_headers =
            "content-type;host;x-amz-content-sha256;x-amz-date".to_string();

        if let Some(ref tok) = self.session_token {
            canonical_headers.push_str(&format!("x-amz-security-token:{}\n", tok));
            signed_headers.push_str(";x-amz-security-token");
        }

        // Canonical request.
        let canonical_request = format!(
            "{}\n{}\n{}\n{}\n{}\n{}",
            method,
            canonical_uri,
            canonical_query,
            canonical_headers,
            signed_headers,
            body_hash
        );

        // String to sign.
        let credential_scope =
            format!("{}/{}/{}/aws4_request", date_str, region, service);
        let canonical_request_hash =
            hex::encode(Sha256::digest(canonical_request.as_bytes()));
        let string_to_sign = format!(
            "AWS4-HMAC-SHA256\n{}\n{}\n{}",
            datetime_str, credential_scope, canonical_request_hash
        );

        // Signing key: HMAC-SHA256 chain.
        let sign_key = {
            let k_date = {
                let mut mac = HmacSha256::new_from_slice(
                    format!("AWS4{}", secret_key).as_bytes(),
                )
                .expect("HMAC init failed");
                mac.update(date_str.as_bytes());
                mac.finalize().into_bytes()
            };
            let k_region = {
                let mut mac = HmacSha256::new_from_slice(&k_date)
                    .expect("HMAC init failed");
                mac.update(region.as_bytes());
                mac.finalize().into_bytes()
            };
            let k_service = {
                let mut mac = HmacSha256::new_from_slice(&k_region)
                    .expect("HMAC init failed");
                mac.update(service.as_bytes());
                mac.finalize().into_bytes()
            };
            let k_signing = {
                let mut mac = HmacSha256::new_from_slice(&k_service)
                    .expect("HMAC init failed");
                mac.update(b"aws4_request");
                mac.finalize().into_bytes()
            };
            k_signing
        };

        let signature = {
            let mut mac =
                HmacSha256::new_from_slice(&sign_key).expect("HMAC init failed");
            mac.update(string_to_sign.as_bytes());
            hex::encode(mac.finalize().into_bytes())
        };

        let authorization = format!(
            "AWS4-HMAC-SHA256 Credential={}/{}, SignedHeaders={}, Signature={}",
            access_key, credential_scope, signed_headers, signature
        );

        headers.insert("Authorization".to_string(), authorization);
        headers.insert("x-amz-date".to_string(), datetime_str);
        headers.insert("x-amz-content-sha256".to_string(), body_hash);
        if let Some(ref tok) = self.session_token {
            headers.insert("x-amz-security-token".to_string(), tok.clone());
        }

        headers
    }

    // -----------------------------------------------------------------------
    // Request body builders
    // -----------------------------------------------------------------------

    fn build_converse_body(request: &ProviderRequest) -> Value {
        let prompt_cache = if Self::model_supports_prompt_caching(&request.model) {
            BedrockPromptCachingOptions::from_provider_options(&request.provider_options)
        } else {
            // `promptCaching` is a Claurst policy request, not a field Bedrock
            // accepts directly. Only turn it into Converse `cachePoint` blocks
            // for model families where Bedrock documents that support; open
            // model adapters like Qwen reject cache points and should keep the
            // turn working without requiring users to rewrite shared settings.
            BedrockPromptCachingOptions::disabled()
        };
        let messages = Self::build_converse_messages(request, &prompt_cache);
        let mut body = json!({
            "messages": messages,
            "inferenceConfig": {
                "maxTokens": request.max_tokens,
                "temperature": request.temperature.unwrap_or(0.7),
                // topP omitted: Bedrock Claude rejects requests that specify
                // both temperature and topP simultaneously.
            }
        });
        if !request.stop_sequences.is_empty()
            && Self::model_supports_stop_sequences(&request.model)
        {
            body["inferenceConfig"]["stopSequences"] = json!(request.stop_sequences);
        }

        // System prompt.
        if let Some(sys) = &request.system_prompt {
            let sys_text = match sys {
                SystemPrompt::Text(t) => t.clone(),
                SystemPrompt::Blocks(blocks) => blocks
                    .iter()
                    .map(|b| b.text.clone())
                    .collect::<Vec<_>>()
                    .join("\n"),
            };
            let mut system = vec![json!({ "text": sys_text })];
            if prompt_cache.cache_system {
                // System instructions are usually stable across an agent
                // session, so this is the first high-value cache boundary.
                // Bedrock counts cache points across tools -> system ->
                // messages; putting the marker here preserves the stable
                // prefix without guessing about user-message volatility.
                system.push(prompt_cache.cache_point());
            }
            body["system"] = json!(system);
        }

        // Tool definitions.
        if !request.tools.is_empty() && Self::model_supports_tool_config(&request.model) {
            let mut tool_specs: Vec<Value> = request
                .tools
                .iter()
                .map(|td| {
                    json!({
                        "toolSpec": {
                            "name": td.name,
                            "description": td.description,
                            "inputSchema": {
                                "json": td.input_schema
                            }
                        }
                    })
                })
                .collect();
            if prompt_cache.cache_tools {
                // Tool schemas are another stable prefix for coding agents.
                // Caching them is especially useful before Knowledge Base and
                // routing work adds more fixed tool definitions to each turn.
                tool_specs.push(prompt_cache.cache_point());
            }
            body["toolConfig"] = json!({ "tools": tool_specs });
        }

        if let Some(thinking) = &request.thinking {
            if !Self::model_supports_reasoning_config(&request.model) {
                // Qwen/DeepSeek/Nova Bedrock models reject Claude-style
                // reasoningConfig. Preserve provider_options so callers can
                // still pass family-specific Bedrock fields, but do not send
                // the incompatible reasoning block.
                merge_bedrock_options(&mut body, &request.provider_options);
                return body;
            }
            body["reasoningConfig"] = json!({
                "type": "enabled",
                "budgetTokens": thinking.budget_tokens,
            });
        }

        merge_bedrock_options(&mut body, &request.provider_options);

        body
    }

    fn build_converse_messages(
        request: &ProviderRequest,
        prompt_cache: &BedrockPromptCachingOptions,
    ) -> Vec<Value> {
        remove_empty_messages(&request.messages)
            .iter()
            .map(|msg| {
                let role = match msg.role {
                    Role::User => "user",
                    Role::Assistant => "assistant",
                };
                let mut content = Self::message_content_to_converse(&msg.content, &msg.role);
                if prompt_cache.cache_messages && !content.is_empty() {
                    // Message checkpoints are opt-in because coding turns
                    // usually change at the tail. This keeps the default cache
                    // policy focused on stable tools/system content while
                    // still allowing long static prompt prefixes to be cached.
                    content.push(prompt_cache.cache_point());
                }
                json!({ "role": role, "content": content })
            })
            .collect()
    }

    fn model_supports_stop_sequences(model: &str) -> bool {
        let model = model.to_ascii_lowercase();
        // Converse accepts stopSequences generically, but several Bedrock
        // open-model adapters reject the field. Keep it only for Claude-family
        // models where it is known to be accepted.
        model.contains("anthropic") || model.contains("claude")
    }

    fn model_supports_tool_config(model: &str) -> bool {
        let model = model.to_ascii_lowercase();
        // This is deliberately allow-listed by family. DeepSeek currently
        // fails better when toolConfig is absent, while Qwen and Nova need the
        // tool schema preserved to perform real agent/tool turns.
        model.contains("anthropic")
            || model.contains("claude")
            || model.contains("nova")
            || model.contains("qwen")
    }

    fn model_supports_reasoning_config(model: &str) -> bool {
        let model = model.to_ascii_lowercase();
        // `reasoningConfig` is the Bedrock Claude thinking surface. Other
        // families expose reasoning differently or reject the field outright.
        model.contains("anthropic") || model.contains("claude")
    }

    fn model_supports_prompt_caching(model: &str) -> bool {
        let model = model.to_ascii_lowercase();
        // Bedrock prompt caching is model-family specific. Claude supports
        // explicit cache points across tools/system/messages. Nova also has
        // Bedrock prompt-caching support, including automatic caching and
        // explicit cache points. Qwen, DeepSeek, and Llama currently reject
        // cache points through Converse, so leave the policy inert for them.
        model.contains("anthropic") || model.contains("claude") || model.contains("nova")
    }

    fn message_content_to_converse(content: &MessageContent, role: &Role) -> Vec<Value> {
        match content {
            MessageContent::Text(t) => vec![json!({ "text": t })],
            MessageContent::Blocks(blocks) => blocks
                .iter()
                .filter_map(|b| Self::content_block_to_converse(b, role))
                .collect(),
        }
    }

    fn content_block_to_converse(block: &ContentBlock, role: &Role) -> Option<Value> {
        match block {
            ContentBlock::Text { text } => Some(json!({ "text": text })),
            ContentBlock::Image { source } => {
                // Bedrock Converse image format.
                let media_type = source
                    .media_type
                    .as_deref()
                    .unwrap_or("image/png")
                    .replace("image/", "");
                if let Some(data) = &source.data {
                    Some(json!({
                        "image": {
                            "format": media_type,
                            "source": {
                                "bytes": data
                            }
                        }
                    }))
                } else if let Some(url) = &source.url {
                    // Bedrock doesn't support URL images natively; skip.
                    debug!("Bedrock does not support URL images: {}", url);
                    None
                } else {
                    None
                }
            }
            ContentBlock::ToolUse { id, name, input } => Some(json!({
                "toolUse": {
                    "toolUseId": id,
                    "name": name,
                    "input": input
                }
            })),
            ContentBlock::ToolResult {
                tool_use_id,
                content,
                is_error,
            } => {
                let result_content = match content {
                    ToolResultContent::Text(t) => vec![json!({ "text": t })],
                    ToolResultContent::Blocks(inner) => inner
                        .iter()
                        .filter_map(|b| Self::content_block_to_converse(b, role))
                        .collect(),
                };
                let status = if is_error.unwrap_or(false) {
                    "error"
                } else {
                    "success"
                };
                Some(json!({
                    "toolResult": {
                        "toolUseId": tool_use_id,
                        "content": result_content,
                        "status": status
                    }
                }))
            }
            ContentBlock::Thinking { .. } => {
                // Bedrock Converse has no provider-neutral thinking block.
                // Serializing local thinking as plain text leaks scratchpad
                // content into follow-up turns, which is especially visible
                // with Qwen-style `<think>` output. Keep the block for local
                // transcript display, but do not round-trip it to Bedrock.
                None
            }
            _ => None,
        }
    }

    // -----------------------------------------------------------------------
    // HTTP helpers
    // -----------------------------------------------------------------------

    fn map_http_error(&self, status: u16, body: &str) -> ProviderError {
        parse_error_response(status, body, &self.id)
    }

    // -----------------------------------------------------------------------
    // Send helpers
    // -----------------------------------------------------------------------

    async fn send_streaming(
        &self,
        request: &ProviderRequest,
    ) -> Result<reqwest::Response, ProviderError> {
        let bedrock_model = self.model_id_with_prefix(&request.model);
        let url = self.endpoint_url(&bedrock_model);

        let body = Self::build_converse_body(request);
        let body_str = serde_json::to_string(&body).unwrap_or_default();

        let now = chrono::Utc::now();
        let auth_headers = self.sign_request("POST", &url, &body_str, &now);

        let mut req_builder = self
            .http_client
            .post(&url)
            .header("Content-Type", "application/json")
            .header("Accept", "application/vnd.amazon.eventstream");

        for (k, v) in &auth_headers {
            req_builder = req_builder.header(k.as_str(), v.as_str());
        }

        let resp = req_builder
            .body(body_str)
            .send()
            .await
            .map_err(|e| ProviderError::Other {
                provider: self.id.clone(),
                message: format!("HTTP request failed: {}", e),
                status: None,
                body: None,
            })?;

        let status = resp.status().as_u16();
        if !(200..300).contains(&(status as usize)) {
            let text = resp.text().await.unwrap_or_default();
            return Err(self.map_http_error(status, &text));
        }

        Ok(resp)
    }

    async fn send_non_streaming(
        &self,
        request: &ProviderRequest,
    ) -> Result<ProviderResponse, ProviderError> {
        let bedrock_model = self.model_id_with_prefix(&request.model);
        // Non-streaming uses /converse (not /converse-stream)
        let url = format!(
            "https://bedrock-runtime.{}.amazonaws.com/model/{}/converse",
            self.region,
            urlencoding::encode(&bedrock_model)
        );

        let body = Self::build_converse_body(request);
        let body_str = serde_json::to_string(&body).unwrap_or_default();

        let now = chrono::Utc::now();
        let auth_headers = self.sign_request("POST", &url, &body_str, &now);

        let mut req_builder = self
            .http_client
            .post(&url)
            .header("Content-Type", "application/json");

        for (k, v) in &auth_headers {
            req_builder = req_builder.header(k.as_str(), v.as_str());
        }

        let resp = req_builder
            .body(body_str)
            .send()
            .await
            .map_err(|e| ProviderError::Other {
                provider: self.id.clone(),
                message: format!("HTTP request failed: {}", e),
                status: None,
                body: None,
            })?;

        let status = resp.status().as_u16();
        let text = resp.text().await.map_err(|e| ProviderError::Other {
            provider: self.id.clone(),
            message: format!("Failed to read response body: {}", e),
            status: Some(status),
            body: None,
        })?;

        if !(200..300).contains(&(status as usize)) {
            return Err(self.map_http_error(status, &text));
        }

        let json_val: Value = serde_json::from_str(&text).map_err(|e| ProviderError::Other {
            provider: self.id.clone(),
            message: format!("Failed to parse response JSON: {}", e),
            status: Some(status),
            body: Some(text.clone()),
        })?;

        Self::parse_converse_response(&json_val, &self.id)
    }

    fn parse_converse_response(
        json: &Value,
        provider_id: &ProviderId,
    ) -> Result<ProviderResponse, ProviderError> {
        // Bedrock Converse non-streaming response shape:
        // { "output": { "message": { "role": "assistant", "content": [...] } },
        //   "stopReason": "end_turn",
        //   "usage": { "inputTokens": N, "outputTokens": M } }

        let message = json
            .get("output")
            .and_then(|o| o.get("message"))
            .ok_or_else(|| ProviderError::Other {
                provider: provider_id.clone(),
                message: "No output.message in Bedrock response".to_string(),
                status: None,
                body: None,
            })?;

        let content_blocks = Self::parse_converse_content(
            message.get("content").and_then(|c| c.as_array()),
        );

        let stop_reason_str = json
            .get("stopReason")
            .and_then(|v| v.as_str())
            .unwrap_or("end_turn");
        let stop_reason = Self::map_stop_reason(stop_reason_str);

        let usage = Self::parse_converse_usage(json.get("usage"));

        Ok(ProviderResponse {
            id: uuid::Uuid::new_v4().to_string(),
            content: content_blocks,
            stop_reason,
            usage,
            model: json
                .get("model")
                .and_then(|v| v.as_str())
                .unwrap_or("")
                .to_string(),
        })
    }

    fn parse_converse_content(content: Option<&Vec<Value>>) -> Vec<ContentBlock> {
        let blocks = match content {
            Some(b) => b,
            None => return vec![],
        };

        let mut content_blocks = Vec::new();
        for b in blocks {
            if let Some(text) = b.get("text").and_then(|v| v.as_str()) {
                content_blocks.extend(split_tagged_thinking_content(text));
                continue;
            }
            if let Some(tu) = b.get("toolUse") {
                let id = tu
                    .get("toolUseId")
                    .and_then(|v| v.as_str())
                    .unwrap_or("")
                    .to_string();
                let name = tu
                    .get("name")
                    .and_then(|v| v.as_str())
                    .unwrap_or("")
                    .to_string();
                let input = tu.get("input").cloned().unwrap_or(json!({}));
                content_blocks.push(ContentBlock::ToolUse { id, name, input });
            }
        }
        content_blocks
    }

    fn map_stop_reason(reason: &str) -> StopReason {
        match reason {
            "end_turn" => StopReason::EndTurn,
            "max_tokens" => StopReason::MaxTokens,
            "tool_use" => StopReason::ToolUse,
            "stop_sequence" => StopReason::StopSequence,
            "content_filtered" => StopReason::ContentFiltered,
            other => StopReason::Other(other.to_string()),
        }
    }

    fn parse_converse_usage(usage: Option<&Value>) -> UsageInfo {
        let u = match usage {
            Some(v) => v,
            None => return UsageInfo::default(),
        };
        parse_bedrock_usage_info(u)
    }
}

#[derive(Debug, Clone)]
struct BedrockPromptCachingOptions {
    cache_tools: bool,
    cache_system: bool,
    cache_messages: bool,
    ttl: Option<String>,
}

impl BedrockPromptCachingOptions {
    fn disabled() -> Self {
        Self {
            cache_tools: false,
            cache_system: false,
            cache_messages: false,
            ttl: None,
        }
    }

    fn from_provider_options(provider_options: &Value) -> Self {
        let Some(value) = provider_options.get("promptCaching") else {
            return Self::disabled();
        };
        if value.as_bool() == Some(false) {
            return Self::disabled();
        }

        let mut options = Self {
            cache_tools: true,
            cache_system: true,
            cache_messages: false,
            ttl: None,
        };

        if value.as_bool() == Some(true) {
            return options;
        }

        let Some(obj) = value.as_object() else {
            return Self::disabled();
        };
        if obj.get("enabled").and_then(Value::as_bool) == Some(false) {
            return Self::disabled();
        }
        options.cache_tools = obj
            .get("tools")
            .or_else(|| obj.get("cacheTools"))
            .and_then(Value::as_bool)
            .unwrap_or(true);
        options.cache_system = obj
            .get("system")
            .or_else(|| obj.get("cacheSystem"))
            .and_then(Value::as_bool)
            .unwrap_or(true);
        options.cache_messages = obj
            .get("messages")
            .or_else(|| obj.get("cacheMessages"))
            .and_then(Value::as_bool)
            .unwrap_or(false);
        options.ttl = obj
            .get("ttl")
            .and_then(Value::as_str)
            .filter(|ttl| matches!(*ttl, "5m" | "1h"))
            .map(str::to_string);
        options
    }

    fn cache_point(&self) -> Value {
        let mut cache_point = json!({ "type": "default" });
        if let Some(ttl) = self.ttl.as_deref() {
            cache_point["ttl"] = json!(ttl);
        }
        json!({ "cachePoint": cache_point })
    }
}

fn split_tagged_thinking_content(text: &str) -> Vec<ContentBlock> {
    // Some Bedrock-hosted open models expose reasoning as tagged text rather
    // than a native Converse content type. Normalize that provider convention
    // into Claurst's internal Thinking block so the TUI can collapse it and
    // future Bedrock requests do not receive it back as ordinary assistant
    // prose. Add new Bedrock family conventions here rather than in the TUI.
    let mut state = BedrockTaggedThinkingState::default();
    let mut blocks = Vec::new();
    for event in state.push_text(0, text) {
        push_tagged_thinking_block(&mut blocks, event);
    }
    for event in state.finish() {
        push_tagged_thinking_block(&mut blocks, event);
    }
    blocks
}

fn push_tagged_thinking_block(blocks: &mut Vec<ContentBlock>, event: StreamEvent) {
    match event {
        StreamEvent::TextDelta { text, .. } if !text.is_empty() => {
            blocks.push(ContentBlock::Text { text });
        }
        StreamEvent::ThinkingDelta { thinking, .. } if !thinking.is_empty() => {
            blocks.push(ContentBlock::Thinking {
                thinking,
                signature: String::new(),
            });
        }
        _ => {}
    }
}

#[derive(Default)]
struct BedrockTaggedThinkingState {
    pending: String,
    in_thinking: bool,
    last_index: usize,
}

impl BedrockTaggedThinkingState {
    fn push_text(&mut self, index: usize, text: &str) -> Vec<StreamEvent> {
        self.last_index = index;
        self.pending.push_str(text);
        self.drain(index, false)
    }

    fn finish(&mut self) -> Vec<StreamEvent> {
        self.drain(self.last_index, true)
    }

    fn drain(&mut self, index: usize, finish: bool) -> Vec<StreamEvent> {
        const OPEN_TAGS: &[&str] = &["<think>", "<thinking>"];
        const CLOSE_TAGS: &[&str] = &["</think>", "</thinking>"];

        let mut events = Vec::new();
        loop {
            let tags = if self.in_thinking { CLOSE_TAGS } else { OPEN_TAGS };
            if let Some((pos, tag_len)) = find_first_tag(&self.pending, tags) {
                let before = self.pending[..pos].to_string();
                self.pending.drain(..pos + tag_len);
                self.push_delta(&mut events, index, before);
                self.in_thinking = !self.in_thinking;
                continue;
            }

            let flush_len = if finish {
                self.pending.len()
            } else {
                safe_tag_flush_len(&self.pending, tags)
            };
            if flush_len > 0 {
                let text = self.pending[..flush_len].to_string();
                self.pending.drain(..flush_len);
                self.push_delta(&mut events, index, text);
            }
            break;
        }

        if finish {
            self.in_thinking = false;
        }

        events
    }

    fn push_delta(&self, events: &mut Vec<StreamEvent>, index: usize, text: String) {
        if text.is_empty() {
            return;
        }
        if self.in_thinking {
            events.push(StreamEvent::ThinkingDelta {
                index,
                thinking: text,
            });
        } else {
            events.push(StreamEvent::TextDelta { index, text });
        }
    }
}

fn find_first_tag(input: &str, tags: &[&str]) -> Option<(usize, usize)> {
    let lower = input.to_ascii_lowercase();
    tags.iter()
        .filter_map(|tag| lower.find(tag).map(|pos| (pos, tag.len())))
        .min_by_key(|(pos, _)| *pos)
}

fn safe_tag_flush_len(input: &str, tags: &[&str]) -> usize {
    if input.is_empty() {
        return 0;
    }
    let max_keep = tags
        .iter()
        .map(|tag| tag.len().saturating_sub(1))
        .max()
        .unwrap_or(0)
        .min(input.len());
    let lower = input.to_ascii_lowercase();
    for keep in (1..=max_keep).rev() {
        let start = input.len().saturating_sub(keep);
        if !input.is_char_boundary(start) {
            continue;
        }
        let suffix = &lower[start..];
        if tags.iter().any(|tag| tag.starts_with(suffix)) {
            return start;
        }
    }
    input.len()
}

fn parse_bedrock_usage_info(u: &Value) -> UsageInfo {
    // Bedrock prompt caching splits input accounting: `inputTokens` is only
    // uncached input, while cache reads and writes are separate billable
    // counters. Mapping them into Claurst's generic cache fields keeps the TUI
    // context/cost meter accurate for Bedrock without changing other providers.
    UsageInfo {
        input_tokens: u
            .get("inputTokens")
            .and_then(|v| v.as_u64())
            .unwrap_or(0),
        output_tokens: u
            .get("outputTokens")
            .and_then(|v| v.as_u64())
            .unwrap_or(0),
        cache_creation_input_tokens: u
            .get("cacheWriteInputTokens")
            .or_else(|| u.get("cacheCreationInputTokens"))
            .and_then(|v| v.as_u64())
            .unwrap_or(0),
        cache_read_input_tokens: u
            .get("cacheReadInputTokens")
            .and_then(|v| v.as_u64())
            .unwrap_or(0),
    }
}

struct BedrockAuth {
    access_key_id: Option<String>,
    secret_access_key: Option<String>,
    session_token: Option<String>,
    bearer_token: Option<String>,
}

fn non_empty_env(name: &str) -> Option<String> {
    std::env::var(name).ok().filter(|value| !value.trim().is_empty())
}

fn sigv4_canonical_uri(path: &str) -> String {
    if path.is_empty() {
        "/".to_string()
    } else {
        aws_uri_encode(path, false)
    }
}

fn aws_uri_encode(value: &str, encode_slash: bool) -> String {
    let mut encoded = String::with_capacity(value.len());
    for byte in value.bytes() {
        match byte {
            b'A'..=b'Z'
            | b'a'..=b'z'
            | b'0'..=b'9'
            | b'-'
            | b'.'
            | b'_'
            | b'~' => encoded.push(byte as char),
            b'/' if !encode_slash => encoded.push('/'),
            _ => encoded.push_str(&format!("%{:02X}", byte)),
        }
    }
    encoded
}

// ---------------------------------------------------------------------------
// LlmProvider impl
// ---------------------------------------------------------------------------

#[async_trait]
impl LlmProvider for BedrockProvider {
    fn id(&self) -> &ProviderId {
        &self.id
    }

    fn name(&self) -> &str {
        "Amazon Bedrock"
    }

    async fn create_message(
        &self,
        request: ProviderRequest,
    ) -> Result<ProviderResponse, ProviderError> {
        self.send_non_streaming(&request).await
    }

    async fn create_message_stream(
        &self,
        request: ProviderRequest,
    ) -> Result<Pin<Box<dyn Stream<Item = Result<StreamEvent, ProviderError>> + Send>>, ProviderError>
    {
        let resp = self.send_streaming(&request).await?;
        let provider_id = self.id.clone();

        // Bedrock Converse streaming uses AWS EventStream binary framing.
        // For simplicity we parse the JSON chunks that appear within the
        // event payload bytes.  Each event is a binary-framed blob containing
        // a JSON payload under the ":event-type" header.
        //
        // We fall back to text-based JSON parsing by scanning for JSON objects
        // in the raw bytes, which works reliably for the common text delta events.
        let s = stream! {
            use futures::StreamExt;

            let mut byte_stream = resp.bytes_stream();
            let mut buf: Vec<u8> = Vec::new();
            let mut message_started = false;
            let mut tagged_thinking = BedrockTaggedThinkingState::default();

            while let Some(chunk_result) = byte_stream.next().await {
                let chunk = match chunk_result {
                    Ok(c) => c,
                    Err(e) => {
                        yield Err(ProviderError::StreamError {
                            provider: provider_id.clone(),
                            message: format!("Stream read error: {}", e),
                            partial_response: None,
                        });
                        return;
                    }
                };

                buf.extend_from_slice(&chunk);

                // AWS EventStream binary framing:
                //   [4 total_len][4 headers_len][4 prelude_crc][headers...][payload...][4 msg_crc]
                // total_len includes all 16 framing bytes.
                // payload starts at byte (12 + headers_len) and ends at (total_len - 4).
                loop {
                    // Need at least 12 bytes for the prelude.
                    if buf.len() < 12 {
                        break;
                    }

                    let total_len = u32::from_be_bytes([buf[0], buf[1], buf[2], buf[3]]) as usize;
                    let headers_len = u32::from_be_bytes([buf[4], buf[5], buf[6], buf[7]]) as usize;

                    // Sanity check: reject obviously corrupt frames.
                    if total_len < 16 || total_len > 1_048_576 || headers_len > total_len {
                        buf.drain(..1);
                        continue;
                    }

                    // Wait until we have the full message.
                    if buf.len() < total_len {
                        break;
                    }

                    // Extract the payload (between headers and trailing CRC).
                    let payload_start = 12 + headers_len;
                    let payload_end = total_len - 4;

                    if payload_start <= payload_end {
                        // Parse the ":event-type" header from the binary headers block.
                        // Header wire format: [u8 name_len][name bytes][u8 type][...value...]
                        // type 7 = string: [u16 value_len][value bytes]
                        let event_type = extract_event_type(&buf[12..12 + headers_len]);

                        let payload = &buf[payload_start..payload_end];
                        if let Ok(val) = serde_json::from_slice::<Value>(payload) {
                            for ev in parse_bedrock_event(&val, event_type.as_deref(), &provider_id, &mut message_started, &mut tagged_thinking) {
                                yield ev;
                            }
                        }
                    }

                    // Consume the full message from the buffer.
                    buf.drain(..total_len);
                }
            }

            // Drain any remaining complete EventStream messages.
            loop {
                if buf.len() < 12 { break; }
                let total_len = u32::from_be_bytes([buf[0], buf[1], buf[2], buf[3]]) as usize;
                let headers_len = u32::from_be_bytes([buf[4], buf[5], buf[6], buf[7]]) as usize;
                if total_len < 16 || total_len > 1_048_576 || headers_len > total_len { break; }
                if buf.len() < total_len { break; }
                let payload_start = 12 + headers_len;
                let payload_end = total_len - 4;
                if payload_start <= payload_end {
                    let event_type = extract_event_type(&buf[12..12 + headers_len]);
                    let payload = &buf[payload_start..payload_end];
                    if let Ok(val) = serde_json::from_slice::<Value>(payload) {
                        for ev in parse_bedrock_event(&val, event_type.as_deref(), &provider_id, &mut message_started, &mut tagged_thinking) {
                            yield ev;
                        }
                    }
                }
                buf.drain(..total_len);
            }

            if message_started {
                for ev in tagged_thinking.finish() {
                    yield Ok(ev);
                }
                yield Ok(StreamEvent::MessageStop);
            }
        };

        Ok(Box::pin(s))
    }

    async fn discover_models(&self) -> Result<Vec<ModelInfo>, ProviderError> {
        Ok(vec![
            ModelInfo {
                id: ModelId::new("anthropic.claude-opus-4-6"),
                provider_id: self.id.clone(),
                name: "Claude Opus 4.6 (Bedrock)".to_string(),
                context_window: 200_000,
                max_output_tokens: 32_000,
                ..Default::default()
            },
            ModelInfo {
                id: ModelId::new("anthropic.claude-sonnet-4-6"),
                provider_id: self.id.clone(),
                name: "Claude Sonnet 4.6 (Bedrock)".to_string(),
                context_window: 200_000,
                max_output_tokens: 16_000,
                ..Default::default()
            },
            ModelInfo {
                id: ModelId::new("anthropic.claude-haiku-4-5-20251001"),
                provider_id: self.id.clone(),
                name: "Claude Haiku 4.5 (Bedrock)".to_string(),
                context_window: 200_000,
                max_output_tokens: 8_192,
                ..Default::default()
            },
        ])
    }

    async fn health_check(&self) -> Result<ProviderStatus, ProviderError> {
        // Lightweight check: GET the list-foundation-models endpoint.
        let url = format!(
            "https://bedrock.{}.amazonaws.com/foundation-models",
            self.region
        );
        let now = chrono::Utc::now();
        // For health check, sign an empty GET body.
        let auth_headers = self.sign_request("GET", &url, "", &now);

        let mut req_builder = self.http_client.get(&url);
        for (k, v) in &auth_headers {
            req_builder = req_builder.header(k.as_str(), v.as_str());
        }

        let resp = req_builder.send().await;
        match resp {
            Ok(r) if r.status().is_success() => Ok(ProviderStatus::Healthy),
            Ok(r) if r.status().as_u16() == 401 || r.status().as_u16() == 403 => {
                Ok(ProviderStatus::Unavailable {
                    reason: "authentication failed".to_string(),
                })
            }
            Ok(r) => Ok(ProviderStatus::Degraded {
                reason: format!("foundation-models returned {}", r.status()),
            }),
            Err(e) => Ok(ProviderStatus::Unavailable {
                reason: e.to_string(),
            }),
        }
    }

    fn capabilities(&self) -> ProviderCapabilities {
        ProviderCapabilities {
            streaming: true,
            tool_calling: true,
            thinking: true,
            image_input: true,
            pdf_input: true,
            audio_input: false,
            video_input: false,
            caching: true,
            structured_output: false,
            system_prompt_style: SystemPromptStyle::TopLevel,
        }
    }
}

// ---------------------------------------------------------------------------
// Bedrock event parsing helper (free function so it can be used in stream!)
// ---------------------------------------------------------------------------

/// Extract the ":event-type" value from an AWS EventStream binary headers block.
/// Header wire format: [u8 name_len][name bytes][u8 type][...value...]
/// type 7 = string: [u16 value_len][value bytes]
fn extract_event_type(headers: &[u8]) -> Option<String> {
    let mut pos = 0;
    while pos < headers.len() {
        if pos >= headers.len() { break; }
        let name_len = headers[pos] as usize;
        pos += 1;
        if pos + name_len > headers.len() { break; }
        let name = &headers[pos..pos + name_len];
        pos += name_len;
        if pos >= headers.len() { break; }
        let htype = headers[pos];
        pos += 1;
        match htype {
            7 => {
                // String: [u16 value_len][value bytes]
                if pos + 2 > headers.len() { break; }
                let vlen = u16::from_be_bytes([headers[pos], headers[pos + 1]]) as usize;
                pos += 2;
                if pos + vlen > headers.len() { break; }
                let value = &headers[pos..pos + vlen];
                pos += vlen;
                if name == b":event-type" {
                    return String::from_utf8(value.to_vec()).ok();
                }
            }
            0 => {} // bool true
            1 => {} // bool false
            2 | 3 | 4 | 5 => { pos += 1; } // byte/short/int/long (simplified)
            6 => { pos += 8; } // timestamp
            8 => { // bytes
                if pos + 2 > headers.len() { break; }
                let vlen = u16::from_be_bytes([headers[pos], headers[pos + 1]]) as usize;
                pos += 2 + vlen;
            }
            _ => break,
        }
    }
    None
}

fn parse_bedrock_event(
    val: &Value,
    event_type: Option<&str>,
    provider_id: &ProviderId,
    message_started: &mut bool,
    tagged_thinking: &mut BedrockTaggedThinkingState,
) -> Vec<Result<StreamEvent, ProviderError>> {
    let mut events = Vec::new();

    // Bedrock Converse streaming events come in several shapes.
    // When event_type is provided (from EventStream headers), use it directly.
    // The payload fields are at the top level (not wrapped in the event-type key).

    // messageStart — flat payload: {"role":"assistant","p":"..."}
    // or wrapped: {"messageStart":{"role":"assistant"}}
    let is_message_start = event_type == Some("messageStart") || val.get("messageStart").is_some();
    if is_message_start {
        let msg_start = val.get("messageStart").unwrap_or(val);
        let role = msg_start
            .get("role")
            .and_then(|v| v.as_str())
            .unwrap_or("assistant");
        let _ = role;
        if !*message_started {
            events.push(Ok(StreamEvent::MessageStart {
                id: uuid::Uuid::new_v4().to_string(),
                model: String::new(),
                usage: UsageInfo::default(),
            }));
            *message_started = true;
        }
        return events;
    }

    // contentBlockStart — flat: {"contentBlockIndex":0,"start":{...},"p":"..."}
    // or wrapped: {"contentBlockStart":{"contentBlockIndex":0,"start":{...}}}
    let is_cb_start = event_type == Some("contentBlockStart") || val.get("contentBlockStart").is_some();
    if is_cb_start {
        let cb_start = val.get("contentBlockStart").unwrap_or(val);
        let index = cb_start
            .get("contentBlockIndex")
            .and_then(|v| v.as_u64())
            .unwrap_or(0) as usize;
        if !*message_started {
            events.push(Ok(StreamEvent::MessageStart {
                id: uuid::Uuid::new_v4().to_string(),
                model: String::new(),
                usage: UsageInfo::default(),
            }));
            *message_started = true;
        }
        let start_val = cb_start.get("start");
        if let Some(tool_use) = start_val.and_then(|s| s.get("toolUse")) {
            let id = tool_use
                .get("toolUseId")
                .and_then(|v| v.as_str())
                .unwrap_or("")
                .to_string();
            let name = tool_use
                .get("name")
                .and_then(|v| v.as_str())
                .unwrap_or("")
                .to_string();
            events.push(Ok(StreamEvent::ContentBlockStart {
                index,
                content_block: ContentBlock::ToolUse {
                    id,
                    name,
                    input: json!({}),
                },
            }));
        } else {
            events.push(Ok(StreamEvent::ContentBlockStart {
                index,
                content_block: ContentBlock::Text { text: String::new() },
            }));
        }
        return events;
    }

    // contentBlockDelta — flat: {"contentBlockIndex":0,"delta":{"text":"..."},"p":"..."}
    // or wrapped: {"contentBlockDelta":{"contentBlockIndex":0,"delta":{"text":"..."}}}
    let is_cb_delta = event_type == Some("contentBlockDelta") || val.get("contentBlockDelta").is_some();
    if is_cb_delta {
        let cb_delta = val.get("contentBlockDelta").unwrap_or(val);
        let index = cb_delta
            .get("contentBlockIndex")
            .and_then(|v| v.as_u64())
            .unwrap_or(0) as usize;
        if !*message_started {
            events.push(Ok(StreamEvent::MessageStart {
                id: uuid::Uuid::new_v4().to_string(),
                model: String::new(),
                usage: UsageInfo::default(),
            }));
            events.push(Ok(StreamEvent::ContentBlockStart {
                index: 0,
                content_block: ContentBlock::Text { text: String::new() },
            }));
            *message_started = true;
        }
        if let Some(delta) = cb_delta.get("delta") {
            if let Some(text) = delta.get("text").and_then(|v| v.as_str()) {
                if !text.is_empty() {
                    events.extend(
                        tagged_thinking
                            .push_text(index, text)
                            .into_iter()
                            .map(Ok),
                    );
                }
            } else if let Some(json_frag) = delta
                .get("toolUse")
                .and_then(|tu| tu.get("input"))
                .and_then(|v| v.as_str())
            {
                if !json_frag.is_empty() {
                    events.push(Ok(StreamEvent::InputJsonDelta {
                        index,
                        partial_json: json_frag.to_string(),
                    }));
                }
            }
        }
        return events;
    }

    // contentBlockStop — flat: {"contentBlockIndex":0,"p":"..."}
    let is_cb_stop = event_type == Some("contentBlockStop") || val.get("contentBlockStop").is_some();
    if is_cb_stop {
        let cb_stop = val.get("contentBlockStop").unwrap_or(val);
        let index = cb_stop
            .get("contentBlockIndex")
            .and_then(|v| v.as_u64())
            .unwrap_or(0) as usize;
        events.extend(tagged_thinking.finish().into_iter().map(Ok));
        events.push(Ok(StreamEvent::ContentBlockStop { index }));
        return events;
    }

    // messageStop — flat: {"stopReason":"end_turn","p":"..."} or wrapped
    let is_msg_stop = event_type == Some("messageStop") || val.get("messageStop").is_some();
    if is_msg_stop {
        let msg_stop = val.get("messageStop").unwrap_or(val);
        let stop_reason_str = msg_stop
            .get("stopReason")
            .and_then(|v| v.as_str())
            .unwrap_or("end_turn");
        let stop_reason = match stop_reason_str {
            "end_turn" => StopReason::EndTurn,
            "max_tokens" => StopReason::MaxTokens,
            "tool_use" => StopReason::ToolUse,
            "stop_sequence" => StopReason::StopSequence,
            other => StopReason::Other(other.to_string()),
        };
        events.extend(tagged_thinking.finish().into_iter().map(Ok));
        events.push(Ok(StreamEvent::MessageDelta {
            stop_reason: Some(stop_reason),
            usage: None,
        }));
        // Do not emit MessageStop here. Bedrock can send the terminal usage
        // metadata after `messageStop`; ending the stream immediately loses
        // cost/context data and can overwrite a `tool_use` turn with a plain
        // stop before the query loop gets the full accounting.
        return events;
    }

    // metadata (usage) — flat payload when event_type is present, wrapped otherwise.
    // The live EventStream decoder sees flat payloads from the AWS binary
    // frame, while tests and some compatibility paths use the wrapped shape.
    let metadata = if event_type == Some("metadata") {
        Some(val)
    } else {
        val.get("metadata")
    };
    if let Some(metadata) = metadata {
        if let Some(usage_val) = metadata.get("usage") {
            let usage = parse_bedrock_usage_info(usage_val);
            events.push(Ok(StreamEvent::MessageDelta {
                stop_reason: None,
                usage: Some(usage),
            }));
        }
        return events;
    }

    // internalServerException / throttlingException
    if let Some(err) = val
        .get("internalServerException")
        .or_else(|| val.get("throttlingException"))
        .or_else(|| val.get("modelStreamErrorException"))
        .or_else(|| val.get("validationException"))
    {
        let message = err
            .get("message")
            .and_then(|v| v.as_str())
            .unwrap_or("Unknown Bedrock error")
            .to_string();
        events.push(Err(ProviderError::StreamError {
            provider: provider_id.clone(),
            message,
            partial_response: None,
        }));
    }

    events
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::ThinkingConfig;
    use claurst_core::types::{Message, MessageContent, ToolDefinition};

    fn test_provider() -> BedrockProvider {
        BedrockProvider {
            id: ProviderId::new(ProviderId::AMAZON_BEDROCK),
            region: "us-west-2".to_string(),
            http_client: reqwest::Client::new(),
            access_key_id: Some("AKIATESTACCESSKEY".to_string()),
            secret_access_key: Some("test-secret-key".to_string()),
            session_token: Some("test-session-token".to_string()),
            bearer_token: None,
        }
    }

    fn test_request(model: &str) -> ProviderRequest {
        ProviderRequest {
            model: model.to_string(),
            messages: vec![Message {
                role: Role::User,
                content: MessageContent::Text("hello".to_string()),
                uuid: None,
                cost: None,
                snapshot_patch: None,
            }],
            system_prompt: None,
            tools: vec![],
            max_tokens: 16,
            temperature: Some(0.0),
            top_p: None,
            top_k: None,
            stop_sequences: vec!["</stop>".to_string()],
            thinking: None,
            provider_options: json!({}),
        }
    }

    fn test_tool() -> ToolDefinition {
        ToolDefinition {
            name: "read_file".to_string(),
            description: "Read a file".to_string(),
            input_schema: json!({
                "type": "object",
                "properties": {
                    "path": { "type": "string" }
                },
                "required": ["path"]
            }),
        }
    }

    #[test]
    fn bedrock_auth_prefers_sigv4_credentials_over_bearer_token() {
        let auth = BedrockProvider::auth_from_values(
            Some("AKIATESTACCESSKEY".to_string()),
            Some("test-secret-key".to_string()),
            Some("test-session-token".to_string()),
            Some("bedrock-bearer-token".to_string()),
        )
        .expect("auth should be configured");

        assert_eq!(auth.access_key_id.as_deref(), Some("AKIATESTACCESSKEY"));
        assert_eq!(auth.secret_access_key.as_deref(), Some("test-secret-key"));
        assert_eq!(auth.session_token.as_deref(), Some("test-session-token"));
        assert!(auth.bearer_token.is_none());
    }

    #[test]
    fn bedrock_auth_uses_bearer_token_only_without_sigv4_credentials() {
        let auth = BedrockProvider::auth_from_values(
            None,
            None,
            None,
            Some("bedrock-bearer-token".to_string()),
        )
        .expect("auth should be configured");

        assert!(auth.access_key_id.is_none());
        assert!(auth.secret_access_key.is_none());
        assert!(auth.session_token.is_none());
        assert_eq!(auth.bearer_token.as_deref(), Some("bedrock-bearer-token"));
    }

    #[test]
    fn sigv4_canonical_uri_reencodes_escaped_model_id_colon() {
        assert_eq!(
            sigv4_canonical_uri("/model/qwen.qwen3-coder-30b-a3b-v1%3A0/converse-stream"),
            "/model/qwen.qwen3-coder-30b-a3b-v1%253A0/converse-stream"
        );
    }

    #[test]
    fn converse_body_omits_stop_sequences_for_qwen_models() {
        let body = BedrockProvider::build_converse_body(&test_request(
            "qwen.qwen3-coder-30b-a3b-v1:0",
        ));

        assert!(body["inferenceConfig"].get("stopSequences").is_none());
    }

    #[test]
    fn converse_body_keeps_stop_sequences_for_anthropic_models() {
        let body = BedrockProvider::build_converse_body(&test_request(
            "anthropic.claude-3-5-sonnet-20241022-v2:0",
        ));

        assert_eq!(body["inferenceConfig"]["stopSequences"], json!(["</stop>"]));
    }

    #[test]
    fn converse_body_keeps_tool_config_for_qwen_models() {
        let mut request = test_request("qwen.qwen3-coder-30b-a3b-v1:0");
        request.tools = vec![test_tool()];

        let body = BedrockProvider::build_converse_body(&request);

        assert!(body["toolConfig"]["tools"].is_array());
    }

    #[test]
    fn converse_body_keeps_tool_config_for_nova_models() {
        let mut request = test_request("amazon.nova-2-lite-v1:0");
        request.tools = vec![test_tool()];

        let body = BedrockProvider::build_converse_body(&request);

        assert!(body["toolConfig"]["tools"].is_array());
    }

    #[test]
    fn converse_body_adds_prompt_cache_points_when_enabled() {
        let mut request = test_request("amazon.nova-2-lite-v1:0");
        request.system_prompt = Some(SystemPrompt::Text("You are Claurst.".to_string()));
        request.tools = vec![test_tool()];
        request.provider_options = json!({
            "promptCaching": {
                "tools": true,
                "system": true,
                "ttl": "5m"
            }
        });

        let body = BedrockProvider::build_converse_body(&request);

        let tools = body["toolConfig"]["tools"].as_array().expect("tools array");
        assert!(tools.iter().any(|tool| tool.get("cachePoint").is_some()));
        let system = body["system"].as_array().expect("system array");
        assert!(system.iter().any(|block| block.get("cachePoint").is_some()));
        assert_eq!(system[1]["cachePoint"]["ttl"], json!("5m"));
        assert!(body.get("promptCaching").is_none());
    }

    #[test]
    fn converse_body_ignores_prompt_cache_points_for_qwen_models() {
        let mut request = test_request("qwen.qwen3-coder-30b-a3b-v1:0");
        request.system_prompt = Some(SystemPrompt::Text("You are Claurst.".to_string()));
        request.tools = vec![test_tool()];
        request.provider_options = json!({
            "promptCaching": {
                "tools": true,
                "system": true,
                "messages": true,
                "ttl": "5m"
            }
        });

        let body = BedrockProvider::build_converse_body(&request);

        let tools = body["toolConfig"]["tools"].as_array().expect("tools array");
        assert!(tools.iter().all(|tool| tool.get("cachePoint").is_none()));
        let system = body["system"].as_array().expect("system array");
        assert!(system.iter().all(|block| block.get("cachePoint").is_none()));
        let content = body["messages"][0]["content"].as_array().expect("content array");
        assert!(content.iter().all(|block| block.get("cachePoint").is_none()));
        assert!(body.get("promptCaching").is_none());
    }

    #[test]
    fn converse_body_keeps_message_cache_points_opt_in() {
        let mut request = test_request("amazon.nova-2-lite-v1:0");
        request.provider_options = json!({ "promptCaching": true });

        let body = BedrockProvider::build_converse_body(&request);

        let content = body["messages"][0]["content"].as_array().expect("content array");
        assert!(content.iter().all(|block| block.get("cachePoint").is_none()));

        request.provider_options = json!({
            "promptCaching": {
                "messages": true,
                "tools": false,
                "system": false
            }
        });
        let body = BedrockProvider::build_converse_body(&request);
        let content = body["messages"][0]["content"].as_array().expect("content array");
        assert!(content.iter().any(|block| block.get("cachePoint").is_some()));
    }

    #[test]
    fn converse_body_omits_local_thinking_blocks_for_bedrock_followups() {
        let mut request = test_request("qwen.qwen3-coder-30b-a3b-v1:0");
        request.messages = vec![Message {
            role: Role::Assistant,
            content: MessageContent::Blocks(vec![
                ContentBlock::Thinking {
                    thinking: "internal plan".to_string(),
                    signature: String::new(),
                },
                ContentBlock::Text {
                    text: "visible answer".to_string(),
                },
            ]),
            uuid: None,
            cost: None,
            snapshot_patch: None,
        }];

        let body = BedrockProvider::build_converse_body(&request);
        let content = body["messages"][0]["content"].as_array().expect("content array");

        assert_eq!(content, &vec![json!({ "text": "visible answer" })]);
    }

    #[test]
    fn converse_body_omits_reasoning_config_for_deepseek_models() {
        let mut request = test_request("deepseek.v3.2");
        request.thinking = Some(ThinkingConfig::enabled(1024));

        let body = BedrockProvider::build_converse_body(&request);

        assert!(body.get("reasoningConfig").is_none());
    }

    #[test]
    fn converse_body_keeps_reasoning_config_for_anthropic_models() {
        let mut request = test_request("anthropic.claude-sonnet-4-6-v1");
        request.thinking = Some(ThinkingConfig::enabled(1024));

        let body = BedrockProvider::build_converse_body(&request);

        assert_eq!(body["reasoningConfig"]["budgetTokens"], json!(1024));
    }

    #[test]
    fn sign_request_uses_aws_sigv4_authorization_header() {
        let provider = test_provider();
        let signed = provider.sign_request(
            "POST",
            "https://bedrock-runtime.us-west-2.amazonaws.com/model/qwen.qwen3-coder-30b-a3b-v1%3A0/converse",
            r#"{"messages":[]}"#,
            &chrono::DateTime::parse_from_rfc3339("2026-01-02T03:04:05Z")
                .unwrap()
                .with_timezone(&chrono::Utc),
        );

        let authorization = signed
            .get("Authorization")
            .expect("Authorization header should be signed");

        assert!(authorization.starts_with("AWS4-HMAC-SHA256 "));
        assert!(authorization.contains(
            "Credential=AKIATESTACCESSKEY/20260102/us-west-2/bedrock/aws4_request"
        ));
        assert!(authorization.contains(
            "SignedHeaders=content-type;host;x-amz-content-sha256;x-amz-date;x-amz-security-token"
        ));
        assert!(authorization.contains("Signature="));
        assert_eq!(signed.get("x-amz-date").map(String::as_str), Some("20260102T030405Z"));
        assert_eq!(
            signed.get("x-amz-security-token").map(String::as_str),
            Some("test-session-token")
        );
    }

    #[test]
    fn parse_converse_usage_maps_bedrock_cache_tokens() {
        let usage = BedrockProvider::parse_converse_usage(Some(&json!({
            "inputTokens": 100,
            "outputTokens": 40,
            "cacheWriteInputTokens": 30,
            "cacheReadInputTokens": 20
        })));

        assert_eq!(usage.input_tokens, 100);
        assert_eq!(usage.output_tokens, 40);
        assert_eq!(usage.cache_creation_input_tokens, 30);
        assert_eq!(usage.cache_read_input_tokens, 20);
    }

    #[test]
    fn parse_converse_content_splits_qwen_tagged_thinking() {
        let blocks = BedrockProvider::parse_converse_content(Some(&vec![json!({
            "text": "<think>inspect files</think>Done."
        })]));

        assert!(matches!(
            &blocks[0],
            ContentBlock::Thinking { thinking, .. } if thinking == "inspect files"
        ));
        assert!(matches!(
            &blocks[1],
            ContentBlock::Text { text } if text == "Done."
        ));
    }

    #[test]
    fn stream_splits_qwen_tagged_thinking_across_chunks() {
        let provider_id = ProviderId::new(ProviderId::AMAZON_BEDROCK);
        let mut message_started = true;
        let mut tagged_thinking = BedrockTaggedThinkingState::default();

        let first_events = parse_bedrock_event(
            &json!({ "contentBlockIndex": 0, "delta": { "text": "<thi" } }),
            Some("contentBlockDelta"),
            &provider_id,
            &mut message_started,
            &mut tagged_thinking,
        );
        assert!(first_events.is_empty());

        let second_events = parse_bedrock_event(
            &json!({ "contentBlockIndex": 0, "delta": { "text": "nk>inspect</think>Done" } }),
            Some("contentBlockDelta"),
            &provider_id,
            &mut message_started,
            &mut tagged_thinking,
        );
        assert!(matches!(
            second_events[0].as_ref().expect("thinking event"),
            StreamEvent::ThinkingDelta { thinking, .. } if thinking == "inspect"
        ));
        assert!(second_events.iter().any(|event| matches!(
            event.as_ref().expect("event"),
            StreamEvent::TextDelta { text, .. } if text == "Done"
        )));

        let stop_events = parse_bedrock_event(
            &json!({ "contentBlockIndex": 0 }),
            Some("contentBlockStop"),
            &provider_id,
            &mut message_started,
            &mut tagged_thinking,
        );
        assert!(stop_events.iter().any(|event| matches!(
            event.as_ref().expect("event"),
            StreamEvent::ContentBlockStop { index } if *index == 0
        )));
    }

    #[test]
    fn stream_message_stop_waits_for_later_metadata_usage() {
        let provider_id = ProviderId::new(ProviderId::AMAZON_BEDROCK);
        let mut message_started = true;
        let mut tagged_thinking = BedrockTaggedThinkingState::default();
        let stop_events = parse_bedrock_event(
            &json!({ "stopReason": "tool_use" }),
            Some("messageStop"),
            &provider_id,
            &mut message_started,
            &mut tagged_thinking,
        );

        assert_eq!(stop_events.len(), 1);
        assert!(matches!(
            stop_events[0].as_ref().expect("event"),
            StreamEvent::MessageDelta {
                stop_reason: Some(StopReason::ToolUse),
                usage: None
            }
        ));

        let usage_events = parse_bedrock_event(
            &json!({
                "usage": {
                    "inputTokens": 100,
                    "outputTokens": 40,
                    "cacheWriteInputTokens": 30,
                    "cacheReadInputTokens": 20
                }
            }),
            Some("metadata"),
            &provider_id,
            &mut message_started,
            &mut tagged_thinking,
        );
        let StreamEvent::MessageDelta {
            stop_reason: None,
            usage: Some(usage),
        } = usage_events[0].as_ref().expect("usage event")
        else {
            panic!("expected usage delta");
        };
        assert_eq!(usage.input_tokens, 100);
        assert_eq!(usage.output_tokens, 40);
        assert_eq!(usage.cache_creation_input_tokens, 30);
        assert_eq!(usage.cache_read_input_tokens, 20);
    }
}
