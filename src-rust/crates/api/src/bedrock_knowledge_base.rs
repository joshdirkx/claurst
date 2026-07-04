use claurst_core::provider_id::ProviderId;
use serde::{Deserialize, Serialize};
use serde_json::{Value, json};

use crate::error_handling::parse_error_response;
use crate::provider_error::ProviderError;

#[derive(Debug, Clone)]
pub struct BedrockKnowledgeBaseClient {
    region: String,
    http_client: reqwest::Client,
    access_key_id: Option<String>,
    secret_access_key: Option<String>,
    session_token: Option<String>,
    bearer_token: Option<String>,
}

#[derive(Debug, Clone)]
pub struct BedrockKnowledgeBaseRetrieveRequest {
    pub knowledge_base_id: String,
    pub query: String,
    pub retrieval_configuration: Option<Value>,
    pub next_token: Option<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct BedrockKnowledgeBaseRetrieveResponse {
    #[serde(skip_serializing_if = "Option::is_none")]
    pub guardrail_action: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub next_token: Option<String>,
    pub retrieval_results: Vec<BedrockKnowledgeBaseRetrievalResult>,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct BedrockKnowledgeBaseRetrievalResult {
    #[serde(skip_serializing_if = "Option::is_none")]
    pub content_type: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub text: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub document_id: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub location: Option<Value>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub metadata: Option<Value>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub score: Option<f64>,
}

struct BedrockAgentRuntimeAuth {
    access_key_id: Option<String>,
    secret_access_key: Option<String>,
    session_token: Option<String>,
    bearer_token: Option<String>,
}

impl BedrockKnowledgeBaseClient {
    pub fn from_env_with_region(region: impl Into<String>) -> Option<Self> {
        let auth = BedrockAgentRuntimeAuth::from_env()?;
        let http_client = reqwest::Client::builder()
            .timeout(crate::request_timeout())
            .build()
            .expect("failed to build reqwest client");

        Some(Self {
            region: region.into(),
            http_client,
            access_key_id: auth.access_key_id,
            secret_access_key: auth.secret_access_key,
            session_token: auth.session_token,
            bearer_token: auth.bearer_token,
        })
    }

    pub fn region(&self) -> &str {
        &self.region
    }

    pub fn retrieve_endpoint_url(&self, knowledge_base_id: &str) -> String {
        format!(
            "https://bedrock-agent-runtime.{}.amazonaws.com/knowledgebases/{}/retrieve",
            self.region,
            urlencoding::encode(knowledge_base_id)
        )
    }

    pub fn build_retrieve_body(request: &BedrockKnowledgeBaseRetrieveRequest) -> Value {
        let mut body = json!({
            "retrievalQuery": {
                "text": request.query
            }
        });

        if let Some(config) = &request.retrieval_configuration {
            // Keep retrieval configuration as caller-provided JSON instead of
            // modeling every Bedrock filter/reranking variant in Claurst. That
            // lets Bedrock add retrieval knobs without forcing a tool schema
            // migration, while unit tests still lock the common request shape.
            body["retrievalConfiguration"] = config.clone();
        }

        if let Some(next_token) = &request.next_token {
            body["nextToken"] = json!(next_token);
        }

        body
    }

    pub async fn retrieve(
        &self,
        request: BedrockKnowledgeBaseRetrieveRequest,
    ) -> Result<BedrockKnowledgeBaseRetrieveResponse, ProviderError> {
        let url = self.retrieve_endpoint_url(&request.knowledge_base_id);
        let body = Self::build_retrieve_body(&request);
        let body_str = serde_json::to_string(&body).unwrap_or_default();
        let now = chrono::Utc::now();
        let auth_headers = self.sign_request("POST", &url, &body_str, &now);

        let mut req_builder = self
            .http_client
            .post(&url)
            .header("Content-Type", "application/json");

        for (key, value) in auth_headers {
            req_builder = req_builder.header(key.as_str(), value.as_str());
        }

        let resp = req_builder
            .body(body_str)
            .send()
            .await
            .map_err(|e| ProviderError::Other {
                provider: ProviderId::new(ProviderId::AMAZON_BEDROCK),
                message: format!("Bedrock Knowledge Base retrieve request failed: {}", e),
                status: None,
                body: None,
            })?;

        let status = resp.status().as_u16();
        let text = resp.text().await.map_err(|e| ProviderError::Other {
            provider: ProviderId::new(ProviderId::AMAZON_BEDROCK),
            message: format!("Failed to read Bedrock Knowledge Base response body: {}", e),
            status: Some(status),
            body: None,
        })?;

        if !(200..300).contains(&(status as usize)) {
            return Err(parse_error_response(
                status,
                &text,
                &ProviderId::new(ProviderId::AMAZON_BEDROCK),
            ));
        }

        let value: Value = serde_json::from_str(&text).map_err(|e| ProviderError::Other {
            provider: ProviderId::new(ProviderId::AMAZON_BEDROCK),
            message: format!(
                "Failed to parse Bedrock Knowledge Base response JSON: {}",
                e
            ),
            status: Some(status),
            body: Some(text.clone()),
        })?;

        Ok(Self::parse_retrieve_response(&value))
    }

    pub fn parse_retrieve_response(value: &Value) -> BedrockKnowledgeBaseRetrieveResponse {
        let retrieval_results = value
            .get("retrievalResults")
            .and_then(Value::as_array)
            .map(|items| items.iter().map(parse_retrieval_result).collect())
            .unwrap_or_default();

        BedrockKnowledgeBaseRetrieveResponse {
            guardrail_action: value
                .get("guardrailAction")
                .and_then(Value::as_str)
                .map(str::to_string),
            next_token: value
                .get("nextToken")
                .and_then(Value::as_str)
                .map(str::to_string),
            retrieval_results,
        }
    }

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
        if let Some(token) = &self.bearer_token {
            headers.insert("Authorization".to_string(), format!("Bearer {}", token));
            return headers;
        }

        let Some(access_key) = self.access_key_id.as_ref() else {
            return headers;
        };
        let Some(secret_key) = self.secret_access_key.as_ref() else {
            return headers;
        };

        let date_str = date.format("%Y%m%d").to_string();
        let datetime_str = date.format("%Y%m%dT%H%M%SZ").to_string();
        let service = "bedrock";
        let region = &self.region;
        let parsed = url::Url::parse(url_str).unwrap_or_else(|_| {
            url::Url::parse("https://bedrock-agent-runtime.us-east-1.amazonaws.com/").unwrap()
        });
        let canonical_uri = sigv4_canonical_uri(parsed.path());
        let canonical_query = parsed.query().unwrap_or("").to_string();
        let body_hash = hex::encode(Sha256::digest(body.as_bytes()));
        let host = parsed.host_str().unwrap_or_default().to_string();
        let content_type = "application/json";

        let mut canonical_headers = format!(
            "content-type:{}\nhost:{}\nx-amz-content-sha256:{}\nx-amz-date:{}\n",
            content_type, host, body_hash, datetime_str
        );
        let mut signed_headers = "content-type;host;x-amz-content-sha256;x-amz-date".to_string();

        if let Some(token) = &self.session_token {
            canonical_headers.push_str(&format!("x-amz-security-token:{}\n", token));
            signed_headers.push_str(";x-amz-security-token");
        }

        let canonical_request = format!(
            "{}\n{}\n{}\n{}\n{}\n{}",
            method, canonical_uri, canonical_query, canonical_headers, signed_headers, body_hash
        );
        let credential_scope = format!("{}/{}/{}/aws4_request", date_str, region, service);
        let canonical_request_hash = hex::encode(Sha256::digest(canonical_request.as_bytes()));
        let string_to_sign = format!(
            "AWS4-HMAC-SHA256\n{}\n{}\n{}",
            datetime_str, credential_scope, canonical_request_hash
        );

        let k_date = {
            let mut mac = HmacSha256::new_from_slice(format!("AWS4{}", secret_key).as_bytes())
                .expect("HMAC init failed");
            mac.update(date_str.as_bytes());
            mac.finalize().into_bytes()
        };
        let k_region = {
            let mut mac = HmacSha256::new_from_slice(&k_date).expect("HMAC init failed");
            mac.update(region.as_bytes());
            mac.finalize().into_bytes()
        };
        let k_service = {
            let mut mac = HmacSha256::new_from_slice(&k_region).expect("HMAC init failed");
            mac.update(service.as_bytes());
            mac.finalize().into_bytes()
        };
        let k_signing = {
            let mut mac = HmacSha256::new_from_slice(&k_service).expect("HMAC init failed");
            mac.update(b"aws4_request");
            mac.finalize().into_bytes()
        };
        let signature = {
            let mut mac = HmacSha256::new_from_slice(&k_signing).expect("HMAC init failed");
            mac.update(string_to_sign.as_bytes());
            hex::encode(mac.finalize().into_bytes())
        };

        headers.insert(
            "Authorization".to_string(),
            format!(
                "AWS4-HMAC-SHA256 Credential={}/{}, SignedHeaders={}, Signature={}",
                access_key, credential_scope, signed_headers, signature
            ),
        );
        headers.insert("x-amz-date".to_string(), datetime_str);
        headers.insert("x-amz-content-sha256".to_string(), body_hash);
        if let Some(token) = &self.session_token {
            headers.insert("x-amz-security-token".to_string(), token.clone());
        }

        headers
    }
}

impl BedrockAgentRuntimeAuth {
    fn from_env() -> Option<Self> {
        let access_key_id = non_empty_env("AWS_ACCESS_KEY_ID");
        let secret_access_key = non_empty_env("AWS_SECRET_ACCESS_KEY");
        let session_token = non_empty_env("AWS_SESSION_TOKEN");
        let bearer_token = non_empty_env("AWS_BEARER_TOKEN_BEDROCK");

        if let (Some(access_key_id), Some(secret_access_key)) = (access_key_id, secret_access_key) {
            return Some(Self {
                access_key_id: Some(access_key_id),
                secret_access_key: Some(secret_access_key),
                session_token,
                bearer_token: None,
            });
        }

        bearer_token.map(|token| Self {
            access_key_id: None,
            secret_access_key: None,
            session_token: None,
            bearer_token: Some(token),
        })
    }
}

fn parse_retrieval_result(value: &Value) -> BedrockKnowledgeBaseRetrievalResult {
    let content = value.get("content");
    BedrockKnowledgeBaseRetrievalResult {
        content_type: content
            .and_then(|content| content.get("type"))
            .and_then(Value::as_str)
            .map(str::to_string),
        text: content
            .and_then(|content| content.get("text"))
            .and_then(Value::as_str)
            .map(str::to_string),
        document_id: value
            .get("documentId")
            .and_then(Value::as_str)
            .map(str::to_string),
        location: value.get("location").cloned(),
        metadata: value.get("metadata").cloned(),
        score: value.get("score").and_then(Value::as_f64),
    }
}

fn non_empty_env(name: &str) -> Option<String> {
    std::env::var(name)
        .ok()
        .filter(|value| !value.trim().is_empty())
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
            b'A'..=b'Z' | b'a'..=b'z' | b'0'..=b'9' | b'-' | b'.' | b'_' | b'~' => {
                encoded.push(byte as char)
            }
            b'/' if !encode_slash => encoded.push('/'),
            _ => encoded.push_str(&format!("%{:02X}", byte)),
        }
    }
    encoded
}

#[cfg(test)]
mod tests {
    use super::*;

    fn test_client() -> BedrockKnowledgeBaseClient {
        BedrockKnowledgeBaseClient {
            region: "us-west-2".to_string(),
            http_client: reqwest::Client::new(),
            access_key_id: Some("AKIATESTACCESSKEY".to_string()),
            secret_access_key: Some("test-secret-key".to_string()),
            session_token: Some("test-session-token".to_string()),
            bearer_token: None,
        }
    }

    #[test]
    fn build_retrieve_body_uses_agent_runtime_shape() {
        let body =
            BedrockKnowledgeBaseClient::build_retrieve_body(&BedrockKnowledgeBaseRetrieveRequest {
                knowledge_base_id: "KB12345678".to_string(),
                query: "What standards apply?".to_string(),
                retrieval_configuration: Some(json!({
                    "vectorSearchConfiguration": {
                        "numberOfResults": 5
                    }
                })),
                next_token: Some("next".to_string()),
            });

        assert_eq!(
            body["retrievalQuery"]["text"],
            json!("What standards apply?")
        );
        assert_eq!(
            body["retrievalConfiguration"]["vectorSearchConfiguration"]["numberOfResults"],
            json!(5)
        );
        assert_eq!(body["nextToken"], json!("next"));
    }

    #[test]
    fn retrieve_endpoint_encodes_knowledge_base_id_as_path_segment() {
        let client = test_client();
        let url = client.retrieve_endpoint_url(
            "arn:aws:bedrock:us-west-2:123456789012:knowledge-base/KB12345678",
        );

        assert_eq!(
            url,
            "https://bedrock-agent-runtime.us-west-2.amazonaws.com/knowledgebases/arn%3Aaws%3Abedrock%3Aus-west-2%3A123456789012%3Aknowledge-base%2FKB12345678/retrieve"
        );
    }

    #[test]
    fn sign_request_uses_bedrock_agent_runtime_host() {
        let client = test_client();
        let signed = client.sign_request(
            "POST",
            "https://bedrock-agent-runtime.us-west-2.amazonaws.com/knowledgebases/KB12345678/retrieve",
            r#"{"retrievalQuery":{"text":"hello"}}"#,
            &chrono::DateTime::parse_from_rfc3339("2026-01-02T03:04:05Z")
                .unwrap()
                .with_timezone(&chrono::Utc),
        );

        let authorization = signed
            .get("Authorization")
            .expect("Authorization header should be signed");

        assert!(authorization.starts_with("AWS4-HMAC-SHA256 "));
        assert!(
            authorization
                .contains("Credential=AKIATESTACCESSKEY/20260102/us-west-2/bedrock/aws4_request")
        );
        assert!(authorization.contains(
            "SignedHeaders=content-type;host;x-amz-content-sha256;x-amz-date;x-amz-security-token"
        ));
        assert!(authorization.contains("Signature="));
        assert_eq!(
            signed.get("x-amz-date").map(String::as_str),
            Some("20260102T030405Z")
        );
    }

    #[test]
    fn parse_retrieve_response_extracts_chunks_and_source_metadata() {
        let response = BedrockKnowledgeBaseClient::parse_retrieve_response(&json!({
            "guardrailAction": "NONE",
            "nextToken": "more",
            "retrievalResults": [{
                "content": {
                    "type": "TEXT",
                    "text": "Use a two-step TeamCity rollout."
                },
                "documentId": "doc-1",
                "location": {
                    "type": "S3",
                    "s3Location": {
                        "uri": "s3://bucket/standards.md"
                    }
                },
                "metadata": {
                    "domain": "teamcity"
                },
                "score": 0.91
            }]
        }));

        assert_eq!(response.guardrail_action.as_deref(), Some("NONE"));
        assert_eq!(response.next_token.as_deref(), Some("more"));
        assert_eq!(response.retrieval_results.len(), 1);
        assert_eq!(
            response.retrieval_results[0].text.as_deref(),
            Some("Use a two-step TeamCity rollout.")
        );
        assert_eq!(response.retrieval_results[0].score, Some(0.91));
    }
}
