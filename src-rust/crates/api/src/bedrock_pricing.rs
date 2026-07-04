use std::collections::HashMap;
use std::path::PathBuf;

use chrono::Utc;
use claurst_core::cost::ModelPricing;
use serde::{Deserialize, Serialize};
use serde_json::{json, Value};
use tracing::debug;

const PRICE_LIST_SERVICE_CODE: &str = "AmazonBedrockFoundationModels";
const PRICING_ENDPOINT_REGION: &str = "us-east-1";
const PRICE_LIST_CACHE_TTL_SECS: i64 = 24 * 60 * 60;

#[derive(Debug, Clone, Serialize, Deserialize)]
struct CachedBedrockPriceList {
    region: String,
    fetched_at_unix_secs: i64,
    price_list: Vec<String>,
}

#[derive(Debug, Deserialize)]
#[serde(rename_all = "PascalCase")]
struct GetProductsResponse {
    price_list: Vec<String>,
    next_token: Option<String>,
}

#[derive(Debug)]
struct AwsPricingAuth {
    access_key_id: String,
    secret_access_key: String,
    session_token: Option<String>,
}

pub async fn resolve_bedrock_pricing(model: &str, region: &str) -> Option<ModelPricing> {
    let region = normalize_region(region);
    if let Some(cached) = load_price_cache(&region) {
        if cache_is_fresh(&cached) {
            if let Some(pricing) = pricing_from_price_list(model, &cached.price_list) {
                return Some(pricing);
            }
        }
    }

    // Pricing is resolved from AWS's central Price List API instead of a
    // region-local Bedrock endpoint. Cache the whole regional product page so
    // switching between allowed models does not make a network call on every
    // TUI model change.
    if let Some(price_list) = fetch_price_list(&region).await {
        save_price_cache(&region, &price_list);
        if let Some(pricing) = pricing_from_price_list(model, &price_list) {
            return Some(pricing);
        }
    }

    // Static values keep the cost meter useful when credentials are missing,
    // SSO has expired, or AWS changes the product text enough that parsing
    // fails. The live price list remains the preferred source when available.
    static_bedrock_pricing(model)
}

fn normalize_region(region: &str) -> String {
    if region.trim().is_empty() {
        "us-east-1".to_string()
    } else {
        region.trim().to_string()
    }
}

fn cache_is_fresh(cache: &CachedBedrockPriceList) -> bool {
    Utc::now().timestamp() - cache.fetched_at_unix_secs < PRICE_LIST_CACHE_TTL_SECS
}

fn cache_path(region: &str) -> Option<PathBuf> {
    dirs::cache_dir().map(|dir| {
        dir.join("claurst")
            .join(format!("bedrock-pricing-{}.json", region))
    })
}

fn load_price_cache(region: &str) -> Option<CachedBedrockPriceList> {
    let path = cache_path(region)?;
    let text = std::fs::read_to_string(path).ok()?;
    let cache: CachedBedrockPriceList = serde_json::from_str(&text).ok()?;
    (cache.region == region).then_some(cache)
}

fn save_price_cache(region: &str, price_list: &[String]) {
    let Some(path) = cache_path(region) else {
        return;
    };
    if let Some(parent) = path.parent() {
        let _ = std::fs::create_dir_all(parent);
    }
    let cache = CachedBedrockPriceList {
        region: region.to_string(),
        fetched_at_unix_secs: Utc::now().timestamp(),
        price_list: price_list.to_vec(),
    };
    if let Ok(text) = serde_json::to_string_pretty(&cache) {
        let _ = std::fs::write(path, text);
    }
}

async fn fetch_price_list(region: &str) -> Option<Vec<String>> {
    let auth = AwsPricingAuth::from_env()?;
    let client = reqwest::Client::builder()
        .timeout(std::time::Duration::from_secs(30))
        .build()
        .ok()?;
    let mut next_token: Option<String> = None;
    let mut price_list = Vec::new();

    loop {
        let mut body = json!({
            "ServiceCode": PRICE_LIST_SERVICE_CODE,
            "FormatVersion": "aws_v1",
            "MaxResults": 100,
            "Filters": [
                {
                    "Type": "TERM_MATCH",
                    "Field": "regionCode",
                    "Value": region,
                }
            ],
        });
        if let Some(token) = next_token.as_deref() {
            body["NextToken"] = json!(token);
        }

        let body_str = serde_json::to_string(&body).ok()?;
        let now = Utc::now();
        let headers = auth.sign_get_products_request(&body_str, &now);

        let mut request = client
            .post("https://api.pricing.us-east-1.amazonaws.com/")
            .body(body_str);
        for (name, value) in headers {
            request = request.header(name, value);
        }

        let response = match request.send().await {
            Ok(response) => response,
            Err(error) => {
                debug!("Bedrock pricing: AWS Price List request failed: {}", error);
                return None;
            }
        };
        if !response.status().is_success() {
            debug!(
                "Bedrock pricing: AWS Price List returned {}",
                response.status()
            );
            return None;
        }
        let parsed: GetProductsResponse = response.json().await.ok()?;
        price_list.extend(parsed.price_list);
        match parsed.next_token {
            Some(token) if !token.is_empty() => next_token = Some(token),
            _ => break,
        }
    }

    Some(price_list)
}

impl AwsPricingAuth {
    fn from_env() -> Option<Self> {
        // The pricing resolver intentionally uses the same environment shape
        // as Bedrock runtime calls. AWS SSO/profile flows should materialize
        // temporary credentials before launch, and long-lived access keys stay
        // only as the fallback operational path.
        Some(Self {
            access_key_id: non_empty_env("AWS_ACCESS_KEY_ID")?,
            secret_access_key: non_empty_env("AWS_SECRET_ACCESS_KEY")?,
            session_token: non_empty_env("AWS_SESSION_TOKEN"),
        })
    }

    fn sign_get_products_request(
        &self,
        body: &str,
        date: &chrono::DateTime<Utc>,
    ) -> HashMap<String, String> {
        use hmac::{Hmac, Mac};
        use sha2::{Digest, Sha256};

        type HmacSha256 = Hmac<Sha256>;

        let date_str = date.format("%Y%m%d").to_string();
        let datetime_str = date.format("%Y%m%dT%H%M%SZ").to_string();
        let body_hash = hex::encode(Sha256::digest(body.as_bytes()));
        let host = "api.pricing.us-east-1.amazonaws.com";
        let content_type = "application/x-amz-json-1.1";
        let target = "AWSPriceListService.GetProducts";

        let mut canonical_headers = format!(
            "content-type:{}\nhost:{}\nx-amz-content-sha256:{}\nx-amz-date:{}\nx-amz-target:{}\n",
            content_type, host, body_hash, datetime_str, target
        );
        let mut signed_headers =
            "content-type;host;x-amz-content-sha256;x-amz-date;x-amz-target".to_string();

        if let Some(token) = self.session_token.as_deref() {
            canonical_headers.push_str(&format!("x-amz-security-token:{}\n", token));
            signed_headers.push_str(";x-amz-security-token");
        }

        let canonical_request = format!(
            "POST\n/\n\n{}\n{}\n{}",
            canonical_headers, signed_headers, body_hash
        );
        let credential_scope = format!(
            "{}/{}/{}/aws4_request",
            date_str, PRICING_ENDPOINT_REGION, "pricing"
        );
        let string_to_sign = format!(
            "AWS4-HMAC-SHA256\n{}\n{}\n{}",
            datetime_str,
            credential_scope,
            hex::encode(Sha256::digest(canonical_request.as_bytes()))
        );

        let k_date = {
            let mut mac =
                HmacSha256::new_from_slice(format!("AWS4{}", self.secret_access_key).as_bytes())
                    .expect("HMAC init failed");
            mac.update(date_str.as_bytes());
            mac.finalize().into_bytes()
        };
        let k_region = {
            let mut mac = HmacSha256::new_from_slice(&k_date).expect("HMAC init failed");
            mac.update(PRICING_ENDPOINT_REGION.as_bytes());
            mac.finalize().into_bytes()
        };
        let k_service = {
            let mut mac = HmacSha256::new_from_slice(&k_region).expect("HMAC init failed");
            mac.update(b"pricing");
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

        let mut headers = HashMap::new();
        headers.insert("Content-Type".to_string(), content_type.to_string());
        headers.insert("X-Amz-Target".to_string(), target.to_string());
        headers.insert("x-amz-date".to_string(), datetime_str);
        headers.insert("x-amz-content-sha256".to_string(), body_hash);
        headers.insert(
            "Authorization".to_string(),
            format!(
                "AWS4-HMAC-SHA256 Credential={}/{}, SignedHeaders={}, Signature={}",
                self.access_key_id, credential_scope, signed_headers, signature
            ),
        );
        if let Some(token) = self.session_token.as_deref() {
            headers.insert("x-amz-security-token".to_string(), token.to_string());
        }
        headers
    }
}

fn pricing_from_price_list(model: &str, price_list: &[String]) -> Option<ModelPricing> {
    let aliases = model_aliases(model);
    for raw in price_list {
        let haystack = normalize_match_text(raw);
        if !aliases.iter().any(|alias| haystack.contains(alias)) {
            continue;
        }
        let Ok(value) = serde_json::from_str::<Value>(raw) else {
            continue;
        };
        if let Some(pricing) = extract_on_demand_token_pricing(&value) {
            return Some(pricing);
        }
    }
    None
}

fn extract_on_demand_token_pricing(value: &Value) -> Option<ModelPricing> {
    let mut input = None;
    let mut output = None;

    let terms = value.get("terms")?.get("OnDemand")?.as_object()?;
    for term in terms.values() {
        let Some(dimensions) = term.get("priceDimensions").and_then(Value::as_object) else {
            continue;
        };
        for dimension in dimensions.values() {
            let description = dimension
                .get("description")
                .and_then(Value::as_str)
                .unwrap_or_default()
                .to_ascii_lowercase();
            let unit = dimension
                .get("unit")
                .and_then(Value::as_str)
                .unwrap_or_default()
                .to_ascii_lowercase();
            if !description.contains("token") && !unit.contains("token") {
                continue;
            }

            let usd = dimension
                .get("pricePerUnit")
                .and_then(|v| v.get("USD"))
                .and_then(Value::as_str)
                .and_then(|s| s.parse::<f64>().ok())?;
            let price_per_mtk = normalize_token_price_to_million(usd, &description, &unit);

            if description.contains("input") && input.is_none() {
                input = Some(price_per_mtk);
            } else if description.contains("output") && output.is_none() {
                output = Some(price_per_mtk);
            }
        }
    }

    Some(ModelPricing {
        input_per_mtk: input?,
        output_per_mtk: output?,
        cache_creation_per_mtk: 0.0,
        cache_read_per_mtk: 0.0,
    })
}

fn normalize_token_price_to_million(price: f64, description: &str, unit: &str) -> f64 {
    let text = format!("{} {}", description, unit);
    if text.contains("1m") || text.contains("1 m") || text.contains("million") {
        price
    } else if text.contains("1k") || text.contains("1 k") || text.contains("1,000") {
        price * 1_000.0
    } else if price < 0.01 {
        price * 1_000.0
    } else {
        price
    }
}

fn model_aliases(model: &str) -> Vec<String> {
    let normalized = normalize_match_text(model);
    let mut aliases = vec![normalized.clone()];

    // Bedrock runtime, Bedrock Mantle, and the Price List API do not use one
    // canonical spelling for every model family. Keep these aliases close to
    // the resolver so new Bedrock model IDs can be added without touching the
    // generic cost tracker.
    if normalized.contains("qwen3coder30b") {
        aliases.push("qwen3coder30ba3b".to_string());
    }
    if normalized.contains("qwen3coder480b") {
        aliases.push("qwen3coder480ba35b".to_string());
    }
    if normalized.contains("qwen3235b") {
        aliases.push("qwen3235ba22b2507".to_string());
    }
    if normalized.contains("qwen332b") {
        aliases.push("qwen332b".to_string());
    }
    if normalized.contains("deepseekv32") || normalized.contains("deepseek32") {
        aliases.push("deepseekv32".to_string());
    }
    if normalized.contains("novamicro") {
        aliases.push("novamicro".to_string());
    }
    if normalized.contains("novalite") {
        aliases.push("novalite".to_string());
    }
    if normalized.contains("novapro") {
        aliases.push("novapro".to_string());
    }

    aliases.sort();
    aliases.dedup();
    aliases
}

fn normalize_match_text(value: &str) -> String {
    value
        .to_ascii_lowercase()
        .chars()
        .filter(|ch| ch.is_ascii_alphanumeric())
        .collect()
}

fn static_bedrock_pricing(model: &str) -> Option<ModelPricing> {
    let normalized = model.to_ascii_lowercase();
    if normalized.contains("qwen3-coder-30b") || normalized.contains("qwen.qwen3-coder-30b") {
        Some(ModelPricing::BEDROCK_QWEN_CODER_30B)
    } else if normalized.contains("qwen3-235b") || normalized.contains("qwen.qwen3-235b") {
        Some(ModelPricing::BEDROCK_QWEN_235B)
    } else if normalized.contains("qwen3-next") || normalized.contains("qwen3-coder-next") {
        Some(ModelPricing::BEDROCK_QWEN_NEXT)
    } else if normalized.contains("deepseek.v3.2") || normalized.contains("deepseek-v3.2") {
        Some(ModelPricing::BEDROCK_DEEPSEEK_V32)
    } else {
        None
    }
}

fn non_empty_env(name: &str) -> Option<String> {
    std::env::var(name)
        .ok()
        .filter(|value| !value.trim().is_empty())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn model_aliases_match_qwen_runtime_id_to_page_name() {
        let aliases = model_aliases("qwen.qwen3-coder-30b-a3b-v1:0");
        assert!(aliases.contains(&"qwen3coder30ba3b".to_string()));
    }

    #[test]
    fn extracts_on_demand_token_prices_from_price_list_product() {
        let product = json!({
            "terms": {
                "OnDemand": {
                    "sku.term": {
                        "priceDimensions": {
                            "input": {
                                "description": "Qwen3 Coder 30B A3B input tokens per 1,000 tokens",
                                "unit": "1K tokens",
                                "pricePerUnit": { "USD": "0.0001545" }
                            },
                            "output": {
                                "description": "Qwen3 Coder 30B A3B output tokens per 1,000 tokens",
                                "unit": "1K tokens",
                                "pricePerUnit": { "USD": "0.000618" }
                            }
                        }
                    }
                }
            }
        });

        let pricing = extract_on_demand_token_pricing(&product).expect("pricing should parse");
        assert_eq!(pricing.input_per_mtk, 0.1545);
        assert_eq!(pricing.output_per_mtk, 0.618);
    }
}
