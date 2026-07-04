use crate::{PermissionLevel, Tool, ToolContext, ToolResult};
use async_trait::async_trait;
use serde::Deserialize;
use serde_json::{Value, json};

pub struct BedrockKnowledgeBaseRetrieveTool;

const MANAGED_KB_STANDARD_RETRIEVE_COST_USD: f64 = 0.001;

#[derive(Debug, Deserialize)]
struct BedrockKnowledgeBaseRetrieveInput {
    query: String,
    #[serde(
        default,
        alias = "knowledgeBase",
        alias = "knowledge_base",
        alias = "knowledgeBaseId",
        alias = "knowledge_base_id"
    )]
    knowledge_base: Option<String>,
    #[serde(default, alias = "numberOfResults", alias = "number_of_results")]
    number_of_results: Option<u32>,
    #[serde(
        default,
        alias = "retrievalConfiguration",
        alias = "retrieval_configuration"
    )]
    retrieval_configuration: Option<Value>,
    #[serde(default)]
    filter: Option<Value>,
    #[serde(default, alias = "nextToken", alias = "next_token")]
    next_token: Option<String>,
}

#[derive(Debug, Clone)]
struct KnowledgeBaseToolConfig {
    name: Option<String>,
    id: String,
    description: Option<String>,
    region: Option<String>,
    number_of_results: Option<u32>,
    retrieval_configuration: Option<Value>,
}

#[derive(Debug)]
struct KnowledgeBaseSelection {
    configured: Option<KnowledgeBaseToolConfig>,
    id: String,
    label: String,
}

#[async_trait]
impl Tool for BedrockKnowledgeBaseRetrieveTool {
    fn name(&self) -> &str {
        "BedrockKnowledgeBaseRetrieve"
    }

    fn description(&self) -> &str {
        "Retrieve relevant chunks from a configured Amazon Bedrock Knowledge Base. \
         Use this when project, organization, or domain knowledge may be stored in \
         Bedrock and should be returned as source chunks for the active model to use."
    }

    fn permission_level(&self) -> PermissionLevel {
        PermissionLevel::ReadOnly
    }

    fn input_schema(&self) -> Value {
        json!({
            "type": "object",
            "properties": {
                "query": {
                    "type": "string",
                    "description": "Natural-language retrieval query for the Bedrock Knowledge Base."
                },
                "knowledge_base": {
                    "type": "string",
                    "description": "Optional configured Knowledge Base name/alias, Bedrock knowledge base id, or ARN. Omit when a default or single Knowledge Base is configured."
                },
                "number_of_results": {
                    "type": "integer",
                    "minimum": 1,
                    "maximum": 20,
                    "description": "Optional result count override. Defaults to configured value or 5."
                },
                "filter": {
                    "type": "object",
                    "description": "Optional Bedrock vector search filter JSON."
                },
                "retrieval_configuration": {
                    "type": "object",
                    "description": "Advanced Bedrock retrievalConfiguration JSON. Overrides number_of_results/filter when supplied."
                },
                "next_token": {
                    "type": "string",
                    "description": "Optional pagination token returned by a previous retrieve call."
                }
            },
            "required": ["query"]
        })
    }

    async fn execute(&self, input: Value, ctx: &ToolContext) -> ToolResult {
        let params: BedrockKnowledgeBaseRetrieveInput = match serde_json::from_value(input) {
            Ok(params) => params,
            Err(e) => return ToolResult::error(format!("Invalid input: {}", e)),
        };
        if params.query.trim().is_empty() {
            return ToolResult::error("query must not be empty");
        }

        let provider_options = ctx
            .config
            .provider_configs
            .get("amazon-bedrock")
            .map(|provider| &provider.options);
        let configured_kbs = parse_configured_knowledge_bases(provider_options);
        let default_kb = default_knowledge_base(provider_options);
        let selection = match select_knowledge_base(
            params.knowledge_base.as_deref(),
            default_kb.as_deref(),
            &configured_kbs,
        ) {
            Ok(selection) => selection,
            Err(message) => return ToolResult::error(message),
        };

        if let Err(e) = ctx.check_permission(
            self.name(),
            &format!("Retrieve from Bedrock Knowledge Base {}", selection.label),
            true,
        ) {
            return ToolResult::error(e.to_string());
        }

        let region = selection
            .configured
            .as_ref()
            .and_then(|kb| kb.region.clone())
            .or_else(|| {
                ctx.config
                    .provider_configs
                    .get("amazon-bedrock")
                    .and_then(|provider| provider.region.clone())
            })
            .or_else(|| std::env::var("AWS_REGION").ok())
            .or_else(|| std::env::var("AWS_DEFAULT_REGION").ok())
            .unwrap_or_else(|| "us-east-1".to_string());
        let number_of_results = match resolve_number_of_results(
            params.number_of_results,
            selection
                .configured
                .as_ref()
                .and_then(|kb| kb.number_of_results),
        ) {
            Ok(value) => value,
            Err(message) => return ToolResult::error(message),
        };
        let retrieval_configuration = resolve_retrieval_configuration(
            params.retrieval_configuration,
            params.filter,
            selection
                .configured
                .as_ref()
                .and_then(|kb| kb.retrieval_configuration.clone()),
            number_of_results,
        );

        let client = match claurst_api::BedrockKnowledgeBaseClient::from_env_with_region(
            region.clone(),
        ) {
            Some(client) => client,
            None => {
                return ToolResult::error(
                    "Bedrock Knowledge Base retrieval requires AWS_ACCESS_KEY_ID and \
                     AWS_SECRET_ACCESS_KEY temporary credentials, or AWS_BEARER_TOKEN_BEDROCK. \
                     Launch Claurst through the Melange Bedrock role wrapper or export AWS credentials.",
                );
            }
        };

        let response = match client
            .retrieve(claurst_api::BedrockKnowledgeBaseRetrieveRequest {
                knowledge_base_id: selection.id.clone(),
                query: params.query.clone(),
                retrieval_configuration: Some(retrieval_configuration.clone()),
                next_token: params.next_token.clone(),
            })
            .await
        {
            Ok(response) => response,
            Err(e) => {
                return ToolResult::error(format!("Bedrock Knowledge Base retrieve failed: {}", e));
            }
        };

        // The tool calls the standard managed Knowledge Base Retrieve API,
        // which AWS prices per API call. Record it as direct service cost so
        // the session meter includes retrieval without distorting token usage.
        ctx.cost_tracker
            .add_service_cost_usd(MANAGED_KB_STANDARD_RETRIEVE_COST_USD);

        let output = json!({
            "knowledge_base": {
                "id": selection.id,
                "name": selection.configured.as_ref().and_then(|kb| kb.name.clone()),
                "description": selection.configured.as_ref().and_then(|kb| kb.description.clone()),
                "region": client.region(),
            },
            "query": params.query,
            "retrieval_configuration": retrieval_configuration,
            "guardrail_action": response.guardrail_action,
            "next_token": response.next_token,
            "results": response
                .retrieval_results
                .into_iter()
                .enumerate()
                .map(|(index, result)| {
                    json!({
                        "index": index + 1,
                        "score": result.score,
                        "document_id": result.document_id,
                        "content_type": result.content_type,
                        "text": result.text.map(|text| truncate_chunk_text(&text)),
                        "location": result.location,
                        "metadata": result.metadata,
                    })
                })
                .collect::<Vec<_>>(),
        });

        ToolResult::success(
            serde_json::to_string_pretty(&output).unwrap_or_else(|_| output.to_string()),
        )
        .with_metadata(json!({
            "type": "bedrock_knowledge_base_retrieve",
            "knowledge_base": selection.label,
            "region": client.region(),
            "result_count": output["results"].as_array().map_or(0, Vec::len),
            "estimated_cost_usd": MANAGED_KB_STANDARD_RETRIEVE_COST_USD,
        }))
    }
}

fn parse_configured_knowledge_bases(
    provider_options: Option<&std::collections::HashMap<String, Value>>,
) -> Vec<KnowledgeBaseToolConfig> {
    let Some(options) = provider_options else {
        return Vec::new();
    };

    let mut configured = Vec::new();
    for key in ["knowledgeBases", "knowledge_bases"] {
        if let Some(items) = options.get(key).and_then(Value::as_array) {
            configured.extend(items.iter().filter_map(parse_knowledge_base_config));
        }
    }
    for key in ["knowledgeBase", "knowledge_base"] {
        if let Some(item) = options.get(key) {
            if let Some(config) = parse_knowledge_base_config(item) {
                configured.push(config);
            }
        }
    }
    configured
}

fn parse_knowledge_base_config(value: &Value) -> Option<KnowledgeBaseToolConfig> {
    let obj = value.as_object()?;
    let id = first_string(obj, &["id", "knowledgeBaseId", "knowledge_base_id"])?;
    Some(KnowledgeBaseToolConfig {
        name: first_string(obj, &["name", "alias"]),
        id,
        description: first_string(obj, &["description"]),
        region: first_string(obj, &["region"]),
        number_of_results: first_u32(obj, &["numberOfResults", "number_of_results"]),
        retrieval_configuration: obj
            .get("retrievalConfiguration")
            .or_else(|| obj.get("retrieval_configuration"))
            .cloned(),
    })
}

fn first_string(obj: &serde_json::Map<String, Value>, keys: &[&str]) -> Option<String> {
    keys.iter()
        .find_map(|key| obj.get(*key).and_then(Value::as_str))
        .filter(|value| !value.trim().is_empty())
        .map(str::to_string)
}

fn first_u32(obj: &serde_json::Map<String, Value>, keys: &[&str]) -> Option<u32> {
    keys.iter()
        .find_map(|key| obj.get(*key).and_then(Value::as_u64))
        .and_then(|value| u32::try_from(value).ok())
}

fn default_knowledge_base(
    provider_options: Option<&std::collections::HashMap<String, Value>>,
) -> Option<String> {
    let options = provider_options?;
    options
        .get("defaultKnowledgeBase")
        .or_else(|| options.get("default_knowledge_base"))
        .and_then(Value::as_str)
        .filter(|value| !value.trim().is_empty())
        .map(str::to_string)
}

fn select_knowledge_base(
    requested: Option<&str>,
    default_kb: Option<&str>,
    configured: &[KnowledgeBaseToolConfig],
) -> Result<KnowledgeBaseSelection, String> {
    let requested = requested
        .filter(|value| !value.trim().is_empty())
        .or(default_kb);

    if let Some(requested) = requested {
        if let Some(config) = configured
            .iter()
            .find(|config| kb_matches(config, requested))
        {
            return Ok(KnowledgeBaseSelection {
                id: config.id.clone(),
                label: config.name.clone().unwrap_or_else(|| config.id.clone()),
                configured: Some(config.clone()),
            });
        }
        return Ok(KnowledgeBaseSelection {
            id: requested.to_string(),
            label: requested.to_string(),
            configured: None,
        });
    }

    match configured {
        [config] => Ok(KnowledgeBaseSelection {
            id: config.id.clone(),
            label: config
                .name
                .clone()
                .unwrap_or_else(|| config.id.clone()),
            configured: Some(config.clone()),
        }),
        [] => Err(
            "No Bedrock Knowledge Base configured. Add provider_configs.amazon-bedrock.options.knowledgeBases or pass knowledge_base explicitly.".to_string(),
        ),
        _ => Err(
            "Multiple Bedrock Knowledge Bases are configured. Pass knowledge_base with a configured name, id, or ARN.".to_string(),
        ),
    }
}

fn kb_matches(config: &KnowledgeBaseToolConfig, requested: &str) -> bool {
    config.id == requested || config.name.as_deref() == Some(requested)
}

fn resolve_number_of_results(
    requested: Option<u32>,
    configured: Option<u32>,
) -> Result<u32, String> {
    let value = requested.or(configured).unwrap_or(5);
    if value == 0 {
        return Err("number_of_results must be at least 1".to_string());
    }
    if value > 20 {
        return Err("number_of_results is capped at 20 to keep tool results usable".to_string());
    }
    Ok(value)
}

fn resolve_retrieval_configuration(
    requested: Option<Value>,
    filter: Option<Value>,
    configured: Option<Value>,
    number_of_results: u32,
) -> Value {
    if let Some(requested) = requested {
        return requested;
    }

    let mut config = configured.unwrap_or_else(|| json!({}));
    if !config.is_object() {
        config = json!({});
    }
    let config_obj = config
        .as_object_mut()
        .expect("retrieval configuration was normalized to an object");
    let vector = config_obj
        .entry("vectorSearchConfiguration".to_string())
        .or_insert_with(|| json!({}));
    if !vector.is_object() {
        *vector = json!({});
    }
    let vector_obj = vector
        .as_object_mut()
        .expect("vector search configuration was normalized to an object");
    vector_obj.insert("numberOfResults".to_string(), json!(number_of_results));
    if let Some(filter) = filter {
        vector_obj.insert("filter".to_string(), filter);
    }
    config
}

fn truncate_chunk_text(text: &str) -> String {
    const MAX_CHARS: usize = 4_000;
    if text.chars().count() <= MAX_CHARS {
        return text.to_string();
    }
    let mut truncated: String = text.chars().take(MAX_CHARS).collect();
    truncated.push_str("\n[truncated]");
    truncated
}

#[cfg(test)]
mod tests {
    use super::*;

    fn provider_options() -> std::collections::HashMap<String, Value> {
        std::collections::HashMap::from([(
            "knowledgeBases".to_string(),
            json!([
                {
                    "name": "melange",
                    "id": "KB12345678",
                    "description": "Melange project knowledge",
                    "region": "us-west-2",
                    "numberOfResults": 3
                }
            ]),
        )])
    }

    #[test]
    fn parses_configured_knowledge_bases() {
        let options = provider_options();
        let configured = parse_configured_knowledge_bases(Some(&options));

        assert_eq!(configured.len(), 1);
        assert_eq!(configured[0].name.as_deref(), Some("melange"));
        assert_eq!(configured[0].id, "KB12345678");
        assert_eq!(configured[0].number_of_results, Some(3));
    }

    #[test]
    fn selects_single_configured_knowledge_base_by_default() {
        let options = provider_options();
        let configured = parse_configured_knowledge_bases(Some(&options));
        let selection = select_knowledge_base(None, None, &configured).expect("selection");

        assert_eq!(selection.id, "KB12345678");
        assert_eq!(selection.label, "melange");
    }

    #[test]
    fn retrieval_configuration_applies_result_count_and_filter() {
        let config = resolve_retrieval_configuration(
            None,
            Some(json!({
                "equals": {
                    "key": "domain",
                    "value": "teamcity"
                }
            })),
            None,
            4,
        );

        assert_eq!(
            config["vectorSearchConfiguration"]["numberOfResults"],
            json!(4)
        );
        assert_eq!(
            config["vectorSearchConfiguration"]["filter"]["equals"]["key"],
            json!("domain")
        );
    }

    #[test]
    fn explicit_retrieval_configuration_wins() {
        let config = resolve_retrieval_configuration(
            Some(json!({
                "vectorSearchConfiguration": {
                    "numberOfResults": 9
                }
            })),
            Some(json!({"equals": {"key": "ignored", "value": true}})),
            None,
            3,
        );

        assert_eq!(
            config["vectorSearchConfiguration"]["numberOfResults"],
            json!(9)
        );
        assert!(config["vectorSearchConfiguration"].get("filter").is_none());
    }
}
