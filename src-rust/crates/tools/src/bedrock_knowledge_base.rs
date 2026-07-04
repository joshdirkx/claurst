use crate::{PermissionLevel, Tool, ToolContext, ToolResult};
use async_trait::async_trait;
use claurst_core::types::ToolDefinition;
use serde::Deserialize;
use serde_json::{Value, json};
use std::collections::HashMap;

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
pub struct BedrockKnowledgeBaseConfig {
    pub name: Option<String>,
    pub id: String,
    pub description: Option<String>,
    pub region: Option<String>,
    pub number_of_results: Option<u32>,
    pub retrieval_configuration: Option<Value>,
}

#[derive(Debug, Clone)]
pub struct BedrockKnowledgeBaseRuntimeConfig {
    pub configured: Vec<BedrockKnowledgeBaseConfig>,
    pub default_knowledge_base: Option<String>,
    pub provider_region: String,
}

impl BedrockKnowledgeBaseRuntimeConfig {
    pub fn is_configured(&self) -> bool {
        !self.configured.is_empty() || self.default_knowledge_base.is_some()
    }

    pub fn labels(&self) -> Vec<String> {
        let mut labels = self
            .configured
            .iter()
            .map(|kb| match &kb.name {
                Some(name) => format!("{} ({})", name, kb.id),
                None => kb.id.clone(),
            })
            .collect::<Vec<_>>();
        if labels.is_empty() {
            if let Some(default) = &self.default_knowledge_base {
                labels.push(default.clone());
            }
        }
        labels
    }

    pub fn default_label(&self) -> Option<String> {
        self.default_knowledge_base.clone().or_else(|| {
            if self.configured.len() == 1 {
                Some(
                    self.configured[0]
                        .name
                        .clone()
                        .unwrap_or_else(|| self.configured[0].id.clone()),
                )
            } else {
                None
            }
        })
    }
}

#[derive(Debug)]
struct KnowledgeBaseSelection {
    configured: Option<BedrockKnowledgeBaseConfig>,
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

    fn to_definition_with_context(
        &self,
        ctx: &ToolContext,
        provider_options: Option<&HashMap<String, Value>>,
    ) -> ToolDefinition {
        let mut definition = self.to_definition();
        let Some(runtime) = resolve_bedrock_knowledge_base_runtime(&ctx.config, provider_options) else {
            return definition;
        };
        if !runtime.is_configured() {
            return definition;
        }

        let labels = runtime.labels().join(", ");
        let default_text = runtime.default_label().unwrap_or_else(|| "none".to_string());
        definition.description = format!(
            "{} Configured for this Bedrock session: aliases or ids [{}], default [{}], region [{}]. \
Use this tool directly for Amazon Bedrock Knowledge Base retrieval; do not look for an MCP server. \
Omit knowledge_base when the default is appropriate.",
            self.description(),
            labels,
            default_text,
            runtime.provider_region,
        );

        if let Some(properties) = definition
            .input_schema
            .get_mut("properties")
            .and_then(Value::as_object_mut)
        {
            if let Some(knowledge_base) = properties
                .get_mut("knowledge_base")
                .and_then(Value::as_object_mut)
            {
                knowledge_base.insert(
                    "description".to_string(),
                    json!(format!(
                        "Optional configured Knowledge Base name/alias, Bedrock knowledge base id, or ARN. Configured aliases or ids: {}. Default: {}. Omit this field when the default is appropriate.",
                        labels,
                        default_text
                    )),
                );
            }
        }

        definition
    }

    async fn execute(&self, input: Value, ctx: &ToolContext) -> ToolResult {
        let params: BedrockKnowledgeBaseRetrieveInput = match serde_json::from_value(input) {
            Ok(params) => params,
            Err(e) => return ToolResult::error(format!("Invalid input: {}", e)),
        };
        if params.query.trim().is_empty() {
            return ToolResult::error("query must not be empty");
        }

        let runtime = resolve_bedrock_knowledge_base_runtime(&ctx.config, None);
        let configured_kbs = runtime
            .as_ref()
            .map(|runtime| runtime.configured.as_slice())
            .unwrap_or(&[]);
        let default_kb = runtime
            .as_ref()
            .and_then(|runtime| runtime.default_label());
        let selection = match select_knowledge_base(
            params.knowledge_base.as_deref(),
            default_kb.as_deref(),
            configured_kbs,
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
                runtime
                    .as_ref()
                    .map(|runtime| runtime.provider_region.clone())
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
                     Launch Claurst with AWS credentials or an AWS role wrapper.",
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

pub fn resolve_bedrock_knowledge_base_runtime(
    config: &claurst_core::config::Config,
    provider_options_override: Option<&HashMap<String, Value>>,
) -> Option<BedrockKnowledgeBaseRuntimeConfig> {
    if !matches!(
        config.provider.as_deref(),
        Some("amazon-bedrock" | "bedrock-mantle")
    ) {
        return None;
    }

    let active_provider_options = provider_options_override.or_else(|| {
        config
            .provider
            .as_deref()
            .and_then(|provider| config.provider_configs.get(provider))
            .map(|provider| &provider.options)
    });
    let amazon_bedrock_options = config
        .provider_configs
        .get("amazon-bedrock")
        .map(|provider| &provider.options);

    let mut configured = parse_configured_knowledge_bases(active_provider_options);
    if configured.is_empty() {
        configured = parse_configured_knowledge_bases(amazon_bedrock_options);
    }

    let default_knowledge_base = default_knowledge_base(active_provider_options)
        .or_else(|| default_knowledge_base(amazon_bedrock_options));

    let provider_region = config
        .provider
        .as_deref()
        .and_then(|provider| config.provider_configs.get(provider))
        .and_then(|provider| provider.region.clone())
        .or_else(|| {
            config
                .provider_configs
                .get("amazon-bedrock")
                .and_then(|provider| provider.region.clone())
        })
        .or_else(|| std::env::var("AWS_REGION").ok())
        .or_else(|| std::env::var("AWS_DEFAULT_REGION").ok())
        .unwrap_or_else(|| "us-east-1".to_string());

    Some(BedrockKnowledgeBaseRuntimeConfig {
        configured,
        default_knowledge_base,
        provider_region,
    })
}

pub fn describe_bedrock_knowledge_base_runtime(
    runtime: &BedrockKnowledgeBaseRuntimeConfig,
) -> String {
    let labels = runtime.labels().join(", ");
    let default_text = runtime.default_label().unwrap_or_else(|| "none".to_string());
    format!(
        "Bedrock KB configured: {} (default: {}, region: {})",
        labels, default_text, runtime.provider_region
    )
}

fn parse_configured_knowledge_bases(
    provider_options: Option<&HashMap<String, Value>>,
) -> Vec<BedrockKnowledgeBaseConfig> {
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

fn parse_knowledge_base_config(value: &Value) -> Option<BedrockKnowledgeBaseConfig> {
    let obj = value.as_object()?;
    let id = first_string(obj, &["id", "knowledgeBaseId", "knowledge_base_id"])?;
    Some(BedrockKnowledgeBaseConfig {
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
    provider_options: Option<&HashMap<String, Value>>,
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
    configured: &[BedrockKnowledgeBaseConfig],
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

fn kb_matches(config: &BedrockKnowledgeBaseConfig, requested: &str) -> bool {
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
    use claurst_core::config::{Config, PermissionMode, ProviderConfig};
    use claurst_core::permissions::AutoPermissionHandler;
    use std::path::PathBuf;
    use std::sync::{
        Arc,
        atomic::AtomicUsize,
    };

    fn provider_options() -> std::collections::HashMap<String, Value> {
        std::collections::HashMap::from([
            (
                "defaultKnowledgeBase".to_string(),
                json!("project-kb"),
            ),
            (
                "knowledgeBases".to_string(),
                json!([
                    {
                        "name": "project-kb",
                        "id": "KB12345678",
                        "description": "Project knowledge",
                        "region": "us-west-2",
                        "numberOfResults": 3
                    }
                ]),
            ),
        ])
    }

    fn config_with_provider(provider: &str) -> Config {
        let mut config = Config {
            provider: Some(provider.to_string()),
            ..Config::default()
        };
        config.provider_configs.insert(
            "amazon-bedrock".to_string(),
            ProviderConfig {
                region: Some("us-west-2".to_string()),
                options: provider_options(),
                ..ProviderConfig::default()
            },
        );
        if provider == "bedrock-mantle" {
            config.provider_configs.insert(
                "bedrock-mantle".to_string(),
                ProviderConfig {
                    region: Some("us-west-2".to_string()),
                    ..ProviderConfig::default()
                },
            );
        }
        config
    }

    fn test_context(config: Config) -> ToolContext {
        ToolContext {
            working_dir: PathBuf::from("/workspace"),
            permission_mode: PermissionMode::BypassPermissions,
            permission_handler: Arc::new(AutoPermissionHandler {
                mode: PermissionMode::BypassPermissions,
            }),
            cost_tracker: claurst_core::cost::CostTracker::new(),
            session_id: "test".to_string(),
            file_history: Arc::new(parking_lot::Mutex::new(
                claurst_core::file_history::FileHistory::new(),
            )),
            current_turn: Arc::new(AtomicUsize::new(0)),
            non_interactive: true,
            mcp_manager: None,
            config,
            managed_agent_config: None,
            completion_notifier: None,
            pending_permissions: None,
            permission_manager: None,
            user_question_tx: None,
        }
    }

    #[test]
    fn parses_configured_knowledge_bases() {
        let options = provider_options();
        let configured = parse_configured_knowledge_bases(Some(&options));

        assert_eq!(configured.len(), 1);
        assert_eq!(configured[0].name.as_deref(), Some("project-kb"));
        assert_eq!(configured[0].id, "KB12345678");
        assert_eq!(configured[0].number_of_results, Some(3));
    }

    #[test]
    fn runtime_resolves_amazon_bedrock_knowledge_base() {
        let config = config_with_provider("amazon-bedrock");
        let runtime = resolve_bedrock_knowledge_base_runtime(&config, None)
            .expect("amazon-bedrock runtime should resolve");

        assert!(runtime.is_configured());
        assert_eq!(runtime.default_label().as_deref(), Some("project-kb"));
        assert_eq!(runtime.configured[0].id, "KB12345678");
        assert_eq!(runtime.provider_region, "us-west-2");
    }

    #[test]
    fn runtime_falls_back_to_amazon_bedrock_config_for_mantle() {
        let config = config_with_provider("bedrock-mantle");
        let runtime = resolve_bedrock_knowledge_base_runtime(&config, None)
            .expect("bedrock-mantle runtime should resolve");

        assert!(runtime.is_configured());
        assert_eq!(runtime.default_label().as_deref(), Some("project-kb"));
        assert_eq!(runtime.configured[0].id, "KB12345678");
    }

    #[test]
    fn runtime_is_not_enabled_for_non_bedrock_providers() {
        let config = Config {
            provider: Some("openai".to_string()),
            ..Config::default()
        };

        assert!(resolve_bedrock_knowledge_base_runtime(&config, None).is_none());
    }

    #[test]
    fn tool_definition_advertises_configured_default_knowledge_base() {
        let ctx = test_context(config_with_provider("amazon-bedrock"));
        let definition = BedrockKnowledgeBaseRetrieveTool.to_definition_with_context(&ctx, None);

        assert!(definition.description.contains("project-kb"));
        assert!(definition.description.contains("KB12345678"));
        assert!(definition.description.contains("do not look for an MCP server"));
        assert!(definition.description.contains("Omit knowledge_base"));
        assert!(definition
            .input_schema
            .to_string()
            .contains("Default: project-kb"));
    }

    #[test]
    fn selects_single_configured_knowledge_base_by_default() {
        let options = provider_options();
        let configured = parse_configured_knowledge_bases(Some(&options));
        let selection = select_knowledge_base(None, None, &configured).expect("selection");

        assert_eq!(selection.id, "KB12345678");
        assert_eq!(selection.label, "project-kb");
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
