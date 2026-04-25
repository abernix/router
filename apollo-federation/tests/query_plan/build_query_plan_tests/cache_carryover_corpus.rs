//! Corpus-based differential tests for condition cache carryover.
//!
//! These tests load ALL pre-composed supergraph fixtures (202 files) and verify
//! that cache carryover via `new_with_previous_cache()` never corrupts plans.
//!
//! Two test strategies:
//! 1. Same-schema carryover: V1 plans queries, V2 is built from the same schema
//!    with V1's cache. Plans must be identical.
//! 2. Warm-cache replay: Plan queries, then re-plan via the same planner. The
//!    second plan (using cached condition resolutions) must match the first.
//!
//! Run with: cargo test -p apollo-federation -- cache_carryover_corpus --nocapture

use apollo_federation::query_plan::query_planner::QueryPlanOptions;
use apollo_federation::query_plan::query_planner::QueryPlanner;
use apollo_federation::query_plan::query_planner::QueryPlannerConfig;
use apollo_federation::Supergraph;

const SUPERGRAPHS_DIR: &str = "tests/query_plan/supergraphs";

/// Generates simple queries from the API schema's Query type.
///
/// For each field on Query, generates `{ fieldName }` (leaf) or
/// `{ fieldName { __typename } }` (composite). Returns up to `max` queries.
fn generate_queries_for_schema(
    api_schema: &apollo_federation::schema::ValidFederationSchema,
    max: usize,
) -> Vec<String> {
    use apollo_compiler::schema::ExtendedType;

    let schema = api_schema.schema();
    let query_type_name = schema
        .root_operation(apollo_compiler::ast::OperationType::Query)
        .map(|n| n.as_str())
        .unwrap_or("Query");

    let query_type = match schema.types.get(query_type_name) {
        Some(ExtendedType::Object(obj)) => obj,
        _ => return vec![],
    };

    let mut queries = Vec::new();
    for (field_name, field_def) in &query_type.fields {
        if queries.len() >= max {
            break;
        }

        // Skip fields with required arguments — we can't generate valid values
        let has_required_args = field_def
            .arguments
            .iter()
            .any(|arg| arg.ty.is_non_null() && arg.default_value.is_none());
        if has_required_args {
            continue;
        }

        // Determine if the field's return type is a leaf or composite
        let inner_type_name = field_def.ty.inner_named_type();
        let is_leaf = matches!(
            schema.types.get(inner_type_name.as_str()),
            Some(ExtendedType::Scalar(_)) | Some(ExtendedType::Enum(_)) | None
        );

        if is_leaf {
            queries.push(format!("{{ {} }}", field_name));
        } else {
            // For composite types, select __typename as a minimal valid selection
            queries.push(format!("{{ {} {{ __typename }} }}", field_name));
        }
    }

    // Also generate a combined query with all leaf fields if we have multiple
    if queries.len() >= 2 {
        let combined_fields: Vec<String> = query_type
            .fields
            .iter()
            .take(max)
            .filter(|(_, fd)| {
                !fd.arguments
                    .iter()
                    .any(|arg| arg.ty.is_non_null() && arg.default_value.is_none())
            })
            .map(|(name, fd)| {
                let inner = fd.ty.inner_named_type();
                let is_leaf = matches!(
                    schema.types.get(inner.as_str()),
                    Some(ExtendedType::Scalar(_)) | Some(ExtendedType::Enum(_)) | None
                );
                if is_leaf {
                    name.to_string()
                } else {
                    format!("{} {{ __typename }}", name)
                }
            })
            .collect();
        if combined_fields.len() >= 2 {
            queries.push(format!("{{ {} }}", combined_fields.join(" ")));
        }
    }

    queries
}

/// Test that carrying over a condition cache from one planner to an identical-schema
/// planner produces byte-identical plans for all queries across all supergraph fixtures.
///
/// This is the broadest correctness test: 202 supergraph fixtures × auto-generated queries.
#[test]
fn carryover_produces_identical_plans_across_corpus() {
    let supergraphs_dir =
        std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join(SUPERGRAPHS_DIR);

    let mut entries: Vec<_> = std::fs::read_dir(&supergraphs_dir)
        .unwrap_or_else(|e| panic!("Cannot read {}: {e}", supergraphs_dir.display()))
        .filter_map(|e| e.ok())
        .filter(|e| {
            e.path()
                .extension()
                .map_or(false, |ext| ext == "graphql")
        })
        .collect();
    entries.sort_by_key(|e| e.file_name());

    let mut total_schemas = 0;
    let mut total_queries_tested = 0;
    let mut skipped_schemas = 0;
    let mut failures: Vec<String> = Vec::new();

    for entry in &entries {
        let path = entry.path();
        let name = path
            .file_stem()
            .unwrap()
            .to_string_lossy()
            .to_string();

        let schema_str = std::fs::read_to_string(&path).unwrap();

        // Parse supergraph — some fixtures may use features we don't handle
        let supergraph = match Supergraph::new(&schema_str) {
            Ok(sg) => sg,
            Err(_) => {
                skipped_schemas += 1;
                continue;
            }
        };

        let api_schema = match supergraph
            .to_api_schema(apollo_federation::ApiSchemaOptions::default())
        {
            Ok(api) => api,
            Err(_) => {
                skipped_schemas += 1;
                continue;
            }
        };

        // Build planner V1
        let planner_v1 = match QueryPlanner::new(&supergraph, QueryPlannerConfig::default()) {
            Ok(p) => p,
            Err(_) => {
                skipped_schemas += 1;
                continue;
            }
        };

        total_schemas += 1;

        // Generate queries
        let queries = generate_queries_for_schema(&api_schema, 10);
        if queries.is_empty() {
            continue;
        }

        // Plan all queries through V1 to populate the cache
        let mut v1_plans: Vec<(String, String)> = Vec::new();
        for query_str in &queries {
            let doc = match apollo_compiler::ExecutableDocument::parse_and_validate(
                api_schema.schema(),
                query_str,
                "corpus_test.graphql",
            ) {
                Ok(d) => d,
                Err(_) => continue,
            };

            match planner_v1.build_query_plan(&doc, None, QueryPlanOptions::default()) {
                Ok(plan) => {
                    v1_plans.push((query_str.clone(), plan.to_string()));
                }
                Err(_) => continue,
            }
        }

        if v1_plans.is_empty() {
            continue;
        }

        // Build planner V2 with V1's cache (same schema)
        let planner_v2 = QueryPlanner::new_with_previous_cache(
            &supergraph,
            QueryPlannerConfig::default(),
            Some(planner_v1.condition_resolver_cache()),
        )
        .expect("V2 planner construction should succeed");

        // Re-plan all queries through V2 and compare
        for (query_str, v1_plan_str) in &v1_plans {
            let doc = apollo_compiler::ExecutableDocument::parse_and_validate(
                api_schema.schema(),
                query_str,
                "corpus_test.graphql",
            )
            .unwrap();

            let v2_plan = planner_v2
                .build_query_plan(&doc, None, QueryPlanOptions::default())
                .expect("V2 planning should succeed");

            let v2_plan_str = v2_plan.to_string();
            if v1_plan_str != &v2_plan_str {
                failures.push(format!(
                    "MISMATCH in [{name}] query: {query_str}\n  V1: {v1_plan_str}\n  V2: {v2_plan_str}"
                ));
            }
            total_queries_tested += 1;
        }
    }

    eprintln!(
        "\n--- Cache carryover corpus test ---\n  \
         Schemas tested: {total_schemas}\n  \
         Schemas skipped: {skipped_schemas}\n  \
         Queries tested: {total_queries_tested}\n  \
         Failures: {}\n",
        failures.len()
    );

    if !failures.is_empty() {
        for f in &failures[..failures.len().min(10)] {
            eprintln!("  {f}\n");
        }
        panic!(
            "{} plan mismatches found in corpus carryover test",
            failures.len()
        );
    }

    // Sanity check: we should have tested a reasonable number of schemas/queries
    assert!(
        total_schemas >= 100,
        "Expected to test at least 100 schemas, only tested {total_schemas}"
    );
    assert!(
        total_queries_tested >= 200,
        "Expected to test at least 200 queries, only tested {total_queries_tested}"
    );
}

/// Test that re-planning through a warm cache (same planner instance) produces
/// identical plans. This validates the Phase 0 invariant that the shared
/// condition resolver cache never changes plan output.
#[test]
fn warm_cache_replay_produces_identical_plans_across_corpus() {
    let supergraphs_dir =
        std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join(SUPERGRAPHS_DIR);

    let mut entries: Vec<_> = std::fs::read_dir(&supergraphs_dir)
        .unwrap_or_else(|e| panic!("Cannot read {}: {e}", supergraphs_dir.display()))
        .filter_map(|e| e.ok())
        .filter(|e| {
            e.path()
                .extension()
                .map_or(false, |ext| ext == "graphql")
        })
        .collect();
    entries.sort_by_key(|e| e.file_name());

    let mut total_schemas = 0;
    let mut total_queries_tested = 0;
    let mut failures: Vec<String> = Vec::new();

    for entry in &entries {
        let path = entry.path();
        let name = path
            .file_stem()
            .unwrap()
            .to_string_lossy()
            .to_string();

        let schema_str = std::fs::read_to_string(&path).unwrap();

        let supergraph = match Supergraph::new(&schema_str) {
            Ok(sg) => sg,
            Err(_) => continue,
        };

        let api_schema = match supergraph
            .to_api_schema(apollo_federation::ApiSchemaOptions::default())
        {
            Ok(api) => api,
            Err(_) => continue,
        };

        let planner = match QueryPlanner::new(&supergraph, QueryPlannerConfig::default()) {
            Ok(p) => p,
            Err(_) => continue,
        };

        total_schemas += 1;

        let queries = generate_queries_for_schema(&api_schema, 10);

        for query_str in &queries {
            let doc = match apollo_compiler::ExecutableDocument::parse_and_validate(
                api_schema.schema(),
                query_str,
                "corpus_test.graphql",
            ) {
                Ok(d) => d,
                Err(_) => continue,
            };

            // Plan twice through the same planner
            let plan1 = match planner.build_query_plan(&doc, None, QueryPlanOptions::default()) {
                Ok(p) => p,
                Err(_) => continue,
            };

            let plan2 = planner
                .build_query_plan(&doc, None, QueryPlanOptions::default())
                .expect("Second plan should succeed if first did");

            let plan1_str = plan1.to_string();
            let plan2_str = plan2.to_string();

            if plan1_str != plan2_str {
                failures.push(format!(
                    "WARM REPLAY MISMATCH in [{name}] query: {query_str}\n  \
                     Plan 1: {plan1_str}\n  Plan 2: {plan2_str}"
                ));
            }

            // Also verify deterministic invariant counters
            if plan1.statistics.evaluated_plan_count.get()
                != plan2.statistics.evaluated_plan_count.get()
            {
                failures.push(format!(
                    "PLAN COUNT MISMATCH in [{name}] query: {query_str}\n  \
                     Plan 1 count: {}, Plan 2 count: {}",
                    plan1.statistics.evaluated_plan_count.get(),
                    plan2.statistics.evaluated_plan_count.get(),
                ));
            }

            total_queries_tested += 1;
        }
    }

    eprintln!(
        "\n--- Warm cache replay corpus test ---\n  \
         Schemas tested: {total_schemas}\n  \
         Queries tested: {total_queries_tested}\n  \
         Failures: {}\n",
        failures.len()
    );

    if !failures.is_empty() {
        for f in &failures[..failures.len().min(10)] {
            eprintln!("  {f}\n");
        }
        panic!(
            "{} warm replay mismatches found in corpus test",
            failures.len()
        );
    }

    assert!(
        total_schemas >= 100,
        "Expected to test at least 100 schemas, only tested {total_schemas}"
    );
}

/// Test cache carryover with interleaved queries — plans query A, then B, then A again
/// on V1, then carries over to V2 and replays. This probes whether cross-query cache
/// pollution produces different plans after carryover.
#[test]
fn carryover_with_interleaved_queries_across_corpus() {
    let supergraphs_dir =
        std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join(SUPERGRAPHS_DIR);

    let mut entries: Vec<_> = std::fs::read_dir(&supergraphs_dir)
        .unwrap_or_else(|e| panic!("Cannot read {}: {e}", supergraphs_dir.display()))
        .filter_map(|e| e.ok())
        .filter(|e| {
            e.path()
                .extension()
                .map_or(false, |ext| ext == "graphql")
        })
        .collect();
    entries.sort_by_key(|e| e.file_name());

    let mut total_schemas = 0;
    let mut total_queries_tested = 0;
    let mut failures: Vec<String> = Vec::new();

    for entry in &entries {
        let path = entry.path();
        let name = path
            .file_stem()
            .unwrap()
            .to_string_lossy()
            .to_string();

        let schema_str = std::fs::read_to_string(&path).unwrap();

        let supergraph = match Supergraph::new(&schema_str) {
            Ok(sg) => sg,
            Err(_) => continue,
        };

        let api_schema = match supergraph
            .to_api_schema(apollo_federation::ApiSchemaOptions::default())
        {
            Ok(api) => api,
            Err(_) => continue,
        };

        let planner_v1 = match QueryPlanner::new(&supergraph, QueryPlannerConfig::default()) {
            Ok(p) => p,
            Err(_) => continue,
        };

        total_schemas += 1;

        let queries = generate_queries_for_schema(&api_schema, 10);
        if queries.len() < 2 {
            continue;
        }

        // Parse all valid docs
        let docs: Vec<(String, apollo_compiler::validation::Valid<apollo_compiler::ExecutableDocument>)> = queries
            .iter()
            .filter_map(|q| {
                apollo_compiler::ExecutableDocument::parse_and_validate(
                    api_schema.schema(),
                    q,
                    "corpus_test.graphql",
                )
                .ok()
                .map(|d| (q.clone(), d))
            })
            .collect();

        if docs.len() < 2 {
            continue;
        }

        // Interleaved planning: A, B, A, C, B, A, ...
        // This maximizes cache cross-pollution
        let mut reference_plans: std::collections::HashMap<String, String> =
            std::collections::HashMap::new();

        for (i, (query_str, doc)) in docs.iter().enumerate() {
            // Plan each query
            let plan = match planner_v1
                .build_query_plan(doc, None, QueryPlanOptions::default())
            {
                Ok(p) => p,
                Err(_) => continue,
            };
            reference_plans.insert(query_str.clone(), plan.to_string());

            // Re-plan earlier queries to check for cross-pollution
            if i > 0 {
                let (prev_q, prev_doc) = &docs[0];
                if let Ok(re_plan) =
                    planner_v1.build_query_plan(prev_doc, None, QueryPlanOptions::default())
                {
                    if let Some(ref_plan) = reference_plans.get(prev_q) {
                        if ref_plan != &re_plan.to_string() {
                            failures.push(format!(
                                "INTERLEAVE POLLUTION in [{name}] after planning query {i}, \
                                 re-plan of query 0 diverged"
                            ));
                        }
                    }
                }
            }
        }

        // Now carry over to V2 and replay all
        let planner_v2 = QueryPlanner::new_with_previous_cache(
            &supergraph,
            QueryPlannerConfig::default(),
            Some(planner_v1.condition_resolver_cache()),
        )
        .expect("V2 planner should succeed");

        for (query_str, doc) in &docs {
            let v2_plan = match planner_v2
                .build_query_plan(doc, None, QueryPlanOptions::default())
            {
                Ok(p) => p,
                Err(_) => continue,
            };

            if let Some(ref_plan) = reference_plans.get(query_str) {
                if ref_plan != &v2_plan.to_string() {
                    failures.push(format!(
                        "CARRYOVER MISMATCH in [{name}] query: {query_str}"
                    ));
                }
                total_queries_tested += 1;
            }
        }
    }

    eprintln!(
        "\n--- Interleaved carryover corpus test ---\n  \
         Schemas tested: {total_schemas}\n  \
         Queries tested: {total_queries_tested}\n  \
         Failures: {}\n",
        failures.len()
    );

    if !failures.is_empty() {
        for f in &failures[..failures.len().min(10)] {
            eprintln!("  {f}\n");
        }
        panic!(
            "{} interleaved carryover mismatches found",
            failures.len()
        );
    }
}
