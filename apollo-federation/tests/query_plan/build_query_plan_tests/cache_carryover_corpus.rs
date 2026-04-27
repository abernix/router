//! Corpus-based differential tests for condition cache carryover.
//!
//! These tests load ALL pre-composed supergraph fixtures (202 files) and verify
//! that cache carryover via `new_with_previous_cache()` never corrupts plans.
//!
//! Test strategies:
//! 1. Same-schema carryover: V1 plans queries, V2 is built from the same schema
//!    with V1's cache. Plans must be identical.
//! 2. Warm-cache replay: Plan queries, then re-plan via the same planner. The
//!    second plan (using cached condition resolutions) must match the first.
//! 3. Schema mutation carryover: V1 plans queries, schema is mutated (field
//!    added), V2 is built from the mutated schema with V1's cache. V2 plans
//!    must match a fresh V2 planner (no carryover) — proving carried-over
//!    entries don't corrupt plans after schema changes.
//!
//! Run with: cargo test -p apollo-federation -- cache_carryover_corpus --nocapture

use apollo_federation::query_plan::query_planner::QueryPlanOptions;
use apollo_federation::query_plan::query_planner::QueryPlanner;
use apollo_federation::query_plan::query_planner::QueryPlannerConfig;
use apollo_federation::Supergraph;

/// Reads all `.graphql` supergraph fixtures, sorted by filename.
fn load_supergraph_entries() -> Vec<std::fs::DirEntry> {
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
    entries
}

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
    let entries = load_supergraph_entries();

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
    let entries = load_supergraph_entries();

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
    let entries = load_supergraph_entries();

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

/// Mutate a supergraph SDL by adding a new leaf field to the first non-Query
/// object type that has a `@join__field` directive. Returns `None` if the
/// schema can't be mutated (e.g., no suitable type found).
///
/// The mutation adds `_carryoverTestField: String @join__field(graph: <FIRST_GRAPH>)`
/// to the first eligible type. This is a minimal, non-breaking schema change
/// that exercises cache carryover across a real schema delta.
fn mutate_supergraph_add_field(schema_sdl: &str) -> Option<String> {
    // Extract the first join__Graph enum value (e.g., "SUBGRAPH1")
    // Must be inside the `enum join__Graph { ... }` block, not the directive definition
    let graph_enum_value = {
        let mut in_enum = false;
        let mut found = None;
        for line in schema_sdl.lines() {
            let trimmed = line.trim();
            if trimmed.starts_with("enum join__Graph") {
                in_enum = true;
                continue;
            }
            if in_enum {
                if trimmed == "}" {
                    break;
                }
                // Lines look like: SUBGRAPH1 @join__graph(name: "Subgraph1", url: "none")
                if trimmed.contains("@join__graph(") {
                    if let Some(name) = trimmed.split_whitespace().next() {
                        found = Some(name.to_string());
                        break;
                    }
                }
            }
        }
        found?
    };

    // Find the first non-Query, non-Mutation, non-Subscription object type
    // that has fields with @join__field. We look for a closing `}` after a
    // `type Foo` block that isn't a root type.
    let mut result = String::with_capacity(schema_sdl.len() + 200);
    let mut inserted = false;
    let mut in_eligible_type = false;
    let mut brace_depth = 0u32;

    for line in schema_sdl.lines() {
        let trimmed = line.trim();

        // Detect type definitions
        if !inserted && trimmed.starts_with("type ") && !trimmed.starts_with("type Query")
            && !trimmed.starts_with("type Mutation")
            && !trimmed.starts_with("type Subscription")
        {
            // Check it has @join__type (it's a federated type, not a helper)
            if trimmed.contains("@join__type") || {
                // The @join__type might be on the next line(s)
                // Heuristic: if this is a real entity type, we'll see @join__field in its body
                true
            } {
                in_eligible_type = true;
                brace_depth = 0;
            }
        }

        if in_eligible_type {
            brace_depth += trimmed.matches('{').count() as u32;
            brace_depth = brace_depth.saturating_sub(trimmed.matches('}').count() as u32);

            // Insert our field just before the closing brace of the first eligible type
            if brace_depth == 0 && trimmed.contains('}') && !inserted {
                result.push_str(&format!(
                    "  _carryoverTestField: String @join__field(graph: {})\n",
                    graph_enum_value
                ));
                inserted = true;
                in_eligible_type = false;
            }
        }

        result.push_str(line);
        result.push('\n');
    }

    if inserted {
        Some(result)
    } else {
        None
    }
}

/// The critical test: schema changes with cache carryover.
///
/// For each supergraph fixture:
/// 1. Build planner V1, plan queries to populate the condition cache
/// 2. Mutate the schema (add a field to a type)
/// 3. Build planner V2-fresh from mutated schema (no cache) — this is the reference
/// 4. Build planner V2-carryover from mutated schema WITH V1's cache
/// 5. Assert: V2-carryover plans == V2-fresh plans for all queries
///
/// This proves that carried-over cache entries from the old schema don't cause
/// the new planner to produce different (wrong) plans. The comparison is against
/// a fresh V2, not against V1 — because the schema changed, V1's plans may
/// legitimately differ from V2's.
#[test]
fn carryover_across_schema_mutation_matches_fresh_planner() {
    let entries = load_supergraph_entries();

    let mut total_schemas = 0;
    let mut total_queries_tested = 0;
    let mut mutated_schemas = 0;
    let mut cache_entries_imported = 0;
    let mut failures: Vec<String> = Vec::new();

    for entry in &entries {
        let path = entry.path();
        let name = path
            .file_stem()
            .unwrap()
            .to_string_lossy()
            .to_string();

        let schema_str = std::fs::read_to_string(&path).unwrap();

        // Parse V1 supergraph
        let supergraph_v1 = match Supergraph::new(&schema_str) {
            Ok(sg) => sg,
            Err(_) => continue,
        };

        let api_v1 = match supergraph_v1
            .to_api_schema(apollo_federation::ApiSchemaOptions::default())
        {
            Ok(api) => api,
            Err(_) => continue,
        };

        let planner_v1 = match QueryPlanner::new(&supergraph_v1, QueryPlannerConfig::default()) {
            Ok(p) => p,
            Err(_) => continue,
        };

        total_schemas += 1;

        // Generate and plan queries through V1 to populate cache
        let queries = generate_queries_for_schema(&api_v1, 10);
        let mut valid_queries: Vec<(
            String,
            apollo_compiler::validation::Valid<apollo_compiler::ExecutableDocument>,
        )> = Vec::new();

        for query_str in &queries {
            let doc = match apollo_compiler::ExecutableDocument::parse_and_validate(
                api_v1.schema(),
                query_str,
                "corpus_test.graphql",
            ) {
                Ok(d) => d,
                Err(_) => continue,
            };

            // Plan through V1 to populate cache
            if planner_v1
                .build_query_plan(&doc, None, QueryPlanOptions::default())
                .is_ok()
            {
                valid_queries.push((query_str.clone(), doc));
            }
        }

        if valid_queries.is_empty() {
            continue;
        }

        // Mutate the schema
        let mutated_sdl = match mutate_supergraph_add_field(&schema_str) {
            Some(s) => s,
            None => continue,
        };

        // Parse mutated schema — if mutation broke the schema, skip
        let supergraph_v2 = match Supergraph::new(&mutated_sdl) {
            Ok(sg) => sg,
            Err(_) => continue,
        };

        let api_v2 = match supergraph_v2
            .to_api_schema(apollo_federation::ApiSchemaOptions::default())
        {
            Ok(api) => api,
            Err(_) => continue,
        };

        mutated_schemas += 1;

        // Build V2-fresh (no carryover) — the reference planner
        let planner_v2_fresh =
            match QueryPlanner::new(&supergraph_v2, QueryPlannerConfig::default()) {
                Ok(p) => p,
                Err(_) => continue,
            };

        // Build V2-carryover (with V1's cache)
        let planner_v2_carryover = match QueryPlanner::new_with_previous_cache(
            &supergraph_v2,
            QueryPlannerConfig::default(),
            Some(planner_v1.condition_resolver_cache()),
        ) {
            Ok(p) => p,
            Err(_) => {
                failures.push(format!(
                    "V2-carryover construction failed for [{name}]"
                ));
                continue;
            }
        };

        cache_entries_imported += planner_v2_carryover.condition_resolver_cache_len();

        // Re-validate queries against V2's API schema (the mutation might have
        // changed the schema in a way that makes some queries invalid, though
        // adding a field shouldn't)
        for (query_str, _) in &valid_queries {
            let doc = match apollo_compiler::ExecutableDocument::parse_and_validate(
                api_v2.schema(),
                query_str,
                "corpus_test.graphql",
            ) {
                Ok(d) => d,
                Err(_) => continue,
            };

            // Plan through V2-fresh (reference)
            let fresh_plan = match planner_v2_fresh
                .build_query_plan(&doc, None, QueryPlanOptions::default())
            {
                Ok(p) => p,
                Err(_) => continue,
            };

            // Plan through V2-carryover (test subject)
            let carryover_plan = match planner_v2_carryover
                .build_query_plan(&doc, None, QueryPlanOptions::default())
            {
                Ok(p) => p,
                Err(e) => {
                    failures.push(format!(
                        "V2-carryover planning failed for [{name}] query: {query_str}: {e}"
                    ));
                    continue;
                }
            };

            let fresh_str = fresh_plan.to_string();
            let carryover_str = carryover_plan.to_string();

            if fresh_str != carryover_str {
                failures.push(format!(
                    "SCHEMA MUTATION CARRYOVER MISMATCH in [{name}] query: {query_str}\n  \
                     Fresh:     {fresh_str}\n  \
                     Carryover: {carryover_str}"
                ));
            }

            total_queries_tested += 1;
        }
    }

    eprintln!(
        "\n--- Schema mutation carryover corpus test ---\n  \
         Schemas tested: {total_schemas}\n  \
         Schemas successfully mutated: {mutated_schemas}\n  \
         Total cache entries imported across all V2 planners: {cache_entries_imported}\n  \
         Queries tested: {total_queries_tested}\n  \
         Failures: {}\n",
        failures.len()
    );

    if !failures.is_empty() {
        for f in &failures[..failures.len().min(10)] {
            eprintln!("  {f}\n");
        }
        panic!(
            "{} schema mutation carryover mismatches found",
            failures.len()
        );
    }

    // Sanity: we should have successfully mutated and tested a good fraction
    assert!(
        mutated_schemas >= 50,
        "Expected to successfully mutate at least 50 schemas, only mutated {mutated_schemas}"
    );
    assert!(
        total_queries_tested >= 100,
        "Expected to test at least 100 queries after mutation, only tested {total_queries_tested}"
    );
}
