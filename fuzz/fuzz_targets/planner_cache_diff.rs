//! Differential fuzz target for query planner cache carryover.
//!
//! Generates random valid GraphQL operations via apollo-smith against
//! a supergraph's API schema, then verifies that:
//! 1. Planning the same operation twice through a warm cache produces identical plans
//! 2. Carrying over the cache to a new planner produces identical plans
//!
//! Run with: cargo fuzz run planner_cache_diff -- -max_len=4096
#![no_main]

use libfuzzer_sys::fuzz_target;

use std::sync::OnceLock;

use apollo_federation::query_plan::query_planner::QueryPlanOptions;
use apollo_federation::query_plan::query_planner::QueryPlanner;
use apollo_federation::query_plan::query_planner::QueryPlannerConfig;
use apollo_federation::Supergraph;

/// Schema fixture — loaded once, reused across all fuzz iterations.
/// Using the example supergraph which has cross-subgraph @key, @requires, @provides.
const SUPERGRAPH_PATH: &str = "examples/graphql/supergraph.graphql";

struct FuzzState {
    supergraph: Supergraph,
    api_schema_sdl: String,
}

static STATE: OnceLock<FuzzState> = OnceLock::new();

fn get_state() -> &'static FuzzState {
    STATE.get_or_init(|| {
        let schema_str =
            std::fs::read_to_string(SUPERGRAPH_PATH).expect("cannot read supergraph");
        let supergraph = Supergraph::new(&schema_str).expect("cannot parse supergraph");
        let api_schema = supergraph
            .to_api_schema(apollo_federation::ApiSchemaOptions::default())
            .expect("cannot build api schema");
        // Serialize the API schema to SDL for apollo-smith
        let api_schema_sdl = api_schema.schema().to_string();
        FuzzState {
            supergraph,
            api_schema_sdl,
        }
    })
}

fuzz_target!(|data: &[u8]| {
    let state = get_state();

    // Generate a valid operation from the fuzz input using apollo-smith
    let operation_str = match generate_operation(data, &state.api_schema_sdl) {
        Some(op) => op,
        None => return, // Invalid fuzz input, skip
    };

    // Build planner V1 and plan the operation
    let planner_v1 = match QueryPlanner::new(&state.supergraph, QueryPlannerConfig::default()) {
        Ok(p) => p,
        Err(_) => return,
    };

    let api_schema = state
        .supergraph
        .to_api_schema(apollo_federation::ApiSchemaOptions::default())
        .unwrap();

    let doc = match apollo_compiler::ExecutableDocument::parse_and_validate(
        api_schema.schema(),
        &operation_str,
        "fuzz.graphql",
    ) {
        Ok(d) => d,
        Err(_) => return, // Operation doesn't validate against the schema
    };

    // Plan 1: cold
    let plan1 = match planner_v1.build_query_plan(&doc, None, QueryPlanOptions::default()) {
        Ok(p) => p,
        Err(_) => return, // Some operations can't be planned (e.g., introspection-only)
    };

    // Plan 2: warm cache (same planner)
    let plan2 = match planner_v1.build_query_plan(&doc, None, QueryPlanOptions::default()) {
        Ok(p) => p,
        Err(_) => panic!("Second plan failed but first succeeded for: {}", operation_str),
    };

    let plan1_str = plan1.to_string();
    let plan2_str = plan2.to_string();

    assert_eq!(
        plan1_str, plan2_str,
        "WARM REPLAY MISMATCH for operation:\n{operation_str}"
    );

    // Plan 3: carryover to new planner
    let planner_v2 = QueryPlanner::new_with_previous_cache(
        &state.supergraph,
        QueryPlannerConfig::default(),
        Some(planner_v1.condition_resolver_cache()),
    )
    .expect("V2 planner construction should succeed");

    // Re-parse against V2's API schema (same schema, but independent planner)
    let plan3 = planner_v2
        .build_query_plan(&doc, None, QueryPlanOptions::default())
        .expect("V2 planning should succeed if V1 did");

    let plan3_str = plan3.to_string();

    assert_eq!(
        plan1_str, plan3_str,
        "CARRYOVER MISMATCH for operation:\n{operation_str}"
    );
});

/// Generate a valid GraphQL operation from fuzz bytes using apollo-smith.
fn generate_operation(data: &[u8], api_schema_sdl: &str) -> Option<String> {
    use std::convert::TryFrom;

    use apollo_parser::Parser;
    use apollo_smith::Document;
    use apollo_smith::DocumentBuilder;
    use libfuzzer_sys::arbitrary::Unstructured;

    let parser = Parser::new(api_schema_sdl);
    let tree = parser.parse();
    if tree.errors().len() > 0 {
        return None;
    }

    let doc = Document::try_from(tree.document()).ok()?;
    let mut u = Unstructured::new(data);
    let mut builder = DocumentBuilder::with_document(&mut u, doc).ok()?;
    let op = builder.operation_definition().ok()??;
    Some(op.into())
}
