# Prompt for Reviewing the Incremental Condition Cache Branch

Use this prompt with Claude Code to review and evaluate the changes on the `abernix/incremental-condition-cache` branch for upstream integration into the Apollo Router's query planner.

---

## Context

The branch `abernix/incremental-condition-cache` contains query planner performance optimizations for `apollo-federation`. The key changes are:

1. **Hoisting `ConditionResolverCache` from per-request to per-schema lifetime** — condition edge resolutions now persist across `build_query_plan` calls, yielding 18-30x warm speedup on condition-heavy queries at 65+ subgraphs.

2. **O(1) field-edge index** — replaces O(out_edges) linear scans in `edge_for_field` with a pre-computed HashMap, cutting 70-78% of open-branches loop time.

3. **SemanticEdgeId** — schema-stable identifiers for QueryGraph edges/nodes that survive graph reconstruction, enabling cache carryover across schema reloads.

4. **Cache carryover on schema reload** — `import_from()` transfers applicable condition resolutions to the new planner on schema change, eliminating cold-start for unchanged edges.

5. **Fuzz infrastructure** (`apollo-federation-fuzz/`) — differential testing, warm-cache measurement, and continuous hot-swap validation modes.

## Review Tasks

### 1. Correctness Audit

Check the following files for soundness:

- `apollo-federation/src/query_graph/condition_resolver.rs` — Review the `SharedConditionResolverCache` and `import_from()` logic. Verify:
  - Guard conditions (empty context, empty excluded_conditions, no extra_conditions) are preserved
  - The "portable entry" filter correctly skips entries with OpPathTree references
  - `SemanticEdgeId` lookup correctness during import

- `apollo-federation/src/query_graph/mod.rs` — Review `SemanticEdgeId` and `SemanticNodeId` definitions. Verify:
  - The identity is complete (captures all semantically-meaningful edge properties)
  - Hash/Eq implementations are sound
  - `edge_index_for_semantic_id()` and `semantic_id_for_edge()` are consistent

- `apollo-federation/src/query_plan/query_planner.rs` — Review how the shared cache is threaded through to traversals

### 2. Performance Validation

Run the measurement test to verify speedup claims:

```bash
# Run the large-schema cache benchmark (65 subgraphs)
cargo test -p apollo-federation measure_large_schema_cache_performance -- --nocapture --ignored

# Run the hot-swap validation (if apollo-federation-fuzz is in the workspace)
cargo run -p apollo-federation-fuzz --bin fuzz -- \
  --report hot-swap --iterations 5 --schema-versions 20 \
  --ops-per-version 50 --preset wide --num-extensions 30
```

### 3. Risk Assessment

Key questions for the reviewer:

- **Lock contention**: The shared cache uses `Arc<Mutex<>>`. Is this acceptable for the router's concurrency model, or should it be `DashMap`? The cache is read-heavy after warmup with O(1) lock holds.

- **Memory growth**: The condition cache is bounded by schema topology (~700 entries at 65 subgraphs). Is this acceptable without explicit LRU eviction? What about ExcludedDestinations combinatorial explosion on types with many keys?

- **Cache carryover safety**: `import_from()` skips entries with `path_tree: Some(...)`. This is conservative — it means the most useful entries (Satisfied with a path tree showing how to resolve the condition) are NOT carried over. They're lazily recomputed. Is this the right tradeoff, or should we invest in OpPathTree remapping?

- **Production workloads**: At scale (billions of unique queries, hundreds of subgraphs), does the condition cache growth pattern hold? The cache is keyed on `(SemanticEdgeId, ExcludedDestinations)` which is O(condition_edges * excluded_destination_patterns). Need to verify this stays bounded.

### 4. Integration Path

The changes are in `apollo-federation` (the planner crate), not `apollo-router`. Integration into the router requires:

- Wiring the `SharedConditionResolverCache` through `QueryPlannerService` lifecycle
- On schema reload in `CachingQueryPlanner`, calling `import_from()` on the new planner's cache with the old planner's cache
- Deciding whether `experimental_reuse_query_plans` interacts with or is superseded by this

### 5. Test Coverage Check

```bash
# Run all federation tests
cargo test -p apollo-federation

# Run the carryover differential tests
cargo test -p apollo-federation cache_carryover

# Run the SemanticEdgeId tests
cargo test -p apollo-federation semantic_edge_id

# Run the fuzzer smoke test
cargo run -p apollo-federation-fuzz --bin fuzz -- \
  --report carryover --iterations 100 --mutate-schema --smoke-fixture
```

### 6. What to Look For in Code Review

- Any place where `EdgeIndex` is used as a long-lived cache key (should be `SemanticEdgeId`)
- Any place where the condition cache is accessed without holding the lock for the minimum duration
- Any assumption that `ExcludedDestinations` patterns are bounded (they may not be for types with many keys across many subgraphs)
- Any test that asserts on `EdgeIndex` numeric values (these are unstable)
- The O(1) field-edge index (`field_edge_index` on `QueryGraph`) — verify it's maintained correctly when the graph is mutated (it shouldn't be; the graph is immutable after construction)
