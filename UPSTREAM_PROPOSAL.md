# Upstream Proposal: Incremental Query Plan Optimization

## Overview

This branch (`abernix/incremental-condition-cache`) contains a series of query planner performance optimizations and supporting infrastructure. The work is structured as independently reviewable PRs that build on each other.

## Suggested PR Sequence

### PR 1: Hoist ConditionResolverCache to per-schema lifetime
**Commits:** `c8dc27af7` through `c8339e910`
**Impact:** 18-30x warm speedup on condition-heavy queries

The `ConditionResolverCache` currently lives per-`QueryPlanningTraversal` — it's created fresh for every `build_query_plan` call and discarded after. This PR hoists it to the `QueryPlanner` level so condition resolutions persist across queries for the lifetime of a schema version.

**Key changes:**
- `condition_resolver.rs`: Add `SharedConditionResolverCache` wrapping `Arc<Mutex<ConditionResolverCache>>`
- `query_planner.rs`: Hold shared cache, pass to each traversal
- `query_planning_traversal.rs`: Accept shared cache instead of creating fresh one

**Safety:** The existing guard conditions (skip caching when context is non-empty, excluded_conditions is non-empty, or extra_conditions is set) are preserved unchanged. The cache is strictly read-more-write-less after warmup. All existing tests pass unchanged.

**Measurements:** At 65 subgraphs (13 core + 50 dept extensions + 3 shareable), condition-heavy queries show:
- product_stress (53 fetches): cold 4,354ms → warm 240ms (18x)
- user_stress (51 fetches): similar speedup
- Cache entries stabilize at ~700 for this fixture

### PR 2: O(1) field-edge index for QueryGraph
**Commits:** `0ce07e9a9`, `5e455d03a`
**Impact:** 70-78% reduction in open-branches loop time

Replaces the O(out_edges) linear scan in `edge_for_field` with a pre-computed `HashMap<(NodeIndex, Name), Vec<EdgeIndex>>`. At 65 subgraphs where entity nodes have 54+ outgoing edges, this eliminates the dominant cost in the indirect-advance phase.

Additional optimization: before trial-advancing each indirect path, check if the tail node (object type) has the field via the O(1) index. 98% of indirect advance calls return None, so this avoids most advance overhead.

**Measurements (A/B, same machine, 65 subgraphs, debug build):**
| Config | product_stress loop us | user_stress loop us |
|---|---|---|
| Baseline | 163,414 | 66,472 |
| Index only | 38,407 (-76%) | 19,694 (-70%) |
| Index + pre-filter | 35,585 (-78%) | 16,905 (-75%) |

### PR 3: SemanticEdgeId for schema-stable cache identity
**Commits:** `b67232136` through `3d432499f`
**Impact:** Foundation for cache carryover across schema reloads

Introduces `SemanticEdgeId` and `SemanticNodeId` — schema-stable identifiers that survive `QueryGraph` reconstruction. Re-keys the `ConditionResolverCache` on `SemanticEdgeId` instead of `EdgeIndex`.

**Why:** `EdgeIndex` is a petgraph-internal numeric index that changes when the graph is rebuilt (even from the same schema). `SemanticEdgeId` captures the edge's semantic content (source type + subgraph, target type + subgraph, transition kind, conditions hash) and is stable across rebuilds.

**Tests:** Comprehensive stability tests verify SemanticEdgeId consistency across independent graph builds from identical schemas.

### PR 4: Condition cache carryover across schema reloads
**Commits:** `b6f4388d3` through `291ebda81`
**Impact:** Near-zero cold-start on schema reload for condition resolution

On schema reload, the new `QueryPlanner` imports applicable cache entries from the old planner via `SharedConditionResolverCache::import_from()`. Only "portable" entries (those without `OpPathTree` references to old-graph indices) are imported.

**Safety:** Entries with `path_tree: Some(...)` are skipped because the `OpPathTree` contains `NodeIndex`/`EdgeIndex` from the old graph. These entries are lazily recomputed on first access. Entries whose `SemanticEdgeId` doesn't exist in the new graph are dropped.

**Tests:**
- Corpus-based differential tests across 179 supergraph fixtures
- Differential fuzz target generating random schemas + operations
- Schema mutation carryover test (add field, verify plans match fresh)

### PR 5: Fuzzing infrastructure (apollo-federation-fuzz)
**Commits:** `93a2aa939` through `45dec2b79`
**Impact:** Comprehensive testing infrastructure for planner correctness

New `apollo-federation-fuzz` crate with multiple test modes:
- `--report warm`: cold vs warm cache measurement
- `--report carryover`: V1→V2 cache carryover correctness
- `--report hot-swap`: continuous V1→V2→...→Vn schema evolution with carryover validation
- Pathological preset schemas (wide/deep/mesh) for stress testing
- Long-running unbounded mode for CI soak testing

## Production Considerations

### Cache Eviction (for billion-query workloads)

The condition cache is naturally bounded by schema topology — entries are keyed on `(SemanticEdgeId, ExcludedDestinations)` which is a finite set determined by the graph structure. At 65 subgraphs we measured ~700 entries; even at hundreds of subgraphs this stays in the low thousands.

However, for future memoization layers (path enumeration, FDG fragments) where keys include selection sets, the cache would need cost-aware eviction:
- Track resolution cost per entry
- Evict cheap-to-recompute entries first
- Retain expensive multi-hop condition chains
- Set max entry count proportional to schema complexity

### Schema Hot-Swap Validation

The `--report hot-swap` mode validates correctness through long chains of schema mutations with cache carryover. It tests:
- Multiple mutation types (add field, rename field, add type, identity)
- Operations that span multiple schema versions
- Cache growth bounds through evolution chains

Recommended CI integration: nightly run with `--iterations 0 --schema-versions 50 --ops-per-version 100 --preset wide --num-extensions 50` for ~30 minutes.

### Lock Contention

The shared cache uses `Arc<Mutex<>>`. For the current condition cache (read-heavy after warmup, ~700 entries), mutex contention is negligible because:
1. Each lock hold is O(1) — a single HashMap lookup
2. The planner runs on the compute thread pool (typically 4-8 threads)
3. Condition resolution is the expensive part; the cache lookup is fast

If contention becomes measurable at higher concurrency, upgrade to `DashMap` sharded by edge identity.

## Measurement Infrastructure

The branch includes extensive phase-timing instrumentation (commits `fae92b666` through `afa0aba9a`) that decomposes `build_query_plan` into:
- `new_ns`: traversal construction
- `open_branches_loop_ns`: the main planning loop
- `best_plan_selection_ns`: plan selection from closed branches  
- `process_ns`: FDG optimization + PlanNode construction

And further subdivides each phase. These timers are behind `#[cfg(test)]` and have zero production cost, but they're invaluable for guiding optimization work. Consider keeping them as permanent test-only instrumentation.
