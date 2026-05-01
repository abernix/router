//! Adapter for the in-tree (HEAD) `apollo-federation` crate.
//!
//! Translates the neutral [`CommonConfig`]/[`CommonOptions`] from
//! [`crate::harness`] into the HEAD version's concrete planner config types,
//! runs the planner, and serializes the resulting `QueryPlan` to JSON.

use std::num::NonZeroU32;

use apollo_compiler::ExecutableDocument;
use apollo_compiler::collections::IndexSet;
use apollo_federation::Supergraph;
use apollo_federation::query_graph::IncrementalQueryGraphResult;
use apollo_federation::query_graph::build_federated_query_graph_incremental;
use apollo_federation::query_plan::query_planner::QueryPlanIncrementalDeliveryConfig;
use apollo_federation::query_plan::query_planner::QueryPlanOptions;
use apollo_federation::query_plan::query_planner::QueryPlanner;
use apollo_federation::query_plan::query_planner::QueryPlannerConfig;
use apollo_federation::query_plan::query_planner::QueryPlannerDebugConfig;
use apollo_federation::ApiSchemaOptions;

use apollo_federation::query_graph::SharedConditionResolverCache;

use crate::harness::{CommonConfig, CommonOptions, HarnessError, PlannerHarness};

const VERSION: &str = "head";

pub struct HeadPlanner {
    supergraph: Supergraph,
    config: QueryPlannerConfig,
    inner: QueryPlanner,
}

impl HeadPlanner {
    /// Returns a reference to the shared condition resolver cache for cache carryover.
    pub fn condition_resolver_cache(&self) -> &SharedConditionResolverCache {
        self.inner.condition_resolver_cache()
    }

    /// Returns the number of entries in the condition resolver cache.
    pub fn condition_resolver_cache_len(&self) -> usize {
        self.inner.condition_resolver_cache_len()
    }

    /// Build a new planner for the given schema, carrying over the condition
    /// resolver cache from a previous planner.
    pub fn build_with_previous_cache(
        supergraph_sdl: &str,
        cfg: &CommonConfig,
        previous_cache: &SharedConditionResolverCache,
    ) -> Result<Self, HarnessError> {
        let supergraph =
            Supergraph::new_with_router_specs(supergraph_sdl).map_err(|e| HarnessError::Supergraph {
                version: VERSION,
                detail: e.to_string(),
            })?;

        let planner_cfg = Self::make_config(cfg);

        let inner = QueryPlanner::new_with_previous_cache(
            &supergraph,
            planner_cfg.clone(),
            Some(previous_cache),
        )
        .map_err(|e| HarnessError::Construct {
            version: VERSION,
            detail: e.to_string(),
        })?;

        Ok(Self {
            supergraph,
            config: planner_cfg,
            inner,
        })
    }

    /// Build a new planner incrementally, reusing unchanged subgraph schemas from
    /// a previous planner. Also carries over the condition resolver cache.
    pub fn build_incremental(
        supergraph_sdl: &str,
        cfg: &CommonConfig,
        previous_planner: &HeadPlanner,
    ) -> Result<Self, HarnessError> {
        let supergraph =
            Supergraph::new_with_router_specs(supergraph_sdl).map_err(|e| HarnessError::Supergraph {
                version: VERSION,
                detail: e.to_string(),
            })?;

        let planner_cfg = Self::make_config(cfg);

        let inner = QueryPlanner::new_incremental(
            &supergraph,
            planner_cfg.clone(),
            Some(&previous_planner.inner),
        )
        .map_err(|e| HarnessError::Construct {
            version: VERSION,
            detail: e.to_string(),
        })?;

        Ok(Self {
            supergraph,
            config: planner_cfg,
            inner,
        })
    }

    /// Build with timing breakdown. Returns (planner, supergraph_parse_ms, planner_build_ms).
    pub fn build_with_timing(
        supergraph_sdl: &str,
        cfg: &CommonConfig,
    ) -> Result<(Self, u128, u128), HarnessError> {
        let t0 = std::time::Instant::now();
        let supergraph =
            Supergraph::new_with_router_specs(supergraph_sdl).map_err(|e| HarnessError::Supergraph {
                version: VERSION,
                detail: e.to_string(),
            })?;
        let parse_ms = t0.elapsed().as_millis();

        let planner_cfg = Self::make_config(cfg);

        let t1 = std::time::Instant::now();
        let inner = QueryPlanner::new(&supergraph, planner_cfg.clone()).map_err(|e| {
            HarnessError::Construct {
                version: VERSION,
                detail: e.to_string(),
            }
        })?;
        let build_ms = t1.elapsed().as_millis();

        Ok((Self {
            supergraph,
            config: planner_cfg,
            inner,
        }, parse_ms, build_ms))
    }

    /// Build incrementally with detailed timing breakdown.
    /// Returns (planner, supergraph_parse_ms, extract_ms, schema_graph_ms, federated_graph_ms).
    pub fn build_incremental_with_timing(
        supergraph_sdl: &str,
        cfg: &CommonConfig,
        previous_planner: &HeadPlanner,
    ) -> Result<(Self, u128, u128, u128, u128), HarnessError> {
        let t0 = std::time::Instant::now();
        let supergraph =
            Supergraph::new_with_router_specs(supergraph_sdl).map_err(|e| HarnessError::Supergraph {
                version: VERSION,
                detail: e.to_string(),
            })?;
        let parse_ms = t0.elapsed().as_millis();

        let planner_cfg = Self::make_config(cfg);

        let inner = QueryPlanner::new_incremental(
            &supergraph,
            planner_cfg.clone(),
            Some(&previous_planner.inner),
        )
        .map_err(|e| HarnessError::Construct {
            version: VERSION,
            detail: e.to_string(),
        })?;

        // Get the timing from the last build
        let timing = inner.last_build_timing();

        Ok((Self {
            supergraph,
            config: planner_cfg,
            inner,
        }, parse_ms, timing.0, timing.1, timing.2))
    }

    /// Serialize the query graph to portable bytes and then reconstruct a full QueryGraph.
    /// Measures both JSON and bincode formats.
    /// Returns (serialize_ms, reconstruct_reuse_ms, bincode_size, json_size, node_count, edge_count, timing_detail).
    pub fn measure_portable_full_roundtrip(
        &self,
        supergraph_sdl: &str,
    ) -> Result<(u128, u128, usize, usize, usize, usize, String), String> {
        use apollo_federation::query_graph::portable::PortableQueryGraph;
        use std::time::Instant;

        let graph = self.inner.query_graph();
        let existing_subgraphs = self.inner.subgraph_schemas();

        // Build the portable representation once.
        let portable = PortableQueryGraph::from_query_graph(graph, supergraph_sdl);
        let node_count = portable.node_count();
        let edge_count = portable.edge_count();

        // Measure bincode serialization.
        let t0 = Instant::now();
        let bincode_bytes = portable.to_bytes();
        let bincode_ser_ms = t0.elapsed().as_millis();
        let bincode_size = bincode_bytes.len();

        // Measure JSON serialization for comparison.
        let t1 = Instant::now();
        let json_bytes = portable.to_json();
        let json_ser_ms = t1.elapsed().as_millis();
        let json_size = json_bytes.len();

        // Measure full roundtrip from bincode with reused schemas (the fast path).
        let supergraph = Supergraph::new_with_router_specs(supergraph_sdl)
            .map_err(|e| format!("supergraph parse error: {e}"))?;
        let supergraph_schema = supergraph.schema.clone();

        let t2 = Instant::now();
        let restored = PortableQueryGraph::from_bytes(&bincode_bytes)
            .map_err(|e| format!("bincode deserialize error: {e}"))?;
        let (_restored_graph, timing) = restored
            .to_query_graph_with_schemas(supergraph_schema.clone(), Some(existing_subgraphs))
            .map_err(|e| format!("reconstruction error: {e}"))?;
        let reconstruct_reuse_ms = t2.elapsed().as_millis();

        // Also measure without reused schemas (cold load from artifact only).
        let t3 = Instant::now();
        let restored2 = PortableQueryGraph::from_bytes(&bincode_bytes)
            .map_err(|e| format!("bincode deserialize error: {e}"))?;
        let (_restored_graph2, timing_cold) = restored2
            .to_query_graph_with_timing(supergraph_schema)
            .map_err(|e| format!("reconstruction error: {e}"))?;
        let reconstruct_cold_ms = t3.elapsed().as_millis();

        let timing_detail = format!(
            "ser: bincode={bincode_ser_ms}ms json={json_ser_ms}ms | \
             cold: {}ms (parse_sg={}ms cond={}ms graph={}ms idx={}ms sem={}ms nlm={}ms) | \
             warm: {}ms (parse_sg={}ms cond={}ms graph={}ms idx={}ms sem={}ms nlm={}ms)",
            reconstruct_cold_ms,
            timing_cold.parse_subgraphs_ms, timing_cold.reconstruct_conditions_ms,
            timing_cold.rebuild_graph_ms,
            timing_cold.restore_indices_ms, timing_cold.compute_semantic_indices_ms,
            timing_cold.compute_non_local_metadata_ms,
            reconstruct_reuse_ms,
            timing.parse_subgraphs_ms, timing.reconstruct_conditions_ms,
            timing.rebuild_graph_ms,
            timing.restore_indices_ms, timing.compute_semantic_indices_ms,
            timing.compute_non_local_metadata_ms,
        );

        Ok((bincode_ser_ms, reconstruct_reuse_ms, bincode_size, json_size, node_count, edge_count, timing_detail))
    }

    fn make_config(cfg: &CommonConfig) -> QueryPlannerConfig {
        let max_evaluated_plans =
            NonZeroU32::new(cfg.max_evaluated_plans.max(1)).unwrap_or(NonZeroU32::new(1).unwrap());

        QueryPlannerConfig {
            generate_query_fragments: cfg.generate_query_fragments,
            subgraph_graphql_validation: cfg.subgraph_validation,
            incremental_delivery: QueryPlanIncrementalDeliveryConfig {
                enable_defer: cfg.incremental_delivery,
            },
            debug: QueryPlannerDebugConfig {
                max_evaluated_plans,
                paths_limit: None,
            },
            type_conditioned_fetching: cfg.type_conditioned_fetching,
        }
    }
}

impl PlannerHarness for HeadPlanner {
    fn version_label() -> &'static str {
        VERSION
    }

    fn build(supergraph_sdl: &str, cfg: &CommonConfig) -> Result<Self, HarnessError> {
        let supergraph =
            Supergraph::new_with_router_specs(supergraph_sdl).map_err(|e| HarnessError::Supergraph {
                version: VERSION,
                detail: e.to_string(),
            })?;

        let planner_cfg = Self::make_config(cfg);

        let inner = QueryPlanner::new(&supergraph, planner_cfg.clone()).map_err(|e| {
            HarnessError::Construct {
                version: VERSION,
                detail: e.to_string(),
            }
        })?;

        Ok(Self {
            supergraph,
            config: planner_cfg,
            inner,
        })
    }

    fn plan(
        &self,
        operation: &str,
        operation_name: Option<&str>,
        opts: &CommonOptions,
    ) -> Result<serde_json::Value, HarnessError> {
        let api_schema = self.inner.api_schema();
        let document = ExecutableDocument::parse_and_validate(
            api_schema.schema(),
            operation,
            "operation.graphql",
        )
        .map_err(|e| HarnessError::Operation {
            version: VERSION,
            detail: e.to_string(),
        })?;

        let op_name = operation_name
            .map(|n| {
                apollo_compiler::Name::new(n).map_err(|e| HarnessError::Operation {
                    version: VERSION,
                    detail: format!("invalid operation name: {e}"),
                })
            })
            .transpose()?;

        let options = QueryPlanOptions {
            override_conditions: opts.override_conditions.clone(),
            check_for_cooperative_cancellation: None,
            non_local_selections_limit_enabled: opts.non_local_selections_limit,
            disabled_subgraph_names: opts.disabled_subgraph_names.iter().cloned().collect::<IndexSet<_>>(),
        };

        let plan = self
            .inner
            .build_query_plan(&document, op_name, options)
            .map_err(|e| HarnessError::Plan {
                version: VERSION,
                detail: e.to_string(),
            })?;

        serde_json::to_value(&plan).map_err(|e| HarnessError::Serialize {
            version: VERSION,
            detail: e.to_string(),
        })
    }
}
