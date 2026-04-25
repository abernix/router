use std::sync::Arc;

use apollo_compiler::Name;
use apollo_compiler::Node;
use apollo_compiler::ast::Type;
use apollo_compiler::collections::IndexMap;
use parking_lot::Mutex;
use petgraph::graph::EdgeIndex;

use crate::error::FederationError;
use crate::operation::SelectionSet;
use crate::query_graph::QueryGraph;
use crate::query_graph::SemanticEdgeId;
use crate::query_graph::graph_path::ExcludedConditions;
use crate::query_graph::graph_path::ExcludedDestinations;
use crate::query_graph::graph_path::operation::OpGraphPathContext;
use crate::query_graph::path_tree::OpPathTree;
use crate::query_plan::QueryPlanCost;

#[derive(Debug, Clone)]
pub(crate) struct ContextMapEntry {
    pub(crate) levels_in_data_path: usize,
    pub(crate) levels_in_query_path: usize,
    pub(crate) path_tree: Option<Arc<OpPathTree>>,
    pub(crate) selection_set: SelectionSet,
    // PORT_NOTE: This field was renamed from the JS name (`paramName`) to better align with naming
    // in ContextCondition.
    pub(crate) argument_name: Name,
    // PORT_NOTE: This field was renamed from the JS name (`argType`) to better align with naming in
    // ContextCondition.
    pub(crate) argument_type: Node<Type>,
    pub(crate) context_id: Name,
}

/// Note that `ConditionResolver`s are guaranteed to be only called for edge with conditions.
pub(crate) trait ConditionResolver {
    fn resolve(
        &mut self,
        edge: EdgeIndex,
        context: &OpGraphPathContext,
        excluded_destinations: &ExcludedDestinations,
        excluded_conditions: &ExcludedConditions,
        extra_conditions: Option<&SelectionSet>,
    ) -> Result<ConditionResolution, FederationError>;
}

#[derive(Debug, Clone)]
pub(crate) enum ConditionResolution {
    Satisfied {
        cost: QueryPlanCost,
        path_tree: Option<Arc<OpPathTree>>,
        context_map: Option<IndexMap<Name, ContextMapEntry>>,
    },
    Unsatisfied {
        reason: Option<UnsatisfiedConditionReason>,
    },
}

impl std::fmt::Display for ConditionResolution {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            ConditionResolution::Satisfied {
                cost,
                path_tree,
                context_map,
            } => {
                writeln!(f, "Satisfied: cost={cost}")?;
                if let Some(path_tree) = path_tree {
                    writeln!(f, "path_tree:\n{path_tree}")?;
                }
                if let Some(context_map) = context_map {
                    writeln!(f, ", context_map:\n{context_map:?}")?;
                }
                Ok(())
            }
            ConditionResolution::Unsatisfied { reason } => {
                writeln!(f, "Unsatisfied: reason={reason:?}")
            }
        }
    }
}

#[derive(Debug, Clone)]
pub(crate) enum UnsatisfiedConditionReason {
    NoPostRequireKey,
    NoSetContext,
}

impl ConditionResolution {
    pub(crate) fn no_conditions() -> Self {
        Self::Satisfied {
            cost: 0.0,
            path_tree: None,
            context_map: None,
        }
    }

    pub(crate) fn unsatisfied_conditions() -> Self {
        Self::Unsatisfied { reason: None }
    }
}

#[derive(Debug, derive_more::IsVariant)]
pub(crate) enum ConditionResolutionCacheResult {
    /// Cache hit.
    Hit(ConditionResolution),
    /// Cache miss; can be inserted into cache.
    Miss,
    /// The value can't be cached; Or, an incompatible value is already in cache.
    NotApplicable,
}

pub(crate) struct ConditionResolverCache {
    // For every edge having a condition, we cache the resolution its conditions when possible.
    // We save resolution with the set of excluded edges that were used to compute it: the reason we do this is
    // that excluded edges impact the resolution, so we should only used a cached value if we know the excluded
    // edges are the same as when caching, and while we could decide to cache only when we have no excluded edges
    // at all, this would sub-optimal for types that have multiple keys, as the algorithm will always at least
    // include the previous key edges to the excluded edges of other keys. In other words, if we only cached
    // when we have no excluded edges, we'd only ever use the cache for the first key of every type. However,
    // as the algorithm always try keys in the same order (the order of the edges in the query graph), including
    // the excluded edges we see on the first ever call is actually the proper thing to do.
    //
    // Keyed on `SemanticEdgeId` (schema-stable) rather than `EdgeIndex` (petgraph-internal) so
    // that cached entries can survive `QueryGraph` reconstruction across schema reloads.
    edge_states: IndexMap<SemanticEdgeId, (ConditionResolution, ExcludedDestinations)>,
}

impl ConditionResolverCache {
    pub(crate) fn new() -> Self {
        Self {
            edge_states: Default::default(),
        }
    }

    /// Check if a cached resolution exists for the given semantic edge id.
    ///
    /// Guard conditions: returns `NotApplicable` if `extra_conditions` is set, `context` is
    /// non-empty, or `excluded_conditions` is non-empty. These cases are not safe to cache.
    pub(crate) fn contains(
        &self,
        semantic_id: &SemanticEdgeId,
        context: &OpGraphPathContext,
        excluded_destinations: &ExcludedDestinations,
        excluded_conditions: &ExcludedConditions,
        extra_conditions: Option<&SelectionSet>,
    ) -> ConditionResolutionCacheResult {
        if extra_conditions.is_some() {
            return ConditionResolutionCacheResult::NotApplicable;
        }
        // We don't cache if there is a context or excluded conditions because those would impact the resolution and
        // we don't want to cache a value per-context and per-excluded-conditions (we also don't cache per-excluded-edges though
        // instead we cache a value only for the first-see excluded edges; see above why that work in practice).
        // TODO: we could actually have a better handling of the context: it doesn't really change how we'd resolve the condition, it's only
        // that the context, if not empty, would have to be added to the trigger of key edges in the resolution path tree when appropriate
        // and we currently don't handle that. But we could cache with an empty context, and then apply the proper transformation on the
        // cached value `pathTree` when the context is not empty. That said, the context is about active @include/@skip and it's not use
        // that commonly, so this is probably not an urgent improvement.
        if !context.is_empty() || !excluded_conditions.is_empty() {
            return ConditionResolutionCacheResult::NotApplicable;
        }

        if let Some((cached_resolution, cached_excluded_destinations)) =
            self.edge_states.get(semantic_id)
        {
            // Cache hit.
            // Ensure we have the same excluded destinations as when we cached the value.
            if cached_excluded_destinations == excluded_destinations {
                return ConditionResolutionCacheResult::Hit(cached_resolution.clone());
            }
            // Otherwise, fall back to non-cached computation
            ConditionResolutionCacheResult::NotApplicable
        } else {
            // Cache miss
            ConditionResolutionCacheResult::Miss
        }
    }

    pub(crate) fn insert(
        &mut self,
        semantic_id: SemanticEdgeId,
        resolution: ConditionResolution,
        excluded_destinations: ExcludedDestinations,
    ) {
        self.edge_states
            .insert(semantic_id, (resolution, excluded_destinations));
    }
}

/// A thread-safe, shared condition resolver cache that persists across query planning invocations
/// for the lifetime of a `QueryPlanner` (i.e., for a single schema version).
///
/// The cache is keyed on `SemanticEdgeId` (schema-stable identity) rather than `EdgeIndex`
/// (petgraph-internal numeric index), enabling cache carryover across schema reloads.
/// `ExcludedDestinations` (which uses subgraph names, not NodeIndex) is stored alongside each
/// entry as a secondary match criterion.
///
/// Guard conditions from the underlying `ConditionResolverCache` still apply: entries are only
/// cached when context is empty, excluded_conditions is empty, and extra_conditions is None.
#[derive(Clone)]
pub struct SharedConditionResolverCache {
    inner: Arc<Mutex<ConditionResolverCache>>,
}

impl SharedConditionResolverCache {
    pub(crate) fn new() -> Self {
        Self {
            inner: Arc::new(Mutex::new(ConditionResolverCache::new())),
        }
    }

    pub(crate) fn contains(
        &self,
        semantic_id: &SemanticEdgeId,
        context: &OpGraphPathContext,
        excluded_destinations: &ExcludedDestinations,
        excluded_conditions: &ExcludedConditions,
        extra_conditions: Option<&SelectionSet>,
    ) -> ConditionResolutionCacheResult {
        self.inner.lock().contains(
            semantic_id,
            context,
            excluded_destinations,
            excluded_conditions,
            extra_conditions,
        )
    }

    pub(crate) fn insert(
        &self,
        semantic_id: SemanticEdgeId,
        resolution: ConditionResolution,
        excluded_destinations: ExcludedDestinations,
    ) {
        self.inner
            .lock()
            .insert(semantic_id, resolution, excluded_destinations);
    }

    /// Returns the number of cached condition resolutions.
    pub(crate) fn len(&self) -> usize {
        self.inner.lock().edge_states.len()
    }

    /// Returns a clone of all cached entries, for cache carryover across schema reloads.
    pub(crate) fn entries(
        &self,
    ) -> IndexMap<SemanticEdgeId, (ConditionResolution, ExcludedDestinations)> {
        self.inner.lock().edge_states.clone()
    }

    /// Import cache entries from a previous schema version's cache.
    ///
    /// Only entries whose `SemanticEdgeId` exists in `new_graph` are imported.
    /// Entries with non-None `path_tree` are skipped because the `OpPathTree`
    /// references `NodeIndex`/`EdgeIndex` values from the old graph which would
    /// be invalid in the new graph. These entries will be lazily recomputed on
    /// first access.
    ///
    /// Returns the number of entries imported.
    pub(crate) fn import_from(
        &self,
        old_cache: &SharedConditionResolverCache,
        new_graph: &crate::query_graph::QueryGraph,
    ) -> usize {
        let old_entries = old_cache.entries();
        let mut imported = 0;

        let mut inner = self.inner.lock();
        for (semantic_id, (resolution, excluded_destinations)) in old_entries {
            // Skip if the edge no longer exists in the new graph
            if new_graph.edge_index_for_semantic_id(&semantic_id).is_none() {
                continue;
            }

            // Only import entries that don't reference old graph indices.
            // Satisfied entries with path_tree: Some(...) contain NodeIndex/EdgeIndex
            // from the old graph — these are not valid in the new graph.
            let is_portable = match &resolution {
                ConditionResolution::Unsatisfied { .. } => true,
                ConditionResolution::Satisfied {
                    path_tree: None, ..
                } => true,
                ConditionResolution::Satisfied {
                    path_tree: Some(_), ..
                } => false,
            };

            if is_portable {
                inner.insert(semantic_id, resolution, excluded_destinations);
                imported += 1;
            }
        }

        imported
    }
}

/// A query plan resolver for edge conditions that caches the outcome per edge.
// PORT_NOTE: This ports the `cachingConditionResolver` function from JS. In JS version, the
//            function creates a closure capturing the QueryPlanningTraversal/ValidationTraversal
//            instance itself The same would be infeasible to implement in Rust due to the cyclic
//            references. Instead, in Rust, it is implemented as `CachingConditionResolver` and
//            `ConditionResolver` traits that will be implemented by `QueryPlanningTraversal` and
//            `ValidationTraversal` structs.
pub(crate) trait CachingConditionResolver {
    fn query_graph(&self) -> &QueryGraph;

    fn resolve_without_cache(
        &self,
        edge: EdgeIndex,
        context: &OpGraphPathContext,
        excluded_destinations: &ExcludedDestinations,
        excluded_conditions: &ExcludedConditions,
        extra_conditions: Option<&SelectionSet>,
    ) -> Result<ConditionResolution, FederationError>;

    fn resolver_cache(&self) -> &SharedConditionResolverCache;

    fn resolve_with_cache(
        &mut self,
        edge: EdgeIndex,
        context: &OpGraphPathContext,
        excluded_destinations: &ExcludedDestinations,
        excluded_conditions: &ExcludedConditions,
        extra_conditions: Option<&SelectionSet>,
    ) -> Result<ConditionResolution, FederationError> {
        // Convert EdgeIndex to SemanticEdgeId for the cache lookup.
        // If the edge has no semantic ID (shouldn't happen for condition edges),
        // fall through to uncached resolution.
        let semantic_id = self.query_graph().semantic_edge_id(edge).cloned();

        let cache_result = if let Some(ref sid) = semantic_id {
            self.resolver_cache().contains(
                sid,
                context,
                excluded_destinations,
                excluded_conditions,
                extra_conditions,
            )
        } else {
            ConditionResolutionCacheResult::NotApplicable
        };

        if let ConditionResolutionCacheResult::Hit(cached_resolution) = cache_result {
            return Ok(cached_resolution);
        }

        let resolution = self.resolve_without_cache(
            edge,
            context,
            excluded_destinations,
            excluded_conditions,
            extra_conditions,
        )?;
        // See if this resolution is eligible to be inserted into the cache.
        if cache_result.is_miss() {
            if let Some(sid) = semantic_id {
                self.resolver_cache()
                    .insert(sid, resolution.clone(), excluded_destinations.clone());
            }
        }
        Ok(resolution)
    }
}

/// Blanket implementation of `ConditionResolver` for any type that implements
/// `CachingConditionResolver`.
impl<T: CachingConditionResolver> ConditionResolver for T {
    fn resolve(
        &mut self,
        edge: EdgeIndex,
        context: &OpGraphPathContext,
        excluded_destinations: &ExcludedDestinations,
        excluded_conditions: &ExcludedConditions,
        extra_conditions: Option<&SelectionSet>,
    ) -> Result<ConditionResolution, FederationError> {
        // Invariant check: The edge must have conditions.
        let graph = &self.query_graph();
        let edge_data = graph.edge_weight(edge)?;
        assert!(
            edge_data.conditions.is_some() || extra_conditions.is_some(),
            "Should not have been called for edge without conditions"
        );

        self.resolve_with_cache(
            edge,
            context,
            excluded_destinations,
            excluded_conditions,
            extra_conditions,
        )
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::query_graph::SemanticNodeId;
    use crate::query_graph::SemanticTransitionKind;
    use crate::query_graph::graph_path::operation::OpGraphPathContext;

    /// Helper to create a test SemanticEdgeId.
    fn test_semantic_edge(id: u32) -> SemanticEdgeId {
        let name = apollo_compiler::Name::new_unchecked(&*format!("Type{id}"));
        SemanticEdgeId {
            head: SemanticNodeId {
                type_name: name.clone(),
                source: format!("subgraph_{id}").into(),
                provide_id: None,
            },
            tail: SemanticNodeId {
                type_name: name,
                source: format!("subgraph_{}", id + 1).into(),
                provide_id: None,
            },
            transition: SemanticTransitionKind::KeyResolution,
            conditions_hash: id as u64,
            override_condition: None,
        }
    }

    #[test]
    fn test_condition_resolver_cache() {
        let mut cache = ConditionResolverCache::new();

        let edge1 = test_semantic_edge(1);
        let empty_context = OpGraphPathContext::default();
        let empty_destinations = ExcludedDestinations::default();
        let empty_conditions = ExcludedConditions::default();

        assert!(
            cache
                .contains(
                    &edge1,
                    &empty_context,
                    &empty_destinations,
                    &empty_conditions,
                    None
                )
                .is_miss()
        );

        cache.insert(
            edge1.clone(),
            ConditionResolution::unsatisfied_conditions(),
            empty_destinations.clone(),
        );

        assert!(
            cache
                .contains(
                    &edge1,
                    &empty_context,
                    &empty_destinations,
                    &empty_conditions,
                    None
                )
                .is_hit(),
        );

        let edge2 = test_semantic_edge(2);

        assert!(
            cache
                .contains(
                    &edge2,
                    &empty_context,
                    &empty_destinations,
                    &empty_conditions,
                    None
                )
                .is_miss()
        );
    }

    #[test]
    fn test_shared_cache_clones_share_state() {
        let cache = SharedConditionResolverCache::new();
        let cache2 = cache.clone();

        let edge1 = test_semantic_edge(1);
        let empty_context = OpGraphPathContext::default();
        let empty_destinations = ExcludedDestinations::default();
        let empty_conditions = ExcludedConditions::default();

        // Insert via first handle
        cache.insert(
            edge1.clone(),
            ConditionResolution::unsatisfied_conditions(),
            empty_destinations.clone(),
        );

        // Read via second handle — should see the entry
        assert!(
            cache2
                .contains(
                    &edge1,
                    &empty_context,
                    &empty_destinations,
                    &empty_conditions,
                    None
                )
                .is_hit()
        );

        assert_eq!(cache.len(), 1);
        assert_eq!(cache2.len(), 1);
    }
}
